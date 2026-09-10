"""Re-rotate cached K so a block can be reused at a different absolute position.

RoPE is a rotation whose angle is linear in the position, so ``R(a) . R(b) == R(a+b)``
and a key cached at ``p`` can be moved to ``p + delta`` by applying ``R(delta)`` to it.
Attention applies RoPE last, immediately before writing the pool, so the pool holds
``R(p) . f(k_raw)`` for whatever normalisation ``f`` the model uses, and

    R(p_new) . f(k_raw) == R(p_new - p) . k_cached[loc]

*exactly* -- ``f`` sits before the rotation and RoPE preserves norms, so there is
nothing to re-run. V carries no position at all and is copied unchanged. Whether a
given model's RoPE really composes like this is measured at startup by
:func:`rope_delta_composable_reason`, not assumed.

This fixes the *position*. It does not fix the *context*: the block's hidden states
were produced under whatever prefix preceded it when it was computed, and no rotation
can restore that. Reusing a rotated block therefore trades some output quality for the
prefill it skips, which is what the pass@1 arm of the experiment measures.

Where the result lands follows one rule: **source owned by the tree, copy; source
owned by this request, rotate in place.** The read paths (the stitch, and the Stage 2
append) rotate a node's KV that other requests hold a ``lock_ref`` on, so they must
allocate; rotating that in place would corrupt every request hitting the same node at
its canonical position. The write paths (re-filing a declined block, and moving the
generated tail to follow it) rotate slots the finished request allocated itself, and
do so in place -- guarded by ``_tree_held_mask``, because a declined block can still
carry a node's own slots.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import host_timer
from sglang.srt.utils.device_timer import DeviceTimer

logger = logging.getLogger(__name__)

# GPU time for the rotation kernel, reported as the ``subctx_rotate_gpu`` stage.
# Off by default and deliberately: it costs a CUDA event pair per call, which lands
# inside the ``subctx_rotate`` host stage measured right beside it. Turn it on for a
# run whose question is what the rotation costs on the GPU, and read the host numbers
# from a run with it off.
_TIME_ROTATE_GPU = bool(os.environ.get("SGLANG_SUBCTX_ROTATE_GPU", ""))


@triton.jit
def _rotate_copy_kv_kernel(
    k_ptrs,  # int64[num_layers], one device pointer per layer's k_buffer
    v_ptrs,  # int64[num_layers]
    k_ref,  # a representative k_buffer; used only for its element type
    v_ref,
    src_loc_ptr,
    dst_loc_ptr,
    cos_ptr,  # fp32[half], the single delta's row of cos_sin_cache
    sin_ptr,  # fp32[half], already negated when delta < 0
    head_num,
    half,
    v_row_elems,
    K_ROW: tl.constexpr,  # head_num * head_dim, elements per K row
    HEAD_DIM: tl.constexpr,
    HEADS_P2: tl.constexpr,
    HALF_P2: tl.constexpr,
    VROW_P2: tl.constexpr,
):
    """One program per (token, layer): rotate that row's K, copy its V.

    Grid is ``(num_locs, num_layers)`` rather than the other way round because CUDA
    caps grid dims 1 and 2 at 65535, and a block can be longer than that while the
    layer count cannot.
    """
    i = tl.program_id(0)
    layer = tl.program_id(1)

    k_elem_ty = k_ref.dtype.element_ty
    v_elem_ty = v_ref.dtype.element_ty
    kb = tl.cast(tl.load(k_ptrs + layer), tl.pointer_type(k_elem_ty))
    vb = tl.cast(tl.load(v_ptrs + layer), tl.pointer_type(v_elem_ty))

    src = tl.load(src_loc_ptr + i).to(tl.int64)
    dst = tl.load(dst_loc_ptr + i).to(tl.int64)

    h = tl.arange(0, HEADS_P2)
    d = tl.arange(0, HALF_P2)
    dm = d < half
    m = (h < head_num)[:, None] & dm[None, :]

    cos = tl.load(cos_ptr + d, mask=dm, other=0.0)
    sin = tl.load(sin_ptr + d, mask=dm, other=0.0)

    # neox pairing: element d of a head rotates against element d + rotary_dim/2.
    lo = h[:, None] * HEAD_DIM + d[None, :]
    hi = lo + half

    k_src = kb + src * K_ROW
    k_dst = kb + dst * K_ROW

    x1 = tl.load(k_src + lo, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(k_src + hi, mask=m, other=0.0).to(tl.float32)

    # Accumulate in fp32: the buffers are bf16, and rounding the product terms
    # separately would cost more than the single rounding on store.
    y1 = x1 * cos[None, :] - x2 * sin[None, :]
    y2 = x2 * cos[None, :] + x1 * sin[None, :]

    tl.store(k_dst + lo, y1.to(k_elem_ty), mask=m)
    tl.store(k_dst + hi, y2.to(k_elem_ty), mask=m)

    vd = tl.arange(0, VROW_P2)
    vm = vd < v_row_elems
    tl.store(
        vb + dst * v_row_elems + vd,
        tl.load(vb + src * v_row_elems + vd, mask=vm, other=0),
        mask=vm,
    )


def delta_cos_sin(
    cos_sin_cache: torch.Tensor, rotary_dim: int, delta: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of ``R(delta)`` as a *unit* rotation, read off the model's own cache.

    A negative delta indexes the cache at ``|delta|`` with the sine negated -- the same
    rotation run backwards, since the cache only holds non-negative rows.

    The row is then normalised per dimension. A plain cache already has unit rows (the
    division is a no-op within fp32 noise), but the YaRN family stores ``mscale * cos``
    and ``mscale * sin`` (``rotary_embedding.py:662-663``). That scalar is already baked
    into the cached K we are about to rotate, so using the row as stored would multiply
    K by ``mscale`` again on every hop. Dividing it out is exact rather than a
    correction: mscale is a scalar, it commutes with the rotation, and it belongs to the
    stored key, not to the delta.
    """
    half = rotary_dim // 2
    row = cos_sin_cache[abs(delta)].float()
    cos = row[:half]
    sin = row[half:rotary_dim]
    if delta < 0:
        sin = -sin
    scale = torch.sqrt(cos * cos + sin * sin)
    return (cos / scale).contiguous(), (sin / scale).contiguous()


def apply_delta_rows(
    rows: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """``R(delta)`` applied to the trailing dim of ``rows``, in fp32. Neox pairing."""
    half = cos.numel()
    src = rows.to(torch.float32)
    x1, x2 = src[..., :half], src[..., half : 2 * half]
    out = torch.empty_like(src)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half : 2 * half] = x2 * cos + x1 * sin
    if src.shape[-1] > 2 * half:  # untouched tail when rotary_dim < head_dim
        out[..., 2 * half :] = src[..., 2 * half :]
    return out


def rotate_copy_kv_native(
    k_buffer,
    v_buffer,
    dst_loc: torch.Tensor,
    src_loc: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Pure-torch reference for :func:`rotate_copy_kv`. Same math, no kernel.

    Kept because it is the oracle the Triton kernel is tested against, and it is the
    only path that runs on CPU (where the sub-context unit tests live).
    """
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        out = apply_delta_rows(k_cache[src_loc], cos, sin)
        k_cache[dst_loc] = out.to(k_cache.dtype)
        v_cache[dst_loc] = v_cache[src_loc]


class KVRotator:
    """Copies a block of cached KV to new slots, rotating K by a position delta.

    Built once per scheduler when the model's RoPE is delta-composable; see
    ``subctx_config.rotation_unsupported_reason`` for what that rules out.
    """

    def __init__(
        self,
        kv_pool,
        cos_sin_cache: torch.Tensor,
        rotary_dim: int,
        use_native: bool = False,
    ):
        self.pool = kv_pool
        self.cos_sin_cache = cos_sin_cache
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2
        # cos_sin_cache is [max_position, rotary_dim]; row `d` is the rotation by d.
        self.max_delta = cos_sin_cache.shape[0] - 1
        self.use_native = use_native or not cos_sin_cache.is_cuda
        self.rotated_tokens = 0
        # Elapsed time is read on a later call, once the end event has completed --
        # never by synchronising, which would stall the very path being priced. The
        # last few calls of a run therefore go unreported; over a run of thousands
        # that is noise, and a blocking read would not be.
        self._gpu_timer = (
            DeviceTimer(reporter=self._report_gpu)
            if _TIME_ROTATE_GPU and host_timer.armed() and not self.use_native
            else None
        )

    @staticmethod
    def _report_gpu(t: float, **metadata) -> None:
        host_timer.add("subctx_rotate_gpu", int(t * 1e9))

    def can_rotate(self, delta: int) -> bool:
        """Whether ``delta`` is representable. delta == 0 needs no rotation at all."""
        return delta != 0 and abs(delta) <= self.max_delta

    def _cos_sin(self, delta: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for ``delta``, reusing the model's own fp32 cos_sin_cache."""
        return delta_cos_sin(self.cos_sin_cache, self.rotary_dim, delta)

    def rotate_into(
        self,
        dst_loc: torch.Tensor,
        src_loc: torch.Tensor,
        delta: int,
        stage: str = "subctx_rotate",
    ) -> None:
        """Write ``R(delta)`` applied to the K at ``src_loc`` (and V verbatim) to ``dst_loc``.

        ``dst_loc`` must come from the allocator: the caller owns those slots and is
        responsible for freeing them. ``src_loc`` is tree-owned and is never written.

        ``stage`` names the host timer this call is charged to. It exists because the
        two call families answer different questions: rotating on the read path is
        latency a request pays before its own prefill, while rotating at finish is
        book-keeping for whoever comes next. Summing them would hide which.
        """
        assert dst_loc.numel() == src_loc.numel(), (
            f"{dst_loc.numel()=} != {src_loc.numel()=}"
        )
        n = dst_loc.numel()
        if n == 0:
            return
        assert delta != 0, "delta == 0 must take the zero-cost path, not the kernel"

        pool = self.pool

        # Timed here rather than at the call sites: every path that rotates goes
        # through this method, so one probe prices the whole mechanism and no caller
        # can be added later that quietly escapes it.
        gpu = (
            self._gpu_timer.wrap(metadata={"tokens": n})
            if self._gpu_timer is not None
            else nullcontext()
        )
        with host_timer.record(stage), gpu:
            cos, sin = self._cos_sin(delta)

            if self.use_native:
                rotate_copy_kv_native(
                    pool.k_buffer, pool.v_buffer, dst_loc, src_loc, cos, sin
                )
            else:
                head_num = pool.head_num
                head_dim = pool.head_dim
                v_row_elems = head_num * pool.v_head_dim
                _rotate_copy_kv_kernel[(n, len(pool.k_buffer))](
                    pool.k_data_ptrs,
                    pool.v_data_ptrs,
                    pool.k_buffer[0],
                    pool.v_buffer[0],
                    src_loc,
                    dst_loc,
                    cos,
                    sin,
                    head_num,
                    self.half,
                    v_row_elems,
                    K_ROW=head_num * head_dim,
                    HEAD_DIM=head_dim,
                    HEADS_P2=triton.next_power_of_2(head_num),
                    HALF_P2=triton.next_power_of_2(self.half),
                    VROW_P2=triton.next_power_of_2(v_row_elems),
                )
        self.rotated_tokens += n


def find_rotary_embedding(model) -> Tuple[Optional[object], Optional[str]]:
    """Return the model's single ``RotaryEmbedding``, or why there isn't one.

    Walks the live model rather than ``rotary_embedding._ROPE_DICT`` because that dict
    is process-global and would also hold a draft model's entry under speculative
    decoding. All layers share one instance via the ``_ROPE_DICT`` memo, so finding
    more than one distinct object means the model mixes rotations and a single delta
    would be wrong for some layer.

    Subclasses are deliberately *not* filtered here. Whether a particular RoPE composes
    under a delta is measured at startup by :func:`rope_delta_composable_reason`, which
    is both stricter than a class name and still right for classes that did not exist
    when this was written.
    """
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding

    found = []
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            if not any(module is f for f in found):
                found.append(module)
    if not found:
        return None, "the model has no RotaryEmbedding module"
    if len(found) > 1:
        return None, f"the model uses {len(found)} distinct RotaryEmbedding instances"
    return found[0], None


# (position, delta) samples for the startup self-test. Spread on purpose, and not only
# near the origin: a cache that is merely *locally* linear -- one that switches inv_freq
# past a threshold, or is several caches concatenated -- matches close to 0 and fails
# far from it. Pairs outside a short cache are skipped, not failed.
_SELFTEST_PAIRS = (
    (0, 1),
    (1, -1),
    (17, 37),
    (600, -400),
    (4096, 4096),
    (8192, -8000),
    (30000, 1000),
    (1000, 30000),
)


def rope_delta_composable_reason(rotary, tol: float = 1e-5) -> Optional[str]:
    """Measure ``R(p + d) . k == R(d) . (R(p) . k)`` on the model's own RoPE.

    This is a measurement standing in for a list of class names. The identity is the
    one thing the whole reuse rests on; it holds for reasons a class name only
    approximates (a per-dimension angle linear in the position, and a unit rotation once
    mscale is divided out), and a name says nothing about a subclass added later. So:
    RoPE a random key at ``p`` through the model's own ``forward_native``, apply our
    delta rotation to the result, and compare against the same key RoPE'd directly at
    ``p + d``.

    The budget is ``tol`` plus a position term, not a flat number. ``cos_sin_cache``
    holds ``position * inv_freq`` in fp32, so its row at position P already carries
    ~``P * 2**-24`` rad of rounding; the identity reads three such rows and cannot hold
    tighter than that. Measured: a plain RoPE with a *float64* cache matches to 8e-8 at
    every sample, and the same law with the shipped fp32 cache drifts to 6e-4 by
    position 30000 -- noise, not a broken law, and the position term is what tells the
    two apart. What is left over is wide: the smallest real breakage is a YaRN mscale
    left in the row, a 13.9% error at scaling factor 4.

    Returns None when every sample fits its budget, else a string naming the worst.
    """
    cache = rotary.cos_sin_cache
    max_pos = cache.shape[0]
    head_size, rotary_dim = rotary.head_size, rotary.rotary_dim
    heads = 2

    pairs = [
        (p, d)
        for p, d in _SELFTEST_PAIRS
        if 0 <= p < max_pos and 0 <= p + d < max_pos and abs(d) < max_pos
    ]
    if not pairs:
        return f"the cos_sin_cache is only {max_pos} rows; too short to self-test"

    device = cache.device
    gen = torch.Generator(device=device).manual_seed(0)
    key = torch.randn(
        len(pairs),
        heads * head_size,
        generator=gen,
        dtype=torch.float32,
        device=device,
    )
    at = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=device)
    shifted = torch.tensor([p + d for p, d in pairs], dtype=torch.long, device=device)

    with torch.no_grad():
        _, cached = rotary.forward_native(at, key.clone(), key.clone())
        _, target = rotary.forward_native(shifted, key.clone(), key.clone())

    worst_ratio, worst = 0.0, (pairs[0], 0.0, 0.0)
    for i, (p, d) in enumerate(pairs):
        cos, sin = delta_cos_sin(cache, rotary_dim, d)
        got = apply_delta_rows(cached[i].view(heads, head_size), cos, sin)
        want = target[i].view(heads, head_size).float()
        err = float((got - want).abs().max()) / max(float(want.abs().max()), 1e-3)
        budget = tol + 4.0 * max(abs(p), abs(p + d)) * 2**-24
        if err / budget > worst_ratio:
            worst_ratio, worst = err / budget, ((p, d), err, budget)

    (p, d), err, budget = worst
    if worst_ratio > 1.0:
        return (
            f"{type(rotary).__name__} is not delta-composable: a key cached at "
            f"position {p} and rotated by {d} differs from RoPE at {p + d} by "
            f"{err:.2e} relative, over a budget of {budget:.2e}"
        )
    logger.info(
        "RoPE delta self-test passed for %s on %d (position, delta) samples; worst was "
        "position %d delta %d at %.2e relative, %.0f%% of its %.2e budget",
        type(rotary).__name__,
        len(pairs),
        p,
        d,
        err,
        100.0 * worst_ratio,
        budget,
    )
    return None
