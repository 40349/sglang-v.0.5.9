"""Re-rotate cached K so a block can be reused at a different absolute position.

RoPE is a rotation whose angle is linear in the position, so ``R(a) . R(b) == R(a+b)``
and a key cached at ``p`` can be moved to ``p + delta`` by applying ``R(delta)`` to it.
Qwen3 computes ``qkv_proj -> q_norm/k_norm -> RoPE -> write pool``
(``models/qwen3.py:137-151``), so the pool holds ``R(p) . RMSNorm(k_raw)`` and

    R(p_new) . RMSNorm(k_raw) == R(p_new - p) . k_cached[loc]

*exactly* -- the norm sits before the rotation and RoPE preserves norms, so there is
nothing to re-run. V carries no position at all and is copied unchanged.

This fixes the *position*. It does not fix the *context*: the block's hidden states
were produced under whatever prefix preceded it when it was computed, and no rotation
can restore that. Reusing a rotated block therefore trades some output quality for the
prefill it skips, which is what the pass@1 arm of the experiment measures.

The rotation always writes to freshly allocated slots. The source slots belong to a
radix tree node that other requests hold a ``lock_ref`` on; rotating in place would
silently corrupt every request that hits the same node at its canonical position.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    half = cos.numel()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        src = k_cache[src_loc].to(torch.float32)
        x1, x2 = src[..., :half], src[..., half : 2 * half]
        out = torch.empty_like(src)
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half : 2 * half] = x2 * cos + x1 * sin
        if src.shape[-1] > 2 * half:  # untouched tail when rotary_dim < head_dim
            out[..., 2 * half :] = src[..., 2 * half :]
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

    def can_rotate(self, delta: int) -> bool:
        """Whether ``delta`` is representable. delta == 0 needs no rotation at all."""
        return delta != 0 and abs(delta) <= self.max_delta

    def _cos_sin(self, delta: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for ``delta``, reusing the model's own fp32 cos_sin_cache.

        Negative deltas index the cache at ``|delta|`` with the sine negated, which is
        the same rotation run backwards -- the cache only holds non-negative rows.
        """
        row = self.cos_sin_cache[abs(delta)]
        cos = row[: self.half].contiguous()
        sin = row[self.half : self.rotary_dim].contiguous()
        if delta < 0:
            sin = -sin
        return cos, sin

    def rotate_into(
        self, dst_loc: torch.Tensor, src_loc: torch.Tensor, delta: int
    ) -> None:
        """Write ``R(delta)`` applied to the K at ``src_loc`` (and V verbatim) to ``dst_loc``.

        ``dst_loc`` must come from the allocator: the caller owns those slots and is
        responsible for freeing them. ``src_loc`` is tree-owned and is never written.
        """
        assert dst_loc.numel() == src_loc.numel(), (
            f"{dst_loc.numel()=} != {src_loc.numel()=}"
        )
        n = dst_loc.numel()
        if n == 0:
            return
        assert delta != 0, "delta == 0 must take the zero-cost path, not the kernel"

        pool = self.pool
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
    """Return the model's single plain ``RotaryEmbedding``, or why there isn't one.

    Walks the live model rather than ``rotary_embedding._ROPE_DICT`` because that dict
    is process-global and would also hold a draft model's entry under speculative
    decoding. All layers share one instance via the ``_ROPE_DICT`` memo, so finding
    more than one distinct object means the model mixes rotations and a single delta
    would be wrong for some layer.
    """
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding

    found = []
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            if type(module) is not RotaryEmbedding:
                # Subclasses are not delta-composable in general: YaRN scales cos/sin
                # by mscale so cache[delta] is not a unit rotation, mrope's delta is a
                # 3-vector, Phi3LongRoPE switches inv_freq at a position threshold.
                return None, (
                    f"{type(module).__name__} is a scaled/variant RoPE whose "
                    "cos_sin_cache[delta] is not a plain rotation"
                )
            if not any(module is f for f in found):
                found.append(module)
    if not found:
        return None, "the model has no RotaryEmbedding module"
    if len(found) > 1:
        return None, f"the model uses {len(found)} distinct RotaryEmbedding instances"
    return found[0], None
