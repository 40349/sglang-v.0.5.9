"""RelayCaching (arXiv 2603.13289) on the sub-context chain storage.

Ported from the official code, github.com/YingshengGeng/RelayCaching (``RelayCache`` in
``RelayCaching/utils/relay_utils.py``, the hooks in ``utils/fixed_qwen3_hijack.py``, the
candidate hand-off in ``llm/gpt_chat.py``). The cached KV a scan finds plays the part of
the upstream agent's decoding KV; each contiguous reused span is one relay chunk.

Decode, per output token (hooks registered before CUDA graph capture):
- layer 0: the slot is marked decoded and its influence reset;
- every attention layer: softmax(q . k * scale) over the request's own output keys, the
  current one included, summed over heads, is added to each of those keys' influence;
- input of ``L_start`` (> 0): the residual stream is stored.

Prefill (``ModelRunner.forward_subctx_relay``), per request:
- before the forward, each span picks its last ``RELAY_SUFFIX`` tokens and its most
  attended-to tokens (``influence_preselect``);
- layers ``[0, L_start)`` run the fresh rows and the reused rows with no stored hidden
  state; at ``L_start`` the others join with their stored one;
- layers ``[L_start, L_det]`` run every row; at ``L_det`` the reused rows whose value
  moved most join the preselected ones (``deviation_select``);
- layers ``(L_det, L_end]`` run the fresh and selected rows, later layers the fresh rows.
Every row writes the K/V it computes into its own slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.mem_cache.subctx_blend import SCORE_CHUNK_ROWS, ProbeRows, probe_rows

# Influence candidates a span offers, its top scorers (``cand_k`` in gpt_chat.py).
CAND_K = 256
# Below this mean value deviation a request defers its selection to the next layer.
DEFER_EPS = 1e-6
# Out-start of a request pool entry that holds no request: past every sequence.
_NO_OUTPUT = 1 << 62


class RelayState:
    """Per-slot state decode leaves for a later prefill, and per-request output starts."""

    def __init__(
        self,
        num_slots: int,
        req_pool_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device,
        start_layer: int,
    ):
        self.start_layer = start_layer
        # Whether decode computed the slot's KV.
        self.decoded = torch.zeros(num_slots, dtype=torch.bool, device=device)
        # Attention the slot received from later output tokens of its request.
        self.influence = torch.zeros(num_slots, dtype=torch.float32, device=device)
        # Residual stream at the input of ``start_layer``, valid where ``decoded``.
        self.hidden = (
            torch.zeros((num_slots, hidden_size), dtype=dtype, device=device)
            if start_layer > 0
            else None
        )
        # Per request pool entry: the position of its first output token.
        self.out_start = torch.full(
            (req_pool_size,), _NO_OUTPUT, dtype=torch.int64, device=device
        )
        # Live tallies: requests prefilled with reused rows, their reused rows, those
        # loaded with a stored hidden state, those run past L_det, deferred selections.
        self.requests = 0
        self.reused_rows = 0
        self.warm_rows = 0
        self.selected_rows = 0
        self.deferred = 0
        # The profile file holds this server's requests only.
        from sglang.srt.utils import subctx_config

        if subctx_config.RELAY_PROFILE:
            open(subctx_config.RELAY_PROFILE, "w").close()

    def report(self) -> str:
        """One-line summary of the tallies."""
        if not self.reused_rows:
            return "RelayCaching: no reused rows yet"
        return (
            f"RelayCaching: {self.requests} requests, {self.reused_rows} reused rows, "
            f"{self.warm_rows} loaded with a stored hidden state, {self.selected_rows} "
            f"({100.0 * self.selected_rows / self.reused_rows:.1f}%) run past L_det, "
            f"{self.deferred} requests deferred their selection"
        )

    @staticmethod
    def bytes_per_token(hidden_size: int, dtype_size: int, start_layer: int) -> int:
        """Bytes one KV slot adds."""
        return 1 + 4 + (hidden_size * dtype_size if start_layer > 0 else 0)

    def mark_prefilled(self, loc: torch.Tensor) -> None:
        """Slots a prefill writes: no stored hidden state, no influence."""
        if loc.numel() == 0:
            return
        loc = loc.long()
        self.decoded.index_fill_(0, loc, False)
        self.influence.index_fill_(0, loc, 0.0)

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy the state at ``src`` to ``dst`` along with their KV."""
        self.decoded[dst] = self.decoded[src]
        self.influence[dst] = self.influence[src]
        if self.hidden is not None:
            self.hidden[dst] = self.hidden[src]

    def set_out_start(self, req_pool_indices: torch.Tensor, prompt_lens: List[int]) -> None:
        """Record where each request's output starts."""
        self.out_start[req_pool_indices] = torch.tensor(
            prompt_lens, dtype=torch.int64
        ).to(self.out_start.device, non_blocking=True)


# ---------------------------------------------------------------------------------------
# Decode: influence and hidden states
# ---------------------------------------------------------------------------------------


@triton.jit
def _influence_kernel(
    q_ptr,
    k_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    out_start_ptr,
    influence_ptr,
    stride_qb,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_r2t,
    scale,
    GROUP: tl.constexpr,
    GROUP_P2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One (request, KV head): softmax of the group's queries over the request's output
    keys, summed over the group, added to each key's influence. A row whose request pool
    entry no longer ends at the slot this step writes adds nothing.
    """
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    rpi = tl.load(req_pool_indices_ptr + b).to(tl.int64)
    hi = tl.load(seq_lens_ptr + b).to(tl.int64)
    lo = tl.load(out_start_ptr + rpi)
    row = req_to_token_ptr + rpi * stride_r2t
    last = tl.load(row + hi - 1).to(tl.int64)
    cur = tl.load(out_cache_loc_ptr + b).to(tl.int64)
    if (lo < hi) & (last == cur):
        offs_g = tl.arange(0, GROUP_P2)
        offs_d = tl.arange(0, BLOCK_D)
        mask_g = offs_g < GROUP
        mask_d = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr
            + b * stride_qb
            + (kv_head * GROUP + offs_g)[:, None] * stride_qh
            + offs_d[None, :],
            mask=mask_g[:, None] & mask_d[None, :],
            other=0.0,
        )

        m = tl.full([GROUP_P2], float("-inf"), dtype=tl.float32)
        s = tl.zeros([GROUP_P2], dtype=tl.float32)
        for start in range(lo, hi, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < hi
            slots = tl.load(row + offs_n, mask=mask_n, other=0).to(tl.int64)
            k = tl.load(
                k_ptr + slots[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)) * scale
            logits = tl.where(mask_n[None, :], logits, float("-inf"))
            m_new = tl.maximum(m, tl.max(logits, 1))
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new[:, None]), 1)
            m = m_new

        for start in range(lo, hi, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < hi
            slots = tl.load(row + offs_n, mask=mask_n, other=0).to(tl.int64)
            k = tl.load(
                k_ptr + slots[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)) * scale
            p = tl.exp(logits - m[:, None]) / s[:, None]
            p = tl.where(mask_g[:, None] & mask_n[None, :], p, 0.0)
            tl.atomic_add(influence_ptr + slots, tl.sum(p, 0), mask=mask_n)


def accumulate_influence(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    out_start: torch.Tensor,
    influence: torch.Tensor,
    scale: float,
) -> None:
    """Add one decode step's attention over each request's output keys to their
    influence. ``q`` is ``[bs, q_heads, head_dim]``, ``k_buffer`` one layer's keys,
    ``out_cache_loc`` the slot each row writes this step.
    """
    bs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    group = num_q_heads // num_kv_heads
    _influence_kernel[(bs, num_kv_heads)](
        q,
        k_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        out_cache_loc,
        out_start,
        influence,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        req_to_token.stride(0),
        scale,
        GROUP=group,
        GROUP_P2=max(16, triton.next_power_of_2(group)),
        HEAD_DIM=head_dim,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        BLOCK_N=64,
    )


def influence_reference(
    q,
    k_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    out_cache_loc,
    out_start,
    influence,
    scale,
) -> None:
    """``accumulate_influence`` in torch, one request at a time."""
    group = q.shape[1] // k_buffer.shape[1]
    for b in range(q.shape[0]):
        rpi = int(req_pool_indices[b])
        lo, hi = int(out_start[rpi]), int(seq_lens[b])
        if lo >= hi or int(req_to_token[rpi, hi - 1]) != int(out_cache_loc[b]):
            continue
        slots = req_to_token[rpi, lo:hi].long()
        keys = k_buffer[slots].float().repeat_interleave(group, dim=1)
        att = torch.einsum("hd,khd->hk", q[b].float(), keys) * scale
        influence.index_add_(0, slots, att.softmax(-1).sum(0))


def _layer_args(args, kwargs):
    """``(hidden_states, forward_batch, residual)`` of a decoder layer call
    ``layer(positions, hidden_states, forward_batch, residual, ...)``.
    """
    names = ("positions", "hidden_states", "forward_batch", "residual")
    values = list(args[:4]) + [kwargs.get(n) for n in names[len(args[:4]) :]]
    return values[1], values[2], values[3]


def register_relay_hooks(model, state: RelayState) -> None:
    """Hook decode into ``model``'s decoder layers and attention modules."""
    from sglang.srt.layers.radix_attention import RadixAttention

    layers = model.model.layers

    def reset_slot(_module, args, kwargs):
        _hidden, fb, _residual = _layer_args(args, kwargs)
        if fb.forward_mode.is_decode():
            loc = fb.out_cache_loc.long()
            state.decoded.index_fill_(0, loc, True)
            state.influence.index_fill_(0, loc, 0.0)

    def store_hidden(_module, args, kwargs):
        hidden, fb, residual = _layer_args(args, kwargs)
        if fb.forward_mode.is_decode():
            state.hidden.index_copy_(0, fb.out_cache_loc.long(), hidden + residual)

    def add_influence(module, args, kwargs, _output):
        q = args[0]
        fb = args[3] if len(args) > 3 else kwargs["forward_batch"]
        if not fb.forward_mode.is_decode():
            return
        accumulate_influence(
            q.reshape(-1, module.tp_q_head_num, module.qk_head_dim),
            fb.token_to_kv_pool.get_key_buffer(module.layer_id),
            fb.req_to_token_pool.req_to_token,
            fb.req_pool_indices,
            fb.seq_lens,
            fb.out_cache_loc,
            state.out_start,
            state.influence,
            module.scaling,
        )

    layers[0].register_forward_pre_hook(reset_slot, with_kwargs=True)
    if state.start_layer > 0:
        layers[state.start_layer].register_forward_pre_hook(store_hidden, with_kwargs=True)
    for layer in layers:
        for module in layer.modules():
            if isinstance(module, RadixAttention):
                module.register_forward_hook(add_influence, with_kwargs=True)


# ---------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------


def _sanitize(scores: torch.Tensor) -> torch.Tensor:
    """NaN and inf to 0, negatives to 0 (``_prepare_scores``)."""
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)


def _group_sum(values: torch.Tensor, group: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32, device=values.device).index_add_(
        0, group, values.float()
    )


def mean_threshold_mask(
    scores: torch.Tensor, group: torch.Tensor, n_groups: int, tau: float
) -> torch.Tensor:
    """Per group: the scores above ``tau`` times the group's mean
    (``get_mask_by_mean_threshold``), the sum, mean and cutoff rounded to the scores'
    dtype. A group summing below 1e-8 keeps none; ``tau`` at most 1e-6 keeps all.
    """
    if tau <= 1e-6:
        return torch.ones_like(scores, dtype=torch.bool)
    s = _sanitize(scores)
    total = _group_sum(s, group, n_groups).to(s.dtype)
    count = _group_sum(torch.ones_like(s), group, n_groups)
    cutoff = ((total.float() * (1.0 / count.clamp(min=1.0))).to(s.dtype).float() * tau).to(
        s.dtype
    )
    return (s > cutoff[group]) & (total >= 1e-8)[group]


def value_cosine(cached: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """Per row and head: ``cos(fresh, cached)`` in the KV dtype."""
    out = torch.empty(fresh.shape[:2], dtype=fresh.dtype, device=fresh.device)
    for lo in range(0, fresh.shape[0], SCORE_CHUNK_ROWS):
        hi = lo + SCORE_CHUNK_ROWS
        out[lo:hi] = F.cosine_similarity(fresh[lo:hi], cached[lo:hi], dim=-1)
    return out


def value_deviation(cached: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """Per row: the mean over heads of ``1 - cos(fresh, cached)``, in the KV dtype."""
    return (1.0 - value_cosine(cached, fresh)).mean(dim=1)


def deviation_select(
    cached: torch.Tensor,
    fresh: torch.Tensor,
    req_of_row: torch.Tensor,
    n_req: int,
    tau_dev: float,
    preselected: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose the reused rows to keep running past this layer (``_fused_deviation_select``).

    Computed in the KV dtype, like the official code on a bf16 model. Per request: if
    ``1 - mean cos`` (the mean rounded to that dtype) is below ``DEFER_EPS`` it keeps every
    row and selects again at the next layer (the second return, per request); in bf16
    that holds for any mean cosine above about 0.998. A request at or above it keeps the
    rows above ``tau_dev`` times the mean deviation, joined by the preselected ones; if
    none is above, none is kept, the preselected ones included.
    """
    cos = value_cosine(cached, fresh)
    score = (1.0 - cos).mean(dim=1)
    count = _group_sum(torch.ones_like(score), req_of_row, n_req)
    mean_cos = (
        _group_sum(cos.float().sum(dim=1), req_of_row, n_req)
        / (count * cos.shape[1]).clamp(min=1.0)
    ).to(cos.dtype)
    deferred = ((1.0 - mean_cos) < DEFER_EPS) & (count > 0)

    mask = mean_threshold_mask(score, req_of_row, n_req, tau_dev)
    some = _group_sum(mask, req_of_row, n_req) > 0
    keep = some[req_of_row] & (mask | preselected)
    keep |= deferred[req_of_row]
    return keep, deferred


def influence_preselect(
    influence: torch.Tensor,
    span_id: torch.Tensor,
    span_off: torch.Tensor,
    span_len: torch.Tensor,
    tau_inf: float,
    suffix: int,
    cand_k: int = CAND_K,
) -> torch.Tensor:
    """Per span: its last ``suffix`` rows and its most attended-to rows
    (``calc_keep_indices``).

    The span's top ``cand_k`` rows by influence are the candidates; ``p`` is the share of
    them above ``tau_inf`` times their mean, the budget ``ceil(len * p)``; the suffix rows
    count inside it and the rest goes to the candidates in rank order.
    """
    n = influence.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=influence.device)
    n_spans = span_len.numel()

    order = torch.argsort(influence, descending=True, stable=True)
    order = order[torch.argsort(span_id[order], stable=True)]
    sorted_span = span_id[order]
    span_first = torch.cumsum(span_len, 0) - span_len
    rank = torch.arange(n, device=influence.device) - span_first[sorted_span]

    n_cand = span_len.clamp(max=cand_k)
    cand = rank < n_cand[sorted_span]
    cand_scores = influence[order].float()
    if tau_inf <= 1e-6:
        opt_k = n_cand.float()
    else:
        s = torch.where(cand, _sanitize(cand_scores), 0.0)
        total = _group_sum(s, sorted_span, n_spans)
        cutoff = total / n_cand.clamp(min=1).float() * tau_inf
        passed = cand & (s > cutoff[sorted_span]) & (total >= 1e-8)[sorted_span]
        opt_k = _group_sum(passed, sorted_span, n_spans)
    p = (opt_k.double() / n_cand.clamp(min=1).double()).float()
    target = torch.ceil(span_len.float() * p).long()
    needed = torch.minimum((target - span_len.clamp(max=suffix)).clamp(min=0), n_cand)

    pick = torch.zeros(n, dtype=torch.bool, device=influence.device)
    pick[order] = rank < needed[sorted_span]
    tail = span_off >= (span_len[span_id] - suffix)
    return pick | tail


# ---------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------


@dataclass
class RelayPlan:
    """The probe rows of one extend batch and how their reused rows group."""

    rows: ProbeRows
    # (L_start, L_det, L_end); None in profile mode.
    layers: Optional[Tuple[int, int, int]]
    # Per reused row: its request, its span, its offset in the span; per span: length.
    req_of_reused: torch.Tensor
    span_id: torch.Tensor
    span_off: torch.Tensor
    span_len: torch.Tensor
    rids: List[str]
    prompt_lens: List[int]

    @property
    def batch_size(self) -> int:
        return len(self.rows.probe_lens)


def build_relay_plan(
    batch, out_cache_loc: torch.Tensor, layers: Optional[Tuple[int, int, int]]
) -> RelayPlan:
    """Build the plan for one extend batch; ``out_cache_loc`` holds the fresh slots."""
    rows = probe_rows(batch, out_cache_loc)
    device = out_cache_loc.device
    counts = [hi - lo for lo, hi in zip(rows.reused_ptr, rows.reused_ptr[1:])]
    lens = [n for spans in rows.span_lens for n in spans]
    assert sum(lens) == sum(counts) == rows.reused_rows.numel(), (
        f"spans cover {sum(lens)} reused rows, the requests {sum(counts)}"
    )
    span_len = torch.tensor(lens, dtype=torch.int64).to(device, non_blocking=True)
    req_of_reused = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64),
        torch.tensor(counts, dtype=torch.int64),
    ).to(device, non_blocking=True)
    span_id = torch.repeat_interleave(
        torch.arange(len(lens), device=device), span_len, output_size=sum(lens)
    )
    span_first = torch.cumsum(span_len, 0) - span_len
    span_off = torch.arange(sum(lens), device=device) - span_first[span_id]
    return RelayPlan(
        rows=rows,
        layers=layers,
        req_of_reused=req_of_reused,
        span_id=span_id,
        span_off=span_off,
        span_len=span_len,
        rids=[r.rid for r in batch.reqs],
        prompt_lens=[len(r.fill_ids) for r in batch.reqs],
    )


# ---------------------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------------------


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """0-based ranks of ``x``, tied values sharing their mean rank."""
    order = torch.argsort(x, stable=True)
    _, counts = torch.unique_consecutive(x[order], return_counts=True)
    ends = torch.cumsum(counts, 0)
    mean_rank = (ends - counts + ends - 1).double() / 2
    ranks = torch.empty(x.numel(), dtype=torch.float64, device=x.device)
    ranks[order] = torch.repeat_interleave(mean_rank, counts)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman's rho of two vectors with average ranks for ties (scipy's
    ``spearmanr``); 1.0 when either is constant or shorter than 2
    (``compute_spearman_correlations``).
    """
    if a.numel() < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 1.0
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    rho = float((ra * rb).sum() / (ra.norm() * rb.norm()))
    return 1.0 if math.isnan(rho) else rho


class RelayProfile:
    """Per contiguous reused span, per layer: how far the cached V of its rows sits from
    the recomputed one. A span is one sample, as one upstream reply is one calibration
    context in the official profiler. Two row groups: every row of the span, and its
    decoded rows only.

    Per layer and group: ``s_tok``, the mean over rows of the mean over heads of
    ``cos(cached, fresh)`` (the paper's Eq. 1-2); ``s_code``, the cosine along the row
    axis per (head, dim), averaged and rounded to the KV dtype (what the official
    profiler computes); ``rho``, per adjacent layer pair, Spearman's rho of the rows'
    ``1 - cos`` over the flattened heads x dim vector.
    """

    def __init__(self, plan: RelayPlan, decoded: torch.Tensor, num_layers: int):
        self.plan = plan
        self.num_layers = num_layers
        rows = plan.rows
        # Per span: (request, span index in the request, its row groups).
        self.spans = []
        for i in range(plan.batch_size):
            lo = rows.reused_ptr[i]
            for j, n in enumerate(rows.span_lens[i]):
                idx = torch.arange(lo, lo + n, device=decoded.device)
                self.spans.append((i, j, {"all": idx, "dec": idx[decoded[lo : lo + n]]}))
                lo += n
        self.out = [
            {g: {"s_tok": [], "s_code": [], "rho": []} for g in ("all", "dec")}
            for _ in self.spans
        ]
        self.prev_dev = None

    def add(self, cached: torch.Tensor, fresh: torch.Tensor) -> None:
        """Record one layer, ``cached`` and ``fresh`` being every reused row's V.
        ``s_code`` and ``rho`` are computed in the KV dtype like the official profiler,
        ``s_tok`` in fp32.
        """
        cos_tok = F.cosine_similarity(fresh.float(), cached.float(), dim=-1).mean(dim=1)
        dev = (1.0 - F.cosine_similarity(fresh.flatten(1), cached.flatten(1), dim=-1)).float()
        for k, (_i, _j, groups) in enumerate(self.spans):
            for name, idx in groups.items():
                rec = self.out[k][name]
                if idx.numel() == 0:
                    continue
                rec["s_tok"].append(float(cos_tok[idx].mean()))
                code = F.cosine_similarity(fresh[idx], cached[idx], dim=0, eps=1e-8)
                rec["s_code"].append(float(code.mean()))
                if self.prev_dev is not None:
                    rec["rho"].append(_spearman(self.prev_dev[idx], dev[idx]))
        self.prev_dev = dev

    def write(self, path: str) -> None:
        """Append one JSON line per reused span."""
        with open(path, "a") as fh:
            for k, (i, j, groups) in enumerate(self.spans):
                line = {
                    "rid": self.plan.rids[i],
                    "span": j,
                    "prompt_len": self.plan.prompt_lens[i],
                    "n_reused": int(groups["all"].numel()),
                    "n_decoded": int(groups["dec"].numel()),
                    "num_layers": self.num_layers,
                    "all": self.out[k]["all"],
                    "dec": self.out[k]["dec"] if groups["dec"].numel() else None,
                }
                fh.write(json.dumps(line) + "\n")
