"""CacheSlide: Contextual Position Encoding (CoPE).

Reference (non-fused) implementation of CoPE (Golovneva et al., 2024, arXiv
2405.18719), the low-positional-sensitivity encoding CacheSlide's CCPE is built
on. RoPE bakes an *absolute* position into q/k before the attention kernel, so a
reused segment that shifts in absolute position drifts hard (PMKD). CoPE instead
derives a *contextual, fractional* position inside attention from gates, so the
same shift perturbs positions far less -- which is what makes cross-position KV
reuse near-lossless.

This module is the foundation the rest of the CoPE work stands on:
  * the LoRA CoPE finetune trains ``ContextualPositionEmbedding.pos_emb``;
  * CCPE pins fixed position ranges onto reuse chunks via the ``positions``
    override on :meth:`ContextualPositionEmbedding.forward`;
  * WCA measures / corrects cached-vs-recomputed KV on top of it.

It is deliberately a plain-PyTorch *batched* reference (materialises the T x T
logits): correct, differentiable, and unit-testable -- NOT fast. Wiring it into
SGLang's varlen attention layout and a fused/triton kernel is a later step.

Untrained behaviour is intentional: ``pos_emb`` is zero-initialised, so before
any finetune the position term is exactly 0 and :func:`cope_attention` reduces
to plain (un-rotated) causal attention. The finetune is what gives it meaning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ContextualPositionEmbedding(nn.Module):
    """CoPE position term: maps attention logits to a per-(query, key) position bias.

    For query ``i`` and key ``j`` (``j <= i``), the contextual position is a reverse
    cumulative sum of the gates over keys::

        p_ij = sum_{l=j}^{i} sigmoid(attn_logit_il)

    so the current token (``j == i``) gets the smallest position and far-back tokens
    get larger ones. ``p_ij`` is fractional and clamped to ``[0, npos_max - 1]``; the
    bias is read from the learnable table ``pos_emb`` (``[head_dim, npos_max]``, one
    key-space vector per integer position) by floor/ceil interpolation.

    Args:
        head_dim: per-head hidden size (matches q/k last dim).
        npos_max: number of integer position slots; also the clamp ceiling. Far-back
            keys beyond this share the top slot (the coarse-order cap CoPE relies on).
    """

    def __init__(self, head_dim: int, npos_max: int, n_heads: Optional[int] = None):
        super().__init__()
        self.head_dim = head_dim
        self.npos_max = npos_max
        # One learnable key-space vector per integer position. Zero-init => untrained
        # module contributes no position bias (reduces to plain causal attention).
        self.pos_emb = nn.Parameter(torch.zeros(head_dim, npos_max))
        # Learnable gate bias, added inside the sigmoid. At init q.k ~ 0 => every gate is
        # ~0.5, so a length-L row accumulates a contextual span of ~L/2. Once that exceeds
        # npos_max the positions clamp, and clamp has ZERO gradient -- the gates of every
        # clamped key are frozen out of training. Biasing the gates down at init keeps the
        # span inside the table so those gradients survive. Zero here = exactly the old
        # behaviour; set it with :meth:`init_gate_bias`.
        shape = (n_heads, 1, 1) if n_heads else (1,)
        self.gate_bias = nn.Parameter(torch.zeros(*shape))

    @torch.no_grad()
    def init_gate_bias(self, target_span: float, seq_len: int) -> float:
        """Bias the gates so a full-length row starts at ~``target_span`` positions.

        Each of the ``seq_len`` keys contributes ``sigmoid(bias)`` on average, so the
        initial span is ``seq_len * sigmoid(bias)``; inverting gives the logit below.
        Returns the value used.
        """
        p = min(max(target_span / max(seq_len, 1), 1e-4), 1 - 1e-4)
        b = math.log(p / (1.0 - p))
        self.gate_bias.fill_(b)
        return b

    def positions_from_gates(
        self, attn_logits: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Contextual positions ``p_ij`` from gates.

        Args:
            attn_logits: scaled q.k logits, ``[..., T, T]``.
            causal_mask: bool, ``[T, T]`` or broadcastable; True where ``j <= i``.

        Returns:
            ``[..., T, T]`` fractional positions in ``[0, npos_max - 1]``.
        """
        # Accumulate in >= fp32: the reverse cumsum sums up to T gates, and in bf16
        # the running total's rounding error (order the total magnitude) can be many
        # whole position slots -- so bf16 inputs would place tokens in the wrong slot.
        pdtype = torch.float64 if attn_logits.dtype == torch.float64 else torch.float32
        gates = torch.sigmoid(attn_logits.to(pdtype) + self.gate_bias.to(pdtype))
        # Keys above the diagonal must not be counted toward any position.
        gates = gates.masked_fill(~causal_mask, 0.0)
        # p_ij = sum_{l=j}^{i} gates_il  ->  reverse cumulative sum along the key axis.
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        return pos.clamp(max=self.npos_max - 1)

    def forward(
        self,
        query: torch.Tensor,
        attn_logits: torch.Tensor,
        causal_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Position logits ``[..., T, T]`` to add to ``attn_logits`` before softmax.

        Args:
            query: ``[..., T, head_dim]``.
            attn_logits: scaled q.k logits, ``[..., T, T]`` (only used when
                ``positions`` is None, to derive gate-based positions).
            causal_mask: bool causal mask (see :meth:`positions_from_gates`).
            positions: optional ``[..., T, T]`` override. This is the CCPE hook --
                supply pinned position ranges (``e*``) for reuse chunks so cached and
                live positions stay aligned instead of being recomputed from gates.
        """
        # Only the positions (and their cumsum) need >= fp32 for correct slot
        # placement. The interpolation -- logits_int, the two gathers, the blend --
        # runs in the caller's dtype: these are all [..., T, T] and fp32 copies OOM at
        # long seq_len. pos_emb stays an fp32/fp64 master param; the cast is
        # differentiable so gradients still reach it.
        pdtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        if positions is None:
            positions = self.positions_from_gates(attn_logits, causal_mask)
        positions = positions.to(pdtype)
        pos_floor_f = positions.floor()
        w = (positions - pos_floor_f).to(query.dtype)
        pos_floor = pos_floor_f.long()
        # ceil == floor + 1 except at integer positions, where w == 0 makes the ceil
        # term vanish anyway; clamp keeps the gather index in range.
        pos_ceil = (pos_floor + 1).clamp_(max=self.npos_max - 1)
        # Per-query logit for every integer position slot: [..., T, npos_max].
        logits_int = torch.matmul(query, self.pos_emb.to(query.dtype))
        logits_floor = logits_int.gather(-1, pos_floor)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        return logits_ceil * w + logits_floor * (1.0 - w)


def cope_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cope: ContextualPositionEmbedding,
    positions: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    q_offset: Optional[int] = None,
) -> torch.Tensor:
    """Reference causal self-attention with CoPE.

    Non-fused: materialises the ``T x T`` logits so the CoPE position term can be
    added inside the softmax (fused RoPE kernels cannot express this). For
    correctness, training, and CKSim measurement -- not production throughput.

    Supports rectangular attention: ``q_len`` (queries) may be shorter than
    ``kv_len`` (keys), with the queries taken to be the trailing ``q_len`` positions
    (``q_offset = kv_len - q_len``). This covers both prefill (q_len == kv_len, the
    square causal case) and decode (q_len == 1 attending to all cached keys).

    Args:
        query: ``[..., q_len, head_dim]``.
        key, value: ``[..., kv_len, head_dim]``.
        cope: the :class:`ContextualPositionEmbedding` supplying the position bias.
        positions: optional CCPE position override forwarded to ``cope``.
        scale: softmax scale; defaults to ``1/sqrt(head_dim)``.

    Returns:
        Attention output ``[..., q_len, head_dim]``.
    """
    head_dim = query.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    q_len, kv_len = query.shape[-2], key.shape[-2]
    # Absolute position of the first query row. Defaults to the trailing-q_len case;
    # query-block tiling passes an explicit offset so each block's rows keep their
    # true absolute positions (and thus the correct causal window + CoPE positions).
    if q_offset is None:
        q_offset = kv_len - q_len

    attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale  # [.., q_len, kv_len]
    # Query row i (absolute position q_offset + i) may attend key col j iff
    # j <= q_offset + i. Reduces to a lower-triangular mask when q_len == kv_len.
    rows = torch.arange(q_len, device=query.device).unsqueeze(-1) + q_offset
    cols = torch.arange(kv_len, device=query.device).unsqueeze(0)
    causal_mask = cols <= rows  # [q_len, kv_len]

    pos_logits = cope(query, attn_logits, causal_mask, positions=positions)
    # Position bias only applies to valid (causal) entries; mask the rest to -inf.
    neg_inf = torch.finfo(attn_logits.dtype).min
    masked = (attn_logits + pos_logits.masked_fill(~causal_mask, 0.0)).masked_fill(
        ~causal_mask, neg_inf
    )
    attn = torch.softmax(masked, dim=-1)
    return torch.matmul(attn, value)