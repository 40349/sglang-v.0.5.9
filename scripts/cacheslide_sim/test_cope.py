"""Numerical unit tests for the CoPE reference module (schedule step 1).

Training is undecided, so we validate the *math* only -- gates, contextual
positions, interpolation, the untrained (zero-init) reduction, and gradient flow.
Drift/CKSim and generation-quality validation both need a CoPE-trained model and
come later.

Run: python scripts/cacheslide_sim/test_cope.py
"""

import math

import torch

from sglang.srt.layers.cope import ContextualPositionEmbedding, cope_attention


def _plain_causal_attention(q, k, v, scale):
    """Reference RoPE-free causal attention -- what CoPE must reduce to at pos_emb=0."""
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    seq_len = q.shape[-2]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
    return torch.matmul(torch.softmax(logits, dim=-1), v)


def test_zero_init_reduces_to_plain_attention():
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 16, 8
    q, k, v = (torch.randn(b, h, t, d, dtype=torch.float64) for _ in range(3))
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=t).double()

    got = cope_attention(q, k, v, cope)
    want = _plain_causal_attention(q, k, v, scale=1.0 / math.sqrt(d))
    assert torch.allclose(got, want, atol=1e-10), (got - want).abs().max()
    print("[ok] zero-init pos_emb reduces to plain causal attention")


def test_positions_bounds_and_monotonicity():
    torch.manual_seed(1)
    b, t, d = 2, 12, 8
    npos = 10
    q, k = (torch.randn(b, t, d, dtype=torch.float64) for _ in range(2))
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=npos).double()

    scale = 1.0 / math.sqrt(d)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    pos = cope.positions_from_gates(logits, causal)

    assert pos.min() >= 0.0 and pos.max() <= npos - 1, (pos.min(), pos.max())
    # Diagonal position p_ii == sigmoid(logit_ii) in (0, 1): nearest token, smallest pos.
    diag = pos[:, torch.arange(t), torch.arange(t)]
    assert (diag > 0).all() and (diag < 1).all(), (diag.min(), diag.max())
    # Within the causal region, p_ij >= p_i(j+1) (extra gate term for smaller j).
    for i in range(1, t):
        row = pos[:, i, : i + 1]
        assert (row[:, :-1] - row[:, 1:] >= -1e-12).all(), i
    print("[ok] contextual positions are bounded and monotonic non-increasing in key")


def test_integer_positions_are_exact_gather():
    torch.manual_seed(2)
    b, t, d = 1, 6, 4
    npos = 8
    q = torch.randn(b, t, d, dtype=torch.float64)
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=npos).double()
    cope.pos_emb.data.normal_()  # non-trivial table

    causal = torch.ones(t, t, dtype=torch.bool).tril()
    positions = torch.randint(0, npos, (b, t, t)).double()  # integer -> w == 0
    out = cope(q, attn_logits=torch.zeros(b, t, t).double(), causal_mask=causal,
               positions=positions)

    logits_int = torch.matmul(q, cope.pos_emb)
    want = logits_int.gather(-1, positions.long())
    assert torch.allclose(out, want, atol=1e-12), (out - want).abs().max()
    print("[ok] integer positions interpolate to an exact table gather")


def test_override_positions_and_gradients():
    torch.manual_seed(3)
    b, h, t, d = 2, 2, 10, 8
    q, k, v = (torch.randn(b, h, t, d, dtype=torch.float64) for _ in range(3))
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=t).double()
    cope.pos_emb.data.normal_()

    # CCPE hook: a pinned (fractional) position field instead of gate-derived ones.
    pinned = torch.rand(b, h, t, t).double() * (t - 1)
    out = cope_attention(q, k, v, cope, positions=pinned)
    assert out.shape == (b, h, t, d)

    out.sum().backward()
    assert cope.pos_emb.grad is not None and cope.pos_emb.grad.abs().sum() > 0
    print("[ok] position override runs and gradients reach pos_emb (trainable)")


if __name__ == "__main__":
    test_zero_init_reduces_to_plain_attention()
    test_positions_bounds_and_monotonicity()
    test_integer_positions_are_exact_gather()
    test_override_positions_and_gradients()
    print("\nAll CoPE reference tests passed.")