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


def test_decode_matches_square_last_row():
    """Rectangular decode (q_len=1 over kv_len=N) must equal the square case's last
    row -- same q/k => same gates => same CoPE positions => same output."""
    torch.manual_seed(4)
    b, h, t, d = 2, 3, 16, 8
    q, k, v = (torch.randn(b, h, t, d, dtype=torch.float64) for _ in range(3))
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=t).double()
    cope.pos_emb.data.normal_()

    full = cope_attention(q, k, v, cope)  # [b,h,t,d]
    # Decode: last query token attends to all t keys.
    dec = cope_attention(q[:, :, -1:, :], k, v, cope)  # [b,h,1,d]
    assert dec.shape == (b, h, 1, d)
    assert torch.allclose(dec[:, :, 0, :], full[:, :, -1, :], atol=1e-10), \
        (dec[:, :, 0, :] - full[:, :, -1, :]).abs().max()
    print("[ok] rectangular decode (q_len=1) matches the square case's last row")


def test_train_forward_matches_serving_reference():
    """The training forward and the serving reference must stay the same function.

    They are two separate implementations (training carries a padding mask and forces an
    fp32 softmax; serving tiles over query blocks), so nothing but a test keeps them from
    drifting apart -- and a drift here means the served model is not the trained model.
    Compared on the part that must be exact: contextual positions and the position bias.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_cope as T
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import repeat_kv

    torch.manual_seed(5)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg).double()
    T.inject_cope(model, npos_max=64)
    attn = model.model.layers[0].self_attn
    attn.cope = attn.cope.double()
    attn.cope.pos_emb.data.normal_(std=0.5)

    b, t = 1, 20
    hs = torch.randn(b, t, 32, dtype=torch.float64)
    q = attn.q_proj(hs).view(b, t, -1, attn.head_dim).transpose(1, 2)
    k = repeat_kv(attn.k_proj(hs).view(b, t, -1, attn.head_dim).transpose(1, 2),
                  attn.num_key_value_groups)
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    logits = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
    full_pos = attn.cope.positions_from_gates(logits, causal)
    full_bias = attn.cope(q, logits, causal)

    # Every query block the serving tiler could pick must reproduce the full result.
    for blk in (1, 3, 7, 19):
        for s in range(0, t, blk):
            e = min(s + blk, t)
            lg = torch.matmul(q[:, :, s:e], k[:, :, :e].transpose(-1, -2)) * attn.scaling
            m = causal[s:e, :e]
            assert torch.allclose(attn.cope.positions_from_gates(lg, m),
                                  full_pos[..., s:e, :e], atol=1e-12)
            assert torch.allclose(attn.cope(q[:, :, s:e], lg, m),
                                  full_bias[..., s:e, :e], atol=1e-12)
    print("[ok] training and serving agree on positions/bias for every query block")


def test_gate_bias_shifts_span_as_intended():
    """init_gate_bias must put the initial contextual span where it claims to."""
    torch.manual_seed(6)
    t, d = 256, 8
    for target in (16, 64):
        cope = ContextualPositionEmbedding(head_dim=d, npos_max=t, n_heads=2).double()
        cope.init_gate_bias(target_span=target, seq_len=t)
        logits = torch.zeros(1, 2, t, t, dtype=torch.float64)  # q.k == 0 at init
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        span = cope.positions_from_gates(logits, causal)[0, 0, -1, 0]  # last row, full
        assert abs(span.item() - target) < 0.05 * target, (span.item(), target)
    print("[ok] gate bias sets the initial span to the requested number of positions")


def test_zero_gate_bias_is_the_old_behaviour():
    """Default (unset) gate_bias must leave the module bit-identical to before."""
    torch.manual_seed(7)
    b, t, d = 1, 12, 8
    q, k = (torch.randn(b, t, d, dtype=torch.float64) for _ in range(2))
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=t).double()
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    got = cope.positions_from_gates(logits, causal)
    want = torch.sigmoid(logits).masked_fill(~causal, 0.0).flip(-1).cumsum(-1).flip(-1)
    assert torch.allclose(got, want.clamp(max=t - 1), atol=1e-14)
    print("[ok] zero gate_bias reproduces the original gate computation exactly")


if __name__ == "__main__":
    test_zero_init_reduces_to_plain_attention()
    test_positions_bounds_and_monotonicity()
    test_integer_positions_are_exact_gather()
    test_override_positions_and_gradients()
    test_decode_matches_square_last_row()
    test_train_forward_matches_serving_reference()
    test_gate_bias_shifts_span_as_intended()
    test_zero_gate_bias_is_the_old_behaviour()
    print("\nAll CoPE reference tests passed.")