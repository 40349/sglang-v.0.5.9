"""CacheSlide step 2b/2c: serve a CoPE-trained model in SGLang.

RoPE is fused into q/k before SGLang's paged attention kernel, and those kernels
cannot add CoPE's gate-derived position bias inside the softmax. So CoPE serving:

  1. skips RoPE in ``LlamaAttention.forward_prepare_native`` (CoPE keys are
     position-free -- position enters at attention time), and
  2. replaces the ``torch_native`` backend's per-request SDPA with cope.py's
     ``cope_attention`` over the gathered paged KV, using the per-layer ``pos_emb``.

The ``torch_native`` backend is the hook because it already gathers full per-request
k/v from the paged pool and runs a non-fused attention -- exactly the shape CoPE
needs. This path is slower than the fused RoPE baseline; the speedup only comes
later from WCA reuse. Enable with :func:`enable_cope_serving` before the server
builds the model (see scripts/cacheslide_sim/serve_cope.py), and launch with
``--attention-backend torch_native``.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from sglang.srt.layers.cope import ContextualPositionEmbedding, cope_attention

# Query-block size for the non-fused CoPE attention. Peak memory is
# O(H * Q_BLOCK * kv_len); tiling keeps it bounded regardless of sequence length.
# Smaller => less memory, more Python-loop overhead. Override via env.
_Q_BLOCK = int(os.environ.get("SGLANG_COPE_Q_BLOCK", "128"))


# ---------------------------------------------------------------------------
# Core: CoPE attention over gathered per-request KV
# ---------------------------------------------------------------------------
def cope_attend_gathered(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cope: ContextualPositionEmbedding,
    scaling: float,
) -> torch.Tensor:
    """CoPE attention for one request. q: [H, q_len, d]; k/v: [H_kv, kv_len, d].

    Expands kv heads for GQA and runs the rectangular (prefill or decode) CoPE
    attention. Returns [H, q_len, d].
    """
    n_heads, n_kv = q.shape[0], k.shape[0]
    if n_heads != n_kv:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
    if k.dtype != q.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)
    if cope.pos_emb.device != q.device:
        cope.to(q.device)

    q_len, kv_len = q.shape[1], k.shape[1]
    base_offset = kv_len - q_len  # absolute position of this request's first query row

    # Full non-fused CoPE attention materialises [H, q_len, kv_len] tensors (several,
    # incl. int64 gather indices) -- OOM for long prompts. Query rows are independent,
    # so tile over query blocks: exact, with peak memory bounded by _Q_BLOCK. Each
    # block only needs keys up to its last row's causal position.
    outs = []
    for start in range(0, q_len, _Q_BLOCK):
        end = min(start + _Q_BLOCK, q_len)
        kv_end = base_offset + end  # last query in block attends up to here (exclusive)
        block = cope_attention(
            q[:, start:end, :],
            k[:, :kv_end, :],
            v[:, :kv_end, :],
            cope,
            scale=scaling,
            q_offset=base_offset + start,
        )
        outs.append(block)
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def _run_cope_extend(query, output, k_cache, v_cache, req_to_token,
                     req_pool_indices, seq_lens, extend_seq_lens, cope, scaling):
    """Prefill: each request's extend queries attend to its full cached KV.

    Mirrors TorchNativeAttnBackend._run_sdpa_forward_extend but swaps SDPA for CoPE.
    query/output: [num_tokens, H, d]; k/v cache: [max_tokens, H_kv, d].
    """
    query = query.movedim(0, query.dim() - 2)  # [H, num_tokens, d]
    start_q = 0
    for si in range(seq_lens.shape[0]):
        extend_len = int(extend_seq_lens[si])
        seq_len_kv = int(seq_lens[si])
        end_q = start_q + extend_len
        per_req_query = query[:, start_q:end_q, :]  # [H, extend_len, d]

        tokens = req_to_token[req_pool_indices[si], :seq_len_kv]
        per_req_key = k_cache[tokens].movedim(0, query.dim() - 2)  # [H_kv, kv_len, d]
        per_req_value = v_cache[tokens].movedim(0, query.dim() - 2)

        out = cope_attend_gathered(per_req_query, per_req_key, per_req_value, cope, scaling)
        output[start_q:end_q, :, :] = out.permute(1, 0, 2)  # [extend_len, H, d]
        start_q = end_q
    return output


def _run_cope_decode(query, output, k_cache, v_cache, req_to_token,
                     req_pool_indices, seq_lens, cope, scaling):
    """Decode: each request's single query attends to its full cached KV."""
    query = query.movedim(0, query.dim() - 2)  # [H, num_seqs, d]
    for si in range(seq_lens.shape[0]):
        seq_len_kv = int(seq_lens[si])
        per_req_query = query[:, si:si + 1, :]  # [H, 1, d]

        tokens = req_to_token[req_pool_indices[si], :seq_len_kv]
        per_req_key = k_cache[tokens].movedim(0, query.dim() - 2)  # [H_kv, kv_len, d]
        per_req_value = v_cache[tokens].movedim(0, query.dim() - 2)

        out = cope_attend_gathered(per_req_query, per_req_key, per_req_value, cope, scaling)
        output[si:si + 1, :, :] = out.permute(1, 0, 2)  # [1, H, d]
    return output


# ---------------------------------------------------------------------------
# Monkey-patches (built as closures over the loaded per-layer CoPE modules)
# ---------------------------------------------------------------------------
def _cope_forward_prepare_native(self, positions, hidden_states):
    """LlamaAttention prepare WITHOUT RoPE (CoPE keys/queries stay un-rotated)."""
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    return q, k, v


def _make_forward_extend(cope_layers):
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        o = torch.empty_like(q) if layer.qk_head_dim == layer.v_head_dim else \
            q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v)
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        _run_cope_extend(
            q_, o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            cope=cope_layers[layer.layer_id],
            scaling=layer.scaling,
        )
        return o
    return forward_extend


def _make_forward_decode(cope_layers):
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q) if layer.qk_head_dim == layer.v_head_dim else \
            q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v)
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        _run_cope_decode(
            q_, o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            cope=cope_layers[layer.layer_id],
            scaling=layer.scaling,
        )
        return o
    return forward_decode


def load_cope_layers(pos_emb_path: str) -> List[Optional[ContextualPositionEmbedding]]:
    """Rebuild per-layer CoPE modules from a cope_pos_emb.pt produced by the merge tool."""
    ckpt = torch.load(pos_emb_path, map_location="cpu")
    npos_max, pos = ckpt["npos_max"], ckpt["pos_emb"]
    # Trained gate bias, if the run used one. Absent for older adapters -> stays 0, the
    # unbiased sigmoid the module defaults to. Serving MUST reproduce whatever bias the
    # training used: dropping it shifts every contextual position and silently
    # invalidates the pos_emb table it was trained against.
    gate = ckpt.get("gate_bias", {})
    n_layers = max(pos) + 1
    layers: List[Optional[ContextualPositionEmbedding]] = [None] * n_layers
    for i, tensor in pos.items():
        head_dim = tensor.shape[0]
        gb = gate.get(i)
        module = ContextualPositionEmbedding(
            head_dim=head_dim, npos_max=npos_max,
            n_heads=gb.shape[0] if gb is not None and gb.dim() == 3 else None,
        )
        module.pos_emb.data.copy_(tensor)
        if gb is not None:
            module.gate_bias.data.copy_(gb)
        layers[i] = module
    return layers


def enable_cope_serving(pos_emb_path: str) -> None:
    """Patch SGLang to serve a CoPE model: drop RoPE + CoPE-ify the torch_native backend.

    Call this BEFORE the server builds the model, and launch with
    ``--attention-backend torch_native``.
    """
    cope_layers = load_cope_layers(pos_emb_path)

    from sglang.srt.models.llama import LlamaAttention
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    # Assigning to a name the class does not define would silently create a NEW method
    # that nothing calls -- RoPE would stay live and the model would run as
    # "RoPE + an untrained CoPE bias" with no error anywhere. Same for the two backend
    # methods. Fail loudly on upstream drift instead.
    import inspect

    for cls, name, repl in ((LlamaAttention, "forward_prepare_native",
                             _cope_forward_prepare_native),
                            (TorchNativeAttnBackend, "forward_extend",
                             _make_forward_extend(cope_layers)),
                            (TorchNativeAttnBackend, "forward_decode",
                             _make_forward_decode(cope_layers))):
        if not hasattr(cls, name):
            raise RuntimeError(
                f"[CoPE] {cls.__name__}.{name} does not exist in this SGLang build, so "
                f"patching it would be a silent no-op (RoPE would stay active). The "
                f"serving patch needs updating for this version."
            )
        # hasattr is not enough. Qwen3MoeAttention.forward_prepare_native takes an extra
        # forward_batch and returns (None, forward_batch, inner_state) rather than
        # (q, k, v), so a patch written against Llama installs cleanly and then dies at
        # the first token. Compare parameter names before replacing anything.
        want = list(inspect.signature(getattr(cls, name)).parameters)
        got = list(inspect.signature(repl).parameters)
        if want != got:
            raise RuntimeError(
                f"[CoPE] {cls.__name__}.{name} takes {want} but the CoPE replacement "
                f"takes {got}. This model's attention is not the one the patch was "
                f"written for -- port _cope_forward_prepare_native before serving it."
            )

    LlamaAttention.forward_prepare_native = _cope_forward_prepare_native
    TorchNativeAttnBackend.forward_extend = _make_forward_extend(cope_layers)
    TorchNativeAttnBackend.forward_decode = _make_forward_decode(cope_layers)

    n_biased = sum(1 for m in cope_layers if m is not None and m.gate_bias.abs().max() > 0)
    print(f"[CoPE] enabled: {len(cope_layers)} layers, "
          f"npos_max={cope_layers[0].npos_max}, gate_bias on {n_biased} layers; "
          f"RoPE disabled, torch_native patched.")


# ---------------------------------------------------------------------------
# Self-test: CoPE-over-gathered-KV helper vs the reference (no server needed)
# ---------------------------------------------------------------------------
def _smoke():
    torch.manual_seed(0)
    n_heads, n_kv, d, kv_len = 4, 2, 8, 12
    cope = ContextualPositionEmbedding(head_dim=d, npos_max=kv_len).double()
    cope.pos_emb.data.normal_()
    k = torch.randn(n_kv, kv_len, d, dtype=torch.float64)
    v = torch.randn(n_kv, kv_len, d, dtype=torch.float64)

    # (a) Decode: single query over full KV == last row of the full prefill.
    q_full = torch.randn(n_heads, kv_len, d, dtype=torch.float64)
    full = cope_attend_gathered(q_full, k, v, cope, scaling=1.0 / (d ** 0.5))
    dec = cope_attend_gathered(q_full[:, -1:, :], k, v, cope, scaling=1.0 / (d ** 0.5))
    assert dec.shape == (n_heads, 1, d)
    err = (dec[:, 0] - full[:, -1]).abs().max().item()
    assert err < 1e-10, err
    print(f"[ok] gathered decode == prefill last row (GQA {n_heads}/{n_kv}, err {err:.1e})")

    # (b) GQA expansion + tiling are exact: a single non-tiled call matches.
    rep = n_heads // n_kv
    man = cope_attention(q_full, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0),
                         cope, scale=1.0 / (d ** 0.5))
    assert torch.allclose(full, man, atol=1e-12)
    print("[ok] GQA head expansion matches manual repeat")

    # (c) Query-block tiling is exact regardless of block size.
    global _Q_BLOCK
    saved = _Q_BLOCK
    try:
        _Q_BLOCK = 3  # force multiple blocks with awkward boundaries
        tiled = cope_attend_gathered(q_full, k, v, cope, scaling=1.0 / (d ** 0.5))
    finally:
        _Q_BLOCK = saved
    assert torch.allclose(tiled, man, atol=1e-12), (tiled - man).abs().max()
    print("[ok] query-block tiling (block=3) equals the non-tiled result")


if __name__ == "__main__":
    _smoke()
    print("\nCoPE serving helpers verified.")
