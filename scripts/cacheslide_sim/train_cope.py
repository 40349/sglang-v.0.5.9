"""CoPE continued-pretraining (CacheSlide schedule, step 5 -- training).

Serving frameworks (vLLM/SGLang) cannot train, so this runs in plain HF
Transformers: it swaps RoPE for CoPE in every attention module, freezes the
backbone, and trains only (a) the per-layer CoPE ``pos_emb`` and (b) LoRA adapters
over q/k/v/o_proj -- exactly the "adapter-based continued pretraining" the paper
describes. The objective is ordinary causal-LM loss; convergence == perplexity
recovering toward the RoPE baseline (the model has re-learned to read position
through CoPE). Export is a small state-dict of the trained tensors, later loaded
into the SGLang serving path (python/sglang/srt/layers/cope.py + wiring).

The CoPE math is imported straight from cope.py so training and serving stay
byte-identical. LoRA and the train loop are hand-rolled (no peft/accelerate).

Quick self-test (no model download, tiny random Llama):
    python scripts/cacheslide_sim/train_cope.py --smoke

Real run (example):
    python scripts/cacheslide_sim/train_cope.py \
        --model meta-llama/Llama-3.1-8B --data_file corpus.txt \
        --max_seq_len 2048 --steps 2000 --batch_size 1 --lora_rank 16 \
        --output_dir ./cope_adapter

At --max_seq_len >= 4096 add ``--tile_q 1024`` (exact; bounds the attention activation
memory that the eager [B,H,T,T] forward would otherwise blow up). The run ends with a
pos_emb coverage report: slots still at their zero init never received a gradient, so
prompts long enough to reach them are served with no position signal there.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    get_cosine_schedule_with_warmup,
)
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv


def attention_class(model):
    """The attention module class to patch, discovered from the model itself.

    Hardcoding LlamaAttention silently no-ops on anything else: isinstance() matches
    nothing, so zero CoPE modules and zero LoRA adapters get attached and the optimizer
    gets an empty parameter list. Reading the class off layer 0 covers Llama, Qwen3,
    Qwen3-MoE and anything else with the same decoder-layer shape.
    """
    return type(model.model.layers[0].self_attn)


def is_cope_attention(module) -> bool:
    """True for an attention module that inject_cope has already fitted with CoPE.

    Used instead of an isinstance() check against one hardcoded class, so the helpers
    stay model-agnostic.
    """
    return hasattr(module, "cope")


# --- Single source of truth for the CoPE math: load cope.py by file path so we do
# --- NOT import the whole sglang package (keeps the training env light).
_COPE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python/sglang/srt/layers/cope.py"
)
_spec = importlib.util.spec_from_file_location("cacheslide_cope", _COPE_PATH)
_cope_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cope_mod)
ContextualPositionEmbedding = _cope_mod.ContextualPositionEmbedding

# Query-block size for the training CoPE forward; 0 = off (single shot). The non-tiled
# forward materialises [B,H,T,T] tensors (int64 gather indices are 8 bytes each) --
# ~137GB just for the two index tensors at T=16384, OOM even on an H200. Query rows are
# independent, so tiling is numerically exact. Each block additionally runs under its
# OWN checkpoint: plain tiling would still keep every block's [B,H,block,T] intermediates
# alive for autograd, so only re-materialising them in backward actually bounds peak
# memory to O(H*block*kv_len). Set from --tile_q.
_TRAIN_Q_BLOCK = 0

# Gate-selectivity regularizer config (set in main). OFF by default -> no stash, no
# overhead, existing runs unchanged. When on, cope_attention_forward stashes the
# detached layer input + valid mask so accumulate_gate_reg_grads can recompute the
# gates OUTSIDE the gradient-checkpointed graph (an aux loss on the internal gates
# would otherwise be severed by checkpointing). See accumulate_gate_reg_grads.
_GATE_REG = {"on": False, "span_target": 512, "lam_span": 0.0, "lam_bimod": 0.0}

# Running "is CoPE actually load-bearing?" meter, filled by _cope_attend_block.
# Both terms go into the same softmax: content = q.(k*scaling), position = q.pos_emb[:,p].
# If |pos| / |attn| stays near 0 the position table is too weak to change any attention
# weight, the LoRA is doing all the work, and the run is not testing CoPE at all. This is
# the single number that says whether the training is doing what it claims.
_DIAG = {"pos": 0.0, "attn": 0.0, "n": 0}


# ---------------------------------------------------------------------------
# 1. CoPE attention -- monkey-patch <Model>Attention.forward (RoPE removed)
# ---------------------------------------------------------------------------
def _cope_attend_block(cope, scaling, query, key, value, valid, start: int, end: int):
    """CoPE attention for query rows ``[start, end)`` against keys ``[0, end)``.

    Exact for any block boundary: row ``i`` only ever attends to keys ``j <= i``, all of
    which lie inside ``[0, end)``, and the CoPE reverse-cumsum runs over that same key
    range -- so a block sees byte-identical gates/positions to the full forward.
    """
    q = query[:, :, start:end, :]
    k = key[:, :, :end, :]
    v = value[:, :, :end, :]
    vb = valid[..., start:end, :end]

    logits = torch.matmul(q, k.transpose(-1, -2)) * scaling  # [B,H,blk,end]
    pos_bias = cope(q, logits, vb)

    if _DIAG["n"] >= 0:  # cheap: two reductions per block, no graph retained
        with torch.no_grad():
            _DIAG["pos"] += pos_bias.detach().abs().mean().item()
            _DIAG["attn"] += logits.detach().abs().mean().item()
            _DIAG["n"] += 1

    neg_inf = torch.finfo(logits.dtype).min
    masked = logits + pos_bias.masked_fill(~vb, 0.0)
    masked = masked.masked_fill(~vb, neg_inf)
    attn = torch.softmax(masked, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)  # [B,H,blk,d]


def _cope_attend_tiled(cope, scaling, query, key, value, valid, seq_len, block: int):
    """Query-block-tiled CoPE attention. Exact; bounds attention activation memory.

    Each block runs under its own non-reentrant checkpoint so its [B,H,block,end]
    intermediates are freed after the forward and re-materialised in backward -- without
    that, tiling alone would leave every block's tensors alive and save nothing.
    """
    from torch.utils.checkpoint import checkpoint

    recompute = torch.is_grad_enabled() and query.requires_grad
    outs = []
    for start in range(0, seq_len, block):
        end = min(start + block, seq_len)
        args = (cope, scaling, query, key, value, valid, start, end)
        outs.append(
            checkpoint(_cope_attend_block, *args, use_reentrant=False)
            if recompute
            else _cope_attend_block(*args)
        )
    return torch.cat(outs, dim=-2)


def cope_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,  # (cos, sin) -- intentionally IGNORED (no RoPE)
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Training-time attention forward with CoPE instead of RoPE.

    Eager (materialised T x T logits) so the CoPE position bias can be added inside
    the softmax. No KV-cache path -- training only. Matches transformers 4.57's
    forward contract, returning ``(attn_output, attn_weights=None)``.
    """
    input_shape = hidden_states.shape[:-1]  # [B, T]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # QK-Norm (Qwen3 / Qwen3-MoE): RMSNorm over the head dim, applied to the [B,T,H,d]
    # view BEFORE the transpose, exactly as the stock forward does. Llama has no q_norm
    # and skips this. Dropping it would leave q/k unnormalised -- the model still runs
    # and still trains, it is just no longer the model whose weights we loaded.
    query = self.q_proj(hidden_states).view(hidden_shape)
    key = self.k_proj(hidden_states).view(hidden_shape)
    if getattr(self, "q_norm", None) is not None:
        query = self.q_norm(query)
    if getattr(self, "k_norm", None) is not None:
        key = self.k_norm(key)
    query = query.transpose(1, 2)  # [B,Hq,T,d]
    key = key.transpose(1, 2)  # [B,Hkv,T,d]
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # A sliding-window layer needs HF to supply the windowed mask; without it the CoPE
    # forward would silently attend to the full history.
    if getattr(self, "sliding_window", None) and attention_mask is None:
        raise RuntimeError(
            f"{type(self).__name__} has sliding_window={self.sliding_window} but no "
            f"attention_mask was passed, so the window cannot be honoured."
        )

    # Grouped-query attention: expand kv heads to match query heads.
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)

    seq_len = query.shape[-2]

    # Validity = causal AND (HF additive mask says "keep"). Used both to gate CoPE
    # positions and to mask the softmax. Built BEFORE the logits (it does not depend on
    # them) so the tiled path never materialises a full [B,H,T,T] logit tensor.
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
    if attention_mask is not None:
        m = attention_mask[..., :seq_len]
        valid = causal & (m > torch.finfo(m.dtype).min / 2)  # [B,1,T,T]
    else:
        valid = causal  # [T,T]

    # Gate-selectivity regularizer needs the gates (sigmoid(q.k)), which are internal
    # to this (possibly gradient-checkpointed) forward. Stash the DETACHED layer input
    # + valid mask so accumulate_gate_reg_grads can recompute the gates outside the
    # checkpoint graph and backprop the reg into this layer's q/k LoRA. Detach is what
    # makes it checkpoint-safe (constant tensor, not freed) and scopes the reg to each
    # layer's own q/k projections. No-op / no memory when the reg is off.
    # Guarded on grad being enabled: under @torch.no_grad() eval there is nothing to
    # backprop, and stashing there would pin a [B,T,hidden] tensor per layer (~1GB over
    # 32 layers at T=4096) that the next training forward would only overwrite.
    if _GATE_REG["on"] and torch.is_grad_enabled():
        self._cope_hs = hidden_states.detach()
        self._cope_valid = valid

    block = _TRAIN_Q_BLOCK
    if block and seq_len > block:
        out = _cope_attend_tiled(
            self.cope, self.scaling, query, key, value, valid, seq_len, block
        )
    else:
        out = _cope_attend_block(
            self.cope, self.scaling, query, key, value, valid, 0, seq_len
        )

    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    out = self.o_proj(out)
    return out, None


def inject_cope(model, npos_max: int, gate_bias_span: float = 0.0,
                seq_len: int = 0) -> None:
    """Attach a per-layer CoPE module and switch LlamaAttention to the CoPE forward.

    ``gate_bias_span > 0`` biases the gates so a full-length row starts at roughly that
    many contextual positions instead of the ~seq_len/2 an unbiased sigmoid gives -- see
    ContextualPositionEmbedding.init_gate_bias. 0 keeps the original behaviour exactly.
    """
    device = next(model.parameters()).device
    attn_cls = attention_class(model)
    n_attached = 0
    for module in model.modules():
        if isinstance(module, attn_cls):
            # Read head_dim off the module, not hidden_size // num_heads: they coincide
            # on Llama-3.1 but not on models with an explicit head_dim (Qwen3-Coder-30B
            # is 2048/32 = 64 derived vs 128 actual), where the derived value would size
            # the pos_emb table wrong and the query @ pos_emb matmul would not even fit.
            head_dim = getattr(module, "head_dim", None) or (
                model.config.hidden_size // model.config.num_attention_heads
            )
            n_heads = getattr(module, "config", model.config).num_attention_heads
            cope = ContextualPositionEmbedding(
                head_dim=head_dim, npos_max=npos_max, n_heads=n_heads
            )
            if gate_bias_span > 0 and seq_len > 0:
                cope.init_gate_bias(gate_bias_span, seq_len)
            # pos_emb trains in fp32 for stability even under a bf16 backbone.
            module.cope = cope.to(device=device)
            last_attn = module  # model.modules() ends on some other module; keep this one
            n_attached += 1
    if n_attached == 0:
        raise RuntimeError(f"no {attn_cls.__name__} modules found -- nothing to patch")
    # Class-level patch: every instance of that attention class now runs CoPE.
    attn_cls.forward = cope_attention_forward
    print(f"CoPE injected into {n_attached} x {attn_cls.__name__} "
          f"(head_dim={last_attn.cope.head_dim}, qk_norm="
          f"{getattr(last_attn, 'q_norm', None) is not None}); RoPE bypassed")


# ---------------------------------------------------------------------------
# 2. Minimal LoRA (no peft) + trainable-parameter selection
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update ``B(A(x)) * alpha/r``.

    ``B`` is zero-initialised so the adapter starts as a no-op (output == base).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.to(base.weight.device)
        # LoRA in fp32 for stable optimisation regardless of backbone dtype.
        self.lora_A.float()
        self.lora_B.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x.float()))) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def add_lora(model, r: int, alpha: int, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Wrap the named attention projections of every attention module with LoRA.

    Requires plain nn.Linear projections: LoRALinear reads base.in_features and
    base.weight, which a 4-bit packed (AWQ / compressed-tensors) module does not have.

    Keyed on the attention CLASS, not on "has a .cope attached": the --keep_rope control
    run never calls inject_cope, and keying on CoPE would silently give it zero adapters.
    """
    attn_cls = attention_class(model)
    for module in model.modules():
        if isinstance(module, attn_cls):
            for name in targets:
                base = getattr(module, name)
                setattr(module, name, LoRALinear(base, r=r, alpha=alpha))


def mark_trainable(model, train_gate_bias: bool = True) -> List[nn.Parameter]:
    """Freeze everything, then unfreeze CoPE ``pos_emb`` + LoRA. Returns trainables.

    ``train_gate_bias=False`` leaves the gate bias at its initialisation value. It is
    still saved by :func:`save_adapter` -- serving must reproduce it either way.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if ("cope.pos_emb" in name
                or ("cope.gate_bias" in name and train_gate_bias)
                or "lora_A" in name or "lora_B" in name):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def accumulate_gate_reg_grads(model, scale: float) -> Dict[str, float]:
    """Backprop the gate-selectivity regularizer into each layer's q/k LoRA.

    Recomputes the CoPE gates ``sigmoid(q.k * scale)`` for every attention layer from
    the DETACHED layer input stashed by ``cope_attention_forward`` (so this is a fresh
    small graph OUTSIDE the gradient-checkpointed backbone -- the only way an aux loss
    on the internal gates survives checkpointing). Detach also scopes the reg to this
    layer's own q/k projections (no backprop into lower layers), which is what we want:
    each layer sculpts its own gates to be sparse+decisive.

    Two terms, per query row ``i`` with contextual span ``span_i = sum_{j<=i} g_ij``:
      * span-cap  ``relu(span_i - S)/S``  -- pushes each row's span under the position
        budget ``S`` (= where pos_emb is well-trained) so long contexts don't overflow
        / mass-collapse onto the top slot;
      * bimodality ``g(1-g)`` -- pushes gates toward 0/1 so ``span`` counts a FEW
        decisive boundaries (selective, the paper's mechanism) not a soft ~0.27 drift.

    Both terms are averaged over REAL query rows only. HF's 4D mask masks pad *keys*,
    not pad *rows*, so a padding row still attends causally to every real token before
    it and would otherwise be scored (and penalised) like a real one. A row is a pad row
    iff its own diagonal key is masked, which is what ``row_ok`` reads off.

    Backprops per layer (one gate tensor live at a time -> bounded peak memory) and
    accumulates into ``.grad`` alongside the main loss. Returns diagnostics: ``span`` is
    the mean over real rows and ``span_max`` the largest single row -- the cap binds on
    the tail, so the mean alone hides it (early rows can never exceed S).
    """
    S = float(_GATE_REG["span_target"])
    lam_s, lam_b = _GATE_REG["lam_span"], _GATE_REG["lam_bimod"]
    tot = {"span": 0.0, "span_max": 0.0, "L_span": 0.0, "L_bimod": 0.0}
    nl = 0
    for m in model.modules():
        if not (is_cope_attention(m) and getattr(m, "_cope_hs", None) is not None):
            continue
        hs = m._cope_hs  # [B, T, hidden], detached (constant)
        valid = m._cope_valid
        B, T = hs.shape[0], hs.shape[1]
        q = m.q_proj(hs).view(B, T, -1, m.head_dim).transpose(1, 2)  # [B,Hq,T,d]
        k = m.k_proj(hs).view(B, T, -1, m.head_dim).transpose(1, 2)  # [B,Hkv,T,d]
        k = repeat_kv(k, m.num_key_value_groups)
        logits = torch.matmul(q, k.transpose(-1, -2)) * m.scaling  # [B,Hq,T,T]
        vb = valid.view(1, 1, T, T) if valid.dim() == 2 else valid  # [B/1,1,T,T]
        row_ok = vb.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # [B/1,1,T,1] real rows
        keep = vb & row_ok  # drop pad keys AND pad rows
        gates = torch.sigmoid(logits.float()) * keep
        span = gates.sum(-1)  # [B,Hq,T]; pad rows are exactly 0 -> relu 0
        n_heads = gates.shape[1]
        n_rows = (row_ok.sum() * n_heads).clamp(min=1).float()
        n_pairs = (keep.sum() * n_heads).clamp(min=1).float()
        L_span = torch.relu(span - S).div(S).sum() / n_rows
        L_bimod = (gates * (1.0 - gates)).sum() / n_pairs
        reg = lam_s * L_span + lam_b * L_bimod
        (scale * reg).backward()
        tot["span"] += (span.detach().sum() / n_rows).item()
        tot["span_max"] += span.detach().max().item()
        tot["L_span"] += L_span.item()
        tot["L_bimod"] += L_bimod.item()
        # Release the stash: it is consumed, and holding it keeps a [B,T,hidden] tensor
        # per layer alive across the optimizer step for nothing.
        m._cope_hs = None
        m._cope_valid = None
        nl += 1
    if nl:
        for kk in tot:
            tot[kk] /= nl
    return tot


# ---------------------------------------------------------------------------
# 3. Data -- fixed-length causal-LM blocks
# ---------------------------------------------------------------------------
class ShareGPTDataset(Dataset):
    """ShareGPT-style HF dataset -> per-conversation token sequences.

    Each row has a ``messages`` list of ``{role, value}``. Roles are mapped to the
    chat-template roles (human->user, gpt->assistant) and the whole conversation is
    rendered with ``tokenizer.apply_chat_template`` so training sees the SAME format
    the server prompts with (the ``<|start_header_id|>...`` structure) -- essential
    for CoPE to learn positions that match the serving distribution.

    Needs a tokenizer that HAS a chat template: the base Llama-3.1-8B does NOT, so
    pass an instruct tokenizer via ``--tokenizer ...-Instruct`` (or use the instruct
    model). One conversation per sample, truncated to ``seq_len`` (padding happens in
    the collate).

    ``offset`` skips the first N rows before taking ``max_samples``. Training reads rows
    ``[0, max_samples)``, so an evaluation that wants genuinely unseen data must pass
    ``offset >= the training run's --max_samples`` -- otherwise it scores the model on
    its own training set.
    """

    def __init__(self, hf_name, split, tokenizer, seq_len, max_samples=None,
                 messages_field="messages", offset: int = 0):
        from datasets import load_dataset

        if tokenizer.chat_template is None:
            raise ValueError(
                "tokenizer has no chat_template; pass --tokenizer <an instruct "
                "tokenizer, e.g. meta-llama/Llama-3.1-8B-Instruct> so conversations "
                "render in the served chat format."
            )
        ds = load_dataset(hf_name, split=split)
        if offset:
            if offset >= len(ds):
                raise SystemExit(
                    f"offset {offset} >= dataset size {len(ds)}: no rows left. Use a "
                    f"smaller --skip_samples or a bigger split."
                )
            ds = ds.select(range(offset, len(ds)))
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.samples: List[torch.Tensor] = []
        kept = skipped = 0
        for row in ds:
            ids = render_sharegpt_row(row[messages_field], tokenizer, seq_len)
            if ids is not None:
                self.samples.append(ids)
                kept += 1
            else:
                skipped += 1
        print(f"ShareGPT: kept {kept} conversations, skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


_SHAREGPT_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
                      "user": "user", "assistant": "assistant"}


def render_sharegpt_row(messages, tokenizer, seq_len, min_len: int = 8):
    """One ShareGPT ``messages`` list -> a truncated id tensor, or None to skip.

    Maps roles to chat-template roles and renders with ``apply_chat_template`` so the
    result matches the served prompt format. Returns None on empty/too-short/bad rows.

    Tolerates the three field schemas in the wild -- ``{from, value}`` (classic
    ShareGPT), ``{role, content}`` (HF chat format), ``{role, value}`` (the hybrid the
    "-renamed" mirrors use). Row parsing is INSIDE the try: a schema mismatch must skip
    the row, not abort the whole dataset build.
    """
    try:
        conv = []
        for m in messages:
            role = m.get("role", m.get("from"))
            content = m.get("content", m.get("value"))
            if role is None or content is None:
                return None
            conv.append(
                {"role": _SHAREGPT_ROLE_MAP.get(role, role), "content": content}
            )
        if not conv:
            return None
        ids = tokenizer.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        return None
    ids = ids[:seq_len]
    if len(ids) < min_len:
        return None
    return torch.tensor(ids, dtype=torch.long)


class TextBlocks(Dataset):
    """Tokenise a plain-text file and pack into contiguous ``seq_len`` blocks."""

    def __init__(self, path: str, tokenizer, seq_len: int):
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        n = (ids.numel() // seq_len) * seq_len
        self.blocks = list(ids[:n].view(-1, seq_len))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return self.blocks[i]


class SyntheticVarLen(Dataset):
    """Variable-length random-token samples -- --smoke only (exercises padding)."""

    def __init__(self, vocab: int, seq_len: int, n: int = 64):
        self.data = [
            torch.randint(0, vocab, (int(torch.randint(seq_len // 2, seq_len + 1, ())),))
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_pad(batch: List[torch.Tensor], pad_id: int):
    """Right-pad a batch of 1-D id tensors. Returns (input_ids, attention_mask, labels).

    Right-padding keeps every row's causal prefix non-empty (no fully-masked softmax
    rows -> no NaNs); pad positions get label -100 so they are ignored by the loss.
    """
    lengths = [x.numel() for x in batch]
    maxlen = max(lengths)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x.numel()
        input_ids[i, :n] = x
        attn[i, :n] = 1
        labels[i, :n] = x
    return input_ids, attn, labels


# ---------------------------------------------------------------------------
# 4. CKSim validation -- reproduce Figure 4(a): drift of cached vs recomputed K
# ---------------------------------------------------------------------------
@torch.no_grad()
def cksim_vs_shift(
    model, segment_ids: torch.Tensor, shifts: List[int], layer_idx: int = -1,
    seed: int = 0,
) -> Dict[int, float]:
    """Mean per-head cosine similarity of a segment's layer-``layer_idx`` keys when
    the segment sits at position 0 vs shifted right by a prefix of length ``shift``.

    ``segment_ids`` is a ``[1, L]`` token-id tensor (kept tokenizer-agnostic so the
    smoke path can pass random ids). Low drift (CKSim staying near 1 as shift grows)
    is exactly the CoPE property that makes shifted KV reuse near-lossless; run on
    the base (unpatched) model to get the RoPE curve for comparison.

    Two things this has to get right to mean anything:
      * hook the FULL ``k_proj`` (base + LoRA). The served keys include the trained LoRA
        delta, so hooking ``.base`` would measure a model that is never served.
      * shifts must be NESTED prefixes of one seeded pool, not a fresh random prefix per
        shift -- otherwise the curve mostly measures prefix *content* and comes out
        non-monotonic in the shift.
    """
    model.eval()
    device = next(model.parameters()).device
    layers = model.model.layers
    layer = layers[layer_idx]
    attn = layer.self_attn

    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        # k_proj output: [B, T, Hkv*d]; keep as-is, we slice/normalise later.
        captured["k"] = out.detach()

    handle = attn.k_proj.register_forward_hook(hook)

    seg = segment_ids.to(device)
    seg_len = seg.shape[1]
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    # One fixed pool; shift s uses its first s tokens, so every shift is a prefix of the
    # next and the only thing varying across the curve is where the segment sits.
    gen = torch.Generator().manual_seed(seed)
    pool = torch.randint(
        0, model.config.vocab_size, (1, max(max(shifts), 1)), generator=gen
    ).to(device)

    def seg_keys(prefix_len: int) -> torch.Tensor:
        ids = seg if prefix_len == 0 else torch.cat([pool[:, :prefix_len], seg], dim=1)
        model(input_ids=ids)
        k = captured["k"][0, prefix_len : prefix_len + seg_len]  # [seg_len, Hkv*d]
        return k.reshape(seg_len, n_kv, -1)  # [seg_len, Hkv, d] -> per-head cosine

    base = seg_keys(0)
    out: Dict[int, float] = {}
    for s in shifts:
        shifted = seg_keys(s)
        cos = torch.nn.functional.cosine_similarity(
            base.float(), shifted.float(), dim=-1
        )  # [seg_len, Hkv]
        out[s] = cos.mean().item()
    handle.remove()
    return out


@torch.no_grad()
def cksim_by_layer(model, segment_ids: torch.Tensor, shift: int, seed: int = 0) -> str:
    """Per-layer key drift for one shift -- where CoPE stops buying you anything.

    CoPE never writes position into k, so layer 0's keys are a pure function of the token
    ids: shifting the segment must leave them BIT-identical (a 1.000 that is not a
    measurement but a structural guarantee -- if it is not 1.000, RoPE is still active).
    Deeper layers drift only because their input hidden states saw a different prefix
    through attention. The layer at which this curve falls away is the honest answer to
    "how far up does position-free KV reuse survive", which a single last-layer number
    cannot tell you.
    """
    model.eval()
    device = next(model.parameters()).device
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    caps: Dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            lambda _m, _i, out, idx=i: caps.__setitem__(idx, out.detach())))

    seg = segment_ids.to(device)
    seg_len = seg.shape[1]
    gen = torch.Generator().manual_seed(seed)
    pool = torch.randint(0, model.config.vocab_size, (1, max(shift, 1)), generator=gen)

    def keys(prefix_len):
        ids = seg if prefix_len == 0 else torch.cat([pool[:, :prefix_len].to(device), seg], 1)
        model(input_ids=ids)
        return {i: v[0, prefix_len:prefix_len + seg_len].reshape(seg_len, n_kv, -1)
                for i, v in caps.items()}

    base, shifted = keys(0), keys(shift)
    for h in handles:
        h.remove()
    per = [torch.nn.functional.cosine_similarity(
        base[i].float(), shifted[i].float(), dim=-1).mean().item() for i in sorted(base)]
    n = len(per)
    # sorted(set(...)): on a shallow model these quarter-points collide, and printing
    # "L0 L0 L1 L1 L1" makes the report look broken.
    show = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    return (f"shift={shift}: " + " ".join(f"L{i}={per[i]:.4f}" for i in show)
            + f"  (layer0 must be 1.0000: CoPE writes no position into k)")


def pos_ratio_and_reset() -> float:
    """Mean |CoPE position bias| / mean |content logit| since the last call.

    Near 0 => the position table cannot move any attention weight, so whatever the loss
    is doing it is not using CoPE. Raise --pos_emb_lr until this climbs.
    """
    n, attn = _DIAG["n"], _DIAG["attn"]
    r = (_DIAG["pos"] / attn) if (n and attn > 0) else 0.0
    _DIAG.update({"pos": 0.0, "attn": 0.0, "n": 0})
    return r


@torch.no_grad()
def pos_emb_coverage(model, bins: int = 8) -> str:
    """Per-slot report of the CoPE tables: how much of the position budget got trained.

    ``pos_emb`` is zero-initialised, so a column still exactly 0 never received a
    gradient -- no training sample ever placed a token at that contextual position. Any
    prompt long enough to reach those slots is served with NO position signal there,
    which is what a length cliff looks like from the inside. Columns whose norm is far
    below the low slots are the soft version of the same problem.
    """
    tables = [m.cope.pos_emb.detach().float() for m in model.modules()
              if is_cope_attention(m)]
    if not tables:
        return "(no CoPE tables found)"
    coln = torch.stack(tables).norm(dim=1)  # [n_layers, npos_max] per-slot column norm
    npos = coln.shape[1]
    live = (coln > 0).any(0)
    highest = int(live.nonzero().max()) if bool(live.any()) else -1
    mean_col = coln.mean(0)
    step = max(1, npos // bins)
    ranges = " ".join(
        f"[{lo}:{min(lo + step, npos)})={mean_col[lo:min(lo + step, npos)].mean():.4f}"
        for lo in range(0, npos, step)
    )
    return (f"slots ever trained: {int(live.sum())}/{npos} (highest={highest})\n"
            f"  mean |column| by slot range: {ranges}")


# ---------------------------------------------------------------------------
# 5. Train helpers
# ---------------------------------------------------------------------------
def infinite(loader):
    """Yield batches forever, re-iterating the loader each epoch (no caching)."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, max_batches: Optional[int] = None) -> float:
    """Mean held-out perplexity over up to ``max_batches`` batches (None = the lot)."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids, attn, labels = (t.to(device) for t in batch)
        total += model(input_ids=input_ids, attention_mask=attn, labels=labels).loss.item()
        n += 1
    if was_training:
        model.train()
    return math.exp(total / max(n, 1))


def save_adapter(model, path, meta: dict, opt=None, sched=None, step=None,
                 best_ppl=None) -> int:
    """Save only the trained tensors (pos_emb + gate_bias + LoRA) plus meta.

    When ``opt``/``sched`` are given the optimizer and scheduler state and the step
    counter go in too, which is what makes ``--resume`` able to continue a killed job
    rather than restart it. Returns the tensor count.
    """
    # Every cope.* parameter goes in whether or not it trains: a frozen gate_bias is
    # still part of the model serving has to reproduce, and dropping it would shift
    # every contextual position at inference.
    adapter = {n: p.detach().cpu() for n, p in model.named_parameters()
               if p.requires_grad or ".cope." in n}
    blob = {**meta, "state": adapter}
    if opt is not None:
        blob.update({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "step": step, "best_ppl": best_ppl})
    torch.save(blob, path)
    return len(adapter)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def _check_tiling_exact(model, device, seq_len: int = 24) -> None:
    """Assert --tile_q changes nothing numerically (--smoke self-test).

    Query-block tiling is only safe if every block reproduces the full forward exactly;
    this runs the same input at several block sizes, including boundaries that do not
    divide the sequence, and compares against the untiled result.
    """
    global _TRAIN_Q_BLOCK
    saved = _TRAIN_Q_BLOCK
    ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    attn = torch.ones(1, seq_len, dtype=torch.long, device=device)
    try:
        _TRAIN_Q_BLOCK = 0
        ref = model(input_ids=ids, attention_mask=attn).logits.float()
        worst = 0.0
        for blk in (1, 5, 7, seq_len - 1):
            _TRAIN_Q_BLOCK = blk
            got = model(input_ids=ids, attention_mask=attn).logits.float()
            worst = max(worst, (ref - got).abs().max().item())
    finally:
        _TRAIN_Q_BLOCK = saved
    print(f"[tiling] max |untiled - tiled| over blocks (1,5,7,{seq_len - 1}) = {worst:.3e}"
          f"  {'OK (exact)' if worst < 1e-5 else 'MISMATCH -- do not use --tile_q'}")


def build_model(args):
    if args.smoke:
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        cfg._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(cfg)
        tok = AutoTokenizer.from_pretrained("gpt2")  # any tokenizer for smoke text
        return model, tok
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Reject quantized checkpoints up front. LoRALinear needs a real nn.Linear
    # (base.in_features / base.weight); AWQ and compressed-tensors ship 4-bit packed
    # weights with neither, and merge_cope_adapter's `proj.weight.data += delta` has
    # nothing to add into. Failing here beats failing after a 60GB download.
    from transformers import AutoConfig

    qcfg = getattr(AutoConfig.from_pretrained(args.model), "quantization_config", None)
    if qcfg:
        method = (qcfg.get("quant_method") if isinstance(qcfg, dict)
                  else getattr(qcfg, "quant_method", "?"))
        raise SystemExit(
            f"{args.model} is quantized (quant_method={method}). This trainer needs "
            f"unquantized nn.Linear projections to attach LoRA to and to merge back "
            f"into. Train the full-precision checkpoint and quantize afterwards if "
            f"serving needs it."
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer id (default: --model). Use an *instruct* "
                         "tokenizer for --hf_dataset so the chat template exists.")
    ap.add_argument("--hf_dataset", default=None,
                    help="ShareGPT-style HF dataset, e.g. "
                         "Crystalcareai/Code-feedback-sharegpt-renamed")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--data_file", default=None, help="UTF-8 plain-text corpus")
    ap.add_argument("--output_dir", default="./cope_adapter")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--npos_max", type=int, default=0,
                    help="CoPE position slots; 0 -> max_seq_len (positions <= seq len)")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos_emb_lr", type=float, default=None,
                    help="separate LR for the CoPE pos_emb (default: same as --lr). "
                         "Set it HIGHER than --lr, not lower. pos_emb is zero-init, and "
                         "while the table is flat the position term is identically 0 and "
                         "the gates get no gradient at all -- the table has to grow first "
                         "or nothing else can learn. (The old advice to use lr/5..lr/10 "
                         "because the reverse-cumsum scales gradients by O(seq_len) is a "
                         "plain-SGD argument; AdamW normalises gradient magnitude away.) "
                         "Watch the pos/attn ratio printed at each eval: if it stays "
                         "<0.05 the model is ignoring CoPE and only the LoRA is training.")
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--batch_size", type=int, default=1, help="micro-batch size")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="micro-batches per optimizer step (effective batch = "
                         "batch_size * grad_accum)")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    # --- Gate-selectivity regularizer (attacks the root cause of the length cliff:
    # --- unregularized gates give span ~= 0.27*L, so long contexts overflow the pos
    # --- budget / mass-collapse. See accumulate_gate_reg_grads.) OFF at 0.
    ap.add_argument("--gate_reg", type=float, default=0.0,
                    help="span-cap coefficient (lambda_span); >0 turns the gate "
                         "regularizer ON. Penalizes relu(span - gate_span_target)/target "
                         "per query row so contextual span stays under the position budget.")
    ap.add_argument("--gate_span_target", type=int, default=512,
                    help="S: target max contextual span per query. Keep < npos_max (the "
                         "well-trained slot range) so trained spans never hit the clamp.")
    ap.add_argument("--gate_bimod", type=float, default=0.0,
                    help="bimodality coefficient (lambda_bimod): penalizes g*(1-g) to push "
                         "gates toward 0/1 (selective/decisive, the paper's mechanism). "
                         "0 = span-cap only (gates shrink uniformly = length-normalized).")
    ap.add_argument("--eval_samples", type=int, default=200,
                    help="held-out conversations reserved for perplexity eval")
    ap.add_argument("--eval_batches", type=int, default=50,
                    help="batches actually scored per eval (0 = the whole held-out "
                         "split). Kept below --eval_samples by default so eval stays "
                         "cheap; raise it if the ppl curve looks noisy.")
    ap.add_argument("--gate_bias_span", type=float, default=0.0,
                    help="bias the gates at init so a full-length row starts at ~this "
                         "many contextual positions (0 = off, the old behaviour). At init "
                         "sigmoid(q.k)~0.5 gives a span of seq_len/2, which overruns "
                         "npos_max and clamps -- and clamp has zero gradient, so every "
                         "clamped key is frozen out of training. Set it below --npos_max, "
                         "e.g. 256 for npos_max=1024.")
    ap.add_argument("--gate_bias_lr", type=float, default=0.0,
                    help="LR for the gate bias; 0 (default) FREEZES it at its init "
                         "value. It is an initialisation device -- letting it train at "
                         "the pos_emb rate moves where tokens land in the table while "
                         "the table is being rewritten, and the two chase each other "
                         "instead of converging. Frozen or <=pos_emb_lr/100.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch/data shuffling so a 24h job is reproducible")
    ap.add_argument("--resume", default=None,
                    help="path to a cope_adapter_last.pt written by this script; "
                         "restores weights, optimizer, scheduler and step counter")
    ap.add_argument("--lora_warmup_steps", type=int, default=0,
                    help="freeze the LoRA for the first N steps so pos_emb is the ONLY "
                         "parameter that can reduce the loss. Without this the LoRA "
                         "absorbs everything: measured on Qwen3-8B, it drove ppl 199.7 "
                         "-> 3.18 while pos/attn stayed at 0.017, i.e. the position table "
                         "never became load-bearing and no learning rate fixed it (a 167x "
                         "range moved pos/attn only 0.005 -> 0.029). Starting from a "
                         "RoPE-less model at ppl ~200, pos_emb alone has to carry the "
                         "positional work before the LoRA is allowed to help.")
    ap.add_argument("--keep_rope", action="store_true",
                    help="CONTROL RUN: train the identical LoRA setup but leave RoPE in "
                         "place and attach no CoPE. Without this the CoPE numbers are "
                         "uninterpretable -- the 'RoPE baseline' is the STOCK model, so a "
                         "finetuned CoPE model beating it says nothing about position. "
                         "Compare this run's ppl against the CoPE run's to find out "
                         "whether position matters on this corpus at all.")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="skip the pre-injection RoPE perplexity measurement")
    ap.add_argument("--tile_q", type=int, default=0,
                    help="query-block size for the CoPE attention (0 = off). Bounds "
                         "attention activation memory to O(H*tile_q*seq_len) by running "
                         "each query block under its own checkpoint. Numerically exact, "
                         "costs ~1 extra attention forward. Use at --max_seq_len >= 4096.")
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--save_every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing -- recommended: eager CoPE attention "
                         "materialises [B,H,T,T], heavy at seq_len 2048.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        import tempfile
        args.max_seq_len, args.device = 32, "cpu"
        # Keep an explicitly-passed --steps so `--smoke --resume ... --steps N` can
        # actually test continuing a run; only override the untouched default.
        if args.steps == ap.get_default("steps"):
            args.steps = 20
        args.grad_accum, args.eval_every, args.save_every = 2, 10, 10
        args.eval_samples = 8
        # Never write smoke output into a real --output_dir (it may hold a trained
        # adapter). Use a throwaway temp dir.
        args.output_dir = tempfile.mkdtemp(prefix="cope_smoke_")

    npos_max = args.npos_max or args.max_seq_len

    global _TRAIN_Q_BLOCK
    _TRAIN_Q_BLOCK = args.tile_q

    _GATE_REG["on"] = args.gate_reg > 0 or args.gate_bimod > 0
    _GATE_REG["span_target"] = args.gate_span_target
    _GATE_REG["lam_span"] = args.gate_reg
    _GATE_REG["lam_bimod"] = args.gate_bimod

    torch.manual_seed(args.seed)

    model, tok = build_model(args)
    model.to(args.device)

    if args.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # Backbone (incl. embeddings) is frozen, so the checkpointed layer inputs would
        # have requires_grad=False and backward through the checkpoint would break
        # ("does not require grad"). This hook re-enables grad on the embedding output
        # so gradients reach pos_emb/LoRA -- exactly what PEFT does under the hood.
        model.enable_input_require_grads()

    # Data comes first: the RoPE baseline below has to be measured on the SAME eval set
    # the CoPE run is scored against, and it has to happen before inject_cope().
    if args.smoke:
        ds = SyntheticVarLen(vocab=model.config.vocab_size, seq_len=args.max_seq_len)
    elif args.hf_dataset:
        ds = ShareGPTDataset(args.hf_dataset, args.hf_split, tok, args.max_seq_len,
                             max_samples=args.max_samples)
    elif args.data_file:
        ds = TextBlocks(args.data_file, tok, args.max_seq_len)
    else:
        raise SystemExit("provide --hf_dataset or --data_file (or --smoke)")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    collate = lambda b: collate_pad(b, pad_id)

    n_eval = min(args.eval_samples, max(1, len(ds) // 5))
    train_ds, eval_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_eval, n_eval],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate)
    print(f"train={len(train_ds)} eval={len(eval_ds)}  "
          f"effective_batch={args.batch_size * args.grad_accum}")

    # ---- RoPE baseline: the pass mark. Convergence is defined as "perplexity recovers
    # ---- toward the RoPE model", so measure that number on this exact eval set BEFORE
    # ---- RoPE is torn out. Without it a final ppl is unreadable: 2.8 could be a triumph
    # ---- or a disaster depending on where the stock model sits.
    baseline_ppl = None
    if not args.skip_baseline:
        baseline_ppl = evaluate(model, eval_loader, args.device,
                                max_batches=args.eval_batches or None)
        print(f"RoPE baseline ppl (stock model, same eval set): {baseline_ppl:.3f}")

    # Calibrate the gate bias against the length the model will ACTUALLY see, not the
    # truncation cap. init_gate_bias solves sigmoid(b) = target/seq_len, so feeding it
    # --max_seq_len when the corpus is shorter starts the span proportionally too low
    # (Code-Feedback medians ~1150 tokens against a 4096 cap -> 3.5x too low).
    import statistics as _stats

    _sample = [ds[i].numel() for i in range(0, len(ds), max(1, len(ds) // 512))][:512]
    typical_len = int(_stats.median(_sample)) if _sample else args.max_seq_len
    print(f"sample length: median {typical_len}  mean {int(_stats.mean(_sample))}  "
          f"max {max(_sample)}  (cap {args.max_seq_len})")
    if args.keep_rope:
        print("CONTROL RUN (--keep_rope): RoPE kept, no CoPE attached; LoRA only")
    else:
        inject_cope(model, npos_max=npos_max, gate_bias_span=args.gate_bias_span,
                    seq_len=typical_len)
    add_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    # Re-home CoPE modules onto the model device/after any wrapping.
    model.to(args.device)
    trainable = mark_trainable(model, train_gate_bias=args.gate_bias_lr > 0)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.3f}%)  npos_max={npos_max}"
          f"{f'  tile_q={args.tile_q}' if args.tile_q else ''}"
          f"{f'  gate_bias_span={args.gate_bias_span:g}' if args.gate_bias_span else ''}")
    # Contextual span grows with sequence length (~0.5*L unbiased), so a table sized to
    # max_seq_len has an upper half no sample can ever reach: those slots stay at their
    # zero init and contribute no position signal when a longer prompt lands on them.
    if not args.keep_rope and npos_max > args.max_seq_len // 2:
        print(f"  NOTE: npos_max={npos_max} vs max_seq_len={args.max_seq_len}: expect "
              f"only slots [0, ~{args.max_seq_len // 2}] to receive gradient. The "
              f"pos_emb coverage report at the end of the run shows what actually "
              f"trained -- serving prompts longer than --max_seq_len will index past it.")
    if not args.keep_rope and not args.gate_bias_span \
            and npos_max < args.max_seq_len // 2:
        print(f"  NOTE: unbiased gates start at a span of ~{args.max_seq_len // 2} but "
              f"npos_max={npos_max}, so most positions clamp at init and clamped keys "
              f"get NO gradient. Consider --gate_bias_span {npos_max // 4}.")

    # Three param groups. pos_emb gates the whole gradient chain: while the table is flat
    # the position term is 0 and so is its derivative w.r.t. the gates, so nothing else
    # can learn until it grows. It wants an LR >= the LoRA one, not lower. gate_bias gets
    # its OWN group because it decides where tokens land in the table -- at the pos_emb
    # rate it moves the target while the table is being rewritten and the two chase each
    # other instead of converging.
    pos_emb_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "cope.pos_emb" in n]
    gate_bias_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "cope.gate_bias" in n]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    pos_emb_lr = args.pos_emb_lr if args.pos_emb_lr is not None else args.lr
    groups = [{"params": lora_params, "lr": args.lr}]
    if pos_emb_params:  # empty under --keep_rope
        groups.append({"params": pos_emb_params, "lr": pos_emb_lr})
    if gate_bias_params:
        groups.append({"params": gate_bias_params, "lr": args.gate_bias_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    group_lrs = [g["lr"] for g in groups]
    gb_note = (f"{args.gate_bias_lr:.2e} ({len(gate_bias_params)} tensors)"
               if gate_bias_params else "FROZEN at init")
    print(f"optimizer: lora_lr={args.lr:.2e} ({len(lora_params)} tensors)  "
          f"pos_emb_lr={pos_emb_lr:.2e} ({len(pos_emb_params)} tensors)  "
          f"gate_bias_lr={gb_note}  "
          f"warmup={int(args.warmup_ratio * args.steps)} steps  clip={args.max_grad_norm}")
    if _GATE_REG["on"]:
        print(f"gate reg: ON  lam_span={args.gate_reg:.3g} span_target={args.gate_span_target} "
              f"lam_bimod={args.gate_bimod:.3g}  (recomputes gates outside grad-ckpt; "
              f"~1.5-2x attn compute)")
    sched = get_cosine_schedule_with_warmup(
        opt, int(args.warmup_ratio * args.steps), args.steps
    )
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"npos_max": npos_max, "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha}

    model.train()
    train_iter = infinite(train_loader)
    grad_seen = {"pos_emb": False, "lora": False}
    best_ppl = float("inf")
    first_loss = None
    start_step = 0

    if args.resume:
        # A 24h SLURM job that dies at hour 23 should not start over. cope_adapter_last.pt
        # carries the optimizer/scheduler/step alongside the weights for exactly this.
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        if "optimizer" in ck:
            # Optimizer state only loads if the param groups still line up. Changing
            # --gate_bias_lr across a resume moves gate_bias between groups, so tolerate
            # the mismatch: the weights and the step counter are the valuable part, and
            # Adam's moments rebuild within a few dozen steps.
            try:
                opt.load_state_dict(ck["optimizer"])
                sched.load_state_dict(ck["scheduler"])
            except ValueError as e:
                print(f"  optimizer state not reusable ({e}); keeping weights and step, "
                      f"restarting Adam moments")
            start_step = ck.get("step", 0) + 1
            best_ppl = ck.get("best_ppl", float("inf"))
            # Command-line LRs override the checkpoint's. Resuming specifically to lower a
            # learning rate is the main reason to resume a run that trained but did not
            # converge, and loading the optimizer state would otherwise restore the exact
            # LR you are trying to change. base_lrs is what the cosine lambda multiplies.
            for g, lr in zip(opt.param_groups, group_lrs):
                g["lr"] = g["initial_lr"] = lr
            sched.base_lrs = list(group_lrs)
            print(f"  LRs set from the command line: "
                  f"{', '.join(f'{lr:.2e}' for lr in group_lrs)}")
        else:
            # cope_adapter_best.pt is written without optimizer state on purpose (it is a
            # weights artifact for serving), so resuming from it silently restarts the
            # step counter. Point at the file that can actually continue a run.
            hint = (" -- use cope_adapter_last.pt to continue a run; best.pt never "
                    "carries optimizer state") if "best" in str(args.resume) else \
                   " (checkpoint predates resume support)"
            print(f"  weights loaded but NO optimizer state, restarting at step 0{hint}")
        print(f"resumed from {args.resume}: step {start_step}, best_ppl {best_ppl:.3f}, "
              f"{len(ck['state'])} tensors loaded ({len(unexpected)} unexpected)")
        if start_step >= args.steps:
            raise SystemExit(
                f"checkpoint is already at step {start_step} of --steps {args.steps}: "
                f"nothing left to run. Raise --steps to continue training it."
            )

    # LoRA warmup: freeze the adapters so the only downhill direction is through pos_emb.
    # requires_grad=False leaves their .grad as None, so AdamW skips them entirely and no
    # stale momentum is applied when they are released.
    if args.lora_warmup_steps and not args.keep_rope:
        for p_ in lora_params:
            p_.requires_grad_(False)
        print(f"LoRA frozen for the first {args.lora_warmup_steps} steps "
              f"({len(lora_params)} tensors); only pos_emb can reduce the loss")

    reg_diag = {"span": 0.0, "span_max": 0.0, "L_span": 0.0, "L_bimod": 0.0}
    eval_batches = args.eval_batches or None
    pos_ratio_and_reset()  # discard whatever the baseline pass accumulated
    for step in range(start_step, args.steps):
        if args.lora_warmup_steps and step == args.lora_warmup_steps and not args.keep_rope:
            for p_ in lora_params:
                p_.requires_grad_(True)
            print(f"step {step:5d}  LoRA released ({len(lora_params)} tensors now training)")
        opt.zero_grad()
        micro_loss = 0.0
        for _ in range(args.grad_accum):
            input_ids, attn, labels = (t.to(args.device) for t in next(train_iter))
            loss = model(
                input_ids=input_ids, attention_mask=attn, labels=labels
            ).loss / args.grad_accum
            loss.backward()
            micro_loss += loss.item()
            # Aux gate-selectivity loss: recompute gates from the detached stash (outside
            # the checkpointed graph) and accumulate its grads into q/k LoRA. Same 1/accum
            # scale so it tracks the effective batch. No-op when the reg is off.
            if _GATE_REG["on"]:
                reg_diag = accumulate_gate_reg_grads(model, scale=1.0 / args.grad_accum)

        # Keep checking until BOTH have been seen: under --lora_warmup_steps the LoRA
        # has no gradient for the first N steps, and a pos_emb-only guard would latch
        # early and report lora=False for the whole run.
        if not (grad_seen["pos_emb"] and grad_seen["lora"]):
            for name, p in model.named_parameters():
                if p.grad is None or p.grad.abs().sum() == 0:
                    continue
                if "cope.pos_emb" in name:
                    grad_seen["pos_emb"] = True
                elif "lora_" in name:
                    grad_seen["lora"] = True

        torch.nn.utils.clip_grad_norm_(
            [t for t in trainable if t.grad is not None], args.max_grad_norm)
        opt.step()
        sched.step()

        if first_loss is None:
            first_loss = micro_loss
        if step % max(1, args.steps // 20) == 0:
            reg_str = ""
            if _GATE_REG["on"]:
                # span_max is the number that says whether the cap is binding; the mean
                # is dragged down by early rows that can never exceed the target.
                reg_str = (f"  span mean {reg_diag['span']:.1f} max "
                           f"{reg_diag['span_max']:.1f} (->{_GATE_REG['span_target']}) "
                           f"L_span {reg_diag['L_span']:.3f} L_bimod {reg_diag['L_bimod']:.3f}")
            print(f"step {step:5d}  loss {micro_loss:.4f}  "
                  f"ppl {math.exp(min(micro_loss, 20)):.1f}  lr {sched.get_last_lr()[0]:.2e}"
                  f"{reg_str}")

        is_last = step == args.steps - 1
        if (step + 1) % args.eval_every == 0 or is_last:
            ratio = pos_ratio_and_reset()
            ppl = evaluate(model, eval_loader, args.device, max_batches=eval_batches)
            tag = ""
            if ppl < best_ppl:
                best_ppl = ppl
                save_adapter(model, out_dir / "cope_adapter_best.pt", meta)
                tag = "  <- best (saved)"
            gap = f" vs stock {baseline_ppl:.2f} ({ppl / baseline_ppl:.2f}x)" \
                if baseline_ppl else ""
            # pos/attn < ~0.05 means the position term cannot shift any attention weight:
            # the LoRA is absorbing the loss and CoPE is along for the ride.
            if args.keep_rope:
                print(f"  [eval] step {step:5d}  ppl {ppl:.2f}{gap}{tag}")
            else:
                flag = "  <- CoPE INERT, raise --pos_emb_lr" if ratio < 0.05 else ""
                print(f"  [eval] step {step:5d}  ppl {ppl:.2f}{gap}  "
                      f"pos/attn {ratio:.3f}{flag}{tag}")
        if (step + 1) % args.save_every == 0 or is_last:
            save_adapter(model, out_dir / "cope_adapter_last.pt", meta,
                         opt=opt, sched=sched, step=step, best_ppl=best_ppl)

    fl = f"{first_loss:.4f}" if first_loss is not None else "n/a (resumed)"
    print(f"done. first_loss={fl}  best_eval_ppl={best_ppl:.2f}"
          + (f"  (RoPE baseline {baseline_ppl:.2f}, ratio "
             f"{best_ppl / baseline_ppl:.2f}x)" if baseline_ppl else ""))
    print(f"gradients reached: pos_emb={grad_seen['pos_emb']} lora={grad_seen['lora']}")
    if not args.keep_rope:
        print(f"pos_emb coverage: {pos_emb_coverage(model)}")

    # CKSim sanity (drift): should print numbers in [-1, 1], ideally high & flat.
    if args.smoke:
        _check_tiling_exact(model, args.device)
        seg_ids = torch.randint(0, model.config.vocab_size, (1, 12))
        shifts = [0, 8, 16]
    else:
        seg_ids = tok(
            "The capital city of France is Paris, which is located in Europe.",
            return_tensors="pt",
        ).input_ids
        shifts = [0, 100, 300, 600, 900]
    cks = cksim_vs_shift(model, seg_ids, shifts=shifts)
    print("CKSim vs shift (CoPE):", {k: round(v, 4) for k, v in cks.items()})
    print("CKSim by layer:", cksim_by_layer(model, seg_ids, shift=max(shifts)))
    print(f"adapters in {out_dir}: cope_adapter_best.pt (lowest eval ppl), "
          f"cope_adapter_last.pt")


if __name__ == "__main__":
    main()