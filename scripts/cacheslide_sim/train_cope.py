"""CoPE continued-pretraining for Llama (CacheSlide schedule, step 5 -- training).

Serving frameworks (vLLM/SGLang) cannot train, so this runs in plain HF
Transformers: it swaps RoPE for CoPE in every LlamaAttention, freezes the
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

# Query-block size for the training CoPE forward. The non-tiled forward materialises
# [B,H,T,T] tensors (int64 gather indices are 8 bytes each) -- ~137GB just for the two
# index tensors at T=16384, OOM even on an H200. Query rows are independent, so tiling
# is exact and bounds peak memory to O(H*block*kv_len). seq_len <= block => one block
# (no tiling). Set from --tile_q.
_TRAIN_Q_BLOCK = 2048


# ---------------------------------------------------------------------------
# 1. CoPE attention -- monkey-patch LlamaAttention.forward (RoPE removed)
# ---------------------------------------------------------------------------
def cope_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,  # (cos, sin) -- intentionally IGNORED (no RoPE)
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Training-time LlamaAttention.forward with CoPE instead of RoPE.

    Eager (materialised T x T logits) so the CoPE position bias can be added inside
    the softmax. No KV-cache path -- training only. Matches transformers 4.57's
    forward contract, returning ``(attn_output, attn_weights=None)``.
    """
    input_shape = hidden_states.shape[:-1]  # [B, T]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hq,T,d]
    key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hkv,T,d]
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Grouped-query attention: expand kv heads to match query heads.
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)

    seq_len = query.shape[-2]
    logits = torch.matmul(query, key.transpose(-1, -2)) * self.scaling  # [B,H,T,T]

    # Validity = causal AND (HF additive mask says "keep"). Used both to gate CoPE
    # positions and to mask the softmax.
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
    if attention_mask is not None:
        m = attention_mask[..., :seq_len]
        valid = causal & (m > torch.finfo(m.dtype).min / 2)  # [B,1,T,T]
    else:
        valid = causal  # [T,T]

    pos_bias = self.cope(query, logits, valid)  # [B,H,T,T]

    neg_inf = torch.finfo(logits.dtype).min
    masked = logits + pos_bias.masked_fill(~valid, 0.0)
    masked = masked.masked_fill(~valid, neg_inf)
    attn = torch.softmax(masked, dim=-1, dtype=torch.float32).to(query.dtype)

    out = torch.matmul(attn, value)  # [B,H,T,d]
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    out = self.o_proj(out)
    return out, None


def inject_cope(model, npos_max: int) -> None:
    """Attach a per-layer CoPE module and switch LlamaAttention to the CoPE forward."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    param_dtype = next(model.parameters()).dtype
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            cope = ContextualPositionEmbedding(head_dim=head_dim, npos_max=npos_max)
            # pos_emb trains in fp32 for stability even under a bf16 backbone.
            module.cope = cope.to(device=next(model.parameters()).device)
    # Class-level patch: every LlamaAttention instance now runs CoPE.
    LlamaAttention.forward = cope_attention_forward


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
    """Wrap the named attention projections of every LlamaAttention with LoRA."""
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            for name in targets:
                base = getattr(module, name)
                setattr(module, name, LoRALinear(base, r=r, alpha=alpha))


def mark_trainable(model) -> List[nn.Parameter]:
    """Freeze everything, then unfreeze CoPE ``pos_emb`` + LoRA. Returns trainables."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if "cope.pos_emb" in name or "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


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
    """

    def __init__(self, hf_name, split, tokenizer, seq_len, max_samples=None,
                 messages_field="messages"):
        from datasets import load_dataset

        if tokenizer.chat_template is None:
            raise ValueError(
                "tokenizer has no chat_template; pass --tokenizer <an instruct "
                "tokenizer, e.g. meta-llama/Llama-3.1-8B-Instruct> so conversations "
                "render in the served chat format."
            )
        ds = load_dataset(hf_name, split=split)
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
    """
    conv = [
        {"role": _SHAREGPT_ROLE_MAP.get(m["role"], m["role"]), "content": m["value"]}
        for m in messages
    ]
    try:
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
    model, segment_ids: torch.Tensor, shifts: List[int], layer_idx: int = -1
) -> Dict[int, float]:
    """Mean per-head cosine similarity of a segment's layer-``layer_idx`` keys when
    the segment sits at position 0 vs shifted right by a prefix of length ``shift``.

    ``segment_ids`` is a ``[1, L]`` token-id tensor (kept tokenizer-agnostic so the
    smoke path can pass random ids). Low drift (CKSim staying near 1 as shift grows)
    is exactly the CoPE property that makes shifted KV reuse near-lossless; run on
    the base (unpatched) model to get the RoPE curve for comparison.
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

    k_proj = attn.k_proj.base if isinstance(attn.k_proj, LoRALinear) else attn.k_proj
    handle = k_proj.register_forward_hook(hook)

    seg = segment_ids.to(device)
    seg_len = seg.shape[1]

    def seg_keys(prefix_len: int) -> torch.Tensor:
        if prefix_len == 0:
            ids = seg
            start = 0
        else:
            prefix = torch.randint(
                0, model.config.vocab_size, (1, prefix_len), device=device
            )
            ids = torch.cat([prefix, seg], dim=1)
            start = prefix_len
        model(input_ids=ids)
        k = captured["k"][0, start : start + seg_len]  # [seg_len, Hkv*d]
        return k

    base = seg_keys(0)
    out: Dict[int, float] = {}
    for s in shifts:
        shifted = seg_keys(s)
        cos = torch.nn.functional.cosine_similarity(
            base.float(), shifted.float(), dim=-1
        )
        out[s] = cos.mean().item()
    handle.remove()
    return out


# ---------------------------------------------------------------------------
# 5. Train helpers
# ---------------------------------------------------------------------------
def infinite(loader):
    """Yield batches forever, re-iterating the loader each epoch (no caching)."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    """Mean held-out perplexity over up to ``max_batches`` batches."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids, attn, labels = (t.to(device) for t in batch)
        total += model(input_ids=input_ids, attention_mask=attn, labels=labels).loss.item()
        n += 1
    if was_training:
        model.train()
    return math.exp(total / max(n, 1))


def save_adapter(model, path, meta: dict) -> int:
    """Save only the trained tensors (pos_emb + LoRA) plus meta. Returns tensor count."""
    adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({**meta, "state": adapter}, path)
    return len(adapter)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
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
                         "The gate reverse-cumsum makes pos_emb gradients scale ~O(seq_len), "
                         "so long-seq runs (>=4096) are more stable with pos_emb_lr < lr, "
                         "e.g. lr/5 .. lr/10.")
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--batch_size", type=int, default=1, help="micro-batch size")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="micro-batches per optimizer step (effective batch = "
                         "batch_size * grad_accum)")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_samples", type=int, default=200,
                    help="held-out conversations for perplexity eval")
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
        args.max_seq_len, args.steps, args.device = 32, 20, "cpu"
        args.grad_accum, args.eval_every, args.save_every = 2, 10, 10
        args.eval_samples = 8
        # Never write smoke output into a real --output_dir (it may hold a trained
        # adapter). Use a throwaway temp dir.
        args.output_dir = tempfile.mkdtemp(prefix="cope_smoke_")

    npos_max = args.npos_max or args.max_seq_len

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

    inject_cope(model, npos_max=npos_max)
    add_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    # Re-home CoPE modules onto the model device/after any wrapping.
    model.to(args.device)
    trainable = mark_trainable(model)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.3f}%)  npos_max={npos_max}")

    # Data
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
        ds, [len(ds) - n_eval, n_eval], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate)
    print(f"train={len(train_ds)} eval={len(eval_ds)}  "
          f"effective_batch={args.batch_size * args.grad_accum}")

    # Two param groups: pos_emb (the global positional backbone -- its reverse-cumsum
    # gradient scales with seq_len) can take a lower LR than the LoRA deltas. The cosine
    # scheduler scales every group by the same factor, so the ratio holds throughout.
    pos_emb_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "cope.pos_emb" in n]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    pos_emb_lr = args.pos_emb_lr if args.pos_emb_lr is not None else args.lr
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": pos_emb_params, "lr": pos_emb_lr}],
        weight_decay=args.weight_decay,
    )
    print(f"optimizer: lora_lr={args.lr:.2e} ({len(lora_params)} tensors)  "
          f"pos_emb_lr={pos_emb_lr:.2e} ({len(pos_emb_params)} tensors)  "
          f"warmup={int(args.warmup_ratio * args.steps)} steps  clip={args.max_grad_norm}")
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

    for step in range(args.steps):
        opt.zero_grad()
        micro_loss = 0.0
        for _ in range(args.grad_accum):
            input_ids, attn, labels = (t.to(args.device) for t in next(train_iter))
            loss = model(
                input_ids=input_ids, attention_mask=attn, labels=labels
            ).loss / args.grad_accum
            loss.backward()
            micro_loss += loss.item()

        if not grad_seen["pos_emb"]:  # one-time sanity that both groups train
            for name, p in model.named_parameters():
                if p.grad is None or p.grad.abs().sum() == 0:
                    continue
                if "cope.pos_emb" in name:
                    grad_seen["pos_emb"] = True
                elif "lora_" in name:
                    grad_seen["lora"] = True

        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        opt.step()
        sched.step()

        if first_loss is None:
            first_loss = micro_loss
        if step % max(1, args.steps // 20) == 0:
            print(f"step {step:5d}  loss {micro_loss:.4f}  "
                  f"ppl {math.exp(min(micro_loss, 20)):.1f}  lr {sched.get_last_lr()[0]:.2e}")

        is_last = step == args.steps - 1
        if (step + 1) % args.eval_every == 0 or is_last:
            ppl = evaluate(model, eval_loader, args.device)
            tag = ""
            if ppl < best_ppl:
                best_ppl = ppl
                save_adapter(model, out_dir / "cope_adapter_best.pt", meta)
                tag = "  <- best (saved)"
            print(f"  [eval] step {step:5d}  ppl {ppl:.2f}{tag}")
        if (step + 1) % args.save_every == 0 or is_last:
            save_adapter(model, out_dir / "cope_adapter_last.pt", meta)

    print(f"done. first_loss={first_loss:.4f}  best_eval_ppl={best_ppl:.2f}")
    print(f"gradients reached: pos_emb={grad_seen['pos_emb']} lora={grad_seen['lora']}")

    # CKSim sanity (drift): should print numbers in [-1, 1], ideally high & flat.
    if args.smoke:
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
    print(f"adapters in {out_dir}: cope_adapter_best.pt (lowest eval ppl), "
          f"cope_adapter_last.pt")


if __name__ == "__main__":
    main()