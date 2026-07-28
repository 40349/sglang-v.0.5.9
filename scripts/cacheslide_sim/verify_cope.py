"""Verify a trained model is really using CoPE (not RoPE) -- CacheSlide.

Three checks:
  1. Wiring: LlamaAttention.forward is the CoPE forward (RoPE is gone).
  2. Learned: the loaded pos_emb tables are non-zero (the finetune wrote them).
  3. Ablation (the real proof): eval perplexity with the trained pos_emb vs with
     pos_emb ZEROED. Zeroing removes CoPE's position signal; if ppl blows up, the
     CoPE positions are load-bearing -- i.e. the model genuinely depends on CoPE.

Run:
    python scripts/cacheslide_sim/verify_cope.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --adapter ./cope_adapter/cope_adapter_best.pt \
        --hf_dataset Crystalcareai/Code-feedback-sharegpt-renamed \
        --max_seq_len 1024 --eval_samples 100 --bf16

Self-test (no download): python scripts/cacheslide_sim/verify_cope.py --smoke
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cope as T  # single source of truth for the CoPE plumbing
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention


def zero_pos_emb(model):
    """Zero every CoPE pos_emb (disables the position signal). Returns saved copies."""
    saved = {}
    for module in model.modules():
        if isinstance(module, LlamaAttention) and hasattr(module, "cope"):
            saved[id(module)] = module.cope.pos_emb.data.clone()
            module.cope.pos_emb.data.zero_()
    return saved


def restore_pos_emb(model, saved):
    for module in model.modules():
        if isinstance(module, LlamaAttention) and hasattr(module, "cope"):
            module.cope.pos_emb.data.copy_(saved[id(module)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--adapter", default=None, help="cope_adapter_best.pt")
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--eval_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # ---- Load base model + tokenizer, or a tiny random Llama for --smoke ----
    if args.smoke:
        args.max_seq_len, args.device, args.eval_samples = 32, "cpu", 8
        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=128)
        cfg._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(cfg)
        tok = AutoTokenizer.from_pretrained("gpt2")
        npos_max, rank, alpha = args.max_seq_len, 8, 16
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="eager",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
        ckpt = torch.load(args.adapter, map_location="cpu")
        npos_max = ckpt["npos_max"]
        rank, alpha = ckpt["lora_rank"], ckpt["lora_alpha"]

    model.to(args.device)
    T.inject_cope(model, npos_max=npos_max)
    T.add_lora(model, r=rank, alpha=alpha)
    model.to(args.device)

    # ---- Check 1: RoPE really replaced by CoPE ----
    is_cope = LlamaAttention.forward is T.cope_attention_forward
    print(f"[1] wiring: LlamaAttention.forward is CoPE (RoPE removed): {is_cope}")

    # ---- Load adapter (real run) or fake a 'trained' pos_emb (smoke) ----
    if args.smoke:
        for m in model.modules():
            if isinstance(m, LlamaAttention):
                m.cope.pos_emb.data.normal_(std=0.1)  # pretend it was trained
    else:
        missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
        adapter_keys = list(ckpt["state"].keys())
        not_loaded = [k for k in adapter_keys if k in set(missing)]
        print(f"[  ] adapter: loaded {len(adapter_keys) - len(not_loaded)}/"
              f"{len(adapter_keys)} tensors, unexpected={len(unexpected)}, "
              f"adapter-keys-missing={len(not_loaded)}")

    # ---- Check 2: pos_emb learned (non-zero) ----
    norms = [m.cope.pos_emb.data.norm().item()
             for m in model.modules() if isinstance(m, LlamaAttention)]
    nonzero = sum(n > 0 for n in norms)
    print(f"[2] learned: {nonzero}/{len(norms)} pos_emb tables non-zero "
          f"(mean L2 {sum(norms)/len(norms):.4f})")

    # ---- Eval data ----
    if args.smoke:
        ds = T.SyntheticVarLen(model.config.vocab_size, args.max_seq_len, n=args.eval_samples)
    else:
        ds = T.ShareGPTDataset(args.hf_dataset, args.hf_split, tok, args.max_seq_len,
                               max_samples=args.eval_samples)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=lambda b: T.collate_pad(b, pad_id))

    # ---- Check 3: ablation ----
    ppl_on = T.evaluate(model, loader, args.device, max_batches=args.eval_samples)
    saved = zero_pos_emb(model)
    ppl_off = T.evaluate(model, loader, args.device, max_batches=args.eval_samples)
    restore_pos_emb(model, saved)

    print(f"[3] ablation: ppl(CoPE on)={ppl_on:.2f}  ppl(pos_emb zeroed)={ppl_off:.2f}"
          f"  ratio={ppl_off / max(ppl_on, 1e-9):.1f}x")
    verdict = ppl_off > ppl_on * 1.5
    print(f"\nVERDICT: CoPE positions are {'LOAD-BEARING (verified)' if verdict else 'NOT clearly used'} "
          f"-- zeroing them {'blew up' if verdict else 'barely changed'} perplexity.")
    if args.smoke:
        print("(smoke: random weights, so the ratio is only a mechanism check)")


if __name__ == "__main__":
    main()
