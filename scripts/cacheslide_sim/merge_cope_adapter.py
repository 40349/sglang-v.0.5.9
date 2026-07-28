"""Merge a trained CoPE adapter into base weights for serving (CacheSlide step 2a).

The adapter (cope_adapter_best.pt) holds LoRA deltas over q/k/v/o plus the CoPE
``pos_emb`` tables. For serving we fold the LoRA into the base linear weights
(so there is zero runtime LoRA overhead) and drop ``pos_emb`` into a side file
that the CoPE-enabled SGLang model loads per layer.

Merge identity (why this is exact): LoRALinear computes
    W x + (alpha/r) * B (A x) = (W + (alpha/r) B A) x
so folding ``W += (alpha/r) * B @ A`` reproduces the trained layer exactly.

Output:
    <output_dir>/                 HF checkpoint with LoRA folded in (safetensors)
    <output_dir>/cope_pos_emb.pt  {"npos_max": int, "pos_emb": {layer_idx: tensor}}

Run:
    python scripts/cacheslide_sim/merge_cope_adapter.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter ./cope_adapter/cope_adapter_best.pt \
        --output_dir ./cope_merged --bf16

Self-test (no download, checks the merge identity numerically):
    python scripts/cacheslide_sim/merge_cope_adapter.py --smoke
"""

from __future__ import annotations

import argparse
import pathlib

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention

_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def merge_lora_into_model(model, state: dict, rank: int, alpha: float) -> int:
    """Fold LoRA deltas from ``state`` into ``model``'s q/k/v/o weights in place.

    Returns the number of projections merged. Expects base ``model`` (plain Linears,
    no LoRA wrappers) and ``state`` keyed like
    ``model.layers.{i}.self_attn.{proj}.lora_{A,B}.weight``.
    """
    scaling = alpha / rank
    merged = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj_name in _TARGETS:
            ka = f"model.layers.{i}.self_attn.{proj_name}.lora_A.weight"
            kb = f"model.layers.{i}.self_attn.{proj_name}.lora_B.weight"
            if ka not in state or kb not in state:
                continue
            A = state[ka]  # [r, in]
            B = state[kb]  # [out, r]
            delta = (B @ A) * scaling  # [out, in]
            proj = getattr(attn, proj_name)
            proj.weight.data += delta.to(proj.weight.dtype).to(proj.weight.device)
            merged += 1
    return merged


def extract_pos_emb(state: dict) -> dict:
    """Pull per-layer ``cope.pos_emb`` tensors out of the adapter state."""
    out = {}
    for k, v in state.items():
        if k.endswith("self_attn.cope.pos_emb"):
            # ...model.layers.{i}.self_attn.cope.pos_emb
            i = int(k.split("model.layers.")[1].split(".")[0])
            out[i] = v
    return out


def _smoke():
    """Verify the merge identity on a random LoRALinear (no model download)."""
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_cope as T

    # fp32 throughout: LoRALinear runs its low-rank path in fp32 internally.
    torch.manual_seed(0)
    in_f, out_f, r, alpha = 32, 48, 8, 16
    base = torch.nn.Linear(in_f, out_f, bias=False)
    lora = T.LoRALinear(base, r=r, alpha=alpha)
    lora.lora_B.weight.data.normal_()  # B starts at 0; make the delta non-trivial

    x = torch.randn(4, in_f)
    ref = lora(x)  # trained-layer output

    merged = torch.nn.Linear(in_f, out_f, bias=False)
    merged.weight.data.copy_(base.weight.data)
    delta = (lora.lora_B.weight @ lora.lora_A.weight) * (alpha / r)
    merged.weight.data += delta
    got = merged(x)

    err = (ref - got).abs().max().item()
    assert err < 1e-4, err
    print(f"[ok] merge identity holds (max err {err:.2e}) -- folding LoRA is exact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tokenizer", default=None, help="default: --model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--output_dir", default="./cope_merged")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    ckpt = torch.load(args.adapter, map_location="cpu")
    state = ckpt["state"]
    rank, alpha, npos_max = ckpt["lora_rank"], ckpt["lora_alpha"], ckpt["npos_max"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    n_merged = merge_lora_into_model(model, state, rank, alpha)
    n_layers = sum(isinstance(m, LlamaAttention) for m in model.modules())
    print(f"merged {n_merged} LoRA projections ({n_merged // 4} layers x 4)")
    assert n_merged == n_layers * len(_TARGETS), "some projections were not merged"

    pos = extract_pos_emb(state)
    assert len(pos) == n_layers, f"pos_emb count {len(pos)} != layers {n_layers}"

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)  # HF checkpoint SGLang can load
    # Save the tokenizer too so the merged dir is self-contained for serving.
    AutoTokenizer.from_pretrained(args.tokenizer or args.model).save_pretrained(out_dir)
    torch.save({"npos_max": npos_max, "pos_emb": pos}, out_dir / "cope_pos_emb.pt")
    print(f"saved merged model + cope_pos_emb.pt ({len(pos)} tables, npos_max={npos_max}) "
          f"-> {out_dir}")
    print("NOTE: the merged checkpoint is a full-size copy of the backbone weights.")


if __name__ == "__main__":
    main()