#!/usr/bin/env python
"""量測 CoPE 的 contextual position p_ij 隨 seq_len 的成長（verify_cope.py 之外的診斷）。

這不是「訓練有效嗎」——那是 verify_cope.py 的消融測試。這裡回答 CoPE 專屬的問題：
  1. p_ij 是否隨 seq_len 線性成長？線性 => gate 不具選擇性，CoPE 退化成縮放過的
     絕對位置，PMKD 沒解掉。次線性/飽和 => 符合論文預期。
  2. 外推到 15.7k prompt 時 span 會不會超過 npos_max（決定服務長 prompt 會不會飽和）。

作法：patch ContextualPositionEmbedding.positions_from_gates，記錄回傳值的統計。
只跑 forward + no_grad，記憶體遠低於訓練，3090 大約可到 seq_len 2048。

用法（沿用 verify_cope.py 的載入方式）：
    python scripts/cacheslide_sim/test_8192.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter cope_adapter/h200_seq8192/cope_adapter_last.pt \
        --hf_dataset Crystalcareai/Code-feedback-sharegpt-renamed \
        --seq_lens 256,512,1024,2048 --n_prompts 8 --bf16
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cope as T  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaAttention  # noqa: E402

ContextualPositionEmbedding = T.ContextualPositionEmbedding

# layer_idx -> list of per-prompt dicts
REC: dict[int, list] = defaultdict(list)
_orig_pfg = ContextualPositionEmbedding.positions_from_gates


def patched_pfg(self, attn_logits, causal_mask):
    p = _orig_pfg(self, attn_logits, causal_mask)
    with torch.no_grad():
        li = getattr(self, "_layer_idx", -1)
        pf = p.float()
        T_ = pf.shape[-1]
        # p[..., i, j] = Σ_{l=j..i} σ(gate).  第 -1 列第 0 欄 = 整段序列的總 gate 質量
        # = 這段 context 在 CoPE 位置軸上的「跨度」。
        span = pf[..., -1, 0].flatten()
        # 因果區內的所有有效位置（上三角是無效的）
        tri = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=pf.device))
        vals = pf[..., tri].flatten()
        # torch.quantile 上限約 2^24 元素；因果區每 head O(T^2)，T>=1024 就爆掉。
        # 百分位數對均勻抽樣不敏感，先抽樣再算。
        if vals.numel() > 2_000_000:
            vals = vals[:: vals.numel() // 2_000_000 + 1]
        ceil_ = self.npos_max - 1
        REC[li].append({
            "T": T_,
            "span_mean": span.mean().item(),
            "span_max": span.max().item(),
            "p_median": vals.median().item(),
            "p_p99": vals.quantile(0.99).item(),
            "clamp_frac": (vals >= ceil_ - 1e-3).float().mean().item(),
        })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--hf_dataset", default="Crystalcareai/Code-feedback-sharegpt-renamed")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--seq_lens", default="256,512,1024,2048")
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    ckpt = torch.load(args.adapter, map_location="cpu", weights_only=False)
    npos_max = ckpt["npos_max"]
    print(f"[info] adapter npos_max={npos_max} lora_rank={ckpt['lora_rank']}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
    model.to(args.device)
    T.inject_cope(model, npos_max=npos_max)
    T.add_lora(model, r=ckpt["lora_rank"], alpha=ckpt["lora_alpha"])
    model.to(args.device)

    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    not_loaded = [k for k in ckpt["state"] if k in set(missing)]
    if not_loaded:
        sys.exit(f"[FAIL] adapter 有 {len(not_loaded)} 個 tensor 沒載進去，例如 {not_loaded[:3]}")
    print(f"[ok] adapter 載入 {len(ckpt['state'])} 個 tensor")

    # 標記層號，讓 hook 知道自己是第幾層
    for i, m in enumerate(model.model.layers):
        m.self_attn.cope._layer_idx = i
    ContextualPositionEmbedding.positions_from_gates = patched_pfg
    model.eval()

    # 用真實 prompt，不用隨機 token
    ds = T.ShareGPTDataset(args.hf_dataset, args.hf_split, tok,
                           max(seq_lens), max_samples=args.n_prompts * 8)
    pool = [ds[i] for i in range(len(ds))]
    pool = [x for x in pool if x.numel() >= max(seq_lens)][:args.n_prompts]
    if not pool:
        sys.exit(f"[FAIL] 資料集裡沒有 >= {max(seq_lens)} token 的對話，改小 --seq_lens")
    print(f"[ok] 取到 {len(pool)} 段夠長的 prompt\n")

    curve = {}
    for L in seq_lens:
        REC.clear()
        with torch.no_grad():
            for ids in pool:
                model(input_ids=ids[:L].unsqueeze(0).to(args.device))
        n_layers = len(REC)
        spans = {li: sum(r["span_mean"] for r in rs) / len(rs) for li, rs in REC.items()}
        clamp = max(max(r["clamp_frac"] for r in rs) for rs in REC.values())
        med = sum(sum(r["p_median"] for r in rs) / len(rs) for rs in REC.values()) / n_layers
        span_all = sum(spans.values()) / n_layers
        curve[L] = span_all
        print(f"seq_len={L:5d}  平均 span={span_all:8.1f}  斜率 span/T={span_all / L:.3f}  "
              f"p 中位數={med:7.1f}  clamp 命中率={clamp:.4%}")
        print(f"             淺層 span L0={spans.get(0, 0):.1f} L1={spans.get(1, 0):.1f}   "
              f"深層 L{n_layers-2}={spans.get(n_layers-2, 0):.1f} L{n_layers-1}={spans.get(n_layers-1, 0):.1f}")

    print("\n" + "=" * 60)
    ls = sorted(curve)
    if len(ls) >= 2:
        import math
        # log-log 斜率：1.0 = 線性，<1 = 次線性（好）
        b = ((math.log(curve[ls[-1]]) - math.log(curve[ls[0]]))
             / (math.log(ls[-1]) - math.log(ls[0])))
        print(f"span ~ T^{b:.2f}   (1.0 = 完全線性)")
        if b > 0.85:
            print("  => 近乎線性。gate 不具選擇性，CoPE 退化成縮放過的絕對位置。")
            print("     平移 Δ token 會讓位置移動約 %.2f·Δ，PMKD 未解決。" % (curve[ls[-1]] / ls[-1]))
            print(f"     外推 15.7k prompt 的 span ≈ {curve[ls[-1]] / ls[-1] * 15700:.0f}"
                  f"（npos_max={npos_max} {'不夠' if curve[ls[-1]] / ls[-1] * 15700 > npos_max else '夠'}）")
        else:
            print("  => 次線性，gate 有選擇性。符合論文預期，CCPE 的 e* 會是穩定窄區間。")
    print("=" * 60)


if __name__ == "__main__":
    main()