#!/usr/bin/env python
"""Find the prompt length where a CoPE server starts degenerating -- CacheSlide.

Truncates the capture's coherent English system_prompt to a range of lengths, appends
one fixed question, and hits /v1/chat/completions (temp 0) on a RUNNING server. If the
output starts looping/repeating past some length, that length is the usable-context
ceiling (the CoPE position cliff). This is a CLIENT -- start the server first:

    SGLANG_COPE_POS_EMB=./cope_merged_gate/cope_pos_emb.pt SGLANG_COPE_Q_BLOCK=64 \
    python -m sglang.launch_server --model-path ./cope_merged_gate \
      --attention-backend torch_native --mem-fraction-static 0.85 --port 30000

then (from anywhere):  python scripts/cacheslide_sim/length_sweep.py
"""
import json
import pathlib

import requests
from transformers import AutoTokenizer

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CAPTURE = "2026-05-04T14-52-26-258Z_127_0001_chat_google_gemma-4-31b-it.json"
QUESTION = "\n\n---\nIn ONE short sentence, what is the main goal described above?"
TARGETS = [800, 1500, 2500, 4000, 5500, 7000]  # prompt tokens


def find_capture() -> str:
    """Locate the capture json: CWD first, then the repo root (script's parents[2])."""
    for cand in (pathlib.Path(CAPTURE),
                 pathlib.Path(__file__).resolve().parents[2] / CAPTURE):
        if cand.exists():
            return str(cand)
    raise SystemExit(
        f"capture json '{CAPTURE}' not found in CWD or repo root -- run from the repo "
        f"root or put the file there."
    )


def degenerate(text: str) -> str:
    """Flag repetition loops. Catches BOTH word-level ('No more No more ...') and
    char-level with no spaces ('-5-5-5-5...'), which the word-split check alone misses."""
    flags = []
    # --- char-level: a short period p dominating the string (e.g. '-5', 'ab') ---
    s = text.strip()
    if len(s) >= 12:
        for p in range(1, 7):
            if len(s) <= p:
                continue
            same = sum(s[i] == s[i - p] for i in range(p, len(s)))
            if same / (len(s) - p) > 0.8:
                flags.append(f"char-rep(p={p})")
                break
    # --- word-level: low unique ratio or a long run of the same token ---
    words = text.split()
    if len(words) >= 6:
        uniq = len(set(words)) / len(words)
        max_rep = cur = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                cur += 1
                max_rep = max(max_rep, cur)
            else:
                cur = 1
        if uniq < 0.4:
            flags.append(f"uniq={uniq:.2f}")
        if max_rep >= 4:
            flags.append(f"rep×{max_rep}")
    return " ".join(flags)


def main():
    cap = json.load(open(find_capture()))
    sysp = cap["context"]["systemPrompt"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    sys_ids = tok(sysp, add_special_tokens=False)["input_ids"]

    print(f"{'target':>7} {'prompt_tok':>10}  {'degen?':>14}  output")
    print("-" * 96)
    for tgt in TARGETS:
        k = min(tgt, len(sys_ids))
        text = tok.decode(sys_ids[:k]) + QUESTION
        payload = {"model": "m", "messages": [{"role": "user", "content": text}],
                   "max_tokens": 40, "temperature": 0}
        try:
            r = requests.post(URL, json=payload, timeout=300).json()
            out = r["choices"][0]["message"]["content"]
            ptok = r["usage"]["prompt_tokens"]
        except Exception as e:
            print(f"{tgt:>7}  ERROR {e}")
            continue
        d = degenerate(out)
        oneline = out.replace("\n", " ")[:70]
        print(f"{tgt:>7} {ptok:>10}  {(d or 'ok'):>14}  {oneline!r}")


if __name__ == "__main__":
    main()
