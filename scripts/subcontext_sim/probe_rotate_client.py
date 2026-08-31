"""Drive both displaced-hit cases against a live server and read the trace back.

Three requests, chosen so that one exercises each rotation path. They differ only in
the system prompt; the user message is identical, so the `messages` block renders to
the same tokens every time and can only ever differ in *where* it sits.

  A: SYS_A + MSG   -> nothing cached yet. Leaves messages in the tree at canonical
                      len(SYS_A).
  B: SYS_B + MSG   -> SYS_B shares a 9-token prefix with SYS_A, so block 0 hits only
                      partially and the stitch stops there. messages is displaced but
                      unreachable in one pass -- STAGE 2: the chunk is cut at the block
                      edge, and once the gap is computed the block is rotated on.
  C: SYS_B + MSG   -> SYS_B now hits in full, and messages is still at canonical
                      len(SYS_A) while its offset is len(SYS_B) -- STAGE 1: rotated
                      straight into the stitch.

Read the numbers from the forward-trace JSONL, not the server log: the sub-context
`print()` traces are block-buffered in the scheduler process and the tail of them is
lost when the server is killed. The JSONL is line-buffered and always complete.

Usage:
    python probe_rotate_client.py <forward-trace.jsonl> [--url URL] [--model NAME]
"""

import json
import sys

import requests

import argparse

DEFAULT_URL = "http://127.0.0.1:30000"
DEFAULT_MODEL = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"

SYS_A = "You are a helpful assistant."
SYS_B = (
    "You are a helpful assistant. "
    + "Always answer carefully, and explain your reasoning step by step. " * 6
)
MSG = "Write a Python function that returns the nth Fibonacci number."


def send(base_url, model, label, system):
    r = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": MSG},
            ],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=180,
    )
    r.raise_for_status()
    d = r.json()
    usage = d.get("usage") or {}
    print(
        f"{label}: prompt={usage.get('prompt_tokens')} "
        f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
        f"text={d['choices'][0]['message']['content'][:48]!r}"
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="the SGLANG_FORWARD_TRACE jsonl the server wrote")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--control",
        action="store_true",
        help="the arm is expected to rotate nothing; do not warn about it",
    )
    args = ap.parse_args()

    requests.post(args.url + "/flush_cache", timeout=30)
    send(args.url, args.model, "A (short sys)      ", SYS_A)
    send(args.url, args.model, "B (long sys, 1st)  ", SYS_B)
    send(args.url, args.model, "C (long sys, 2nd)  ", SYS_B)

    rows = []
    with open(args.trace) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("type") and row.get("mode", "").startswith("extend"):
                rows.append(row)

    print(
        f"\n{'ct':>4} {'new':>6} {'cached':>7} {'matched':>8} {'moved':>6} {'rot':>5}"
    )
    tot = {"new_tokens": 0, "cached_tokens": 0, "moved_tokens": 0, "rotated_tokens": 0}
    for r in rows:
        print(
            f"{r['ct']:>4} {r['new_tokens']:>6} {r['cached_tokens']:>7} "
            f"{r['matched_tokens']:>8} {r['moved_tokens']:>6} "
            f"{r.get('rotated_tokens', '-'):>5}"
        )
        for k in tot:
            tot[k] += r.get(k, 0)
    print("\ntotals: " + "  ".join(f"{k}={v}" for k, v in tot.items()))
    if args.control:
        if tot["rotated_tokens"]:
            print("\nWARNING: the control arm rotated tokens; it is not a control.")
    elif tot["rotated_tokens"] == 0:
        print(
            "\nNo rotation happened. Check the server logged "
            "'Sub-context KV rotation ENABLED' and that both env vars are set."
        )


if __name__ == "__main__":
    main()
