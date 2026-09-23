#!/usr/bin/env python3
"""Sub-context round-1 driver.

Reads an openclaw normalized-request JSON capture, splits it into the three
sub-contexts (system_prompt / tools / messages), and POSTs one `/generate` request
to a running sglang server with the `sub_contexts` field.

The server tokenizes each block separately, matches/inserts it in its own radix
namespace, and decodes the turn-1 output. Start the server with SGLANG_SUBCTX_TRACE=1
to see the per-block matches, and SGLANG_DUMP_TREE=1 to print the tree.

Usage:
    conda activate sglangv59
    python scripts/subcontext_sim/round1_from_json.py \
        --json <capture>.json --url http://127.0.0.1:30000 --max-new-tokens 128

The blocks use a hand-written Qwen-style template, not the model's chat template.
"""
import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """Flatten the (last) user message's content into plain text."""
    text_parts: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
    return "\n".join(p for p in text_parts if p)


def build_sub_contexts(capture: Dict[str, Any]) -> List[Dict[str, str]]:
    """Turn a normalized-request capture into ordered sub-contexts."""
    ctx = capture.get("context", {})
    system_prompt = ctx.get("systemPrompt", "") or ""
    tools = ctx.get("tools", []) or []
    messages = ctx.get("messages", []) or []

    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    user_text = _extract_user_text(messages)

    # Ordered blocks; each is tokenized + cached in its own radix namespace.
    return [
        {
            "content": f"<|im_start|>system\n{system_prompt}\n\n# Available tools:\n",
            "extra_key": "system_prompt",
        },
        {
            "content": f"{tools_json}\n<|im_end|>\n",
            "extra_key": "tools",
        },
        {
            "content": (
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "extra_key": "messages",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the openclaw normalized-request JSON capture.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:30000",
        help="Base URL of the running sglang server.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request payload without sending it.",
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        capture = json.load(f)

    sub_contexts = build_sub_contexts(capture)

    print("=== Sub-context round-1 blocks ===")
    for sc in sub_contexts:
        preview = sc["content"][:80].replace("\n", "\\n")
        print(f"  extra_key={sc['extra_key']:<14} chars={len(sc['content']):>6}  {preview!r}")

    payload = {
        "sub_contexts": sub_contexts,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "stream": False,
    }

    if args.dry_run:
        print("\n=== Payload (dry run) ===")
        print(json.dumps(payload, ensure_ascii=False)[:2000])
        return 0

    endpoint = args.url.rstrip("/") + "/generate"
    print(f"\nPOST {endpoint}")
    resp = requests.post(endpoint, json=payload, timeout=600)
    resp.raise_for_status()
    out = resp.json()

    print("\n=== Decoded turn-1 output ===")
    text = out.get("text") if isinstance(out, dict) else None
    print(text if text is not None else json.dumps(out, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
