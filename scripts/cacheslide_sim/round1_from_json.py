#!/usr/bin/env python3
"""CacheSlide round-1 driver.

Reads an openclaw normalized-request JSON capture, splits it into the three
CacheSlide sub-contexts (system_prompt / tools / messages), and POSTs a single
`/generate` request to a running sglang server with the `sub_contexts` field.

This exercises Stage 0 of the CacheSlide pipeline end-to-end: the server tokenizes
each block independently, matches/inserts it in its own radix namespace, prints the
three per-namespace radix trees (TRACE-1..4 + pretty_print), and decodes the turn-1
output. See ~/.claude/plans/linked-wandering-bonbon.md.

Usage:
    conda activate sglangv59
    python scripts/cacheslide_sim/round1_from_json.py \
        --json 2026-05-04T14-52-26-258Z_127_0001_chat_google_gemma-4-31b-it.json \
        --url http://127.0.0.1:30000 --max-new-tokens 128

Note: contents are concatenated as a lightly-structured prompt so each block has a
clean boundary. Full chat-template fidelity for the target model is refined in a
later stage; Stage 0 only needs the pipeline shape to run.
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
    """Turn a normalized-request capture into ordered CacheSlide sub-contexts."""
    ctx = capture.get("context", {})
    system_prompt = ctx.get("systemPrompt", "") or ""
    tools = ctx.get("tools", []) or []
    messages = ctx.get("messages", []) or []

    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    user_text = _extract_user_text(messages)

    # Llama-3 chat format: these ARE special tokens for the Llama-3.1 tokenizer
    # (<|start_header_id|>=128006, <|eot_id|>=128009, ...), matching how the CoPE
    # adapter was trained (apply_chat_template on the instruct tokenizer). The old
    # ChatML <|im_start|> markers are NOT special tokens here -- they got split into
    # subword junk, which is why the model echoed the tool JSON instead of answering.
    def hdr(role: str) -> str:
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

    eot = "<|eot_id|>"

    # Ordered blocks; each is tokenized + cached in its own radix namespace.
    return [
        {
            "content": f"<|begin_of_text|>{hdr('system')}{system_prompt}\n\n# Available tools:\n",
            "extra_key": "system_prompt",
        },
        {
            "content": f"{tools_json}{eot}",
            "extra_key": "tools",
        },
        {
            "content": f"{hdr('user')}{user_text}{eot}{hdr('assistant')}",
            "extra_key": "messages",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        default="2026-05-04T14-52-26-258Z_127_0001_chat_google_gemma-4-31b-it.json",
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

    print("=== CacheSlide round-1 sub-contexts ===")
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
