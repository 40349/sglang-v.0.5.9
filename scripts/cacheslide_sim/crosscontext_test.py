#!/usr/bin/env python3
"""Cross-context reuse test.

Goal: reproduce the cross-context approximation error (the WCA baseline). Two requests
share the SAME `tools` and `messages` sub-contexts but have DIFFERENT `system_prompt`.

Because each block is cached in its own radix namespace, the second request's
`tools`/`messages` KV is silently replaced (in `RadixCache._cache_unfinished_sub_contexts`)
with the FIRST request's cached slots -- which were computed under a different preceding
system_prompt. So the second request decodes against out-of-context KV.

How to use (two server sessions, to get a clean comparison):

  # (1) CORRECT baseline: run only the varied request on a fresh server.
  python scripts/cacheslide_sim/crosscontext_test.py --order B

  # (2) CONTAMINATED: restart the server, then prime with A and run B.
  python scripts/cacheslide_sim/crosscontext_test.py --order A,B

Compare B's output between (1) and (2). If they differ, the tools/messages KV cached
under system_prompt_A leaked into B -> cross-context contamination reproduced.

Notes:
  * Keep A's and B's system_prompt the SAME TOKEN LENGTH to isolate context
    contamination from RoPE position-shift. Use --pad-system to pad the shorter one.
    (A rough char-based pad; exact token-length matching is a later refinement.)
"""
import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
    return "\n".join(p for p in parts if p)


def build_sub_contexts(capture: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    """Same block layout as round1_from_json.py, but with an overridable system_prompt."""
    ctx = capture.get("context", {})
    tools = ctx.get("tools", []) or []
    messages = ctx.get("messages", []) or []

    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    user_text = _extract_user_text(messages)

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


def send(url: str, sub_contexts: List[Dict[str, str]], max_new_tokens: int,
         temperature: float, label: str) -> str:
    payload = {
        "sub_contexts": sub_contexts,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "stream": False,
    }
    endpoint = url.rstrip("/") + "/generate"
    sp_preview = sub_contexts[0]["content"][:60].replace("\n", "\\n")
    print(f"\n### Request {label}  (system_prompt: {sp_preview!r}...)")
    print(f"POST {endpoint}")
    resp = requests.post(endpoint, json=payload, timeout=600)
    resp.raise_for_status()
    out = resp.json()
    text = out.get("text") if isinstance(out, dict) else None
    text = text if text is not None else json.dumps(out, ensure_ascii=False)[:2000]
    print(f"--- Output {label} ---\n{text}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--json",
        default="2026-05-04T14-52-26-258Z_127_0001_chat_google_gemma-4-31b-it.json",
        help="Base openclaw normalized-request JSON capture.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--order", default="A,B",
        help="Comma-separated request order, e.g. 'A,B' (prime then vary) or 'B' (baseline).",
    )
    parser.add_argument(
        "--variant-system", default=None,
        help="system_prompt text for variant B. If omitted, B reuses the original "
             "system_prompt but with a marker sentence swapped in (still same tools/messages).",
    )
    parser.add_argument(
        "--pad-system", action="store_true",
        help="Pad the shorter system_prompt with spaces so A and B are ~equal char length "
             "(rough isolation of context-contamination from RoPE position-shift).",
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        capture = json.load(f)

    original_system = (capture.get("context", {}) or {}).get("systemPrompt", "") or ""

    # Variant B's system_prompt: user-supplied, or a minimal mutation of the original
    # (swap the first line) so it is a *different context* of similar length.
    if args.variant_system is not None:
        variant_system = args.variant_system
    else:
        lines = original_system.split("\n")
        lines[0] = "You are a different assistant with different priorities." if lines else original_system
        variant_system = "\n".join(lines)

    if args.pad_system:
        n = max(len(original_system), len(variant_system))
        original_system = original_system.ljust(n)
        variant_system = variant_system.ljust(n)

    variants = {
        "A": build_sub_contexts(capture, original_system),
        "B": build_sub_contexts(capture, variant_system),
    }

    print("=== Cross-context reuse test ===")
    print(f"  system_prompt A chars={len(original_system)}")
    print(f"  system_prompt B chars={len(variant_system)}")
    print(f"  order={args.order}")

    outputs = {}
    for label in [s.strip() for s in args.order.split(",") if s.strip()]:
        if label not in variants:
            print(f"!! unknown request label {label!r}, expected A or B", file=sys.stderr)
            return 2
        outputs[label] = send(
            args.url, variants[label], args.max_new_tokens, args.temperature, label
        )

    if "A" in outputs and "B" in outputs:
        print("\n=== NOTE ===")
        print("B ran AFTER A primed the tools/messages namespaces. B's tools/messages KV "
              "was reused from A's (different system_prompt) context. Compare this B "
              "output against B run alone on a fresh server (--order B) to see the "
              "cross-context contamination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
