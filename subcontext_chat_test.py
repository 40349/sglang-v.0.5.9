"""Smoke test for CacheSlide sub-contexts on the OpenAI chat endpoint.

Mimics the mini-swe-agent loop: a fixed system prompt plus a message list that grows
by one assistant/user pair per turn. Watch the server log while this runs -- each
request should print TRACE-2 with extra_keys=['system_prompt_key', 'messages_key'] and
TRACE-4 with a per-namespace hit, where system_prompt_key hits 100% from turn 2 on and
messages_key hits the whole previous tail.

Usage: python subcontext_chat_test.py [base_url]
"""

import sys

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM = (
    "You are a helpful assistant that solves software issues by running bash "
    "commands. Reply with exactly one command in a ```bash block."
)

FOLLOWUPS = [
    "<returncode>0</returncode>\n<output>django  docs  tests  setup.py</output>",
    "<returncode>0</returncode>\n<output>django/core/validators.py:230</output>",
    "<returncode>0</returncode>\n<output>patch applied</output>",
]


def main():
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Fix django__django-11099 in /testbed."},
    ]

    for turn, followup in enumerate([None] + FOLLOWUPS):
        if followup is not None:
            messages.append({"role": "user", "content": followup})

        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 48,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(
            f"turn {turn}: prompt_tokens={usage.get('prompt_tokens')} "
            f"cached_tokens={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')}"
        )
        print(f"  reply: {reply.strip()[:100]!r}")
        messages.append({"role": "assistant", "content": reply})

    print(
        "\nDone. In the server log, check that every request printed TRACE-2 with "
        "two extra_keys and that TRACE-4 shows a growing messages_key hit."
    )


if __name__ == "__main__":
    main()
