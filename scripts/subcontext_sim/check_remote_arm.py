"""Verify a sglang server is the sub-context arm the caller asked for.

Reads the ``sub_context`` block of ``/server_info`` (absent on a server not running
this fork). Exits non-zero with the reason on any mismatch.

Usage:
    check_remote_arm.py URL --arm off|cdc[@ratio[,ratio...]] [--audit B]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def _ratios(v: str) -> list:
    return [float(r) for r in v.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="server base URL, e.g. http://140.118.202.100:8000")
    ap.add_argument(
        "--arm",
        required=True,
        help="an arm as sglang_server.sh takes it: off, cdc or cdc@ratio[,ratio...]",
    )
    ap.add_argument(
        "--audit",
        type=_bool,
        default=None,
        help="require the audit on or off (also catches a stale server still holding "
        "the port)",
    )
    args = ap.parse_args()
    name, _, ratio = args.arm.partition("@")
    if name not in ("off", "cdc") or (ratio and name != "cdc"):
        ap.error(f"unknown arm {args.arm!r}; want off, cdc or cdc@ratio")
    want_split = name == "cdc"
    want_ratios = _ratios(ratio or "0")

    base = args.url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/server_info", timeout=30) as r:
            info = json.load(r)
    except Exception as exc:  # noqa: BLE001 -- the reason is what the caller needs
        print(f"REFUSING: cannot read {base}/server_info: {exc}", file=sys.stderr)
        return 1

    sub = info.get("sub_context")
    if sub is None:
        print(
            "REFUSING: /server_info reports no sub_context block, so this server is "
            "not running the fork. The usual cause is `sglang serve` without "
            "PYTHONPATH pointing at the checkout: it imports the pip-installed "
            "sglang, which has no sub-context code, and serves happily.",
            file=sys.stderr,
        )
        return 1

    if "split_mode" in sub:
        print(
            "REFUSING: the server reports `split_mode`, so it runs a build from before "
            "the on/rot/idx arms were removed (2026-10-14), whatever arm it serves. "
            "Update that checkout and restart the server.",
            file=sys.stderr,
        )
        return 1

    if sub["split_enabled"] != want_split:
        print(
            f"REFUSING: asked for {args.arm} (split={want_split}), but the server "
            f"reports {sub}",
            file=sys.stderr,
        )
        return 1

    got = sub.get("topk_ratios")
    if got is None or len(got) != len(want_ratios) or any(
        abs(float(g) - w) > 1e-9 for g, w in zip(got, want_ratios)
    ):
        print(
            f"REFUSING: asked for topk_ratios={want_ratios} but the server reports "
            f"topk_ratios={got!r}. Ratios are swept within one arm name, so a mismatch "
            "here produces a table whose rows are all the same run.",
            file=sys.stderr,
        )
        return 1

    if args.audit is not None and bool(sub.get("audit")) != args.audit:
        print(
            f"REFUSING: asked for audit={args.audit} but the server reports "
            f"audit={sub.get('audit')!r} (pid {sub.get('pid')}). The usual cause is a "
            "previous job still holding the port: the new job's bind fails, uvicorn "
            "shuts its HTTP layer down without failing the job, and this client "
            "reaches the OLD server -- same arm, different build. `squeue -u $USER`, "
            "scancel the old job, resubmit.",
            file=sys.stderr,
        )
        return 1

    print(f"  arm OK: {sub}")
    print(f"  model:  {info.get('model_path')}")
    print(f"  KV pool: max_total_num_tokens={info.get('max_total_num_tokens')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
