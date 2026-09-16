"""Verify a remote sglang server is the sub-context arm the caller asked for.

The single-machine flow greps the server log for "Sub-context split DISABLED". With
MASLab on one box and the server on another that log is unreadable, so this asks
``/server_info`` instead -- the same guard by another route.

The guard is not ceremony. An arm that silently ran as a different arm is worse than
no arm at all, because it still produces a table that looks like a valid comparison.
Two ways that happens here:

- the server was started with ``sglang serve`` and no PYTHONPATH, so it imported the
  pip-installed sglang, which has no sub-context code at all; and
- MASLab reaches the server through its own ``model_api_config.json``, not through the
  URL this script was given, so a stale entry there sends the traffic elsewhere.

Both are checked. Exits non-zero with the reason on any mismatch.

Usage:
    check_remote_arm.py URL --split true|false --rotate true|false \
        [--index true|false] [--maslab-config PATH --model NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="server base URL, e.g. http://140.118.202.100:30000")
    ap.add_argument("--split", type=_bool, required=True)
    ap.add_argument(
        "--split-mode",
        choices=("blocks", "cdc"),
        default=None,
        help="require the boundaries to come from roles (blocks) or content (cdc)",
    )
    ap.add_argument("--rotate", type=_bool, required=True)
    ap.add_argument(
        "--index",
        type=_bool,
        default=None,
        help="require the content-addressed index (and the sparse prefill that goes "
        "with it) to be on or off. Left unchecked by default so callers written "
        "before it existed keep meaning what they meant.",
    )
    ap.add_argument(
        "--audit",
        type=_bool,
        default=None,
        help="require the server to report the finish-path audit as on/off. The arm "
        "flags alone cannot tell a freshly started server from a previous job that "
        "still owns the port -- they match either way.",
    )
    ap.add_argument("--maslab-config")
    ap.add_argument("--model")
    args = ap.parse_args()

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

    if sub["split_enabled"] != args.split or sub["rotate"] != args.rotate:
        print(
            f"REFUSING: asked for split={args.split} rotate={args.rotate}, but the "
            f"server reports {sub}",
            file=sys.stderr,
        )
        return 1

    if args.index is not None and bool(sub.get("index")) != args.index:
        print(
            f"REFUSING: asked for index={args.index} but the server reports "
            f"index={sub.get('index')!r}. A server old enough to have no `index` key "
            "at all reports None here, which is the same failure as the missing "
            "sub_context block above: it is not this checkout.",
            file=sys.stderr,
        )
        return 1

    if args.split_mode is not None and sub.get("split_mode") != args.split_mode:
        print(
            f"REFUSING: asked for split_mode={args.split_mode} but the server reports "
            f"split_mode={sub.get('split_mode')!r}.",
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
    # The KV pool size decides whether the A/B measures the split or the eviction
    # policy, and it is not comparable across boxes -- record what this server got.
    print(f"  KV pool: max_total_num_tokens={info.get('max_total_num_tokens')}")

    if args.maslab_config and args.model:
        want = base + "/v1"
        cfg = json.load(open(args.maslab_config))
        entry = cfg.get(args.model)
        if entry is None:
            print(
                f"REFUSING: {args.model} is not in {args.maslab_config}",
                file=sys.stderr,
            )
            return 1
        urls = [c["model_url"] for c in entry["model_list"]]
        if want not in urls:
            print(
                f"REFUSING: MASLab would send {args.model} to {urls}, not {want}.\n"
                f"  Fix model_url in {args.maslab_config}",
                file=sys.stderr,
            )
            return 1
        print(f"  MASLab -> {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
