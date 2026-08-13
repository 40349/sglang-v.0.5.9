#!/usr/bin/env python3
"""A/B harness for sub-context KV reuse.

Two subcommands:

  replay   Re-send a captured SWE-bench chat sequence to a server, in order.
  report   Aggregate the CUDA-event forward traces the server wrote, and diff
           a baseline run against a sub-context run.

Why replay instead of just timing two SWE-bench runs: an agent loop is closed.
The moment one sampled token differs the two runs take different actions, run
different bash commands and end with different numbers of turns, so their total
GPU time compares two different conversations. Replaying one captured request
sequence against both configs fixes the input and leaves KV reuse as the only
variable.

Replay pins every turn to the same fixed number of generated tokens
(--gen-tokens, with ignore_eos) rather than letting the model generate freely.
Free generation makes each turn's decode length depend on the config, which adds
a large, uncontrolled block of decode time on top of the prefill effect actually
being measured.

Do NOT lower --gen-tokens to 1 to "measure prefill only". A request that finishes
without ever going through cache_unfinished_req never gets sub_context_last_nodes
set, so RadixCache.cache_finished_req falls through to the stock insert and files
the prompt under extra_key=None instead of the per-namespace keys. The next
request then misses every namespace, and the run reports 0% reuse -- measuring
the harness, not the cache. Pass --full to use each captured request's own
max_tokens instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, Iterator, List, Optional


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_requests(path: str, limit: Optional[int]) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def cmd_replay(args: argparse.Namespace) -> int:
    reqs = load_requests(args.requests, args.limit)
    if not reqs:
        print(f"no requests in {args.requests}", file=sys.stderr)
        return 1

    base = args.url.rstrip("/")

    # Warm the GPU clocks before the measured window. Measured empirically: the
    # first replay of a session reports ~14% more decode GPU time than an
    # identical later one, purely from the card ramping up -- enough to swamp the
    # effect being measured and to flip its sign depending on run order.
    for _ in range(args.warmup):
        try:
            _post(
                f"{base}/v1/chat/completions",
                {
                    "model": args.model or reqs[0].get("model", ""),
                    "messages": [{"role": "user", "content": "warm up"}],
                    "max_tokens": 128,
                    "ignore_eos": True,
                    "stream": False,
                },
                args.timeout,
            )
        except Exception as e:
            print(f"warning: warmup request failed ({e})", file=sys.stderr)
            break
    if args.warmup:
        print(f"warmed up ({args.warmup} generations)")

    if not args.no_flush:
        try:
            urllib.request.urlopen(f"{base}/flush_cache", timeout=30).read()
            print("flushed radix cache")
            time.sleep(1.0)
        except urllib.error.URLError as e:
            print(f"warning: flush_cache failed ({e})", file=sys.stderr)

    # Tell the report where the measured window starts, so the warm-up generations
    # above are not counted. Same machine as the server, one short appended line.
    if args.trace:
        try:
            with open(args.trace, "a", buffering=1) as f:
                f.write(json.dumps({"type": "measure_start"}) + "\n")
        except OSError as e:
            print(f"warning: cannot mark {args.trace} ({e})", file=sys.stderr)

    rows = []
    t_start = time.perf_counter()
    for i, body in enumerate(reqs):
        body = dict(body)
        body["stream"] = False
        if not args.full:
            # Same decode length for every turn in both configs, so any difference
            # in total GPU time comes from prefill.
            body["max_tokens"] = args.gen_tokens
            body["ignore_eos"] = True
        if args.model:
            body["model"] = args.model

        t0 = time.perf_counter()
        try:
            resp = _post(f"{base}/v1/chat/completions", body, args.timeout)
        except Exception as e:
            print(f"  request {i} failed: {e}", file=sys.stderr)
            return 1
        dt = time.perf_counter() - t0

        usage = resp.get("usage") or {}
        prompt = usage.get("prompt_tokens", 0) or 0
        # Only reported when the server ran with --enable-cache-report; otherwise
        # prompt_tokens_details is absent. Keep that distinct from a real zero --
        # printing 0 here looks exactly like "the cache is broken".
        details = usage.get("prompt_tokens_details")
        cached = None if details is None else (details.get("cached_tokens") or 0)
        rows.append(
            {
                "i": i,
                "latency_s": round(dt, 4),
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
            }
        )
        if cached is None:
            shown = f"cached={'n/a':>6}  (server lacks --enable-cache-report)"
        else:
            shown = (f"cached={cached:6d}  "
                     f"({100.0 * cached / prompt if prompt else 0:5.1f}%)")
        print(f"  [{i:3d}] {dt * 1000:8.1f} ms  prompt={prompt:6d}  {shown}")

    total = time.perf_counter() - t_start
    tot_prompt = sum(r["prompt_tokens"] for r in rows)
    if any(r["cached_tokens"] is None for r in rows):
        cache_str = "cached n/a (use --enable-cache-report; the forward trace has it either way)"
    else:
        tot_cached = sum(r["cached_tokens"] for r in rows)
        cache_str = (f"cached {tot_cached} tok "
                     f"({100.0 * tot_cached / tot_prompt if tot_prompt else 0:.1f}%)")
    print(
        f"\n{len(rows)} requests in {total:.1f}s | prompt {tot_prompt} tok, {cache_str} | "
        f"client latency sum {sum(r['latency_s'] for r in rows):.2f}s"
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def iter_runs(path: str) -> Iterator[List[dict]]:
    """Split a trace file into runs, delimited by the run_start marker."""
    current: List[dict] = []
    started = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = row.get("type")
            if kind == "run_start":
                if started and current:
                    yield current
                current, started = [], True
                continue
            if kind == "measure_start":
                current = []  # drop warm-up passes
                continue
            current.append(row)
    if current:
        yield current


def summarize(rows: List[dict]) -> Dict[str, float]:
    by_mode: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_mode["extend" if r.get("mode", "").startswith("extend") else r["mode"]].append(r)

    ext = by_mode.get("extend", [])
    dec = by_mode.get("decode", [])
    ext_ms = sum(r["gpu_ms"] for r in ext)
    dec_ms = sum(r["gpu_ms"] for r in dec)
    new_tok = sum(r["new_tokens"] for r in ext)
    cached_tok = sum(r["cached_tokens"] for r in ext)
    # Older traces predate matched_tokens; fall back to "matched == reused".
    matched_tok = sum(r.get("matched_tokens", r["cached_tokens"]) for r in ext)

    return {
        "matched_tokens": matched_tok,
        "discarded_tokens": matched_tok - cached_tok,
        "prefill_passes": len(ext),
        "prefill_gpu_ms": ext_ms,
        "prefill_new_tokens": new_tok,
        "prefill_cached_tokens": cached_tok,
        "prefill_total_tokens": new_tok + cached_tok,
        "hit_rate": (100.0 * cached_tok / (new_tok + cached_tok)) if (new_tok + cached_tok) else 0.0,
        "us_per_new_token": (1000.0 * ext_ms / new_tok) if new_tok else 0.0,
        "decode_passes": len(dec),
        "decode_gpu_ms": dec_ms,
        "decode_tokens": sum(r["new_tokens"] for r in dec),
        "total_gpu_ms": ext_ms + dec_ms,
        "prefill_ms_median": statistics.median([r["gpu_ms"] for r in ext]) if ext else 0.0,
    }


def _fmt(v: float) -> str:
    return f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"


def cmd_report(args: argparse.Namespace) -> int:
    def pick(path: str) -> Dict[str, float]:
        runs = list(iter_runs(path))
        if not runs:
            print(f"no traced forward passes in {path}", file=sys.stderr)
            sys.exit(1)
        if args.all_runs:
            rows = [r for run in runs for r in run]
        else:
            rows = runs[-1]
            if len(runs) > 1:
                print(f"note: {path} holds {len(runs)} runs, using the last "
                      f"(--all-runs to merge)")
        return summarize(rows)

    a = pick(args.baseline)
    label_a = "baseline (no split)"

    if not args.treatment:
        print(f"\n=== {label_a} ===")
        for k, v in a.items():
            print(f"  {k:24s} {_fmt(v)}")
        return 0

    b = pick(args.treatment)
    label_b = "sub-context"

    keys = [
        ("prefill_passes", "prefill passes", ""),
        ("prefill_total_tokens", "prefill tokens seen", "tok"),
        ("prefill_new_tokens", "  ...actually computed", "tok"),
        ("prefill_cached_tokens", "  ...from radix cache", "tok"),
        ("matched_tokens", "matched in the tree", "tok"),
        ("discarded_tokens", "  ...matched but DROPPED", "tok"),
        ("hit_rate", "cache hit rate (reuse only)", "%"),
        ("prefill_gpu_ms", "PREFILL GPU time", "ms"),
        ("prefill_ms_median", "  median pass", "ms"),
        ("us_per_new_token", "  us / computed token", "us"),
        ("decode_passes", "decode passes", ""),
        ("decode_gpu_ms", "decode GPU time", "ms"),
        ("total_gpu_ms", "TOTAL GPU time", "ms"),
    ]

    w = 26
    print(f"\n{'':{w}} {label_a:>18} {label_b:>18} {'delta':>18}")
    print("-" * (w + 58))
    for key, label, unit in keys:
        va, vb = a[key], b[key]
        if key in ("hit_rate",):
            delta = f"{vb - va:+.1f} pp"
        elif va:
            delta = f"{100.0 * (vb - va) / va:+.1f}%"
        else:
            delta = "n/a"
        print(f"{label:{w}} {_fmt(va):>18} {_fmt(vb):>18} {delta:>18}")

    if a["prefill_gpu_ms"]:
        saved = a["prefill_gpu_ms"] - b["prefill_gpu_ms"]
        print(
            f"\nprefill GPU time saved: {saved:,.1f} ms "
            f"({100.0 * saved / a['prefill_gpu_ms']:+.1f}%)"
        )
    if a["prefill_passes"] != b["prefill_passes"]:
        print(
            "\nWARNING: the two runs did not execute the same number of prefill\n"
            "passes, so they probably did not see the same request sequence.\n"
            "Compare only replays of one captured trace."
        )
    return 0


STAGE_NOTES = {
    "tpl_render": "chat template render (INCLUDES subctx_split)",
    "subctx_split": "  of which: finding block boundaries",
    "match": "prefix match / per-namespace stitch",
    "cache_unfinished": "insert prompt into the tree",
    "cache_finished": "release locks / free tail",
}


def load_stages(prefix: str) -> Dict[str, Dict[str, float]]:
    """Merge the per-process stage files a run wrote (<prefix>.http, .scheduler)."""
    merged: Dict[str, Dict[str, float]] = {}
    import glob

    hits = sorted(glob.glob(f"{prefix}.*"))
    if not hits:
        print(f"no stage files matching {prefix}.*", file=sys.stderr)
        sys.exit(1)
    for path in hits:
        with open(path) as f:
            doc = json.load(f)
        for stage, s in doc["stages"].items():
            merged[stage] = s
    return merged


def cmd_stages(args: argparse.Namespace) -> int:
    a = load_stages(args.baseline)
    b = load_stages(args.treatment)

    print(f"\n{'stage':<18} {'baseline':>22} {'sub-context':>22} {'added':>10}")
    print(f"{'':<18} {'calls   mean_us   tot_ms':>22} {'calls   mean_us   tot_ms':>22}")
    print("-" * 76)

    total_added = 0.0
    for stage in sorted(set(a) | set(b), key=lambda s: -(b.get(s, {}).get("total_ms", 0))):
        sa, sb = a.get(stage), b.get(stage)
        fa = (f"{sa['count']:>5} {sa['mean_us']:>9.1f} {sa['total_ms']:>8.1f}"
              if sa else f"{'-':>5} {'-':>9} {'-':>8}")
        fb = (f"{sb['count']:>5} {sb['mean_us']:>9.1f} {sb['total_ms']:>8.1f}"
              if sb else f"{'-':>5} {'-':>9} {'-':>8}")
        added = (sb["total_ms"] if sb else 0.0) - (sa["total_ms"] if sa else 0.0)
        # subctx_split is nested inside tpl_render; counting both double-counts it.
        if stage != "subctx_split":
            total_added += added
        print(f"{stage:<18} {fa:>22} {fb:>22} {added:>+9.1f}ms")

    print("-" * 76)
    print(f"{'TOTAL host overhead added':<18} {'':>45} {total_added:>+9.1f}ms")
    print("\n(subctx_split is nested inside tpl_render and excluded from the total)")
    for stage, note in STAGE_NOTES.items():
        if stage in a or stage in b:
            print(f"  {stage:<18} {note}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="re-send a captured chat sequence")
    r.add_argument("requests", help="jsonl written by SGLANG_CAPTURE_REQUESTS")
    r.add_argument("--url", default="http://127.0.0.1:30000")
    r.add_argument("--model", default=None, help="override the model field")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--timeout", type=float, default=1800.0)
    r.add_argument("--gen-tokens", type=int, default=64,
                   help="force exactly this many generated tokens per turn "
                        "(default 64; values near 1 break sub-context caching, "
                        "see module docstring)")
    r.add_argument("--full", action="store_true",
                   help="use each captured request's own max_tokens instead")
    r.add_argument("--warmup", type=int, default=3,
                   help="dummy generations to ramp GPU clocks before measuring "
                        "(default 3; the cache is flushed after them)")
    r.add_argument("--trace", default=None,
                   help="the server's SGLANG_FORWARD_TRACE path; marks where the "
                        "measured window begins so warm-up is excluded")
    r.add_argument("--no-flush", action="store_true")
    r.add_argument("--out", default=None, help="write per-request stats here")
    r.set_defaults(func=cmd_replay)

    s = sub.add_parser("report", help="aggregate/diff forward traces")
    s.add_argument("baseline", help="trace from the SGLANG_DISABLE_SUBCONTEXT=1 run")
    s.add_argument("treatment", nargs="?", help="trace from the sub-context run")
    s.add_argument("--all-runs", action="store_true",
                   help="merge every run in the file instead of the last")
    s.set_defaults(func=cmd_report)

    t = sub.add_parser("stages", help="diff host-side (CPU) stage timings")
    t.add_argument("baseline", help="SGLANG_STAGE_TRACE prefix from the baseline run")
    t.add_argument("treatment", help="SGLANG_STAGE_TRACE prefix from the sub-context run")
    t.set_defaults(func=cmd_stages)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
