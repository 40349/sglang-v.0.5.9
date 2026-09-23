#!/usr/bin/env python3
"""A/B harness for sub-context KV reuse.

Subcommands:

  replay   Re-send a captured chat sequence to a server, in order.
  report   Aggregate the CUDA-event forward traces and diff baseline vs sub-context.
  stages   Diff the host-side (CPU) stage timers and cost the mechanism itself.
  parity   Compare two arms' generated text.
  summary  One table over all three: the serving metrics an outside reader asks for
           (hit rate, TTFT, end-to-end latency, throughput) plus what the split adds.

`replay` re-sends one captured request sequence, so every arm sees the same input.
Each turn is pinned to --gen-tokens with ignore_eos (--full uses each request's own
max_tokens). Keep --gen-tokens at 2 or more: a request that finishes at prefill never
inserts its prompt into the namespaces, so the next request misses them. Decode GPU
time is the control and should not move between arms.

Replay streams by default so TTFT can be measured (--no-stream reports end-to-end
latency only). --concurrency defaults to 1; above 1, reuse depends on how requests
interleave.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Tuple


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


def _post_stream(url: str, payload: dict, timeout: float) -> dict:
    """Send one request with SSE streaming and time the first token.

    TTFT is the time to the first chunk carrying generated text (``content`` or
    ``reasoning_content``); the role-only opening chunk is skipped. ``text`` holds
    ``content`` only.
    """
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    parts: List[str] = []
    usage: dict = {}
    ttft = None
    finish = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            # The usage chunk arrives last and carries no choices.
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                parts.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {
        "latency_s": time.perf_counter() - t0,
        "ttft_s": ttft,
        "text": "".join(parts),
        "usage": usage,
        "finish_reason": finish,
    }


def _pct(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (not interpolated)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[k]


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
    if args.concurrency < 1:
        print("--concurrency must be at least 1", file=sys.stderr)
        return 1
    if args.concurrency > 1 and args.flush_every:
        print("--flush-every needs --concurrency 1 to mean anything", file=sys.stderr)
        return 1

    base = args.url.rstrip("/")

    # Warm up the GPU clocks before the measured window.
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

    # Mark the start of the measured window in the forward trace (same machine).
    if args.trace:
        try:
            with open(args.trace, "a", buffering=1) as f:
                f.write(json.dumps({"type": "measure_start"}) + "\n")
        except OSError as e:
            print(f"warning: cannot mark {args.trace} ({e})", file=sys.stderr)

    # And for the host stage timers, which watch for this file.
    if args.stage_trace:
        try:
            with open(f"{args.stage_trace}.mark", "w") as f:
                f.write(str(time.time()))
        except OSError as e:
            print(f"warning: cannot mark {args.stage_trace}.mark ({e})", file=sys.stderr)

    # One request start to finish; the row carries its capture index.
    def send(i: int, body: dict) -> dict:
        # --flush-every: no reuse in any arm (diagnosis only; ruins the timings).
        if args.flush_every and i:
            try:
                urllib.request.urlopen(f"{base}/flush_cache", timeout=30).read()
                time.sleep(0.2)
            except urllib.error.URLError as e:
                print(f"warning: flush_cache failed ({e})", file=sys.stderr)
        body = dict(body)
        if not args.full:
            body["max_tokens"] = args.gen_tokens
            body["ignore_eos"] = True
        if args.model:
            body["model"] = args.model

        try:
            if args.stream:
                res = _post_stream(f"{base}/v1/chat/completions", body, args.timeout)
            else:
                body["stream"] = False
                t0 = time.perf_counter()
                resp = _post(f"{base}/v1/chat/completions", body, args.timeout)
                dt = time.perf_counter() - t0
                try:
                    text = resp["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    text = ""
                res = {
                    "latency_s": dt,
                    "ttft_s": None,
                    "text": text,
                    "usage": resp.get("usage") or {},
                    "finish_reason": None,
                }
        except Exception as e:
            raise RuntimeError(f"request {i} failed: {e}") from e

        text = res["text"]
        usage = res["usage"] or {}
        prompt = usage.get("prompt_tokens", 0) or 0
        # None when the server gives no prompt_tokens_details (no --enable-cache-report).
        details = usage.get("prompt_tokens_details")
        cached = None if details is None else (details.get("cached_tokens") or 0)
        row = {
            "i": i,
            "latency_s": round(res["latency_s"], 4),
            "ttft_s": None if res["ttft_s"] is None else round(res["ttft_s"], 4),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
            **({"text": text} if args.save_text else {}),
        }
        if cached is None:
            shown = f"cached={'n/a':>6}  (server lacks --enable-cache-report)"
        else:
            shown = (f"cached={cached:6d}  "
                     f"({100.0 * cached / prompt if prompt else 0:5.1f}%)")
        ttft_s = "  ttft=%7.1f ms" % (res["ttft_s"] * 1000.0) if res["ttft_s"] else ""
        print(f"  [{i:3d}] {res['latency_s'] * 1000:8.1f} ms{ttft_s}  "
              f"prompt={prompt:6d}  {shown}")
        return row

    t_start = time.perf_counter()
    try:
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(send, i, b) for i, b in enumerate(reqs)]
                rows = [f.result() for f in futures]
        else:
            rows = [send(i, b) for i, b in enumerate(reqs)]
    except RuntimeError as e:
        print(f"  {e}", file=sys.stderr)
        return 1
    total = time.perf_counter() - t_start

    tot_prompt = sum(r["prompt_tokens"] for r in rows)
    tot_completion = sum(r["completion_tokens"] for r in rows)
    # "n/a" only when no row has cache details; a missing row counts as 0.
    missing = sum(1 for r in rows if r["cached_tokens"] is None)
    tot_cached = sum(r["cached_tokens"] or 0 for r in rows)
    if missing == len(rows):
        cache_str = "cached n/a (use --enable-cache-report; the forward trace has it either way)"
        hit_rate = None
    else:
        hit_rate = (100.0 * tot_cached / tot_prompt) if tot_prompt else 0.0
        cache_str = f"cached {tot_cached} tok ({hit_rate:.1f}%)"
        if missing:
            cache_str += f" [{missing} request(s) reported no cache detail, counted as 0]"

    lat = [r["latency_s"] for r in rows]
    ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    meta = {
        "requests": len(rows),
        "concurrency": args.concurrency,
        "stream": bool(args.stream),
        "gen_tokens": None if args.full else args.gen_tokens,
        "wall_s": round(total, 4),
        "prompt_tokens": tot_prompt,
        "cached_tokens": None if missing == len(rows) else tot_cached,
        "completion_tokens": tot_completion,
        "hit_rate_pct": hit_rate,
        # Over the replay's wall time; at concurrency 1 this is 1/latency.
        "requests_per_s": (len(rows) / total) if total else 0.0,
        "output_tokens_per_s": (tot_completion / total) if total else 0.0,
        "total_tokens_per_s": ((tot_prompt + tot_completion) / total) if total else 0.0,
        "latency_ms": {
            "mean": 1000.0 * statistics.fmean(lat) if lat else None,
            "p50": 1000.0 * _pct(lat, 0.50) if lat else None,
            "p95": 1000.0 * _pct(lat, 0.95) if lat else None,
        },
        "ttft_ms": {
            "mean": 1000.0 * statistics.fmean(ttft) if ttft else None,
            "p50": 1000.0 * _pct(ttft, 0.50) if ttft else None,
            "p95": 1000.0 * _pct(ttft, 0.95) if ttft else None,
        },
    }

    print(
        f"\n{len(rows)} requests in {total:.1f}s (concurrency {args.concurrency}) | "
        f"prompt {tot_prompt} tok, {cache_str} | "
        f"client latency sum {sum(lat):.2f}s"
    )
    if ttft:
        print(f"  TTFT     mean {meta['ttft_ms']['mean']:8.1f} ms  "
              f"p50 {meta['ttft_ms']['p50']:8.1f}  p95 {meta['ttft_ms']['p95']:8.1f}")
    else:
        print("  TTFT     n/a (--no-stream: a non-streamed reply arrives all at once)")
    print(f"  latency  mean {meta['latency_ms']['mean']:8.1f} ms  "
          f"p50 {meta['latency_ms']['p50']:8.1f}  p95 {meta['latency_ms']['p95']:8.1f}")
    print(f"  through  {meta['requests_per_s']:.2f} req/s  "
          f"{meta['output_tokens_per_s']:.1f} output tok/s  "
          f"{meta['total_tokens_per_s']:.1f} total tok/s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2)
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
    # Older traces lack `discarded_tokens`; derive it from `matched_tokens` there.
    if any("discarded_tokens" in r for r in ext):
        discarded_tok = sum(r.get("discarded_tokens", 0) for r in ext)
    else:
        discarded_tok = sum(r.get("matched_tokens", r["cached_tokens"]) for r in ext) - cached_tok
    matched_tok = cached_tok + discarded_tok
    # The fields below are None (not 0) when the trace predates them.
    # Dropped hits that were matched at another position.
    moved_tok = (
        sum(r.get("moved_tokens", 0) for r in ext)
        if any("moved_tokens" in r for r in ext)
        else None
    )
    # Displaced hits rotated into place; part of cached_tokens.
    rotated_tok = (
        sum(r.get("rotated_tokens", 0) for r in ext)
        if any("rotated_tokens" in r for r in ext)
        else None
    )
    # Declined blocks rotated back and filed at finish (reused by later requests).
    reinserted_tok = (
        sum(r.get("reinserted_tokens", 0) for r in ext)
        if any("reinserted_tokens" in r for r in ext)
        else None
    )
    sub_reqs = sum(r["sub_reqs"] for r in ext) if all("sub_reqs" in r for r in ext) else None

    return {
        "matched_tokens": matched_tok,
        "discarded_tokens": discarded_tok,
        "moved_tokens": moved_tok,
        "rotated_tokens": rotated_tok,
        "reinserted_tokens": reinserted_tok,
        "sub_reqs": sub_reqs,
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
        ("prefill_total_tokens", "prefill prompt tokens", "tok"),
        ("prefill_new_tokens", "  ...actually computed", "tok"),
        ("prefill_cached_tokens", "  ...from radix cache", "tok"),
        ("matched_tokens", "matched in the tree", "tok"),
        ("discarded_tokens", "  ...matched but DROPPED", "tok"),
        ("moved_tokens", "     ...dropped as MOVED", "tok"),
        ("rotated_tokens", "  ...MOVED but ROTATED in", "tok"),
        ("reinserted_tokens", "REVERSE-ROTATED into tree", "tok"),
        ("hit_rate", "hit rate (cached / prompt)", "%"),
        ("prefill_gpu_ms", "PREFILL GPU time", "ms"),
        ("prefill_ms_median", "  median pass", "ms"),
        ("us_per_new_token", "  us / computed token", "us"),
        ("decode_passes", "decode passes", ""),
        ("decode_gpu_ms", "decode GPU time", "ms"),
        ("total_gpu_ms", "TOTAL GPU time", "ms"),
    ]

    def gate_absent(arm: Dict[str, float]) -> bool:
        """True when no request in this arm took the split path."""
        return arm["sub_reqs"] == 0

    w = 26
    print(f"\n{'':{w}} {label_a:>18} {label_b:>18} {'delta':>18}")
    print("-" * (w + 58))
    for key, label, unit in keys:
        va, vb = a[key], b[key]
        na_a, na_b = va is None, vb is None
        if key == "discarded_tokens":
            na_a, na_b = na_a or gate_absent(a), na_b or gate_absent(b)
        if na_a or na_b:
            delta = "n/a"
        elif key in ("hit_rate",):
            delta = f"{vb - va:+.1f} pp"
        elif va:
            delta = f"{100.0 * (vb - va) / va:+.1f}%"
        else:
            delta = "n/a"
        sa = "n/a" if na_a else _fmt(va)
        sb = "n/a" if na_b else _fmt(vb)
        print(f"{label:{w}} {sa:>18} {sb:>18} {delta:>18}")

    if a["prefill_gpu_ms"]:
        saved = a["prefill_gpu_ms"] - b["prefill_gpu_ms"]
        print(
            f"\nprefill GPU time saved: {saved:,.1f} ms "
            f"({100.0 * saved / a['prefill_gpu_ms']:+.1f}%)"
        )
    if (a["discarded_tokens"] or 0) < 0 or (b["discarded_tokens"] or 0) < 0:
        print(
            "\nNOTE: a negative DROPPED count means this trace predates the fix that\n"
            "records the drop at stitch time. It was derived by subtracting a\n"
            "per-pass length from a per-request one, so every chunked-prefill\n"
            "continuation pass reads one chunk short. Re-run to get a real number;\n"
            "the other rows are unaffected."
        )
    # Different chunking (e.g. Stage 2): cross-check reuse against the prompt total.
    if a["prefill_passes"] != b["prefill_passes"]:
        prompt = min(a["prefill_total_tokens"], b["prefill_total_tokens"])
        print(
            f"\nNOTE: pass counts differ ({a['prefill_passes']:,} vs "
            f"{b['prefill_passes']:,}), so the two arms did NOT chunk the same way.\n"
            "That is expected when one arm cuts chunks at block boundaries; it does\n"
            "not by itself mean the request sequences differed, and the token rows\n"
            "above stay comparable -- each is counted once per request. Cross-check\n"
            "against the prompt total, which no counter can move:"
        )
        for label, arm in ((label_a, a), (label_b, b)):
            print(
                f"  {label:>20}: computed {arm['prefill_new_tokens']:,} of "
                f"{prompt:,} prompt tokens -> reuse "
                f"{100.0 * (prompt - arm['prefill_new_tokens']) / prompt:.1f}%"
            )
        print(
            "  (denominator is the smaller 'seen' total: chunking only inflates it,\n"
            "   so this is exact when that arm ran one pass per request. It assumes\n"
            "   BOTH arms replayed the same capture, which `toggle` guarantees and\n"
            "   two `record` runs do not -- there the streams differ and only each\n"
            "   arm's own 'seen' total is its denominator.)"
        )
    return 0


STAGE_NOTES = {
    "tpl_render": "chat template render (INCLUDES subctx_split)",
    "subctx_split": "  DECOMPOSITION: finding the block boundaries",
    "match": "prefix match / per-namespace stitch",
    "subctx_stitch": "  the split's own match+stitch, inside match",
    "subctx_scan": "  the index's scan + lookups, inside match",
    "subctx_lookup": "    LOOKUP: one match_prefix per namespace",
    "subctx_rotate": "    ROTATE: copy a displaced block, rotated, before prefill",
    "subctx_rotate_finish": "  ROTATE: rotate a block back at finish, to file it",
    "subctx_rotate_gpu": "  ROTATE on the GPU (SGLANG_SUBCTX_ROTATE_GPU=1; NOT host time)",
    "cache_unfinished": "insert prompt into the tree",
    "cache_finished": "release locks / free tail",
    "subctx_rev_rotate": "  of which: reverse-rotate a refused block and file it",
}

# Stages nested in another counted stage: shown, but kept out of the total.
BREAKDOWN = {
    "subctx_stitch": "match",
    "subctx_scan": "match",
    "subctx_lookup": "subctx_stitch",
    "subctx_rotate": "match / cache_unfinished",
    "subctx_rev_rotate": "cache_finished",
    "subctx_rotate_finish": "cache_finished",
    "subctx_rotate_gpu": "GPU, not host",
}

# child -> enclosing stage. Only the child enters the total (the parent's delta is
# noisier); cmd_stages prints the residual parent - child for both arms.
NESTED = {"subctx_split": "tpl_render"}


def load_stages(
    prefix: str, missing_ok: bool = False
) -> tuple[Dict[str, Dict[str, float]], bool]:
    """Merge the per-process stage files a run wrote (<prefix>.http.json, .scheduler.json).

    Returns the merged stages and whether they are the post-warm-up ``measured``
    window, used only when every process has it.
    """
    hits = sorted(glob.glob(f"{prefix}.*"))
    hits = [h for h in hits if not h.endswith(".mark")]
    if not hits:
        if missing_ok:
            return {}, False
        print(f"no stage files matching {prefix}.*", file=sys.stderr)
        sys.exit(1)

    docs = []
    for path in hits:
        with open(path) as f:
            docs.append((path, json.load(f)))
    measured = all("measured" in doc for _, doc in docs)

    merged: Dict[str, Dict[str, float]] = {}
    for path, doc in docs:
        for stage, s in doc["measured" if measured else "stages"].items():
            # A stage reported by two processes would overwrite the first.
            if stage in merged:
                print(f"warning: stage {stage!r} reported by more than one "
                      f"process (last seen in {path}); numbers will be wrong",
                      file=sys.stderr)
            merged[stage] = s
    return merged, measured


def cmd_stages(args: argparse.Namespace) -> int:
    a, a_measured = load_stages(args.baseline)
    b, b_measured = load_stages(args.treatment)
    label_a, label_b = "baseline", "sub-context"

    if not (a_measured and b_measured):
        print("\nNOTE: at least one arm has no post-warm-up window, so these counts\n"
              "      start at process launch and include the replay warm-up. The\n"
              "      forward trace excludes it, so the two are not over the same\n"
              "      window. Re-run with a --stage-trace passed to `replay`.")

    print(f"\n{'stage':<18} {'baseline':>22} {'sub-context':>22} {'added':>10}")
    print(f"{'':<18} {'calls   mean_us   tot_ms':>22} {'calls   mean_us   tot_ms':>22}")
    print("-" * 76)

    total_added = 0.0
    skewed = []
    for stage in sorted(set(a) | set(b), key=lambda s: -(b.get(s, {}).get("total_ms", 0))):
        sa, sb = a.get(stage), b.get(stage)
        fa = (f"{sa['count']:>5} {sa['mean_us']:>9.1f} {sa['total_ms']:>8.1f}"
              if sa else f"{'-':>5} {'-':>9} {'-':>8}")
        fb = (f"{sb['count']:>5} {sb['mean_us']:>9.1f} {sb['total_ms']:>8.1f}"
              if sb else f"{'-':>5} {'-':>9} {'-':>8}")
        # Per-call means scaled to a common call count, not raw totals.
        n_ref = max(sa["count"] if sa else 0, sb["count"] if sb else 0)
        norm = lambda s: (s["mean_us"] * n_ref / 1000.0) if s else 0.0
        added = norm(sb) - norm(sa)
        if sa and sb and sa["count"] != sb["count"]:
            skewed.append((stage, sa["count"], sb["count"]))
        if stage in NESTED.values():
            note = "  (level only, see below)"
        elif stage in BREAKDOWN:
            note = f"  (inside {BREAKDOWN[stage]})"
        else:
            total_added += added
            note = ""
        print(f"{stage:<18} {fa:>22} {fb:>22} {added:>+9.1f}ms{note}")

    print("-" * 76)
    print(f"{'TOTAL host overhead added':<18} {'':>45} {total_added:>+9.1f}ms")
    # Requests: cache_finished runs once per finished request.
    n_req = max(
        (arm["cache_finished"]["count"] for arm in (a, b) if "cache_finished" in arm),
        default=0,
    )
    if n_req:
        print(f"{'  per request':<18} {'':>45} {1000.0 * total_added / n_req:>+9.1f}us")

    print("\n(per-call means scaled to a common call count)")
    for child, parent in NESTED.items():
        if child not in b:
            continue
        print(f"\n{child} is nested inside {parent}, so only one enters the total:")
        print(f"  {child:<20} counted  -- measured directly, baseline is the disabled path")
        print(f"  {parent:<20} EXCLUDED -- its delta is a difference of two ~13ms numbers")
        # Parent minus child is the work both arms share; its delta should be ~0.
        for label, arm in ((label_a, a), (label_b, b)):
            if parent in arm:
                net = arm[parent]["mean_us"] - arm.get(child, {}).get("mean_us", 0.0)
                print(f"  {parent} minus {child}, {label:<20} {net:>10.1f} us/call")
        if parent in a and parent in b:
            resid = ((b[parent]["mean_us"] - b.get(child, {}).get("mean_us", 0.0))
                     - (a[parent]["mean_us"] - a.get(child, {}).get("mean_us", 0.0)))
            print(f"  -> residual (should be ~0)              {resid:>+10.1f} us/call")

    for stage, ca, cb in skewed:
        print(f"  WARNING: {stage} ran {ca} times in baseline but {cb} in sub-context")
    print()
    for stage, note in STAGE_NOTES.items():
        if stage in a or stage in b:
            print(f"  {stage:<18} {note}")
    return 0


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #


def load_client(path: str) -> Tuple[dict, List[dict]]:
    """Read a replay's --out file: ``{"meta", "rows"}``, or an older bare row list."""
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return {}, doc
    return doc.get("meta") or {}, doc.get("rows") or []


def cmd_parity(args: argparse.Namespace) -> int:
    """Compare the two arms' generated text, request by request (by hash).

    Identical text is expected only where the reused KV is exact (e.g. an arm against
    itself); rotated or cross-namespace reuse changes the KV and can change the text.
    """
    _, a = load_client(args.baseline)
    _, b = load_client(args.treatment)

    if len(a) != len(b):
        print(f"MISMATCH: {len(a)} baseline rows vs {len(b)} treatment rows")
        return 1

    bad = [i for i, (x, y) in enumerate(zip(a, b))
           if x.get("text_sha1") != y.get("text_sha1")]
    if not any("text_sha1" in x for x in a):
        print("no text_sha1 in the results -- replay predates the field, rerun to compare")
        return 1

    print(f"{len(a) - len(bad)}/{len(a)} requests generated identical text")
    if not bad:
        print("PARITY OK")
        return 0

    print(f"\nDIVERGED at {len(bad)} request(s): {bad[:20]}{' ...' if len(bad) > 20 else ''}")
    i = bad[0]
    if "text" in a[i] and "text" in b[i]:
        ta, tb = a[i]["text"], b[i]["text"]
        n = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
        print(f"\nfirst divergence, request {i}, at char {n}:")
        print(f"  common prefix: ...{ta[max(0, n - 60):n]!r}")
        print(f"  baseline then: {ta[n:n + 80]!r}")
        print(f"  sub-context:   {tb[n:n + 80]!r}")
    else:
        print("(rerun replay with --save-text to see what diverged)")
    return 1


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

# File stem -> column label, in table order.
ARM_STEMS = (
    ("base", "baseline"),
    ("sub", "sub-context"),
    ("rot", "+ rotation"),
    ("idx", "+ index"),
    ("cdc", "+ content-cut blocks"),
)


def _load_arm(dirname: str, suffix: str, stem: str) -> Optional[dict]:
    client = os.path.join(dirname, f"client_{stem}{suffix}.json")
    trace = os.path.join(dirname, f"trace_{stem}{suffix}.jsonl")
    stage = os.path.join(dirname, f"stage_{stem}{suffix}")
    if not os.path.exists(client) and not os.path.exists(trace):
        return None

    meta: dict = {}
    rows: List[dict] = []
    if os.path.exists(client):
        meta, rows = load_client(client)
    gpu = None
    if os.path.exists(trace):
        runs = list(iter_runs(trace))
        if runs:
            gpu = summarize(runs[-1])
    stages, measured = load_stages(stage, missing_ok=True)
    return {
        "client": client if os.path.exists(client) else None,
        "meta": meta,
        "rows": rows,
        "gpu": gpu,
        "stages": stages,
        "stages_measured": measured,
    }


def _client_stat(arm: dict, *path: str) -> Optional[float]:
    """Read a value from the replay's ``meta``, else recompute it from the rows.

    Only the request count, hit rate and latency can be recomputed; others are None.
    """
    meta, rows = arm["meta"], arm["rows"]
    node: object = meta
    for key in path:
        if not isinstance(node, dict) or key not in node:
            node = None
            break
        node = node[key]
    if node is not None:
        return node

    if path == ("requests",):
        return len(rows) or None
    if path == ("hit_rate_pct",):
        prompt = sum(r.get("prompt_tokens") or 0 for r in rows)
        if not prompt or all(r.get("cached_tokens") is None for r in rows):
            return None
        return 100.0 * sum(r.get("cached_tokens") or 0 for r in rows) / prompt
    if path[0] == "latency_ms" and rows:
        lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        if not lat:
            return None
        if path[1] == "mean":
            return 1000.0 * statistics.fmean(lat)
        return 1000.0 * _pct(lat, 0.50 if path[1] == "p50" else 0.95)
    return None


def _gpu_stat(arm: dict, key: str) -> Optional[float]:
    gpu = arm["gpu"]
    return None if gpu is None else gpu.get(key)


def _stage_us_per_req(arm: dict, stage: str, n_req: Optional[float]) -> Optional[float]:
    s = arm["stages"].get(stage)
    if s is None or not n_req:
        return None
    return 1000.0 * s["total_ms"] / n_req


def _added_us_per_req(
    arm: dict, base: dict, stage: str, n_req: Optional[float]
) -> Optional[float]:
    """Per-request cost this arm added at ``stage`` over the baseline (per-call
    means scaled to a common call count, as in `stages`).
    """
    sa, sb = base["stages"].get(stage), arm["stages"].get(stage)
    if sb is None and sa is None:
        return None
    if not n_req:
        return None
    n_ref = max(sa["count"] if sa else 0, sb["count"] if sb else 0)
    norm = lambda s: (s["mean_us"] * n_ref / 1000.0) if s else 0.0
    return 1000.0 * (norm(sb) - norm(sa)) / n_req


# label, unit, how to get it, how to compare it to the baseline.
#   "pp"  percentage points   "pct" relative percent   None  no comparison
_SUMMARY_METRICS = (
    ("WORKLOAD", None, None, None),
    ("requests replayed", "", lambda a, b, n: _client_stat(a, "requests"), None),
    ("concurrency", "", lambda a, b, n: _client_stat(a, "concurrency"), None),
    ("prompt tokens", "tok", lambda a, b, n: _client_stat(a, "prompt_tokens"), None),
    ("generated tokens", "tok", lambda a, b, n: _client_stat(a, "completion_tokens"), None),

    ("KV CACHE REUSE", None, None, None),
    ("hit rate (client, cached/prompt)", "%",
     lambda a, b, n: _client_stat(a, "hit_rate_pct"), "pp"),
    ("hit rate (server, cached/prompt)", "%",
     lambda a, b, n: _gpu_stat(a, "hit_rate"), "pp"),
    ("prefill tokens computed", "tok",
     lambda a, b, n: _gpu_stat(a, "prefill_new_tokens"), "pct"),
    ("prefill tokens from the cache", "tok",
     lambda a, b, n: _gpu_stat(a, "prefill_cached_tokens"), "pct"),
    # Part of the row above; the dropped rows are in `report`.
    ("  MOVED but rotated in", "tok",
     lambda a, b, n: _gpu_stat(a, "rotated_tokens"), "pct"),

    ("TIME TO FIRST TOKEN", None, None, None),
    ("mean", "ms", lambda a, b, n: _client_stat(a, "ttft_ms", "mean"), "pct"),
    ("p50", "ms", lambda a, b, n: _client_stat(a, "ttft_ms", "p50"), "pct"),
    ("p95", "ms", lambda a, b, n: _client_stat(a, "ttft_ms", "p95"), "pct"),

    ("END-TO-END LATENCY", None, None, None),
    ("mean", "ms", lambda a, b, n: _client_stat(a, "latency_ms", "mean"), "pct"),
    ("p50", "ms", lambda a, b, n: _client_stat(a, "latency_ms", "p50"), "pct"),
    ("p95", "ms", lambda a, b, n: _client_stat(a, "latency_ms", "p95"), "pct"),

    ("THROUGHPUT", None, None, None),
    ("requests / s", "", lambda a, b, n: _client_stat(a, "requests_per_s"), "pct"),
    ("output tokens / s", "", lambda a, b, n: _client_stat(a, "output_tokens_per_s"), "pct"),
    ("total tokens / s", "", lambda a, b, n: _client_stat(a, "total_tokens_per_s"), "pct"),
    ("wall clock", "s", lambda a, b, n: _client_stat(a, "wall_s"), "pct"),

    ("GPU TIME (CUDA events)", None, None, None),
    ("prefill", "ms", lambda a, b, n: _gpu_stat(a, "prefill_gpu_ms"), "pct"),
    ("decode  (control: expect ~0%)", "ms",
     lambda a, b, n: _gpu_stat(a, "decode_gpu_ms"), "pct"),
    ("total", "ms", lambda a, b, n: _gpu_stat(a, "total_gpu_ms"), "pct"),
)

# Host cost each arm added over the baseline, per request.
#
# (label, stage, own, counted). `own`: the baseline does not run the stage, so its
# whole cost is added; otherwise the baseline's cost is subtracted. `counted`: enters
# the total (nested stages do not; subctx_split does, see NESTED).
_OVERHEAD_METRICS = (
    ("decomposition (subctx_split)", "subctx_split", True, True),
    ("match, total added", "match", False, True),
    ("  index scan (subctx_scan)", "subctx_scan", True, False),
    ("  lookup (subctx_lookup)", "subctx_lookup", True, False),
    ("  assembly (stitch - lookup - rotate)", None, True, False),
    ("  rotate, before prefill", "subctx_rotate", True, False),
    ("insert into tree, added", "cache_unfinished", False, True),
    ("release / free, added", "cache_finished", False, True),
    ("  reverse-rotate and re-file", "subctx_rev_rotate", True, False),
    ("  rotate, at finish", "subctx_rotate_finish", True, False),
    ("[GPU] rotation kernel", "subctx_rotate_gpu", True, False),
)


def _overhead_value(
    arm: dict, base: dict, stage: Optional[str], own: bool, n_req: Optional[float]
) -> Optional[float]:
    if stage is None:  # assembly: what the stitch spends outside its own children
        stitch = _stage_us_per_req(arm, "subctx_stitch", n_req)
        if stitch is None:
            return None
        lookup = _stage_us_per_req(arm, "subctx_lookup", n_req) or 0.0
        rotate = _stage_us_per_req(arm, "subctx_rotate", n_req) or 0.0
        return stitch - lookup - rotate
    if own:  # a stage the baseline does not have at all
        return _stage_us_per_req(arm, stage, n_req)
    return _added_us_per_req(arm, base, stage, n_req)


def _split_ratio_stem(stem: str) -> Tuple[str, Optional[int]]:
    """``"idx_r15"`` -> ``("idx", 15)``; any other stem -> ``(stem, None)``."""
    base, _, tail = stem.rpartition("_r")
    if base and tail.isdigit():
        return base, int(tail)
    return stem, None


def cmd_summary(args: argparse.Namespace) -> int:
    want = [s.strip() for s in args.arms.split(",") if s.strip()]
    known = dict(ARM_STEMS)
    unknown = [s for s in want if _split_ratio_stem(s)[0] not in known]
    if unknown:
        print(f"unknown arm(s) {', '.join(unknown)}; want any of "
              f"{', '.join(s for s, _ in ARM_STEMS)}, optionally with a recompute "
              f"ratio appended as _rNN (e.g. idx_r15)", file=sys.stderr)
        return 1
    arms = []
    # Arm order first, then ratio, so a sweep of one arm reads left to right.
    for stem in sorted(want, key=lambda s: (
        [k for k, _ in ARM_STEMS].index(_split_ratio_stem(s)[0]),
        _split_ratio_stem(s)[1] or 0,
    )):
        base_stem, ratio = _split_ratio_stem(stem)
        arm = _load_arm(args.dir, args.suffix, stem)
        if arm is not None:
            arm["label"] = known[base_stem] + (
                f" +{ratio}% recompute" if ratio else ""
            )
            arms.append(arm)
    if not arms:
        print(f"no arms found under {args.dir} with suffix {args.suffix!r} "
              f"(looking for client_base{args.suffix}.json and friends, "
              f"limited to {args.arms})", file=sys.stderr)
        return 1
    base = arms[0]
    if base["label"] != dict(ARM_STEMS)["base"]:
        print(f"WARNING: no baseline arm in {args.arms}; the deltas below are "
              f"against {base['label']}, not against the split being off.")

    W, V, D = 40, 13, 10
    head = f"{'':<{W}}" + "".join(
        f"{a['label']:>{V}}" + ("" if i == 0 else f"{'vs base':>{D}}")
        for i, a in enumerate(arms)
    )
    print()
    print(head)
    print("-" * len(head))

    def fmt(v: Optional[float], unit: str) -> str:
        if v is None:
            return "n/a"
        if unit == "%":
            return f"{v:.1f}%"
        if unit in ("ms", "s"):
            return f"{v:,.1f}"
        if abs(v) >= 1000 or float(v).is_integer():
            return f"{v:,.0f}"
        return f"{v:,.2f}"

    def delta(v: Optional[float], v0: Optional[float], kind: Optional[str]) -> str:
        if kind is None or v is None or v0 is None:
            return ""
        if kind == "pp":
            return f"{v - v0:+.1f} pp"
        if not v0:
            return "n/a"
        return f"{100.0 * (v - v0) / v0:+.1f}%"

    n_req = {id(a): _client_stat(a, "requests") for a in arms}

    for label, unit, getter, cmp_kind in _SUMMARY_METRICS:
        if getter is None:  # section heading
            print(f"\n{label}")
            continue
        v0 = getter(base, base, n_req[id(base)])
        line = f"  {label:<{W - 2}}"
        for i, a in enumerate(arms):
            v = getter(a, base, n_req[id(a)])
            shown = fmt(v, unit)
            if v is not None and unit and unit != "%":
                shown = f"{shown} {unit}"
            line += f"{shown:>{V}}"
            if i:
                line += f"{delta(v, v0, cmp_kind):>{D}}"
        print(line)

    print("\nHOST OVERHEAD ADDED BY THE SPLIT (us / request)")
    print("  (indented rows are inside the row above them and are not added again;")
    print("   the GPU row is device time and never enters a host total)")
    totals = {id(a): 0.0 for a in arms}
    counted = {id(a): False for a in arms}
    for label, stage, own, in_total in _OVERHEAD_METRICS:
        line = f"  {label:<{W - 2}}"
        for i, a in enumerate(arms):
            v = _overhead_value(a, base, stage, own, n_req[id(a)])
            if v is not None and in_total and a is not base:
                totals[id(a)] += v
                counted[id(a)] = True
            if v is not None:
                shown = f"{v:+.1f}"
            elif not a["stages"]:
                shown = "n/a"  # no stage files: unknown, not zero
            else:
                shown = "-"  # this arm does not run that stage
            line += f"{shown:>{V}}"
            if i:
                line += f"{'':>{D}}"
        print(line)
    line = f"  {'TOTAL added per request':<{W - 2}}"
    for i, a in enumerate(arms):
        shown = "ref" if a is base else (f"{totals[id(a)]:+.1f}" if counted[id(a)] else "n/a")
        line += f"{shown:>{V}}"
        if i:
            line += f"{'':>{D}}"
    print(line)

    notes = []
    if not all(a["stages"] for a in arms):
        notes.append("some arms have no stage files, so their host overhead is n/a; "
                     "pass --stage-trace to `replay` to get them")
    if any(a["stages"] and not a["stages_measured"] for a in arms):
        notes.append("at least one arm's stage counters start at process launch and "
                     "include the replay warm-up, while the GPU trace excludes it")
    if any(a["stages"].get("subctx_rotate") for a in arms) and not any(
        a["stages"].get("subctx_rotate_gpu") for a in arms
    ):
        notes.append("the rotation kernel's GPU time is unmeasured; re-run the "
                     "server with SGLANG_SUBCTX_ROTATE_GPU=1 for it (it costs a "
                     "CUDA event pair per rotation, so read the host numbers from "
                     "a run with it off)")
    if not any(_client_stat(a, "ttft_ms", "mean") for a in arms):
        notes.append("no TTFT: those replays either ran with --no-stream or predate "
                     "streaming, and a reply that arrives all at once has no first "
                     "token to time")
    if (_client_stat(base, "concurrency") or 1) == 1:
        notes.append("concurrency 1: throughput here is 1/latency by construction, "
                     "not serving capacity -- re-run with --concurrency for that")
    notes.append("replay pins the generated length with ignore_eos, so this says "
                 "nothing about answer quality: pass@1 comes from `run_mas.sh eval` "
                 "on a record run, counted as regressions against the baseline")
    print()
    for n in notes:
        print(f"NOTE: {n}")
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
                        "(default 64; keep it >= 2, see module docstring)")
    r.add_argument("--full", action="store_true",
                   help="use each captured request's own max_tokens instead")
    r.add_argument("--warmup", type=int, default=3,
                   help="dummy generations to ramp GPU clocks before measuring "
                        "(default 3; the cache is flushed after them)")
    r.add_argument("--trace", default=None,
                   help="the server's SGLANG_FORWARD_TRACE path; marks where the "
                        "measured window begins so warm-up is excluded")
    r.add_argument("--stage-trace", default=None,
                   help="the server's SGLANG_STAGE_TRACE prefix; same purpose as "
                        "--trace, for the host-side timers")
    r.add_argument("--no-flush", action="store_true")
    r.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                   help="stream the reply so TTFT can be measured (default on); "
                        "--no-stream restores the old single-response behaviour and "
                        "reports no TTFT")
    r.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight at once (default 1); above 1, reuse "
                        "depends on how requests interleave")
    r.add_argument("--flush-every", action="store_true",
                   help="empty the radix tree before every request, so neither "
                        "arm reuses anything (diagnosis only -- destroys timings)")
    r.add_argument("--out", default=None, help="write per-request stats here")
    r.add_argument("--save-text", action="store_true",
                   help="also store each completion in --out, so `parity` can "
                        "show what diverged and not just that something did")
    r.set_defaults(func=cmd_replay)

    s = sub.add_parser("report", help="aggregate/diff forward traces")
    s.add_argument("baseline", help="trace from the SGLANG_DISABLE_SUBCONTEXT=1 run")
    s.add_argument("treatment", nargs="?", help="trace from the sub-context run")
    s.add_argument("--all-runs", action="store_true",
                   help="merge every run in the file instead of the last")
    s.set_defaults(func=cmd_report)

    y = sub.add_parser("parity", help="compare two arms' generated text")
    y.add_argument("baseline", help="--out json from the baseline arm")
    y.add_argument("treatment", help="--out json from the sub-context arm")
    y.set_defaults(func=cmd_parity)

    m = sub.add_parser("summary",
                       help="one table over every arm of a completed toggle run")
    m.add_argument("dir", nargs="?", default=".",
                   help="directory holding client_/trace_/stage_ files (default .)")
    m.add_argument("--suffix", default="",
                   help="the _<tag>_ab suffix run_mas.sh gave the run's files")
    m.add_argument("--arms", default="base,sub,rot,idx,cdc",
                   help="which arms to put in the table, comma separated "
                        "(base, sub, rot, idx, cdc). An arm whose files are missing is left "
                        "out either way; this drops one that IS there. Deltas are "
                        "always against the leftmost column")
    m.set_defaults(func=cmd_summary)

    t = sub.add_parser("stages", help="diff host-side (CPU) stage timings")
    t.add_argument("baseline", help="SGLANG_STAGE_TRACE prefix from the baseline run")
    t.add_argument("treatment", help="SGLANG_STAGE_TRACE prefix from the sub-context run")
    t.set_defaults(func=cmd_stages)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
