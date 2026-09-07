#!/usr/bin/env python3
"""A/B harness for sub-context KV reuse.

Two subcommands:

  replay   Re-send a captured chat sequence to a server, in order.
  report   Aggregate the CUDA-event forward traces and diff baseline vs sub-context.

Replay rather than timing two live agent runs: an agent loop is closed, so the
moment one sampled token differs the runs take different actions and their GPU
totals compare different conversations. Replaying one captured sequence fixes the
input and leaves KV reuse as the only variable.

Every turn is pinned to --gen-tokens (with ignore_eos); free generation makes decode
length depend on the arm and swamps the prefill effect being measured.

Do NOT lower --gen-tokens to 1 to "measure prefill only": a request that never goes
through cache_unfinished_req gets no sub_context_last_nodes, so cache_finished_req
files the prompt under extra_key=None and the next request misses every namespace --
0% reuse, measuring the harness. Use --full for each request's own max_tokens.

The functional floor is 2, the useful floor higher. Decode is the control: the
mechanism cannot touch decode kernels, so the decode delta reads out this run's
drift. Measured at 32 (1600 decode passes) drift was 0.02%/-0.11% clock-locked; at
8 (400 passes) it was +3.4%. 32 keeps prefill at ~28% of GPU time and the control
working.

--gen-tokens also caps a known asymmetry: the baseline caches its generated output
while the split frees it, and regenerated tokens often match the recorded reply
verbatim, so the baseline reuses them next turn -- 168 tokens at 32, 89 at 8. A real
agent loop has no such bound.
"""

from __future__ import annotations

import argparse
import hashlib
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

    # The host timers live in the server processes and cannot see the line above,
    # so they watch for this file instead. Same instant, same window.
    if args.stage_trace:
        try:
            with open(f"{args.stage_trace}.mark", "w") as f:
                f.write(str(time.time()))
        except OSError as e:
            print(f"warning: cannot mark {args.stage_trace}.mark ({e})", file=sys.stderr)

    rows = []
    t_start = time.perf_counter()
    for i, body in enumerate(reqs):
        # Bisection aid: with the tree emptied before every request there is no
        # reuse in either arm, so the two arms must compute byte-identical
        # prefills. Divergence that survives this is in the split path itself,
        # not in what it chose to reuse. Ruins every timing number -- diagnosis
        # only, never a measurement run.
        if args.flush_every and i:
            try:
                urllib.request.urlopen(f"{base}/flush_cache", timeout=30).read()
                time.sleep(0.2)
            except urllib.error.URLError as e:
                print(f"warning: flush_cache failed ({e})", file=sys.stderr)
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

        # Hash the completion so the two arms can be compared token-for-token.
        # Sub-context only changes WHICH KV slots get reused; the reused KV has
        # to be bit-identical, so at temperature 0 any divergence here is
        # cache corruption. This is a sharper correctness test than pass@1 and
        # it costs one hash per request.
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
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
                "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
                **({"text": text} if args.save_text else {}),
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
    # A cold request has no `prompt_tokens_details` at all, so a single genuine zero
    # used to turn the whole run's summary into "n/a" and hide a real hit rate.
    # Only ALL rows missing means the server ran without --enable-cache-report.
    missing = sum(1 for r in rows if r["cached_tokens"] is None)
    if missing == len(rows):
        cache_str = "cached n/a (use --enable-cache-report; the forward trace has it either way)"
    else:
        tot_cached = sum(r["cached_tokens"] or 0 for r in rows)
        cache_str = (f"cached {tot_cached} tok "
                     f"({100.0 * tot_cached / tot_prompt if tot_prompt else 0:.1f}%)")
        if missing:
            cache_str += f" [{missing} request(s) reported no cache detail, counted as 0]"
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
    # Three trace vintages. Newest records the drop directly and is the only one
    # that survives chunked prefill; the middle one derived it by subtracting a
    # per-pass length from a per-request one, which goes negative by a chunk on
    # every continuation pass; the oldest has neither field, so "matched" can
    # only mean "reused" and the drop is unknowable rather than zero.
    if any("discarded_tokens" in r for r in ext):
        discarded_tok = sum(r.get("discarded_tokens", 0) for r in ext)
    else:
        discarded_tok = sum(r.get("matched_tokens", r["cached_tokens"]) for r in ext) - cached_tok
    matched_tok = cached_tok + discarded_tok
    # The share of the drop that is a position mismatch: matched in the tree but
    # computed at a different absolute position, so it cannot be stitched as-is.
    # None (not 0) when the trace predates the field -- "no moved hits" and "this
    # trace cannot say" are different claims.
    moved_tok = (
        sum(r.get("moved_tokens", 0) for r in ext)
        if any("moved_tokens" in r for r in ext)
        else None
    )
    # The share of the position mismatch that was WON BACK by rotating the block's K
    # to the position it is reused at. These tokens are counted in cached_tokens, so
    # moved + rotated is the whole displaced population and `rotated` is the arm's
    # headline number. None (not 0) when the trace predates the field.
    rotated_tok = (
        sum(r.get("rotated_tokens", 0) for r in ext)
        if any("rotated_tokens" in r for r in ext)
        else None
    )
    # Tokens of a block the tree had refused, rotated back to the position it holds
    # and filed there when the request finished. Not part of this run's cached_tokens
    # -- the payoff lands on *later* requests, as a longer hit and a cached reply.
    # None (not 0) when the trace predates the field.
    reinserted_tok = (
        sum(r.get("reinserted_tokens", 0) for r in ext)
        if any("reinserted_tokens" in r for r in ext)
        else None
    )
    # None, not 0, when the trace predates the field: "no request took the split
    # path" and "the trace cannot say" are different claims and print differently.
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
        ("prefill_total_tokens", "prefill tokens seen", "tok"),
        ("prefill_new_tokens", "  ...actually computed", "tok"),
        ("prefill_cached_tokens", "  ...from radix cache", "tok"),
        ("matched_tokens", "matched in the tree", "tok"),
        ("discarded_tokens", "  ...matched but DROPPED", "tok"),
        ("moved_tokens", "     ...dropped as MOVED", "tok"),
        ("rotated_tokens", "  ...MOVED but ROTATED in", "tok"),
        ("reinserted_tokens", "REVERSE-ROTATED into tree", "tok"),
        ("hit_rate", "hit rate (of tokens SEEN)", "%"),
        ("prefill_gpu_ms", "PREFILL GPU time", "ms"),
        ("prefill_ms_median", "  median pass", "ms"),
        ("us_per_new_token", "  us / computed token", "us"),
        ("decode_passes", "decode passes", ""),
        ("decode_gpu_ms", "decode GPU time", "ms"),
        ("total_gpu_ms", "TOTAL GPU time", "ms"),
    ]

    def gate_absent(arm: Dict[str, float]) -> bool:
        """True when this arm never ran the contiguity gate, so a drop of zero is
        not a measurement. None means the trace is too old to tell."""
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
    # `prefill tokens seen` is per PASS, so chunked prefill counts a request's
    # already-covered prefix once per continuation: the same capture reads 921,205
    # tokens unchunked and 1,092,928 when Stage 2 cuts at block boundaries. Anything
    # divided by it -- the hit rate above -- moves with the chunking regime and not
    # with how much was reused. `actually computed` does not: every token is computed
    # exactly once whatever the pass count.
    #
    # So state the reuse against a denominator that cannot move. The prompt total is a
    # property of the capture, which this command does not have, but chunking can only
    # inflate `seen`, so the smaller of the two arms bounds it from above and is exact
    # whenever that arm ran one pass per request.
    if a["prefill_passes"] != b["prefill_passes"]:
        prompt = min(a["prefill_total_tokens"], b["prefill_total_tokens"])
        print(
            f"\nNOTE: pass counts differ ({a['prefill_passes']:,} vs "
            f"{b['prefill_passes']:,}), so the two arms did NOT chunk the same way.\n"
            "That is expected when one arm cuts chunks at block boundaries; it does\n"
            "not by itself mean the request sequences differed. But it does mean\n"
            "'tokens seen' and the hit rate above are NOT comparable between the two\n"
            "columns -- a continuation pass re-counts the prefix it inherits.\n"
            "Compare 'actually computed', which is one entry per token either way:"
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
    "subctx_split": "  of which: finding block boundaries",
    "match": "prefix match / per-namespace stitch",
    "cache_unfinished": "insert prompt into the tree",
    "cache_finished": "release locks / free tail",
}

# child -> enclosing stage. The two overlap, so only one may enter the total.
#
# Take the http-side cost from the child. subctx_split is measured directly and
# reproduces to 0.4% across runs; tpl_render's delta is a difference between two
# ~13 ms numbers whose own spread (~800 us) swamps the ~680 us being resolved.
# Both estimate the same quantity -- the parent minus the nested child is the jinja
# work, identical in both arms and duly ~0 -- so this picks the better estimator,
# not a different number. cmd_stages prints the residual to keep that checkable.
NESTED = {"subctx_split": "tpl_render"}


def load_stages(prefix: str) -> tuple[Dict[str, Dict[str, float]], bool]:
    """Merge the per-process stage files a run wrote (<prefix>.http.json, .scheduler.json).

    Returns the merged stages and whether they came from the post-warm-up
    window. ``measured`` is preferred but only when *every* process has it: a
    mix of windows across processes is worse than one honest wide window.
    """
    import glob

    hits = sorted(glob.glob(f"{prefix}.*"))
    hits = [h for h in hits if not h.endswith(".mark")]
    if not hits:
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
            # Stage names are global, not per-process; a collision would have one
            # process silently overwrite the other's numbers.
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
        # Compare per-call cost scaled to a common call count, never raw totals.
        # The arms can legitimately differ by a call or two -- a dump can land
        # between a nested stage's add() and its parent's -- and subtracting
        # unequal totals turns that off-by-one into phantom milliseconds.
        n_ref = max(sa["count"] if sa else 0, sb["count"] if sb else 0)
        norm = lambda s: (s["mean_us"] * n_ref / 1000.0) if s else 0.0
        added = norm(sb) - norm(sa)
        if sa and sb and sa["count"] != sb["count"]:
            skewed.append((stage, sa["count"], sb["count"]))
        if stage in NESTED.values():
            note = "  (level only, see below)"
        else:
            total_added += added
            note = ""
        print(f"{stage:<18} {fa:>22} {fb:>22} {added:>+9.1f}ms{note}")

    print("-" * 76)
    print(f"{'TOTAL host overhead added':<18} {'':>45} {total_added:>+9.1f}ms")
    n_req = max((s["count"] for s in list(a.values()) + list(b.values())), default=0)
    if n_req:
        print(f"{'  per request':<18} {'':>45} {1000.0 * total_added / n_req:>+9.1f}us")

    print("\n(per-call means scaled to a common call count)")
    for child, parent in NESTED.items():
        if child not in b:
            continue
        print(f"\n{child} is nested inside {parent}, so only one enters the total:")
        print(f"  {child:<20} counted  -- measured directly, baseline is the disabled path")
        print(f"  {parent:<20} EXCLUDED -- its delta is a difference of two ~13ms numbers")
        # The parent minus the nested child is the work both arms do identically.
        # It should be zero; anything else means the split is not the whole story.
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


def cmd_parity(args: argparse.Namespace) -> int:
    """Compare what the two arms generated, request by request.

    At temperature 0 with the same weights, the arms must produce the same
    tokens: the split changes which cached KV is reused, and reused KV that is
    not bit-identical shows up as a different sampled token. A mismatch here is
    a correctness bug, not noise -- unlike a pass@1 difference, which a single
    flipped token can cause without anything being wrong.
    """
    with open(args.baseline) as f:
        a = json.load(f)
    with open(args.treatment) as f:
        b = json.load(f)

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
    r.add_argument("--stage-trace", default=None,
                   help="the server's SGLANG_STAGE_TRACE prefix; same purpose as "
                        "--trace, for the host-side timers")
    r.add_argument("--no-flush", action="store_true")
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

    y = sub.add_parser("parity", help="check two arms generated identical text")
    y.add_argument("baseline", help="--out json from the baseline arm")
    y.add_argument("treatment", help="--out json from the sub-context arm")
    y.set_defaults(func=cmd_parity)

    t = sub.add_parser("stages", help="diff host-side (CPU) stage timings")
    t.add_argument("baseline", help="SGLANG_STAGE_TRACE prefix from the baseline run")
    t.add_argument("treatment", help="SGLANG_STAGE_TRACE prefix from the sub-context run")
    t.set_defaults(func=cmd_stages)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
