"""Pick RelayCaching's L_start, L_det, L_end from a relay@profile run.

Reads the JSON lines ``RelayProfile`` writes (one per request with reused tokens),
averages the per-layer similarity and the adjacent-layer Spearman rho over requests, and
applies the official layer-range profiler's rules
(github.com/YingshengGeng/RelayCaching, experiments/observation/layer_selection.py,
MIT License, Copyright (c) 2025 Yingsheng Geng, Yuchong Gao, Weihong Wu, Guyue Liu,
Jiang Liu): ``auto_select_middle_layer_range`` for L_start / L_end and
``select_detection_layer_from_correlations`` for L_det. The defaults are the profiler
CLI's (``run_layer_profiler.py``).

Usage:
    python relay_layers.py PROFILE.jsonl [--group dec|all] [--metric code|tok]
        [--tau-low 0.99] [--stable-multiplier 0.5] [--consecutive-stables 2]
        [--min-rise-samples 3] [--right-tail-layers 5] [--min-reused N]

``--group dec`` uses the reused tokens decode computed (the paper's decoding KV);
``--metric code`` the similarity the official profiler computes, ``tok`` the paper's
per-token one.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

import numpy as np

# rho before layer 0 (``_RHO_PRIOR_BEFORE_LAYER0``).
RHO_BEFORE_LAYER0 = 0.0


def select_layer_range(
    sims: List[float],
    lower_threshold: float,
    upper_sigma_multiplier: float,
    upper_consecutive_count: int,
    upper_min_samples: int,
    right_tail_layers: int,
) -> Tuple[int, int]:
    """(L_start, L_end), both inclusive (``auto_select_middle_layer_range``)."""
    num_layers = len(sims)

    last_high = -1
    for i in range(num_layers):
        if sims[i] >= lower_threshold:
            last_high = i
    lower = last_high if last_high >= 0 else 0

    valley = int(np.argmin(sims))
    tail = max(1, min(int(right_tail_layers), num_layers))
    tail_mean = float(np.mean(sims[-tail:]))
    tail_std = float(np.std(sims[-tail:]))
    entry = tail_mean - upper_sigma_multiplier * tail_std

    upper = num_layers
    n = 0
    rise_mean = 0.0
    stable_count = 0
    stable_start = None
    for i in range(valley + 1, num_layers):
        rise = sims[i] - sims[i - 1]
        n += 1
        rise_mean += (rise - rise_mean) / n
        if n >= upper_min_samples:
            stable = sims[i] >= entry and abs(rise) < max(tail_std * 2, 0.02)
            if stable:
                if stable_count == 0:
                    stable_start = i - 1
                stable_count += 1
                if stable_count >= upper_consecutive_count:
                    upper = stable_start
                    break
            else:
                stable_count = 0
                stable_start = None

    end = min(num_layers - 1, max(lower, upper))
    if upper >= num_layers:
        end = num_layers - 1
    if lower > end:
        lower = int(num_layers * 0.25)
        end = int(num_layers * 0.85) - 1
    min_range = max(3, num_layers // 10)
    if end - lower + 1 < min_range:
        mid = (lower + end) // 2
        lower = max(0, mid - min_range // 2)
        end = min(num_layers - 1, mid + min_range // 2)
    return lower, end


def _rho_at(correlations: Dict[int, float], layer: int) -> float:
    if layer in correlations:
        return correlations[layer]
    if layer < 0:
        return RHO_BEFORE_LAYER0
    raise KeyError(f"no Spearman rho for pair-start layer {layer}")


def second_order_diffs(correlations: Dict[int, float]) -> Dict[int, float]:
    """alpha_l = rho_l - 2 rho_{l-1} + rho_{l-2} (``compute_second_order_diffs``)."""
    if not correlations:
        return {}
    out = {}
    for l in range(1, max(correlations) + 1):
        if l in correlations:
            out[l] = _rho_at(correlations, l) - 2.0 * _rho_at(correlations, l - 1) + _rho_at(
                correlations, l - 2
            )
    return out


def select_detection_layer(
    correlations: Dict[int, float], start: int, end: int, num_layers: int
) -> Tuple[int, Dict[int, float]]:
    """L_det (``select_detection_layer_from_correlations``)."""
    alpha = second_order_diffs(correlations)
    pair_end = min(end, num_layers - 2)
    scan = sorted(l for l in alpha if l <= pair_end)
    after_start = [l for l in scan if l > start]
    det = start + 1
    for cur, nxt in zip(scan, scan[1:]):
        if alpha[cur] >= 0 and alpha[nxt] < 0:
            det = nxt
            break
    else:
        if after_start:
            if alpha[after_start[0]] < 0:
                det = start + 1
            else:
                det = min(after_start, key=lambda l: alpha[l])
    if det <= start:
        det = start + 1
    return min(det, end), alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("profile")
    ap.add_argument("--group", choices=["dec", "all"], default="dec")
    ap.add_argument("--metric", choices=["code", "tok"], default="code")
    ap.add_argument("--tau-low", type=float, default=0.99)
    ap.add_argument("--stable-multiplier", type=float, default=0.5)
    ap.add_argument("--consecutive-stables", type=int, default=2)
    ap.add_argument("--min-rise-samples", type=int, default=3)
    ap.add_argument("--right-tail-layers", type=int, default=5)
    ap.add_argument(
        "--min-reused", type=int, default=2, help="skip requests with fewer rows in the group"
    )
    args = ap.parse_args()

    sims, rhos = [], []
    num_layers = None
    seen = 0
    with open(args.profile) as fh:
        for line in fh:
            rec = json.loads(line)
            seen += 1
            group = rec.get(args.group)
            n = rec["n_decoded"] if args.group == "dec" else rec["n_reused"]
            if group is None or n < args.min_reused:
                continue
            if num_layers is None:
                num_layers = rec["num_layers"]
            sims.append(group["s_code" if args.metric == "code" else "s_tok"])
            rhos.append(group["rho"])
    if not sims:
        print(f"no request in {args.profile} has {args.min_reused}+ rows in group "
              f"{args.group!r} ({seen} lines read)", file=sys.stderr)
        return 1

    sim = np.mean(np.array(sims), axis=0).tolist()
    corr = {l: float(v) for l, v in enumerate(np.mean(np.array(rhos), axis=0))}
    start, end = select_layer_range(
        sim,
        args.tau_low,
        args.stable_multiplier,
        args.consecutive_stables,
        args.min_rise_samples,
        args.right_tail_layers,
    )
    det, alpha = select_detection_layer(corr, start, end, num_layers)

    print(f"{len(sims)} of {seen} requests, group {args.group}, metric {args.metric}, "
          f"{num_layers} layers")
    print(f"{'layer':>5} {'sim':>8} {'rho(l,l+1)':>11} {'alpha':>8}")
    for l in range(num_layers):
        r = f"{corr[l]:11.4f}" if l in corr else f"{'':>11}"
        a = f"{alpha[l]:8.4f}" if l in alpha else f"{'':>8}"
        print(f"{l:5d} {sim[l]:8.4f} {r} {a}")
    print(f"L_start {start}  L_det {det}  L_end {end}   ->  ARM=relay@{start},{det},{end}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
