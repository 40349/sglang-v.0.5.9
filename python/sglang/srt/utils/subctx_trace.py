"""Opt-in tracing of the sub-context path.

These probes fire once per radix insert, once per namespace match and once per
request. Left always on they do more than make the log noisy:

* Several sites build their message with real work -- slicing a token list for a
  preview, zipping the segment lists -- so the cost is paid even when nobody
  reads the output.
* Some of those sites sit *inside* the regions ``host_timer`` measures
  (``match``, ``cache_unfinished``, ``cache_finished``), so the debug output is
  billed to the very mechanism the timing is meant to cost.
* The sub-context sites fire only when the split is active, so in an A/B they
  land on one arm and show up as the mechanism being expensive.

Gate on ``SGLANG_SUBCTX_TRACE``; off by default. Guard at the call site, never
by passing a ready-built string, or the f-string is evaluated regardless:

    if TRACE_ON:
        trace(f"[TRACE-4 ...] hit={_preview(seg_ids[:hit])}")
"""

from __future__ import annotations

import os

# Read once at import: this is a benchmarking switch, set before the server
# starts, and the guard sits in hot paths where a dict lookup per call is waste.
TRACE_ON = os.environ.get("SGLANG_SUBCTX_TRACE", "") not in ("", "0")


def trace(msg: str) -> None:
    """Emit one trace line on stdout.

    print() rather than logging: the servers are launched with ``python -u`` and
    their stdout is redirected to the run log, which is where these lines are
    read back from.
    """
    print(msg)
