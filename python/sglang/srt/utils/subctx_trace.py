"""Opt-in tracing of the sub-context path, gated on ``SGLANG_SUBCTX_TRACE``.

Off by default: these probes fire per insert/match/request, several build their
message with real work, and some sit inside the regions ``host_timer`` measures --
left on they bill debug output to the mechanism being timed, on one A/B arm only.

Guard at the call site, never by passing a ready-built string:

    if TRACE_ON:
        trace(f"[TRACE-4 ...] hit={_preview(seg_ids[:hit])}")
"""

from __future__ import annotations

import os

# Read once at import: set before the server starts, and the guard sits in hot paths.
TRACE_ON = os.environ.get("SGLANG_SUBCTX_TRACE", "") not in ("", "0")


def trace(msg: str) -> None:
    """Emit one trace line on stdout.

    print() rather than logging: servers run under ``python -u`` with stdout
    redirected to the run log, which is where these lines are read back from.
    """
    print(msg)
