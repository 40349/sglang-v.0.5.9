"""Opt-in tracing of the sub-context path, gated on ``SGLANG_SUBCTX_TRACE``.

Off by default; some probes sit inside regions ``host_timer`` measures. Guard at the
call site so the message is not built when tracing is off:

    if TRACE_ON:
        trace(f"[TRACE-4 ...] hit={_preview(seg_ids[:hit])}")
"""

from __future__ import annotations

import os

# Read once at import.
TRACE_ON = os.environ.get("SGLANG_SUBCTX_TRACE", "") not in ("", "0")


def trace(msg: str) -> None:
    """Print one trace line to stdout (the server's run log)."""
    print(msg)
