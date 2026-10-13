"""Selective recompute: recompute the reused tokens whose keys moved most.

Layers ``[0, TOPK_LAYER]`` run each prompt from its probe start on (the probe): past the
exact prefix of a sparse layout, past the reused prefix of any other request. Every row
writes the K/V it computes into its own slot, a reused row over its copied KV. Each
scored layer ranks the reused rows still running by how far their key moved from the
cached one, keeps the top ``TOPK_RATIOS[i]`` of the request's reused tokens, and the
layers after it run only the fresh plus kept rows.

Counts, offsets and allocations are fixed before the forward; only the choice of
positions is made on the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

# Rows scored per step, bounding the fp32 temporaries.
SCORE_CHUNK_ROWS = 8192


def _to_device(values: List[int], device) -> torch.Tensor:
    """``values`` as an int64 tensor on ``device``, copied without a host sync."""
    return torch.tensor(values, dtype=torch.int64).to(device, non_blocking=True)


@dataclass
class BlendPlan:
    """Everything the forward needs to score and cut down, per scored layer.

    Built by ``build_plan`` before the forward. ``rows`` and ``reused`` track what is
    still running; ``select`` advances them past one scored layer.
    """

    # The scored layers, consecutive and ascending.
    check_layers: List[int]

    # Per request: probe rows, the position of the first one, and its first row in the
    # batch.
    probe_lens: List[int]
    probe_starts: List[int]
    row_offsets: List[int]

    # Per request: fresh tokens; and per scored layer, the reused tokens kept after it.
    fresh_counts: List[int]
    keep_counts: List[List[int]]

    # Per probe row: the slot its K/V is written to. Fresh rows have new slots, reused
    # rows the request's own copy.
    probe_cache_loc: torch.Tensor

    # Probe rows of the fresh positions, and of the reused ones; ``reused_ptr`` splits the
    # reused rows per request. Both ascending.
    fresh_rows: torch.Tensor
    reused_rows: torch.Tensor
    reused_ptr: List[int]

    # Rows still running, ascending, and the reused ones among them as indices into
    # ``reused_rows``, grouped per request. Set by ``start``.
    rows: Optional[torch.Tensor] = None
    reused: Optional[torch.Tensor] = None
    # Scored layers passed so far.
    stage: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.probe_lens)

    def start(self) -> None:
        """Every probe row runs, every reused one among them."""
        self.rows = torch.arange(
            sum(self.probe_lens), dtype=torch.int64, device=self.probe_cache_loc.device
        )
        self.reused = torch.arange(
            self.reused_rows.numel(), dtype=torch.int64, device=self.rows.device
        )
        self.stage = 0

    def reused_counts(self) -> List[int]:
        """Per request: the reused rows still running."""
        if self.stage == 0:
            return [hi - lo for lo, hi in zip(self.reused_ptr, self.reused_ptr[1:])]
        return self.keep_counts[self.stage - 1]

    def lens(self) -> List[int]:
        """Per request: the rows still running."""
        if self.stage == 0:
            return list(self.probe_lens)
        return [f + k for f, k in zip(self.fresh_counts, self.keep_counts[self.stage - 1])]

    def running_keys(self, key_buffer: torch.Tensor) -> torch.Tensor:
        """The keys ``key_buffer`` holds at the slots of the reused rows still running."""
        return key_buffer[self.probe_cache_loc[self.reused_rows[self.reused]]]

    def select(self, cached: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
        """Keep the top scorers of this scored layer, and return the running rows to keep.

        ``cached`` and ``fresh`` are this layer's keys of the reused rows still running,
        before and after the layer wrote them. The score is ``sum((fresh - cached) ** 2)``
        over heads and dims. The returned indices address the rows that ran this layer.
        """
        keep_counts = self.keep_counts[self.stage]
        score = torch.empty(fresh.shape[0], dtype=torch.float32, device=fresh.device)
        for lo in range(0, fresh.shape[0], SCORE_CHUNK_ROWS):
            hi = lo + SCORE_CHUNK_ROWS
            diff = fresh[lo:hi].float() - cached[lo:hi].float()
            score[lo:hi] = diff.square_().flatten(1).sum(dim=1)

        kept: List[torch.Tensor] = []
        lo = 0
        for i, count in enumerate(self.reused_counts()):
            segment = self.reused[lo : lo + count]
            keep = keep_counts[i]
            if keep == count:
                kept.append(segment)
            elif keep > 0:
                top = torch.topk(score[lo : lo + count], k=keep).indices
                kept.append(torch.sort(segment[top]).values)
            lo += count
        reused = torch.cat(kept) if kept else self.reused[:0]

        rows = torch.sort(torch.cat([self.fresh_rows, self.reused_rows[reused]])).values
        keep_rows = torch.searchsorted(self.rows, rows)
        self.rows = rows
        self.reused = reused
        self.stage += 1
        return keep_rows


def build_plan(
    batch,
    out_cache_loc: torch.Tensor,
    ratios: List[float],
    check_layer: int,
) -> BlendPlan:
    """Build the plan for one extend batch.

    ``out_cache_loc`` holds the fresh tokens' slots. ``ratios`` gives one scored layer
    each, from ``check_layer`` on.
    """
    device = out_cache_loc.device

    probe_lens: List[int] = []
    probe_starts: List[int] = []
    row_offsets: List[int] = []
    fresh_counts: List[int] = []
    keep_counts: List[List[int]] = [[] for _ in ratios]

    probe_loc_parts: List[torch.Tensor] = []
    fresh_rows_parts: List[torch.Tensor] = []
    reused_rows_parts: List[torch.Tensor] = []
    reused_ptr: List[int] = [0]

    row_cursor = 0
    fresh_cursor = 0
    for i, req in enumerate(batch.reqs):
        start = req.sub_context_probe_start()
        rows = len(req.fill_ids) - start
        fresh = batch.subctx_fresh_positions[i]
        assert not fresh or fresh[0] >= start, (
            f"{req.rid}: computes position {fresh[0]}, before its probe start {start}"
        )

        row_offsets.append(row_cursor)
        probe_lens.append(rows)
        probe_starts.append(start)
        fresh_counts.append(len(fresh))

        fresh_t = _to_device(fresh, device)
        fresh_rows_parts.append(fresh_t - start + row_cursor)
        loc = torch.empty(rows, dtype=out_cache_loc.dtype, device=device)
        loc[fresh_t - start] = out_cache_loc[fresh_cursor : fresh_cursor + len(fresh)]
        fresh_cursor += len(fresh)

        reused = req.sub_context_reused_positions()
        assert len(fresh) + len(reused) == rows, (
            f"{req.rid}: {len(fresh)} fresh and {len(reused)} reused positions for "
            f"{rows} probe rows"
        )
        if reused:
            assert reused[0] >= start, (
                f"{req.rid}: scores position {reused[0]}, before its probe start {start}"
            )
            assert all(req.sub_context_reused_ours()), (
                f"{req.rid}: a scored run is not this request's own copy, and the probe "
                "writes over the slots it reads"
            )
            reused_t = _to_device(reused, device)
            reused_rows_parts.append(reused_t - start + row_cursor)
            loc[reused_t - start] = torch.cat(
                [s.to(device) for s in req.sub_context_reused_slots()]
            ).to(loc.dtype)
        probe_loc_parts.append(loc)
        for counts, n in zip(keep_counts, req.sub_context_topk_counts(ratios)):
            counts.append(n)
        reused_ptr.append(reused_ptr[-1] + len(reused))
        row_cursor += rows

    empty = torch.empty(0, dtype=torch.int64, device=device)
    return BlendPlan(
        check_layers=list(range(check_layer, check_layer + len(ratios))),
        probe_lens=probe_lens,
        probe_starts=probe_starts,
        row_offsets=row_offsets,
        fresh_counts=fresh_counts,
        keep_counts=keep_counts,
        probe_cache_loc=torch.cat(probe_loc_parts),
        fresh_rows=torch.cat(fresh_rows_parts) if fresh_rows_parts else empty,
        reused_rows=torch.cat(reused_rows_parts) if reused_rows_parts else empty,
        reused_ptr=reused_ptr,
    )
