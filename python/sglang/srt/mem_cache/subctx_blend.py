"""Selective recompute: recompute the reused tokens whose keys moved most.

Layers ``[0, check_layer]`` run the whole prompt (the probe). At ``check_layer`` each
reused token's fresh key is compared with its cached key, the top ``TOPK_RATIO`` per
request are selected, and the remaining layers run only the fresh plus selected tokens.

Counts, offsets and allocations are fixed before the forward; only the choice of
positions is made on the device, with no host sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BlendPlan:
    """Everything the forward needs to score, select, and cut down.

    Built by ``build_plan`` before the forward; ``select`` fills the ``sel_*`` fields.
    """

    check_layer: int

    # Rows per request: the probe (whole prompt), and the selection (fresh tokens plus
    # the recomputed ones).
    probe_lens: List[int]
    sel_lens: List[int]

    # Where the probe layers write KV, per probe row; reused rows go to dummy slot 0.
    probe_cache_loc: torch.Tensor

    # Per reused position, in the same order: its probe row, its cached slot, and its
    # flat offset into ``req_to_token``. ``reused_ptr`` splits them per request.
    reused_rows: torch.Tensor
    reused_slots: torch.Tensor
    reused_r2t: torch.Tensor
    reused_ptr: List[int]

    # Per reused position: whether its slot is this request's rotated copy.
    reused_ours: torch.Tensor

    # Per request: recomputed-token count, and fresh positions. `topk_slots` holds the
    # new slots for all of them.
    topk_counts: List[int]
    topk_slots: torch.Tensor
    topk_ptr: List[int]
    fresh_positions: List[torch.Tensor]

    req_pool_indices: List[int]
    row_offsets: List[int]

    # Filled by ``select`` at the scored layer.
    sel_rows: Optional[torch.Tensor] = None
    sel_positions: Optional[torch.Tensor] = None
    sel_topk_rows: Optional[torch.Tensor] = None
    sel_topk_slots: Optional[torch.Tensor] = None
    # Rotated-copy slots the selection displaced; freed by the caller.
    orphaned_slots: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.probe_lens)

    @property
    def total_topk(self) -> int:
        return self.topk_ptr[-1]

    def select(self, k: torch.Tensor, key_buffer: torch.Tensor) -> None:
        """Score the reused positions and select the top ``topk_counts`` per request.

        The score is ``sum((k_fresh - k_cached) ** 2)`` over heads and dims, where
        ``k`` is this layer's post-RoPE key per probe row and ``key_buffer`` still holds
        the cached key at the reused slots. Selected positions are merged with the fresh
        ones and sorted ascending.
        """
        if self.reused_rows.numel() == 0 or self.total_topk == 0:
            self._select_nothing()
            return

        diff = k[self.reused_rows] - key_buffer[self.reused_slots]
        score = (diff * diff).sum(dim=tuple(range(1, diff.dim())), dtype=torch.float32)

        device = k.device
        sel_rows: List[torch.Tensor] = []
        sel_positions: List[torch.Tensor] = []
        topk_rows: List[torch.Tensor] = []
        for i in range(self.batch_size):
            lo, hi = self.reused_ptr[i], self.reused_ptr[i + 1]
            count = self.topk_counts[i]
            fresh = self.fresh_positions[i]
            if count == 0:
                chosen_pos = fresh.new_empty((0,))
                chosen_idx = self.reused_rows.new_empty((0,))
            else:
                local = torch.topk(score[lo:hi], k=count).indices
                chosen_idx = local + lo
                chosen_pos = (
                    self.reused_rows[chosen_idx] - self.row_offsets[i]
                ).to(fresh.dtype)
            positions, _ = torch.sort(torch.cat([fresh, chosen_pos]))
            sel_positions.append(positions)
            sel_rows.append(positions + self.row_offsets[i])
            topk_rows.append(chosen_idx)

        self.sel_positions = torch.cat(sel_positions)
        self.sel_rows = torch.cat(sel_rows).to(device=device, dtype=torch.int64)
        self.sel_topk_rows = torch.cat(topk_rows)
        self.sel_topk_slots = self.topk_slots

    def commit(self, k: torch.Tensor, v: torch.Tensor, forward_batch, layer) -> None:
        """Move each selected position to a slot of its own, and fill that slot.

        Called after the scored layer's attention. Layers before ``check_layer`` are
        copied from the cached slot, this layer gets the fresh K/V, and later layers are
        written by the second range. Then req_to_token is re-pointed. The j-th selected
        position gets ``topk_slots[j]``.
        """
        from sglang.srt.mem_cache.memory_pool import move_kv_cache_native

        if self.sel_topk_rows is None or self.sel_topk_rows.numel() == 0:
            return

        pool = forward_batch.token_to_kv_pool
        chosen = self.sel_topk_rows
        slots = self.topk_slots

        # Layers before this one: copy from the cached slot.
        move_kv_cache_native(
            pool.k_buffer[: self.check_layer],
            pool.v_buffer[: self.check_layer],
            slots,
            self.reused_slots[chosen],
        )

        rows = self.reused_rows[chosen]
        pool.set_kv_buffer(layer, slots, k[rows], v[rows])

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        displaced = self.reused_slots[chosen]
        req_to_token.view(-1)[self.reused_r2t[chosen]] = slots.to(req_to_token.dtype)

        # Displaced rotated copies are no longer in req_to_token; the caller frees them.
        # Displaced tree slots stay with the tree.
        self.orphaned_slots = displaced[self.reused_ours[chosen]]

    def _select_nothing(self) -> None:
        """The selection is just the fresh tokens: nothing was reused, or ratio 0."""
        device = self.probe_cache_loc.device
        positions = torch.cat(self.fresh_positions) if self.fresh_positions else None
        rows = [
            self.fresh_positions[i] + self.row_offsets[i]
            for i in range(self.batch_size)
        ]
        self.sel_positions = positions
        self.sel_rows = torch.cat(rows).to(device=device, dtype=torch.int64)
        self.sel_topk_rows = self.reused_rows.new_empty((0,))
        self.sel_topk_slots = self.topk_slots.new_empty((0,))


def build_plan(
    batch,
    out_cache_loc: torch.Tensor,
    req_pool_indices: List[int],
    topk_slots: torch.Tensor,
    check_layer: int,
) -> BlendPlan:
    """Build the plan for one extend batch.

    ``out_cache_loc`` holds the fresh tokens' slots, ``topk_slots`` the slots for the
    reused tokens that will be recomputed.
    """
    device = out_cache_loc.device
    req_to_token = batch.req_to_token_pool.req_to_token
    stride = req_to_token.stride(0)

    probe_lens: List[int] = []
    sel_lens: List[int] = []
    topk_counts: List[int] = []
    row_offsets: List[int] = []
    fresh_positions: List[torch.Tensor] = []

    probe_loc_parts: List[torch.Tensor] = []
    reused_rows_parts: List[torch.Tensor] = []
    reused_slots_parts: List[torch.Tensor] = []
    reused_r2t_parts: List[torch.Tensor] = []
    reused_ours_parts: List[torch.Tensor] = []
    reused_ptr: List[int] = [0]
    topk_ptr: List[int] = [0]

    row_cursor = 0
    fresh_cursor = 0
    for i, req in enumerate(batch.reqs):
        n = len(req.fill_ids)
        fresh = batch.subctx_fresh_positions[i]
        count = req.sub_context_topk_count()
        req_idx = req_pool_indices[i]

        row_offsets.append(row_cursor)
        probe_lens.append(n)
        sel_lens.append(len(fresh) + count)
        topk_counts.append(count)
        topk_ptr.append(topk_ptr[-1] + count)

        fresh_t = torch.tensor(fresh, dtype=torch.int64, device=device)
        fresh_positions.append(fresh_t)

        # Reused rows write to the padded dummy slot 0.
        loc = torch.zeros(n, dtype=out_cache_loc.dtype, device=device)
        loc[fresh_t] = out_cache_loc[fresh_cursor : fresh_cursor + len(fresh)]
        fresh_cursor += len(fresh)
        probe_loc_parts.append(loc)

        reused = req.sub_context_reused_positions()
        if reused:
            reused_t = torch.tensor(reused, dtype=torch.int64, device=device)
            reused_rows_parts.append(reused_t + row_cursor)
            reused_slots_parts.append(
                torch.cat([s.to(device) for s in req.sub_context_reused_slots()]).long()
            )
            reused_r2t_parts.append(reused_t + req_idx * stride)
            reused_ours_parts.append(
                torch.tensor(
                    req.sub_context_reused_ours(), dtype=torch.bool, device=device
                )
            )
        reused_ptr.append(reused_ptr[-1] + len(reused))
        row_cursor += n

    empty = torch.empty(0, dtype=torch.int64, device=device)
    return BlendPlan(
        check_layer=check_layer,
        probe_lens=probe_lens,
        sel_lens=sel_lens,
        probe_cache_loc=torch.cat(probe_loc_parts),
        reused_rows=torch.cat(reused_rows_parts) if reused_rows_parts else empty,
        reused_slots=torch.cat(reused_slots_parts) if reused_slots_parts else empty,
        reused_r2t=torch.cat(reused_r2t_parts) if reused_r2t_parts else empty,
        reused_ours=(
            torch.cat(reused_ours_parts)
            if reused_ours_parts
            else empty.to(torch.bool)
        ),
        reused_ptr=reused_ptr,
        topk_counts=topk_counts,
        topk_slots=topk_slots,
        topk_ptr=topk_ptr,
        fresh_positions=fresh_positions,
        req_pool_indices=req_pool_indices,
        row_offsets=row_offsets,
    )
