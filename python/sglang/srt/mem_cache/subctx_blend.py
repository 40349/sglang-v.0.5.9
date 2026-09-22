"""Recompute the reused tokens whose keys moved the most, and keep the rest cached.

Rotation (``rotate_kv.py``) corrects a reused block's *position*. It cannot correct the
*context*: the block's KV was computed with different tokens in front of it, and no
rotation expresses that. This module buys the difference back for a fixed fraction of
the reused tokens -- the ones whose key, recomputed here, has moved furthest from what
the cache holds.

The obstacle is that the reuse path never recomputes a reused token, so there is no
fresh key to compare the cached one against. So the first layers run the whole prompt,
one layer is enough to have mixed the new context in, and the token dimension is cut
back down immediately after. The extra rows exist to produce keys; their KV is thrown
away.

Everything in here is shaped by one rule: **no host sync in the middle of a forward.**
The fraction is fixed, so every count, offset and allocation is known before the pass
starts. Only which positions win is decided on the device, and nothing downstream needs
to read that on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BlendPlan:
    """Everything the forward needs to score, select, and cut down.

    Built on the host in ``prepare_for_extend``; the ``sel_*`` fields are the only ones
    filled during the forward, by ``select``.
    """

    check_layer: int

    # Rows per request in each of the two token sets. Host-known: the probe is the
    # whole prompt, and the selection is the fresh tokens plus a fixed fraction of the
    # reused ones.
    probe_lens: List[int]
    sel_lens: List[int]

    # Where layers up to and including the scored one write their KV. One entry per
    # probe row; reused rows point at the padded dummy slot, because their real slots
    # belong to the radix tree and are shared with every other request holding that
    # block.
    probe_cache_loc: torch.Tensor

    # The reused positions, as three parallel views of the same ordering: the probe row
    # holding this position's fresh key, the cache row holding its stale one, and the
    # flat offset into ``req_to_token`` that addresses it. A score index means all
    # three.
    reused_rows: torch.Tensor
    reused_slots: torch.Tensor
    reused_r2t: torch.Tensor
    reused_ptr: List[int]

    # Whether each reused row sits on a copy made for this request (a rotated block)
    # rather than on the tree's own row. A recomputed position moves off its row, and
    # a copy nobody else holds is unreferenced the moment it does.
    reused_ours: torch.Tensor

    # Per request: how many reused tokens are recomputed, the slots held for them, and
    # the fresh positions they will be merged with.
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
    # Rows the selection displaced that belonged to this request. Nothing points at
    # them any more; the caller hands them back after the scored range.
    orphaned_slots: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.probe_lens)

    @property
    def total_topk(self) -> int:
        return self.topk_ptr[-1]

    def select(self, k: torch.Tensor, key_buffer: torch.Tensor) -> None:
        """Score the reused positions against the cache and pick the top fraction.

        ``k`` is this layer's freshly computed key for every probe row, already rotated
        to its new position; ``key_buffer`` is the pool's key for this layer, which at
        the reused slots still holds what the block was cached with. Their squared
        difference, summed over the head dimension, is the deviation the recompute is
        meant to undo.

        Sorted ascending before it is used anywhere: the extend kernel takes each
        query's position as its whole causal rule, and the row order has to agree with
        the position order for the last row of a request to still be its last token.
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
                # Indices into this request's slice of the reused arrays, which is also
                # an index into its cache rows and its req_to_token offsets.
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
        """Give the selected positions rows of their own, and fill every layer of them.

        Called *after* the scored layer's own attention, which is still reading the
        cache rows being replaced here.

        A selected position is about to stop pointing at the radix tree's row and start
        pointing at one of this request's, which decode will then read at **every**
        layer -- not just the ones recomputed after the cut. So the layers before the
        scored one are seeded from the cache first. Miss that and decode reads
        uninitialised pool memory for those layers, silently.

        The writes below use the same ordering of ``sel_topk_rows`` throughout, which is
        what pairs a chosen position with the slot that will hold it.
        """
        from sglang.srt.mem_cache.memory_pool import move_kv_cache_native

        if self.sel_topk_rows is None or self.sel_topk_rows.numel() == 0:
            return

        pool = forward_batch.token_to_kv_pool
        chosen = self.sel_topk_rows
        slots = self.topk_slots

        # Layers before this one: whatever the cache holds is the best available, and
        # for layer 0 it is exactly right -- that key is a function of the token and
        # its position alone. Layers after this one are written by the second range,
        # and this layer is written just below, so neither needs copying.
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

        # What the position was pointing at a moment ago. A tree row stays where it is
        # -- other requests are still served from it -- but a rotated copy was made for
        # this request alone, and the write above was the last thing referencing it.
        # The free paths walk `req_to_token`, so they will never see it again: it has to
        # be handed back explicitly, which the caller does once the pass is out of the
        # attention backend.
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
    """Lay out the two token sets before the forward runs.

    Called once per extend batch, with the slots the allocator has just handed out:
    ``out_cache_loc`` for the tokens that were going to be computed anyway, and
    ``topk_slots`` for the reused ones that will be recomputed on top.
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

        # The padded slot 0 absorbs every row whose KV must not be written. Duplicate
        # indices race each other there and the result is garbage nobody reads, which
        # is what that slot is for.
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
