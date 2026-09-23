"""GPU test for the attention a sub-context sparse prefill runs.

The kernel masks by each query's position (``q_positions``) instead of by index.
Checked:

- against a reference that attends by position (a query misses nothing it should see),
- with garbage ahead of every query, the output does not change (a query sees nothing
  ahead of its position),
- ``q_positions`` agrees with the same rule as a materialised ``custom_mask``.

Usage:
    python test_sparse_extend.py
    python -m pytest test_sparse_extend.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=30, suite="stage-b-test-small-1-gpu-amd")

import unittest

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd_unified,
)


def position_causal_mask(positions: torch.Tensor, kv_len: int) -> torch.Tensor:
    """``mask[q, j] = positions[q] >= j``, flattened the way the kernel reads it."""
    keys = torch.arange(kv_len, device=positions.device, dtype=positions.dtype)
    return (positions[:, None] >= keys[None, :]).to(torch.uint8).view(-1)


def reference_attention(q, k, v, positions, scale):
    """What each query should get: softmax over the keys at or behind its position."""
    out = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=torch.float32, device=q.device)
    groups = q.shape[1] // k.shape[1]
    for i, pos in enumerate(positions.tolist()):
        keys = k[: pos + 1].float()  # [pos+1, kv_heads, dim]
        values = v[: pos + 1].float()
        for head in range(q.shape[1]):
            kv_head = head // groups
            logits = (q[i, head].float() @ keys[:, kv_head].T) * scale
            out[i, head] = torch.softmax(logits, dim=-1) @ values[:, kv_head]
    return out


class TestSparsePositionCausalAttention(unittest.TestCase):
    """Attention over one gather list covering the whole prompt, masked by position."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs a GPU")
        torch.manual_seed(20260915)

    def _run(
        self, kv_len, positions, heads=4, kv_heads=4, dim=64, k=None, v=None,
        mode="positions",
    ):
        """``mode`` picks how the causal rule reaches the kernel: as ``q_positions``,
        which is what production passes, or as a materialised ``custom_mask``."""
        device = "cuda"
        dtype = torch.bfloat16
        if k is None:
            k = torch.randn(kv_len, kv_heads, dim, dtype=dtype, device=device)
        if v is None:
            v = torch.randn(kv_len, kv_heads, dim, dtype=dtype, device=device)
        positions = torch.tensor(positions, dtype=torch.int64, device=device)
        q = torch.randn(len(positions), heads, dim, dtype=dtype, device=device)
        o = torch.empty_like(q)
        scale = dim**-0.5
        by_mask = mode == "mask"

        extend_attention_fwd_unified(
            q,
            o,
            k,
            v,
            qo_indptr=torch.tensor([0, len(positions)], dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, kv_len], dtype=torch.int32, device=device),
            # `req_to_token` is indexed by position, so in a real pass entry j of this
            # list is whatever slot holds position j. Here the mapping is the identity,
            # which keeps the test about the mask.
            kv_indices=torch.arange(kv_len, dtype=torch.int64, device=device),
            # Zero because the kernel only consults it for the index-based causal rule
            # the mask replaces.
            prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
            max_len_extend=len(positions),
            custom_mask=position_causal_mask(positions, kv_len) if by_mask else None,
            mask_indptr=(
                torch.tensor(
                    [0, len(positions) * kv_len], dtype=torch.int64, device=device
                )
                if by_mask
                else None
            ),
            q_positions=None if by_mask else positions,
            sm_scale=scale,
            is_causal=False,
        )
        return q, k, v, o, positions, scale

    def test_scattered_queries_match_attending_by_position(self):
        """The gaps between three reused blocks, attending over the whole prompt."""
        kv_len = 300
        positions = list(range(0, 40)) + list(range(90, 150)) + list(range(260, 300))
        q, k, v, o, pos, scale = self._run(kv_len, positions)
        want = reference_attention(q, k, v, pos, scale)
        torch.testing.assert_close(o.float(), want, rtol=2e-2, atol=2e-2)

    def test_a_single_trailing_query_sees_the_whole_prompt(self):
        """The degenerate case the `input_len - 1` rule produces: everything reused but
        the last token, which still has to run to produce logits."""
        kv_len = 512
        q, k, v, o, pos, scale = self._run(kv_len, [kv_len - 1])
        want = reference_attention(q, k, v, pos, scale)
        torch.testing.assert_close(o.float(), want, rtol=2e-2, atol=2e-2)

    def test_grouped_query_attention_maps_heads_to_the_right_kv_head(self):
        kv_len = 200
        positions = list(range(0, 30)) + list(range(120, 200))
        q, k, v, o, pos, scale = self._run(kv_len, positions, heads=8, kv_heads=2)
        want = reference_attention(q, k, v, pos, scale)
        torch.testing.assert_close(o.float(), want, rtol=2e-2, atol=2e-2)

    def test_positions_and_a_materialised_mask_agree(self):
        """``q_positions`` and ``custom_mask`` agree over a layout with three gaps
        (to within an ulp of bfloat16 on a few elements).
        """
        kv_len = 300
        positions = list(range(0, 40)) + list(range(90, 150)) + list(range(260, 300))
        torch.manual_seed(1)
        _, _, _, by_pos, _, _ = self._run(kv_len, positions)
        torch.manual_seed(1)
        _, _, _, by_mask, _, _ = self._run(kv_len, positions, mode="mask")
        torch.testing.assert_close(by_pos, by_mask, rtol=1e-2, atol=1e-3)

    def test_a_query_cannot_see_past_its_own_position(self):
        """Changing only what sits ahead of every query leaves the output bit-identical."""
        kv_len = 256
        positions = list(range(0, 20)) + list(range(60, 100))
        horizon = max(positions)

        device = "cuda"
        k = torch.randn(kv_len, 4, 64, dtype=torch.bfloat16, device=device)
        v = torch.randn(kv_len, 4, 64, dtype=torch.bfloat16, device=device)

        torch.manual_seed(7)
        _, _, _, first, _, _ = self._run(kv_len, positions, k=k, v=v)

        future = torch.randn_like(k[horizon + 1 :]) * 50.0
        k[horizon + 1 :] = future
        v[horizon + 1 :] = future
        torch.manual_seed(7)
        _, _, _, second, _, _ = self._run(kv_len, positions, k=k, v=v)

        self.assertTrue(
            torch.equal(first, second),
            "output moved when KV ahead of every query changed -- the mask lets a "
            "query attend to its own future",
        )


if __name__ == "__main__":
    unittest.main()
