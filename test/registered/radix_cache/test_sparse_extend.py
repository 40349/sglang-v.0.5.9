"""
GPU test for the attention a sub-context sparse prefill runs.

The claim under test: when reused blocks sit wherever the prompt puts them, a prefill
can still be exact. Everything about that rests on the mask. The kernel's own causal
rule compares *indices* -- query i may see key i and everything before it -- and a
sparse prefill breaks that correspondence, because its queries are the gaps between
reused blocks and are not at the indices their positions imply. So causality is handed
over to an explicit mask, and if that mask is wrong attention still returns a number.

Two properties, because a mask can fail in two directions:

- too strict, and a query misses keys it should have attended to. Caught by comparing
  against a reference that attends by position.
- too permissive, and a query attends to a key *ahead* of it -- reading KV that, in a
  real prefill, has not been computed. Caught by filling the positions behind a query
  with garbage and requiring its output not to move.

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
    """``mask[q, j] = positions[q] >= j``, flattened the way the kernel reads it.

    This is the production rule, kept here in one line so the tests below exercise the
    shape and the kernel's addressing of it rather than a second implementation.
    """
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

    def _run(self, kv_len, positions, heads=4, kv_heads=4, dim=64, k=None, v=None):
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
            custom_mask=position_causal_mask(positions, kv_len),
            mask_indptr=torch.tensor(
                [0, len(positions) * kv_len], dtype=torch.int64, device=device
            ),
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

    def test_a_query_cannot_see_past_its_own_position(self):
        """The direction a reference comparison alone would not catch.

        In a real sparse prefill the positions ahead of a query hold KV that has not
        been computed yet, or a reused block that is causally in its future. If the mask
        let either through, attention would still return a plausible number. So: run
        twice, changing only what sits ahead of every query, and require bit-identical
        output.
        """
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
