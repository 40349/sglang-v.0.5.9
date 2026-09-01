"""
GPU test for the sub-context KV rotation kernel.

The claim under test is the one the whole feature rests on: because RoPE's angle is
linear in the position, re-rotating a cached key by ``delta`` reproduces the key that
would have been written at ``position + delta``. Everything else about the feature is
bookkeeping around that identity, covered on CPU by ``test_sub_context_unit.py``.

The oracle is the model's own ``RotaryEmbedding.forward_native``: the same raw key is
RoPE'd at ``p`` (and cached) and at ``p + delta`` (the target), and the kernel has to
turn the first into the second. V must come through byte-identical -- it carries no
position at all.

Usage:
    python test_rotate_kv.py
    python -m pytest test_rotate_kv.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=60, suite="stage-b-test-small-1-gpu-amd")

import unittest

import torch

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# `get_rope` reads the global server args while building its inv_freq.
try:
    from sglang.srt.server_args import get_global_server_args

    get_global_server_args()
except ValueError:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.mem_cache.rotate_kv import KVRotator


class FakePool:
    """The five attributes ``KVRotator`` touches, over freshly allocated buffers."""

    def __init__(self, size, layers, head_num, head_dim, dtype, device):
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.k_buffer = [
            torch.zeros((size, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layers)
        ]
        self.v_buffer = [
            torch.zeros((size, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layers)
        ]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=device
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=device
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestRotateKV(unittest.TestCase):
    # (layers, kv_heads, head_dim, tokens, start_position, delta)
    CASES = [
        (4, 8, 128, 37, 0, 40),  # Qwen3-8B shape; the block-grew-by-40 case
        (4, 4, 128, 200, 593, 107),  # Qwen3-Coder shape; a head-length shift
        (2, 8, 128, 16, 500, -400),  # backwards: canonical after the target
        (2, 4, 128, 1, 0, 1),  # a single token, the smallest delta
        (2, 8, 128, 64, 100, 30000),  # far apart, still inside cos_sin_cache
    ]

    def _run(self, layers, head_num, head_dim, n, p0, delta, dtype, native):
        device = "cuda"
        rope = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=40960,
            base=1000000,
            rope_scaling=None,
            dtype=dtype,
        ).to(device)
        # A subclass here would mean the delta identity does not hold.
        self.assertIs(type(rope), RotaryEmbedding)

        pool = FakePool(4 * n + 8, layers, head_num, head_dim, dtype, device)
        torch.manual_seed(0)
        src = torch.arange(1, n + 1, dtype=torch.int64, device=device)
        dst = torch.arange(n + 1, 2 * n + 1, dtype=torch.int64, device=device)

        want_k = []
        for layer in range(layers):
            k_raw = torch.randn(n, head_num * head_dim, dtype=dtype, device=device)
            unused_q = torch.zeros_like(k_raw)

            at_p0 = torch.arange(p0, p0 + n, dtype=torch.int64, device=device)
            _, k_cached = rope.forward_native(at_p0, unused_q.clone(), k_raw.clone())
            pool.k_buffer[layer][src] = k_cached.view(n, head_num, head_dim)
            pool.v_buffer[layer][src] = torch.randn(
                n, head_num, head_dim, dtype=dtype, device=device
            )

            at_target = torch.arange(
                p0 + delta, p0 + delta + n, dtype=torch.int64, device=device
            )
            _, k_target = rope.forward_native(
                at_target, unused_q.clone(), k_raw.clone()
            )
            want_k.append(k_target.view(n, head_num, head_dim))

        KVRotator(pool, rope.cos_sin_cache, head_dim, use_native=native).rotate_into(
            dst, src, delta
        )

        for layer in range(layers):
            got = pool.k_buffer[layer][dst].float()
            ref = want_k[layer].float()
            # bf16 keeps ~8 mantissa bits, so the bound is relative to the values.
            tol = 3e-2 * max(ref.abs().max().item(), 1.0)
            self.assertLess(
                (got - ref).abs().max().item(),
                tol,
                f"K mismatch at layer {layer} (delta={delta})",
            )
            self.assertTrue(
                torch.equal(pool.v_buffer[layer][dst], pool.v_buffer[layer][src]),
                f"V was modified at layer {layer}; it carries no position",
            )

    def test_triton_matches_rope_at_the_shifted_position(self):
        for case in self.CASES:
            with self.subTest(case=case):
                self._run(*case, dtype=torch.bfloat16, native=False)

    def test_native_path_matches_too(self):
        """The torch fallback must agree; it is the kernel's reference."""
        for case in self.CASES:
            with self.subTest(case=case):
                self._run(*case, dtype=torch.bfloat16, native=True)

    def test_source_block_is_left_untouched(self):
        """The source belongs to a shared tree node: rotating must not write to it."""
        device = "cuda"
        rope = get_rope(128, 128, 40960, 1000000, rope_scaling=None).to(device)
        pool = FakePool(64, 2, 4, 128, torch.bfloat16, device)
        src = torch.arange(1, 9, dtype=torch.int64, device=device)
        dst = torch.arange(20, 28, dtype=torch.int64, device=device)
        for layer in range(2):
            pool.k_buffer[layer][src] = torch.randn(
                8, 4, 128, dtype=torch.bfloat16, device=device
            )
        before = [pool.k_buffer[i][src].clone() for i in range(2)]

        KVRotator(pool, rope.cos_sin_cache, 128).rotate_into(dst, src, 41)

        for layer in range(2):
            self.assertTrue(torch.equal(pool.k_buffer[layer][src], before[layer]))

    def test_in_place_rotation_matches_the_copying_one(self):
        """``dst is src``: the write path rotates a block where it already lies.

        Nothing new is allocated there -- the slots are the finishing request's own --
        so the kernel has to be correct when source and destination are the same row.
        Each program loads its row before it stores to it, so this holds, and it is
        cheap enough to keep proving.
        """
        device = "cuda"
        for native in (False, True):
            with self.subTest(native=native):
                rope = get_rope(128, 128, 40960, 1000000, rope_scaling=None).to(device)
                pool = FakePool(64, 3, 4, 128, torch.bfloat16, device)
                loc = torch.arange(1, 17, dtype=torch.int64, device=device)
                other = torch.arange(20, 36, dtype=torch.int64, device=device)
                torch.manual_seed(1)
                for layer in range(3):
                    block = torch.randn(
                        16, 4, 128, dtype=torch.bfloat16, device=device
                    )
                    pool.k_buffer[layer][loc] = block
                    pool.k_buffer[layer][other] = block  # the same K, twice
                    pool.v_buffer[layer][loc] = torch.randn(
                        16, 4, 128, dtype=torch.bfloat16, device=device
                    )
                    pool.v_buffer[layer][other] = pool.v_buffer[layer][loc]
                v_before = [pool.v_buffer[i][loc].clone() for i in range(3)]

                rotator = KVRotator(
                    pool, rope.cos_sin_cache, 128, use_native=native
                )
                rotator.rotate_into(loc, loc, -23)  # in place
                rotator.rotate_into(other, other.clone(), -23)  # separate tensors

                for layer in range(3):
                    self.assertTrue(
                        torch.equal(
                            pool.k_buffer[layer][loc], pool.k_buffer[layer][other]
                        ),
                        f"in-place K differs from the two-tensor result at {layer}",
                    )
                    self.assertTrue(
                        torch.equal(pool.v_buffer[layer][loc], v_before[layer]),
                        f"in-place V was disturbed at layer {layer}",
                    )

    def test_delta_bounds(self):
        rope = get_rope(128, 128, 40960, 1000000, rope_scaling=None).to("cuda")
        rotator = KVRotator(FakePool(8, 1, 4, 128, torch.bfloat16, "cuda"), rope.cos_sin_cache, 128)
        self.assertFalse(rotator.can_rotate(0))  # nothing to do, and not the kernel's job
        self.assertTrue(rotator.can_rotate(-5))
        self.assertTrue(rotator.can_rotate(rotator.max_delta))
        self.assertFalse(rotator.can_rotate(rotator.max_delta + 1))


if __name__ == "__main__":
    unittest.main()
