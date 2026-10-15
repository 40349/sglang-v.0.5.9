"""Unit tests for the replay harness's turn ordering (``subcontext_bench.py replay``).

Checked:

- each turn is linked to its conversation's previous turn: interleaved conversations
  stay apart, also when they share their first message, a conversation restarted from
  its first message is a new one, and a turn's reply is read from its next turn,
- reply lengths: the reply is templated as the server templates it (content None as
  "", list content as its text, tool-call arguments parsed), a message the template
  rejects is counted as plain text, turns without a next turn take the median, and
  each length is capped at the request's own max_tokens and at least 2,
- a pinned request carries nothing else that ends or sizes the generation,
- the concurrent runner never starts a turn before its previous turn finished, keeps
  at most N in flight, reaches N when turns are independent, stops on an error, and
  refuses a ``prev`` that is not an earlier index,
- a capture whose last line was cut short loads without it,
- ``--gen-tokens`` takes a token count of at least 2 or ``reply``.

Usage:
    python test_subcontext_bench_replay.py
    python -m pytest test_subcontext_bench_replay.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")

import argparse
import importlib.util
import json
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "subcontext_bench", Path(__file__).resolve().parents[3] / "subcontext_bench.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def turn(task, k, system=None, tag=None):
    """Request body of turn k (0-based) of a conversation about ``task``; ``tag``
    tells apart two runs of the same task."""
    tag = task if tag is None else tag
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": f"task {task}"})
    for j in range(k):
        msgs.append({"role": "assistant", "content": f"{tag} reply {j}"})
        msgs.append({"role": "user", "content": f"{tag} output {j}"})
    return {"messages": msgs}


def contents(reply):
    return [None if r is None else r["content"] for r in reply]


class TestLinkTurns(unittest.TestCase):
    def test_interleaved_conversations_stay_apart(self):
        reqs = [turn("A", 0), turn("B", 0), turn("A", 1), turn("B", 1), turn("A", 2)]
        prev, reply = bench.link_turns(reqs)
        self.assertEqual(prev, [None, None, 0, 1, 2])
        self.assertEqual(contents(reply), ["A reply 0", "B reply 0", "A reply 1", None, None])

    def test_two_runs_of_one_task_stay_apart(self):
        order = [("a", 0), ("b", 0), ("b", 1), ("a", 1), ("a", 2), ("b", 2), ("a", 3), ("b", 3)]
        reqs = [turn("T", k, tag=tag) for tag, k in order]
        prev, reply = bench.link_turns(reqs)
        # Turn 0 of both runs is the same prompt, so either one is a valid previous turn.
        self.assertIn(prev[3], (0, 1))
        self.assertEqual(prev[2], 1)
        self.assertEqual(prev[4], 3)
        self.assertEqual(prev[5], 2)
        self.assertEqual(prev[6], 4)
        self.assertEqual(prev[7], 5)
        self.assertEqual(reply[2]["content"], "b reply 1")
        self.assertEqual(reply[4]["content"], "a reply 2")

    def test_a_restarted_conversation_is_a_new_one(self):
        reqs = [turn("A", 0), turn("A", 1), turn("A", 0), turn("A", 1)]
        prev, reply = bench.link_turns(reqs)
        self.assertEqual(prev, [None, 0, None, 2])
        self.assertEqual(contents(reply), ["A reply 0", None, "A reply 0", None])

    def test_the_first_user_message_tells_conversations_apart(self):
        reqs = [turn("A", 0, system="s"), turn("B", 0, system="s"), turn("B", 1, system="s")]
        prev, _ = bench.link_turns(reqs)
        self.assertEqual(prev, [None, None, 1])


class FakeTokenizer:
    """Renders a conversation as one token per word of its JSON; refuses a message
    whose content is not a string, as some real templates do."""

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=True):
        assert return_dict is False
        self.seen.append(messages)
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise TypeError("content must be a string")
        text = json.dumps(messages) + (" assistant" if add_generation_prompt else "")
        return text.split()

    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestReplyLengths(unittest.TestCase):
    def count_words(self, m):
        return len(m["content"].split())

    def test_turns_without_a_reply_take_the_median(self):
        reply = [{"content": "a b c"}, None, {"content": "x y"}, {"content": "p q r s"}]
        counts, median, capped, raised = bench.reply_lengths(
            reply, self.count_words, [None] * 4
        )
        self.assertEqual((median, capped, raised), (3, 0, 0))
        self.assertEqual(counts, [3, 3, 2, 4])

    def test_lengths_are_capped_at_max_tokens_and_at_least_two(self):
        reply = [{"content": "a b c d e"}, {"content": ""}, {"content": "a b c"}]
        counts, _, capped, raised = bench.reply_lengths(reply, self.count_words, [4, None, 0])
        self.assertEqual(counts, [4, 2, 3])
        self.assertEqual((capped, raised), (1, 1))

    def test_a_median_filled_turn_is_capped_too(self):
        reply = [{"content": "a b c d e"}, None, {"content": "a b c d e f g"}]
        counts, median, capped, _ = bench.reply_lengths(reply, self.count_words, [None, 3, None])
        self.assertEqual((counts, median, capped), ([5, 3, 7], 7, 1))

    def test_no_reply_at_all_is_an_error(self):
        with self.assertRaises(ValueError):
            bench.reply_lengths([None, None], self.count_words, [None, None])


class TestRenderedReplyTokens(unittest.TestCase):
    def test_tool_call_arguments_are_templated_as_parsed_json(self):
        tok = FakeTokenizer()
        count = bench.RenderedReplyTokens(tok)
        call = {"id": "call_1", "type": "function",
                "function": {"name": "ls", "arguments": '{"path": "/tmp"}'}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        self.assertGreater(count(message), 0)
        self.assertEqual(count.fallbacks, 0)
        templated = tok.seen[-1][-1]
        self.assertEqual(templated["content"], "")
        self.assertEqual(templated["tool_calls"][0]["function"]["arguments"], {"path": "/tmp"})
        self.assertEqual(call["function"]["arguments"], '{"path": "/tmp"}')

    def test_list_content_is_templated_as_its_text(self):
        tok = FakeTokenizer()
        count = bench.RenderedReplyTokens(tok)
        message = {"role": "assistant",
                   "content": [{"type": "text", "text": "a b"}, {"type": "text", "text": " c"}]}
        count(message)
        self.assertEqual(tok.seen[-1][-1]["content"], "a b c")
        self.assertEqual(count.fallbacks, 0)

    def test_a_rejected_message_is_counted_as_plain_text(self):
        class Refusing(FakeTokenizer):
            def apply_chat_template(self, messages, **kw):
                if len(messages) > 1:
                    raise ValueError("unsupported message")
                return super().apply_chat_template(messages, **kw)

        count = bench.RenderedReplyTokens(Refusing())
        self.assertEqual(count({"role": "assistant", "content": "a b c"}), 3)
        self.assertEqual(count.fallbacks, 1)


class TestPinLength(unittest.TestCase):
    def test_only_the_pinned_length_ends_the_generation(self):
        body = {"messages": [], "max_tokens": 4096, "max_completion_tokens": 4096,
                "stop": ["<x>"], "stop_token_ids": [1], "min_tokens": 5,
                "response_format": {"type": "json_object"}, "temperature": 0.0}
        pinned = bench.pin_length(body, 37)
        self.assertEqual(pinned, {"messages": [], "max_tokens": 37, "ignore_eos": True,
                                  "temperature": 0.0})
        self.assertIn("stop", body)


class TestLoadRequests(unittest.TestCase):
    def write(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_a_cut_last_line_is_skipped(self):
        path = self.write(['{"a": 1}', '{"a": 2}', '{"a": 3, "messages": [{"ro'])
        self.assertEqual(bench.load_requests(path), [{"a": 1}, {"a": 2}])

    def test_a_bad_line_before_the_last_is_an_error(self):
        path = self.write(['{"a": 1}', '{"a": ', '{"a": 3}'])
        with self.assertRaises(json.JSONDecodeError):
            bench.load_requests(path)


class TestRunInTurnOrder(unittest.TestCase):
    def run_logged(self, n_workers, prev, delay=lambda i: 0.002, fail_at=None):
        log, lock = [], threading.Lock()
        in_flight = [0, 0]  # current, max

        def send(i):
            with lock:
                log.append(("start", i))
                in_flight[0] += 1
                in_flight[1] = max(in_flight[1], in_flight[0])
            time.sleep(delay(i))
            with lock:
                in_flight[0] -= 1
                log.append(("end", i))
            if i == fail_at:
                raise RuntimeError(f"request {i} failed")
            return {"i": i}

        rows = bench.run_in_turn_order(n_workers, prev, send)
        return rows, log, in_flight[1]

    def test_a_turn_starts_after_its_previous_turn_ends(self):
        rng = random.Random(0)
        reqs = []
        progress = {t: 0 for t in "ABCDEFGH"}
        while any(v < 6 for v in progress.values()):
            task = rng.choice([t for t, v in progress.items() if v < 6])
            reqs.append(turn(task, progress[task]))
            progress[task] += 1
        prev, _ = bench.link_turns(reqs)
        rows, log, peak = self.run_logged(4, prev, delay=lambda i: rng.random() * 0.004)
        self.assertEqual([r["i"] for r in rows], list(range(len(reqs))))
        at = {event: k for k, event in enumerate(log)}
        for i, p in enumerate(prev):
            if p is not None:
                self.assertLess(at[("end", p)], at[("start", i)])
        self.assertLessEqual(peak, 4)

    def test_independent_turns_fill_every_worker(self):
        _, _, peak = self.run_logged(5, [None] * 20, delay=lambda i: 0.02)
        self.assertEqual(peak, 5)

    def test_one_conversation_runs_one_turn_at_a_time(self):
        prev = [None] + list(range(9))
        _, log, peak = self.run_logged(4, prev)
        self.assertEqual(peak, 1)
        self.assertEqual([i for kind, i in log if kind == "start"], list(range(10)))

    def test_an_error_is_raised(self):
        with self.assertRaisesRegex(RuntimeError, "request 3 failed"):
            self.run_logged(3, [None] * 10, fail_at=3)

    def test_a_prev_that_is_not_earlier_is_refused(self):
        for prev in ([None, 1], [1, None], [None, 5]):
            with self.assertRaises(ValueError):
                bench.run_in_turn_order(2, prev, lambda i: {"i": i})


class TestGenTokensArg(unittest.TestCase):
    def test_a_count_or_reply(self):
        self.assertEqual(bench._gen_tokens_arg("32"), 32)
        self.assertEqual(bench._gen_tokens_arg("2"), 2)
        self.assertEqual(bench._gen_tokens_arg("reply"), "reply")
        for bad in ("1", "0", "-1", "lots"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench._gen_tokens_arg(bad)


if __name__ == "__main__":
    unittest.main()
