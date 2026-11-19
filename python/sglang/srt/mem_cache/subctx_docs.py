"""Documents a request marks in its prompt: token spans, each kept as one block.

A chat client marks a document by a character span of a message's content; it becomes
the span of the tokens whose text lies wholly inside it. A ``/generate`` client gives
token spans directly.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

Span = Tuple[int, int]


def valid_doc_spans(spans: Sequence[Sequence[int]], input_len: int) -> bool:
    """Whether ``spans`` are ``[start, end)`` pairs inside the prompt, increasing and
    not overlapping."""
    if not isinstance(spans, (list, tuple)):
        return False
    prev = 0
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return False
        a, b = span
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)):
            return False
        if not (prev <= a < b <= input_len):
            return False
        prev = b
    return True


def doc_block_ends(
    ends: Sequence[int], spans: Sequence[Sequence[int]], input_len: int
) -> List[int]:
    """Block ends with a cut at each document's start and end and none inside one.

    ``spans`` as ``valid_doc_spans`` accepts. The last end is ``input_len``.
    """
    starts = [a for a, _ in spans]
    out = set()
    for e in ends:
        i = bisect.bisect_left(starts, e) - 1
        if i >= 0 and e < spans[i][1]:
            continue
        out.add(e)
    for a, b in spans:
        if a > 0:
            out.add(a)
        out.add(b)
    out.add(input_len)
    return sorted(out)


def message_doc_spans(
    tokenizer, ids: Sequence[int], content: str, char_spans: Sequence[Sequence[int]]
) -> Optional[List[Span]]:
    """Token spans, relative to ``ids``, of documents given as ``[start, end)``
    character spans of ``content``.

    ``ids`` are one message's tokens as rendered, ``content`` among them verbatim. A
    span holds the tokens whose text lies wholly inside the character span; spans that
    hold no token are left out. None if ``ids`` do not decode and re-encode to
    themselves, or ``content`` is not in their text.
    """
    text = tokenizer.decode(list(ids), skip_special_tokens=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(ids):
        return None
    base = text.find(content)
    if base < 0:
        return None
    offsets = enc["offset_mapping"]
    first_char = [o[0] for o in offsets]
    out: List[Span] = []
    for s, e in char_spans:
        lo, hi = base + s, base + e
        a = bisect.bisect_left(first_char, lo)
        b = a
        while b < len(offsets) and offsets[b][1] <= hi:
            b += 1
        if b > a:
            out.append((a, b))
    return out


def prompt_doc_spans(
    tokenizer,
    ids: Sequence[int],
    marker: int,
    contents: Sequence[Optional[str]],
    char_spans: Dict[int, Sequence[Sequence[int]]],
    memo: Optional["OrderedDict"] = None,
    memo_size: int = 0,
) -> Tuple[List[List[int]], List[int]]:
    """Token spans in ``ids``, increasing, of documents given per message as character
    spans of its content (``char_spans[m]``), and the messages left out.

    Message m's tokens run from the m-th ``marker`` (the first message at 0) to the
    next; ``contents[m]`` is its content. A message is left out when it has no text
    content or ``message_doc_spans`` cannot place its spans. ``memo`` keeps placed
    messages by their tokens and spans, at most ``memo_size``.
    """
    starts = [0] + [i for i, t in enumerate(ids) if t == marker and i > 0]
    out: List[List[int]] = []
    left_out: List[int] = []
    for m in sorted(char_spans):
        content = contents[m] if 0 <= m < len(contents) else None
        if not isinstance(content, str) or m >= len(starts):
            left_out.append(m)
            continue
        a = starts[m]
        b = starts[m + 1] if m + 1 < len(starts) else len(ids)
        chars = tuple(sorted(tuple(c) for c in char_spans[m]))
        key = (b - a, hash(tuple(ids[a:b])), chars)
        spans = memo.get(key, ...) if memo is not None else ...
        if spans is ...:
            spans = message_doc_spans(tokenizer, ids[a:b], content, chars)
            if memo is not None:
                memo[key] = spans
                if len(memo) > memo_size:
                    memo.popitem(last=False)
        else:
            memo.move_to_end(key)
        if spans is None:
            left_out.append(m)
            continue
        out += [[a + s, a + e] for s, e in spans]
    out.sort()
    return out, left_out


def doc_relation(
    spans: Sequence[Sequence[int]], starts: Sequence[int], start: int, end: int
) -> int:
    """How ``[start, end)`` lies against the documents: 0 if it overlaps none, 1 if it
    is one of them exactly, -1 if it overlaps one otherwise. ``starts`` are the spans'
    starts."""
    i = bisect.bisect_left(starts, end) - 1
    if i < 0 or spans[i][1] <= start:
        return 0
    a, b = spans[i]
    if a == start and b == end:
        return 1
    return -1
