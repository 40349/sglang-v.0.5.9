"""CacheBlend's quality tasks (2WikiMQA, MuSiQue: F1; SAMSum: Rouge-L) against a server.

The paper computes every passage's KV on its own, then answers from a prompt that
concatenates them. Over HTTP the same thing is: send each passage alone first (the
index files its chunks), then the full prompt. On `off` the full prompt shares no
prefix with the passages, so `off` is the full prefill the paper compares against.

Prompts, answer parsing, scoring and generation lengths follow CacheBlend's
example/blend_{wikimqa,musique,samsum}.py. What differs: the chat template (the split
only runs on /v1/chat/completions), so the paper's `[INST] ... [/INST]` becomes one
user message each for the prefix, every passage and the question -- cdc cuts before
every message, so a passage is found whole, as the paper reuses it; F1 tokenizes with
the served model's tokenizer; thinking is off.
Each run's reuse rate is printed: near zero on idx/cdc means the passages were not
found and the run measured nothing. Requests are streamed, so each answer records its
TTFT as well as its end-to-end latency.

    python blend_quality.py run http://127.0.0.1:30000          all three tasks
    python blend_quality.py compare ab_out/quality/<model> off cdc cdc_r15

`run` names its output <task>_<arm>.json after the server's own arm (as
sglang_server.sh tags it), so the arm is set once, on the server.

Needs `transformers` and `rouge_score`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
import urllib.request

DATA_FILES = {"wikimqa": "wikimqa_s.json", "musique": "musique_s.json", "samsum": "samsum.json"}

TASKS = {
    "wikimqa": dict(
        prefix="Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n",
        query="\n\nAnswer the question based on the given passages. Answer the question "
        "within 5 words. Do NOT repeat the question or output any other words. Question: ",
        max_tokens=32,
        metric="f1",
    ),
    "musique": dict(
        prefix="You will be asked a question after reading several passages. Please directly "
        "answer the question based on the given passages. Do NOT repeat the question. The "
        "answer should be within 5 words..\nPassages:\n",
        query="\n\nAnswer the question directly based on the given passages. Do NOT repeat "
        "the question. The answer should be within 5 words. \nQuestion:",
        max_tokens=32,
        metric="f1",
    ),
    "samsum": dict(
        prefix="Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n",
        query=None,
        max_tokens=128,
        metric="rougeL",
        max_ctx_len=3400,
    ),
}


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def normalize_question(q: str) -> str:
    if not q.endswith("?"):
        q = q + "?"
    return q[0].lower() + q[1:]


def build_prompt(task: str, ex: dict, tokenizer) -> tuple[list[str], str]:
    """Passages and the query text, as CacheBlend's build_{qa,fewshot}_prompt."""
    cfg = TASKS[task]
    if task == "samsum":
        docs = [c["text"] for c in ex["ctxs"]]
        q = "\n\n" + ex["question"]
        # Drop the middle example until the passages fit.
        lens = [len(tokenizer.encode(d, add_special_tokens=False)) for d in docs]
        while docs and sum(lens) > cfg["max_ctx_len"]:
            i = len(docs) // 2
            del docs[i], lens[i]
        return docs, q
    docs = [f"{c['title']}\n\n{c['text']}\n\n" for c in ex["ctxs"]]
    return docs, f"{cfg['query']}{normalize_question(ex['question'])}\nAnswer:"


def parse_generation(s: str) -> str:
    s = s.lstrip("\n").split("\n")[0]
    if not s.split():
        return s
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split()[0].startswith("No") or s.split()[0].startswith("no"):
        s = "No"
    return s


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_f1(pred: str, gold: str, tokenizer) -> float:
    pred = parse_generation(pred)
    gold_toks = tokenizer.encode(normalize_answer(gold), add_special_tokens=False)
    pred_toks = tokenizer.encode(normalize_answer(pred), add_special_tokens=False)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score(task: str, text: str, answers: list, tokenizer, scorer) -> float:
    if task == "samsum":
        text = text.lstrip("\n").split("\n")[0]
        return max(scorer.score(a, text)["rougeL"].fmeasure for a in answers)
    golds = [a[0] if isinstance(a, list) else a for a in answers]
    return max(compute_f1(text, g, tokenizer) for g in golds)


def chat(
    base: str, model: str, contents: list[str], max_tokens: int, thinking: bool, ignore_eos: bool = False
) -> tuple[str, dict, float, float]:
    """One streamed request: (text, usage, TTFT, end-to-end latency)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": c} for c in contents],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    text, usage, ttft = [], {}, None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or ():
                # The server sends its first chunk once the first token is out.
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append((choice.get("delta") or {}).get("content") or "")
    return "".join(text), usage, ttft, time.perf_counter() - t0


def arm_tag(sub: dict) -> str:
    """The server's arm as sglang_server.sh tags it: off, on, rot, idx, cdc, + _rNN."""
    if not sub.get("split_enabled"):
        return "off"
    if sub.get("index"):
        tag = "cdc" if sub.get("split_mode") == "cdc" else "idx"
    else:
        tag = "rot" if sub.get("rotate") else "on"
    ratio = float(sub.get("topk_ratio") or 0)
    return f"{tag}_r{int(ratio * 100 + 0.5):02d}" if ratio > 0 else tag


def run_task(task, args, base, model, info, tokenizer, scorer) -> None:
    cfg = TASKS[task]
    data = json.load(open(os.path.join(args.inputs, DATA_FILES[task])))
    if args.limit:
        data = data[: args.limit]
    out = os.path.join(
        args.out_dir, info["model_path"].split("/")[-1], f"{task}_{arm_tag(info['sub_context'])}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"  task:   {task}  samples: {len(data)}  prime: {not args.no_prime}")

    rows = []
    for i, ex in enumerate(data):
        docs, q = build_prompt(task, ex, tokenizer)
        if not docs:
            continue
        if not args.no_prime:
            # Two tokens, EOS ignored: on the split arms a request that finishes during
            # its prefill step is never filed, and its passage would not be found.
            for d in docs:
                chat(base, model, [d], 2, args.thinking, ignore_eos=True)
        text, usage, ttft, latency = chat(
            base, model, [cfg["prefix"], *docs, q], cfg["max_tokens"], args.thinking
        )
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rows.append(
            dict(
                idx=i,
                prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=cached,
                ttft_s=round(ttft, 4),
                latency_s=round(latency, 4),
                output=text,
                score=score(task, text, ex["answers"], tokenizer, scorer),
            )
        )
        if (i + 1) % 10 == 0 or i + 1 == len(data):
            mean = sum(r["score"] for r in rows) / len(rows)
            reuse = sum(r["cached_tokens"] for r in rows) / max(1, sum(r["prompt_tokens"] or 0 for r in rows))
            print(f"  [{i + 1}/{len(data)}] {cfg['metric']} {mean:.4f}  reused {reuse:.1%} of prompt tokens")

    summary = dict(
        task=task,
        metric=cfg["metric"],
        n=len(rows),
        score=sum(r["score"] for r in rows) / len(rows),
        reuse=sum(r["cached_tokens"] for r in rows) / max(1, sum(r["prompt_tokens"] or 0 for r in rows)),
        ttft_s=sum(r["ttft_s"] for r in rows) / len(rows),
        latency_s=sum(r["latency_s"] for r in rows) / len(rows),
        sub_context=info["sub_context"],
        model_path=info["model_path"],
    )
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    print(f"  {task}: {cfg['metric']} {summary['score']:.4f}  reused {summary['reuse']:.1%}  -> {out}")


def cmd_run(args) -> int:
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer

    base = args.url.rstrip("/").removesuffix("/v1")
    info = get(base + "/server_info")
    if info.get("sub_context") is None:
        print("REFUSING: /server_info has no sub_context block; this is not the fork.", file=sys.stderr)
        return 1
    model = args.model or get(base + "/v1/models")["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or info["model_path"])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    print(f"  server: {info['model_path']}  arm: {arm_tag(info['sub_context'])}")
    for task in args.task:
        run_task(task, args, base, model, info, tokenizer, scorer)
    return 0


def compare_pair(a: dict, b: dict, name_b: str) -> None:
    sa, sb = a["summary"], b["summary"]
    rb = {r["idx"]: r for r in b["rows"]}
    pairs = [(r, rb[r["idx"]]) for r in a["rows"] if r["idx"] in rb]
    diffs = [y["score"] - x["score"] for x, y in pairs]
    same = sum(x["output"] == y["output"] for x, y in pairs)
    worse = sum(d < -1e-9 for d in diffs)
    better = sum(d > 1e-9 for d in diffs)
    print(
        f"  {name_b:10}{sb['score']:>9.4f}{sb['score'] - sa['score']:>+9.4f}{sb['reuse']:>9.1%}"
        f"{sb.get('ttft_s', float('nan')):>9.3f}s{sb['latency_s']:>9.3f}s   same {same}/{len(pairs)}  worse {worse}  better {better}"
    )


def cmd_compare(args) -> int:
    for task in sorted(TASKS):
        paths = [os.path.join(args.dir, f"{task}_{arm}.json") for arm in args.arms]
        if not all(os.path.exists(p) for p in paths):
            continue
        runs = [json.load(open(p)) for p in paths]
        base = runs[0]["summary"]
        print(f"{task} ({base['metric']}), baseline {args.arms[0]}, {base['n']} samples")
        print(f"  {'arm':10}{'score':>9}{'delta':>9}{'reuse':>9}{'ttft':>10}{'latency':>10}")
        print(
            f"  {args.arms[0]:10}{base['score']:>9.4f}{'':>9}{base['reuse']:>9.1%}"
            f"{base.get('ttft_s', float('nan')):>9.3f}s{base['latency_s']:>9.3f}s"
        )
        for arm, run in zip(args.arms[1:], runs[1:]):
            compare_pair(runs[0], run, arm)
    return 0


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url", help="server base URL, e.g. http://127.0.0.1:30000")
    r.add_argument("--task", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    r.add_argument("--inputs", default=os.path.expanduser("~/Desktop/MiaoChen/CacheBlend/inputs"),
                   help="CacheBlend's inputs/ directory")
    r.add_argument("--out-dir", default=os.path.join(repo, "ab_out/quality"),
                   help="written to <out-dir>/<model>/<task>_<arm>.json")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--model", default=None, help="default: what /v1/models lists")
    r.add_argument("--tokenizer", default=None, help="default: the server's model_path")
    r.add_argument("--thinking", action="store_true", help="leave Qwen3 thinking on")
    r.add_argument("--no-prime", action="store_true", help="skip sending each passage alone first")
    c = sub.add_parser("compare")
    c.add_argument("dir", help="ab_out/quality/<model>")
    c.add_argument("arms", nargs="+", help="file tags, baseline first, e.g. off cdc cdc_r15")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
