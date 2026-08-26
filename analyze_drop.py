#!/usr/bin/env python3
"""Explain the contiguity rule's discards from a SUBCTX_TRACE server log.

`_stitch_sub_contexts` reuses a segment's hit only while the reused slots still
form a contiguous prefix from position 0. With a [system][messages] split that
collapses to one rule: the messages hit is spendable if and only if the system
segment hit in full. So every discarded token is a messages hit gated by the
much smaller system block, and the question is why that small block ever misses.

Two candidate causes, with different fixes:
  eviction    the system block is in the tree but gets pushed out under memory
              pressure -> pin the namespace, or reserve for it.
  never there the block is not being inserted correctly -> a bug, and fixing it
              would recover the discards outright.

Eviction predicts discards that START LATE and track KV-pool occupancy.
A broken insert predicts discards spread evenly from the first request.
"""
import re, sys
from collections import OrderedDict, defaultdict

MATCH = re.compile(
    r"sub-context match rid=(\S+)\s+extra_key='([^']+)'\s+hit=(\d+)/(\d+)")
USAGE = re.compile(r"token usage: ([0-9.]+)")

log = sys.argv[1]
reqs = OrderedDict()          # rid -> {key: (hit, seg_len)}, last stitch wins
usage_trail = []              # KV pool occupancy over the run

with open(log, errors="replace") as f:
    for line in f:
        m = MATCH.search(line)
        if m:
            rid, key, hit, ln = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            # A re-scheduled request stitches again; keep the latest attempt.
            if rid in reqs and key in reqs[rid]:
                reqs[rid] = {key: (hit, ln)}
            else:
                reqs.setdefault(rid, {})[key] = (hit, ln)
            continue
        u = USAGE.search(line)
        if u:
            usage_trail.append(float(u.group(1)))

rows = [(rid, d) for rid, d in reqs.items()
        if "system_prompt_key" in d and "messages_key" in d]
print(f"requests with a two-segment split: {len(rows)} (of {len(reqs)} traced)\n")
if not rows:
    sys.exit("no two-segment requests in the trace")

tot_sys_len = tot_sys_hit = tot_msg_len = tot_msg_hit = 0
discarded = spent = 0
n_full = n_partial = n_miss = 0
first_bad = None
bad_positions = []

for i, (rid, d) in enumerate(rows):
    sh, sl = d["system_prompt_key"]
    mh, ml = d["messages_key"]
    tot_sys_len += sl; tot_sys_hit += sh
    tot_msg_len += ml; tot_msg_hit += mh
    if sh == sl:                      # system fully hit -> messages spendable
        n_full += 1; spent += mh
    else:
        if sh == 0: n_miss += 1
        else:       n_partial += 1
        discarded += mh               # the whole messages hit is thrown away
        if mh and first_bad is None: first_bad = i
        if mh: bad_positions.append(i)

W = 34
print(f"{'system segment, tokens seen':<{W}} {tot_sys_len:>10,}")
print(f"{'system segment, tokens hit':<{W}} {tot_sys_hit:>10,}  ({100*tot_sys_hit/tot_sys_len:.1f}%)")
print(f"{'messages segment, tokens seen':<{W}} {tot_msg_len:>10,}")
print(f"{'messages segment, tokens hit':<{W}} {tot_msg_hit:>10,}  ({100*tot_msg_hit/tot_msg_len:.1f}%)")
print()
print(f"{'system hit IN FULL -> msg spendable':<{W}} {n_full:>10,} requests")
print(f"{'system hit PARTIALLY -> msg dropped':<{W}} {n_partial:>10,} requests")
print(f"{'system MISSED -> msg dropped':<{W}} {n_miss:>10,} requests")
print()
print(f"{'messages tokens actually spent':<{W}} {spent:>10,}")
print(f"{'messages tokens DISCARDED':<{W}} {discarded:>10,}")
if spent + discarded:
    print(f"{'discard share of matched messages':<{W}} {100*discarded/(spent+discarded):>9.1f}%")

print("\n--- when do the discards happen? ---")
if first_bad is None:
    print("  never: the system segment hit in full on every request")
else:
    n = len(rows)
    print(f"  first discarding request: #{first_bad} of {n}  ({100*first_bad/n:.0f}% into the run)")
    q = [0, 0, 0, 0]
    for p in bad_positions:
        q[min(3, p * 4 // n)] += 1
    print(f"  discarding requests by quarter of the run: "
          f"Q1={q[0]}  Q2={q[1]}  Q3={q[2]}  Q4={q[3]}")
    print("  eviction predicts these cluster late; a broken insert predicts them even.")

if usage_trail:
    print(f"\n  KV pool occupancy over the run: "
          f"min {min(usage_trail):.2f}  median {sorted(usage_trail)[len(usage_trail)//2]:.2f}  "
          f"max {max(usage_trail):.2f}")
