#!/bin/bash
# MASLab arm of the sub-context A/B.
#
#   TAG=ag_he_on  ./run_mas.sh record   run MASLab end to end and record its traffic
#   TAG=ag_he_ab  ./run_mas.sh toggle   replay a capture, split OFF then ON
#   TAG=ag_he_on  ./run_mas.sh eval     pass@1 for a record run's results
#
# Everything lands in ab_out/maslab. SWE-bench is run_swe.sh.
#
# REMOTE SERVER. Set SERVER_URL to run MASLab here against a server on another box
# (the H200): `record` then drives it over HTTP and `eval` scores the local results,
# neither of which needs the GPU. `toggle` does NOT work that way -- it restarts the
# server three times with different env vars and reads traces the server writes on its
# own disk -- so run `toggle` on the server box, where the capture already is.
#
#   SERVER_URL=http://140.118.202.100:30000 ARM=rot \
#     METHOD=agentverse MAS_CONFIG= DATASET=humaneval TAG=av_he_rot ./run_mas.sh record
#
# ARM must match how the remote server was started (`ARM=... sbatch sglang_server.sh`);
# `record` reads /server_info and refuses if it does not.
#
# METHOD / MAS_CONFIG / DATASET / TAG / REQUESTS / SUBCTX_OFF are overridable so
# run_matrix.sh can drive four combinations through this script; edit the rest here.
set -euo pipefail

# Overridable so the same script runs on the 3090 (MASLab side) and on the H200
# (where `toggle` has to run, next to the server and its traces).
REPO=${REPO:-/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9}
OUT=${OUT:-$REPO/ab_out/maslab}
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}
QUANT=${QUANT-moe_wna16}   # set empty on a box with the VRAM for bf16 weights
PORT=${PORT:-30000}
CTXLEN=${CTXLEN:-16384}
GEN_TOKENS=${GEN_TOKENS:-32}   # replay generates a fixed length so both arms do equal work
ENV=${ENV:-sglangv59}
CONDA_SH=${CONDA_SH:-/home/t2503-3090/miniconda3/etc/profile.d/conda.sh}
MASLAB=${MASLAB:-/home/t2503-3090/Desktop/MiaoChen/MASLab}
MAS_MODEL=${MAS_MODEL:-Qwen3-Coder-30B-A3B}
MAS_TEMP=${MAS_TEMP:-0.0}

METHOD=${METHOD:-autogen}
# `-` not `:-`: an explicitly empty MAS_CONFIG must stay empty. agentverse picks its
# own config from the dataset (agentverse_humaneval.py:10 defaults to
# config_humaneval), and substituting autogen's config_code there is a FileNotFoundError.
MAS_CONFIG=${MAS_CONFIG-config_code}
DATASET=${DATASET:-humaneval}
TAG=${TAG:-}
SUF=${TAG:+_$TAG}
REQUESTS=${REQUESTS:-$OUT/requests$SUF.jsonl}
INFER=${INFER:-$OUT/infer$SUF.jsonl}   # overridable so a pre-split run can still be scored

# Empty => this box runs the server too (the original single-machine setup).
SERVER_URL=${SERVER_URL:-}
if [ -n "$SERVER_URL" ]; then REMOTE=1; else REMOTE=0; SERVER_URL=http://127.0.0.1:$PORT; fi

# Which arm. Same three words as sglang_server.sh, so one word means one thing on
# both boxes: against a remote server this is what `record` asserts /server_info
# reports, and locally it is what `launch` starts. `toggle` drives all three arms
# itself and refuses to be told one. Leaving ARM unset keeps the older
# SUBCTX_OFF / SUBCTX_ROTATE spelling working.
ARM=${ARM:-}
case "$ARM" in
  "")  ;;
  on)  SUBCTX_OFF=""; SUBCTX_ROTATE=""; SUBCTX_ROTATE_ACROSS="" ;;
  off) SUBCTX_OFF="1"; SUBCTX_ROTATE=""; SUBCTX_ROTATE_ACROSS="" ;;
  rot) SUBCTX_OFF=""; SUBCTX_ROTATE="1"; SUBCTX_ROTATE_ACROSS="1" ;;
  *)   echo "REFUSING: unknown ARM='$ARM' (want on|off|rot)"; exit 1 ;;
esac

mkdir -p "$OUT"
source "$CONDA_SH"
conda activate $ENV
export PYTHONNOUSERSITE=1        # ~/.local has a broken torch dist-info ahead of the env
export PYTHONPATH=$REPO/python   # run THIS checkout, not the installed sglang

# Wait for a server on another box and refuse if it is not the arm we asked for.
remote_check() {
  echo -n "waiting for $SERVER_URL"
  for _ in $(seq 1 300); do
    if curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then echo " up"; break; fi
    echo -n .; sleep 2
  done
  if ! curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
    echo " UNREACHABLE"
    echo "  A Slurm job runs on whichever compute node was allocated, so its address"
    echo "  changes every submission -- read the URL out of the job log. If that node"
    echo "  is not routable from here, the server box has to run the client too."
    exit 1
  fi
  python "$REPO/scripts/subcontext_sim/check_remote_arm.py" "$SERVER_URL" \
    --split "$( [ -n "${SUBCTX_OFF:-}" ] && echo false || echo true )" \
    --rotate "$( [ -n "${SUBCTX_ROTATE:-}" ] && echo true || echo false )" \
    ${SUBCTX_AUDIT:+--audit true} \
    --maslab-config "$MASLAB/model_api_configs/model_api_config.json" \
    --model "$MAS_MODEL"
}

# The server command lives here, once. Callers set LOG, and optionally CAPTURE /
# TRACE / STAGE / SUBCTX_OFF / SUBCTX_TRACE / SUBCTX_ROTATE / SUBCTX_ROTATE_ACROSS
# before calling.
launch() {
  pkill -f "[s]glang\.launch_server" 2>/dev/null || true
  sleep 6
  if [ -n "${TRACE:-}" ]; then rm -f "$TRACE"; fi
  if [ -n "${STAGE:-}" ]; then rm -f "$STAGE".*; fi
  SGLANG_CAPTURE_REQUESTS=${CAPTURE:-} \
  SGLANG_FORWARD_TRACE=${TRACE:-} \
  SGLANG_STAGE_TRACE=${STAGE:-} \
  SGLANG_DISABLE_SUBCONTEXT=${SUBCTX_OFF:-} \
  SGLANG_SUBCTX_TRACE=${SUBCTX_TRACE:-} \
  SGLANG_SUBCONTEXT_ROTATE=${SUBCTX_ROTATE:-} \
  SGLANG_SUBCONTEXT_ROTATE_ACROSS=${SUBCTX_ROTATE_ACROSS:-} \
  nohup python -u -m sglang.launch_server \
    --model-path $MODEL \
    --context-length $CTXLEN \
    ${QUANT:+--quantization $QUANT} \
    --tool-call-parser qwen3_coder \
    --enable-cache-report \
    --port $PORT \
    --mem-fraction-static 0.85 \
    > "$LOG" 2>&1 &
  echo -n "  waiting"
  for _ in $(seq 1 300); do
    if grep -q "fired up and ready" "$LOG"; then echo " ready"; return 0; fi
    if ! pgrep -f "[s]glang\.launch_server" > /dev/null; then
      echo " DIED"; tail -30 "$LOG"; exit 1
    fi
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -30 "$LOG"; exit 1
}

stop() {
  pkill -TERM -f "[s]glang\.launch_server" 2>/dev/null || true   # TERM so timers flush
  sleep 8
}

# Replay the capture against whatever server is up. Callers set CLIENT (+ TRACE/STAGE).
replay() {
  python $REPO/subcontext_bench.py replay "$REQUESTS" \
    --url http://127.0.0.1:$PORT \
    --trace "${TRACE:-}" --stage-trace "${STAGE:-}" --out "$CLIENT" \
    --model $MODEL --gen-tokens $GEN_TOKENS --save-text
}

case "${1:-}" in
  record)
    : "${TAG:?set TAG so this capture does not overwrite another combination}"
    # RESUME keeps both files so MASLab's own reserve_unprocessed_queries can pick
    # up where a killed run stopped; the capture then holds both attempts.
    if [ -z "${RESUME:-}" ]; then rm -f "$INFER"; [ "$REMOTE" = 1 ] || rm -f "$REQUESTS"; fi
    if [ "$REMOTE" = 1 ]; then
      # The server is on another box, already up with its arm baked in at start. The
      # capture and traces land on ITS disk, which is also where `toggle` needs them.
      remote_check | tee "$OUT/serverinfo$SUF.txt"
    else
      LOG=$OUT/server_record$SUF.log CAPTURE=$REQUESTS launch
      if [ -n "${SUBCTX_OFF:-}" ]; then
        grep -q "Sub-context split DISABLED" $OUT/server_record$SUF.log \
          || { echo "REFUSING: SUBCTX_OFF set but the split did not report itself disabled"; exit 1; }
      fi
      # The KV pool size decides whether the A/B measures the split or the eviction
      # policy. Record what the server actually got: 30B AWQ on 24GB leaves little.
      grep -m1 -o "max_total_num_tokens=[0-9]*" $OUT/server_record$SUF.log \
        | tee $OUT/kvpool$SUF.txt || echo "WARNING: could not read KV pool size"
    fi
    # MASLab must not import sglang from the fork tree.
    ( unset PYTHONPATH; cd $MASLAB && python inference.py \
        --method_name "$METHOD" \
        ${MAS_CONFIG:+--method_config_name "$MAS_CONFIG"} \
        --test_dataset_name "$DATASET" \
        --model_name $MAS_MODEL \
        --model_temperature $MAS_TEMP \
        --output_path "$INFER" )
    # A run whose server went away mid-flight still writes a full-length results file,
    # with `None` where the calls failed -- and `evaluate.py` then reports an accuracy
    # over only the rows it could score, which looks like a clean number. Refuse to call
    # that a recording.
    #
    # But an empty response is not automatically that. AgentVerse's `parse_solver`
    # does `re.findall(r"```.*?\n(.+?)```", out)[-1]`, so a reply that runs out of
    # tokens mid-fence raises IndexError -- the model answered, the method could not
    # use the answer. That is a real result (a failure) and every arm has some: off 4,
    # on 1, rot 2 on agentverse/humaneval. Refusing on those would make a clean run
    # unrecordable, and dropping them would score each arm on its own denominator.
    python - "$INFER" <<'EOF' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
gone, unparsed = [], []
for r in rows:
    if r.get("response"):
        continue
    err = r.get("error") or ""
    calls = sum(v.get("num_llm_calls", 0) for v in (r.get("token_stats") or {}).values())
    # Classify on the exception TYPE, not on words anywhere in the traceback: matching
    # bare status codes read "503" out of a source line number and refused a clean run.
    tail = [ln for ln in err.strip().splitlines() if ln and not ln[0].isspace()]
    kind = tail[-1].split(":", 1)[0].lower() if tail else ""
    transport = not err or calls == 0 or any(
        w in kind
        for w in ("connection", "timeout", "apierror", "apistatus", "internalserver",
                  "serviceunavailable", "badgateway", "remotedisconnected",
                  "protocolerror", "oserror", "httpx")
    )
    (gone if transport else unparsed).append(r.get("task_id"))
if gone:
    print(f"REFUSING: {len(gone)}/{len(rows)} results never reached the model -- the "
          f"server went away before this run finished (first: {gone[0]}, last: "
          f"{gone[-1]}). Any pass@1 over the rest would be scored on a subset. Re-run.",
          file=sys.stderr)
    raise SystemExit(1)
if unparsed:
    print(f"  {len(rows)} results, {len(unparsed)} the method could not parse "
          f"({', '.join(unparsed[:5])}) -- scored as failures, not dropped")
else:
    print(f"  all {len(rows)} results present")
EOF
    if [ "$REMOTE" = 1 ]; then
      echo "inference results -> $INFER"
      echo "the capture and traces are on the server box, under its traces/ dir;"
      echo "run 'toggle' there -- it needs them and it restarts the server per arm."
    else
      stop
      echo "captured $(wc -l < "$REQUESTS" 2>/dev/null || echo 0) requests -> $REQUESTS"
      echo "inference results -> $INFER"
    fi
    ;;

  toggle)
    # Three arms means three server restarts, and the traces it compares are written
    # on the server's own disk. Neither is reachable from another box.
    [ -z "$ARM" ] || { echo "REFUSING: toggle runs all three arms; do not set ARM"; exit 1; }
    [ "$REMOTE" = 1 ] && {
      echo "REFUSING: 'toggle' restarts the server per arm and reads traces it writes"
      echo "locally, so it must run ON the server box. The capture is already there;"
      echo "unset SERVER_URL and run this script from the checkout on that machine."
      exit 1
    }
    [ -s "$REQUESTS" ] || { echo "no capture at $REQUESTS; run '$0 record' first"; exit 1; }
    echo "captured $(wc -l < "$REQUESTS") requests"

    echo; echo "======== SPLIT OFF ========"
    SUBCTX_OFF=1 LOG=$OUT/server_off$SUF.log \
      TRACE=$OUT/trace_base$SUF.jsonl STAGE=$OUT/stage_base$SUF launch
    # An arm meant to be the baseline that silently ran with the split on is worse
    # than no arm at all: it looks like a valid comparison.
    grep -q "Sub-context split DISABLED" $OUT/server_off$SUF.log \
      || { echo "REFUSING: the split did not report itself disabled"; exit 1; }
    TRACE=$OUT/trace_base$SUF.jsonl STAGE=$OUT/stage_base$SUF \
      CLIENT=$OUT/client_base$SUF.json replay

    echo; echo "======== SPLIT ON ========"
    LOG=$OUT/server_on$SUF.log \
      TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF launch
    TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF \
      CLIENT=$OUT/client_sub$SUF.json replay

    # Third arm: the split plus position compensation. Only run when asked, so the
    # two-arm comparison stays exactly what it was before rotation existed.
    if [ -n "${ROTATE:-}" ]; then
      echo; echo "======== SPLIT ON + ROTATE ========"
      SUBCTX_ROTATE=1 SUBCTX_ROTATE_ACROSS=${ACROSS:-} LOG=$OUT/server_rot$SUF.log \
        TRACE=$OUT/trace_rot$SUF.jsonl STAGE=$OUT/stage_rot$SUF launch
      # Same reasoning as the baseline guard: an arm that silently ran without the
      # rotation it is named for is worse than no arm at all.
      grep -q "Sub-context KV rotation ENABLED" $OUT/server_rot$SUF.log \
        || { echo "REFUSING: rotation did not report itself enabled"; exit 1; }
      TRACE=$OUT/trace_rot$SUF.jsonl STAGE=$OUT/stage_rot$SUF \
        CLIENT=$OUT/client_rot$SUF.json replay
    fi

    stop

    echo; echo "======== GPU (CUDA events) ========"
    python $REPO/subcontext_bench.py report $OUT/trace_base$SUF.jsonl $OUT/trace_sub$SUF.jsonl
    echo; echo "======== HOST (CPU stages) ========"
    python $REPO/subcontext_bench.py stages $OUT/stage_base$SUF $OUT/stage_sub$SUF
    echo; echo "======== PARITY (generated text) ========"
    python $REPO/subcontext_bench.py parity \
      $OUT/client_base$SUF.json $OUT/client_sub$SUF.json || true

    if [ -n "${ROTATE:-}" ]; then
      echo; echo "======== ROTATE vs SPLIT (GPU) ========"
      python $REPO/subcontext_bench.py report $OUT/trace_sub$SUF.jsonl $OUT/trace_rot$SUF.jsonl
      echo; echo "======== ROTATE vs BASELINE (GPU) ========"
      python $REPO/subcontext_bench.py report $OUT/trace_base$SUF.jsonl $OUT/trace_rot$SUF.jsonl
      echo; echo "======== ROTATE host cost ========"
      python $REPO/subcontext_bench.py stages $OUT/stage_sub$SUF $OUT/stage_rot$SUF
      # Rotation fixes the position a block is reused at, not the context it was
      # computed under, so divergence here is expected and is the number to read,
      # not a failure. pass@1 from a `record` run is what prices it.
      echo; echo "======== PARITY: rotate vs split (divergence EXPECTED) ========"
      python $REPO/subcontext_bench.py parity \
        $OUT/client_sub$SUF.json $OUT/client_rot$SUF.json || true
    fi
    ;;

  eval)
    # pass@1 for a record run's results. Kept separate from toggle on purpose: the
    # replay generates a fixed length with ignore_eos, so its output is not a real
    # attempt at the benchmark. Quality comes from the record run, which generated
    # freely.
    : "${TAG:?set TAG to the run you want scored}"
    [ -s "$INFER" ] || { echo "no results at $INFER; run '$0 record' first"; exit 1; }
    ( unset PYTHONPATH; cd $MASLAB && python evaluate.py \
        --eval_protocol code \
        --model_name $MAS_MODEL \
        --tested_dataset_name "$DATASET" \
        --tested_infer_path "$INFER" \
        --overwrite )
    # `evaluate.py` leaves eval_score None where the method produced nothing and reports
    # accuracy over the rest, so each arm gets its own denominator -- on agentverse it
    # scored off 152/160 = 95.00% and on 154/163 = 94.48%, which reverses over the whole
    # set. A row the method could not answer is a failure of that arm's run, so score
    # every row.
    python - "$OUT/xverify_eval$SUF.jsonl" <<'EOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(f"  (no {p} to re-score)"); raise SystemExit(0)
rows = [json.loads(l) for l in open(p)]
ok = sum(1 for r in rows if r.get("eval_score") == 1)
unscored = sum(1 for r in rows if r.get("eval_score") is None)
print(f"  pass@1 over ALL {len(rows)}: {ok}/{len(rows)} = {ok / len(rows):.2%}"
      f"   ({unscored} unanswered, counted as failures)")
EOF
    ;;

  *)
    echo "usage: METHOD=.. DATASET=.. TAG=.. $0 {record|toggle|eval}"
    echo "  record  METHOD=autogen MAS_CONFIG=config_code DATASET=humaneval TAG=ag_he_on $0 record"
    echo "  toggle  REQUESTS=$OUT/requests_ag_he_on.jsonl TAG=ag_he_ab $0 toggle"
    echo "  eval    TAG=ag_he_on DATASET=humaneval $0 eval"
    echo "  add ARM=off|rot to run any of them on another arm (default: on)"
    exit 1;;
esac
