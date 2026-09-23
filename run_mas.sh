#!/bin/bash
# MASLab arm of the sub-context A/B. SWE-bench is run_swe.sh.
#
#   ARM=on TAG=ag_he  ./run_mas.sh record   run MASLab end to end and record its traffic
#   TAG=ag_he_ab      ./run_mas.sh toggle   replay a capture, split OFF then ON
#   ARM=on TAG=ag_he  ./run_mas.sh eval     pass@1 for a record run's results
#
# Arms, same words as sglang_server.sh: off (no split), on (split), rot (+ rotation
# and Stage 2). `toggle` runs off and on (and rot with ROTATE=1) itself; do not set ARM.
#
# An arm's results land in ab_out/maslab/<arm>/ named ..._<tag>_<arm>, both halves from
# $ARM. Everything printed is also appended to
# $OUT/<mode><suffix>.txt; CONSOLE=path moves it, CONSOLE= turns it off.
#
# Re-print the summary without re-running, and choose the columns:
#
#   python subcontext_bench.py summary ab_out/maslab --suffix _ag_he_ab --arms base,rot
#
# CONC=8 replays with 8 requests in flight, for serving capacity rather than a
# per-request number.
#
# REMOTE SERVER. SERVER_URL runs MASLab here against a server on another box (the
# H200); `record` checks /server_info against ARM. `toggle` must run on the server box.
#
#   SERVER_URL=http://140.118.202.100:30000 ARM=rot \
#     METHOD=agentverse MAS_CONFIG= DATASET=humaneval TAG=av_he ./run_mas.sh record
set -euo pipefail

# Overridable so the same script runs on the 3090 and on the H200.
REPO=${REPO:-/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9}
OUT=${OUT:-$REPO/ab_out/maslab}
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}
QUANT=${QUANT-moe_wna16}   # set empty on a box with the VRAM for bf16 weights
PORT=${PORT:-30000}
CTXLEN=${CTXLEN:-16384}
# 0.85 leaves a 30B AWQ workable on 24GB; an H200 can take 0.9.
MEMFRAC=${MEMFRAC:-0.85}
GEN_TOKENS=${GEN_TOKENS:-32}   # fixed generated length per replayed request
# 1 request in flight; the throughput row is then 1/latency by construction.
CONC=${CONC:-1}
ENV=${ENV:-sglangv59}
# Ask conda itself, with the 3090 layout as the fallback.
if [ -z "${CONDA_SH:-}" ]; then
  _base=${CONDA_EXE:-}; _base=${_base%/bin/conda}
  [ -n "$_base" ] || _base=$(conda info --base 2>/dev/null || true)
  if [ -n "$_base" ] && [ -r "$_base/etc/profile.d/conda.sh" ]; then
    CONDA_SH=$_base/etc/profile.d/conda.sh
  else
    CONDA_SH=/home/t2503-3090/miniconda3/etc/profile.d/conda.sh
  fi
fi
MASLAB=${MASLAB:-/home/t2503-3090/Desktop/MiaoChen/MASLab}
MAS_MODEL=${MAS_MODEL:-Qwen3-Coder-30B-A3B}
MAS_TEMP=${MAS_TEMP:-0.0}

METHOD=${METHOD:-autogen}
# `-` not `:-`: an explicitly empty MAS_CONFIG stays empty. agentverse picks its own
# config from the dataset.
MAS_CONFIG=${MAS_CONFIG-config_code}
DATASET=${DATASET:-humaneval}
TAG=${TAG:-}

# Empty: this box runs the server too.
SERVER_URL=${SERVER_URL:-}
if [ -n "$SERVER_URL" ]; then REMOTE=1; else REMOTE=0; SERVER_URL=http://127.0.0.1:$PORT; fi

# Leaving ARM unset keeps the older SUBCTX_OFF / SUBCTX_ROTATE spelling working.
ARM=${ARM:-}
case "$ARM" in
  "")  ;;
  on)  SUBCTX_OFF=""; SUBCTX_ROTATE=""; SUBCTX_ROTATE_ACROSS="" ;;
  off) SUBCTX_OFF="1"; SUBCTX_ROTATE=""; SUBCTX_ROTATE_ACROSS="" ;;
  rot) SUBCTX_OFF=""; SUBCTX_ROTATE="1"; SUBCTX_ROTATE_ACROSS="1" ;;
  *)   echo "REFUSING: unknown ARM='$ARM' (want on|off|rot)"; exit 1 ;;
esac

# Both the directory and the suffix come from $ARM, not from TAG.
[ -z "$ARM" ] || OUT=$OUT/$ARM
SUF=${TAG:+_$TAG}${ARM:+_$ARM}
REQUESTS=${REQUESTS:-$OUT/requests$SUF.jsonl}
INFER=${INFER:-$OUT/infer$SUF.jsonl}   # override to score a run recorded elsewhere

mkdir -p "$OUT"
[ -r "$CONDA_SH" ] || {
  echo "no conda.sh at $CONDA_SH"
  echo "set CONDA_SH=<conda base>/etc/profile.d/conda.sh   (conda info --base)"
  exit 1
}
source "$CONDA_SH"
conda activate $ENV 2>/dev/null || {
  echo "no conda env '$ENV' under $CONDA_SH; set ENV=<name>. available:"
  conda env list
  exit 1
}
export PYTHONNOUSERSITE=1        # ignore ~/.local
export PYTHONUNBUFFERED=1
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

# Every process the server spawns: setproctitle renames the scheduler to
# `sglang::scheduler`, which holds the weights and the KV pool. `[s]` stops the
# pattern matching whatever shell carries it.
SGLANG_PROCS='[s]glang::|[s]glang\.launch_server|[s]glang\.bench|[s]glang\.srt'

# TERM first and then wait: the host timers dump on SIGTERM.
kill_servers() {
  pkill -TERM -f "$SGLANG_PROCS" 2>/dev/null || true
  for _ in $(seq 1 40); do
    pgrep -f "$SGLANG_PROCS" > /dev/null 2>&1 || return 0
    sleep 1
  done
  echo "  WARNING: sglang processes outlived SIGTERM; escalating to KILL."
  echo "  Their stage counters stop at whatever was last flushed."
  pkill -KILL -f "$SGLANG_PROCS" 2>/dev/null || true
  sleep 5
}

# The UUID of the card this run gets, read from CUDA_VISIBLE_DEVICES, which may hold
# an index or a UUID.
target_gpu_uuid() {
  local first
  first=${CUDA_VISIBLE_DEVICES:-0}
  first=${first%%,*}
  case "$first" in
    GPU-*|MIG-*) echo "$first"; return 0 ;;
    ""|*[!0-9]*) return 0 ;;
  esac
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v i="$first" '$1 == i {print $2; exit}'
}

require_free_vram() {
  command -v nvidia-smi > /dev/null 2>&1 || return 0
  local uuid row total free need
  # `|| true` as in report_gpu: failing to probe means say nothing.
  uuid=$(target_gpu_uuid) || true
  [ -n "$uuid" ] || return 0
  row=$(nvidia-smi --query-gpu=uuid,memory.total,memory.free \
          --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v u="$uuid" '$1 == u {print $2, $3; exit}') || true
  [ -n "$row" ] || return 0
  total=${row%% *}; free=${row##* }
  case "$total$free" in *[!0-9]*|"") return 0 ;; esac   # not a number: say nothing
  need=$(awk -v t="$total" -v f="$MEMFRAC" 'BEGIN{printf "%d", t * f}')
  [ "$free" -ge "$need" ] && return 0
  echo "REFUSING: the GPU this job was given (${uuid}) has ${free} MiB free, but"
  echo "  --mem-fraction-static $MEMFRAC wants ${need} MiB of its ${total} MiB."
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Nothing scheduled this run onto a card, so it defaulted to index 0.
    echo "  CUDA_VISIBLE_DEVICES is unset, so this fell back to the node's GPU 0."
    echo "  If you meant to run inside a Slurm allocation, you are not in one"
    echo "  (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}); get one, or name a free card"
    echo "  yourself with CUDA_VISIBLE_DEVICES=<n>."
  fi
  echo "  Holding it now:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
      2>/dev/null \
    | awk -F', *' -v u="$uuid" '$1 == u {print $2", "$3}' \
    | while IFS= read -r row; do
        pid=$(echo "${row%%,*}" | tr -d ' ')
        echo "    pid ${row}   owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" \
             "cmd=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')"
      done
  echo "  If they are yours:      bash $REPO/scripts/killall_sglang.sh   (or scancel"
  echo "                          the job holding them -- a Slurm job that ended"
  echo "                          badly leaves its scheduler behind)"
  echo "  If they are not:        this card is shared -- the timings would be a"
  echo "                          property of both jobs. Get a node to yourself, or"
  echo "                          lower MEMFRAC and accept that only the reuse"
  echo "                          numbers survive, not the latency ones."
  exit 1
}

# Say which physical card the server got. With no Slurm allocation and no
# CUDA_VISIBLE_DEVICES, sglang falls back to device 0 of the node.
report_gpu() {
  # Every value below is a pipeline in a command substitution; under `set -e -o
  # pipefail` a non-zero status would end the script before the `[ -n ... ]` guards run.
  command -v nvidia-smi > /dev/null 2>&1 || { echo "  (no nvidia-smi; GPU not reported)"; return 0; }
  local pid cvd uuid index
  pid=$(pgrep -f '[s]glang::scheduler' | head -1) || true
  [ -n "$pid" ] || { echo "  (no sglang::scheduler process; GPU not reported)"; return 0; }
  cvd=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | sed -n 's/^CUDA_VISIBLE_DEVICES=//p') || true
  uuid=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
         | awk -F', *' -v p="$pid" '$1 == p {print $2; exit}') || true
  index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', *' -v u="$uuid" '$2 == u {print $1; exit}') || true
  echo "  scheduler pid $pid on physical GPU ${index:-?} (${uuid:-unknown})," \
       "CUDA_VISIBLE_DEVICES=${cvd:-<unset>}"
  [ -n "$cvd" ] || echo "  NOTE: unset, so this fell back to the node's device 0."
  return 0
}

# Callers set LOG, and optionally CAPTURE / TRACE / STAGE / SUBCTX_OFF / SUBCTX_TRACE /
# SUBCTX_ROTATE / SUBCTX_ROTATE_ACROSS / ROTATE_GPU (CUDA events per rotation).
launch() {
  kill_servers
  require_free_vram
  # on_exit stops the server only if this run started one.
  LAUNCHED=1
  if [ -n "${TRACE:-}" ]; then rm -f "$TRACE"; fi
  if [ -n "${STAGE:-}" ]; then rm -f "$STAGE".*; fi
  SGLANG_CAPTURE_REQUESTS=${CAPTURE:-} \
  SGLANG_FORWARD_TRACE=${TRACE:-} \
  SGLANG_STAGE_TRACE=${STAGE:-} \
  SGLANG_DISABLE_SUBCONTEXT=${SUBCTX_OFF:-} \
  SGLANG_SUBCTX_TRACE=${SUBCTX_TRACE:-} \
  SGLANG_SUBCONTEXT_ROTATE=${SUBCTX_ROTATE:-} \
  SGLANG_SUBCTX_ROTATE_GPU=${ROTATE_GPU:-} \
  SGLANG_SUBCONTEXT_ROTATE_ACROSS=${SUBCTX_ROTATE_ACROSS:-} \
  nohup python -u -m sglang.launch_server \
    --model-path $MODEL \
    --context-length $CTXLEN \
    ${QUANT:+--quantization $QUANT} \
    --tool-call-parser qwen3_coder \
    --enable-cache-report \
    --port $PORT \
    --mem-fraction-static $MEMFRAC \
    > "$LOG" 2>&1 &
  echo -n "  waiting"
  for _ in $(seq 1 300); do
    if grep -q "fired up and ready" "$LOG"; then echo " ready"; report_gpu; return 0; fi
    if ! pgrep -f "[s]glang\.launch_server" > /dev/null; then
      echo " DIED"; tail -30 "$LOG"; exit 1
    fi
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -30 "$LOG"; exit 1
}

stop() {
  kill_servers
}

# Replay the capture against whatever server is up. Callers set CLIENT (+ TRACE/STAGE).
replay() {
  python $REPO/subcontext_bench.py replay "$REQUESTS" \
    --url http://127.0.0.1:$PORT \
    --trace "${TRACE:-}" --stage-trace "${STAGE:-}" --out "$CLIENT" \
    --model $MODEL --gen-tokens $GEN_TOKENS --concurrency $CONC --save-text
}

# The trap is installed unconditionally.
LAUNCHED=0
TEE_PID=
on_exit() {
  if [ "$LAUNCHED" = 1 ]; then
    echo; echo "cleaning up the server this run started"
    kill_servers
  fi
  # Closing stdout lets `tee` see EOF and flush.
  exec 1>&- 2>&- || true
  [ -n "$TEE_PID" ] && wait "$TEE_PID" 2>/dev/null
  return 0
}
trap on_exit EXIT

# The console output is the result. Appended, not truncated; the header says which
# run is which.
CONSOLE=${CONSOLE-$OUT/${1:-run}$SUF.txt}
if [ -n "$CONSOLE" ]; then
  {
    echo "### $(date -Is)  $0 ${*:-}"
    echo "### ARM=${ARM:-} TAG=${TAG:-} ROTATE=${ROTATE:-} ACROSS=${ACROSS:-}" \
         "CONC=$CONC MEMFRAC=$MEMFRAC GEN_TOKENS=$GEN_TOKENS MODEL=$MODEL"
  } >> "$CONSOLE"
  exec > >(tee -a "$CONSOLE") 2>&1
  # `tee` outlives the shell's last write; on_exit waits for it.
  TEE_PID=$!
  echo "console -> $CONSOLE"
fi

case "${1:-}" in
  record)
    : "${TAG:?set TAG so this capture does not overwrite another combination}"
    # RESUME keeps both files for MASLab's reserve_unprocessed_queries; the capture
    # then holds both attempts.
    if [ -z "${RESUME:-}" ]; then rm -f "$INFER"; [ "$REMOTE" = 1 ] || rm -f "$REQUESTS"; fi
    if [ "$REMOTE" = 1 ]; then
      # The capture and traces land on the server's disk, where `toggle` needs them.
      remote_check | tee "$OUT/serverinfo$SUF.txt"
    else
      LOG=$OUT/server_record$SUF.log CAPTURE=$REQUESTS launch
      if [ -n "${SUBCTX_OFF:-}" ]; then
        grep -q "Sub-context split DISABLED" $OUT/server_record$SUF.log \
          || { echo "REFUSING: SUBCTX_OFF set but the split did not report itself disabled"; exit 1; }
      fi
      # MEMFRAC sets the KV pool size.
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
    # Refuse a run in which requests never reached the server; answers the method could
    # not parse are kept and scored as failures.
    python - "$INFER" <<'EOF' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
gone, unparsed = [], []
for r in rows:
    if r.get("response"):
        continue
    err = r.get("error") or ""
    calls = sum(v.get("num_llm_calls", 0) for v in (r.get("token_stats") or {}).values())
    # Classify on the exception type, not on words anywhere in the traceback.
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
    [ -z "$ARM" ] || { echo "REFUSING: toggle runs every arm itself; do not set ARM"; exit 1; }
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
    # Ask the server what it is rather than trusting the switches.
    grep -q "Sub-context split DISABLED" $OUT/server_off$SUF.log \
      || { echo "REFUSING: the split did not report itself disabled"; exit 1; }
    TRACE=$OUT/trace_base$SUF.jsonl STAGE=$OUT/stage_base$SUF \
      CLIENT=$OUT/client_base$SUF.json replay

    echo; echo "======== SPLIT ON ========"
    LOG=$OUT/server_on$SUF.log \
      TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF launch
    TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF \
      CLIENT=$OUT/client_sub$SUF.json replay

    # Only when asked.
    if [ -n "${ROTATE:-}" ]; then
      echo; echo "======== SPLIT ON + ROTATE ========"
      SUBCTX_ROTATE=1 SUBCTX_ROTATE_ACROSS=${ACROSS:-} LOG=$OUT/server_rot$SUF.log \
        TRACE=$OUT/trace_rot$SUF.jsonl STAGE=$OUT/stage_rot$SUF launch
      grep -q "Sub-context KV rotation ENABLED" $OUT/server_rot$SUF.log \
        || { echo "REFUSING: rotation did not report itself enabled"; exit 1; }
      # ACROSS: whether prefill chunks are cut at block edges (Stage 2).
      want_across=$([ -n "${ACROSS:-}" ] && echo True || echo False)
      grep -q "across-recompute=$want_across" $OUT/server_rot$SUF.log \
        || { echo "REFUSING: asked for ACROSS=${ACROSS:-<unset>} but the server reported"; \
             grep -o "across-recompute=[A-Za-z]*" $OUT/server_rot$SUF.log | head -1; exit 1; }
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

    # With ROTATE the summary shows base,rot; `--arms base,sub,rot` adds the split.
    echo; echo "======== SUMMARY ========"
    python $REPO/subcontext_bench.py summary "$OUT" --suffix "$SUF" \
      ${ROTATE:+--arms base,rot}

    if [ -n "${ROTATE:-}" ]; then
      echo; echo "======== ROTATE vs SPLIT (GPU) ========"
      python $REPO/subcontext_bench.py report $OUT/trace_sub$SUF.jsonl $OUT/trace_rot$SUF.jsonl
      echo; echo "======== ROTATE vs BASELINE (GPU) ========"
      python $REPO/subcontext_bench.py report $OUT/trace_base$SUF.jsonl $OUT/trace_rot$SUF.jsonl
      echo; echo "======== ROTATE host cost ========"
      python $REPO/subcontext_bench.py stages $OUT/stage_sub$SUF $OUT/stage_rot$SUF
      echo; echo "======== PARITY: rotate vs split (divergence EXPECTED) ========"
      python $REPO/subcontext_bench.py parity \
        $OUT/client_sub$SUF.json $OUT/client_rot$SUF.json || true
    fi
    ;;

  eval)
    # Scores a record run (the replay pins the length with ignore_eos).
    : "${TAG:?set TAG to the run you want scored}"
    [ -s "$INFER" ] || { echo "no results at $INFER; run '$0 record' first"; exit 1; }
    echo ">> scoring $INFER${ARM:+ (arm $ARM)}"
    ( unset PYTHONPATH; cd $MASLAB && python evaluate.py \
        --eval_protocol code \
        --model_name $MAS_MODEL \
        --tested_dataset_name "$DATASET" \
        --tested_infer_path "$INFER" \
        --overwrite )
    # Re-score over every row, counting eval_score None as a failure.
    python - "$OUT/xverify_eval$SUF.jsonl" <<'EOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(f"  (no {p} to re-score)"); raise SystemExit(0)
rows = [json.loads(l) for l in open(p)]
ok = sum(1 for r in rows if r.get("eval_score") == 1)
unscored = sum(1 for r in rows if r.get("eval_score") is None)
print(f"  pass@1 over ALL {len(rows)}: {ok}/{len(rows)} = {ok / max(len(rows), 1):.2%}"
      f"   ({unscored} unanswered, counted as failures)")
EOF
    ;;

  *)
    echo "usage: ARM=on|off|rot METHOD=.. DATASET=.. TAG=.. $0 {record|toggle|eval}"
    echo "  record  ARM=on METHOD=autogen MAS_CONFIG=config_code DATASET=humaneval TAG=ag_he $0 record"
    echo "  eval    ARM=on TAG=ag_he DATASET=humaneval $0 eval"
    echo "  toggle  REQUESTS=<capture> TAG=ag_he_ab $0 toggle   (no ARM; ROTATE=1 adds rot)"
    echo "  CONC=N replays with N requests in flight; the summary's throughput row"
    echo "  is 1/latency at the default CONC=1"
    echo "  an arm's files live in ab_out/maslab/<arm>/ and are suffixed _<tag>_<arm>"
    exit 1;;
esac
