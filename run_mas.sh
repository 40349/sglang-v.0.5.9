#!/bin/bash
# MASLab arm of the sub-context A/B. SWE-bench is run_swe.sh.
#
#   ARM=on TAG=ag_he  ./run_mas.sh record   run MASLab end to end and record its traffic
#   TAG=ag_he_ab      ./run_mas.sh toggle   replay a capture, split OFF then ON
#   ARM=on TAG=ag_he  ./run_mas.sh eval     pass@1 for a record run's results
#
# Arms, same words as sglang_server.sh: off (no split), on (split), rot (+ rotation
# and Stage 2). `toggle` drives all three itself and refuses to be told one.
#
# An arm's results land in ab_out/maslab/<arm>/ named ..._<tag>_<arm>, both halves from
# $ARM rather than from anything typed. Everything printed is also appended to
# $OUT/<mode><suffix>.txt; CONSOLE=path moves it, CONSOLE= turns it off.
#
# Re-print the summary without re-running, and choose the columns:
#
#   python subcontext_bench.py summary ab_out/maslab --suffix _ag_he_ab --arms base,rot
#
# CONC=8 replays with 8 requests in flight, for serving capacity rather than a
# per-request number; expect the rotation arm to stop reproducing exactly when you do.
#
# REMOTE SERVER. SERVER_URL runs MASLab here against a server on another box (the
# H200); ARM must match how that server was started, and `record` reads /server_info
# and refuses if it does not. `toggle` does NOT work that way -- it restarts the server
# per arm and reads traces written on its own disk -- so run it on the server box.
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
# Decides whether an A/B measures the split or the eviction policy, and is a property
# of the box: 0.85 leaves a 30B AWQ workable on 24GB, an H200 can take 0.9.
MEMFRAC=${MEMFRAC:-0.85}
GEN_TOKENS=${GEN_TOKENS:-32}   # replay generates a fixed length so both arms do equal work
# 1 keeps every arm reproducible and makes the throughput row 1/latency by construction.
CONC=${CONC:-1}
ENV=${ENV:-sglangv59}
# Not one hardcoded path: on the H200 conda arrives through `ml load miniconda3` and
# sits elsewhere. Ask conda itself, with the 3090 layout as the fallback.
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
# `-` not `:-`: an explicitly empty MAS_CONFIG must stay empty. agentverse picks its own
# config from the dataset, and substituting autogen's config_code is a FileNotFoundError.
MAS_CONFIG=${MAS_CONFIG-config_code}
DATASET=${DATASET:-humaneval}
TAG=${TAG:-}

# Empty => this box runs the server too (the original single-machine setup).
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

# Both the directory and the suffix come from $ARM, so nothing typed can put an arm's
# results anywhere but its own directory: a stale TAG=av_he_rot on an ARM=on run once
# wrote the on arm over the rot results and left no evidence but an mtime.
[ -z "$ARM" ] || OUT=$OUT/$ARM
SUF=${TAG:+_$TAG}${ARM:+_$ARM}
REQUESTS=${REQUESTS:-$OUT/requests$SUF.jsonl}
INFER=${INFER:-$OUT/infer$SUF.jsonl}   # overridable so a pre-split run can still be scored

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
export PYTHONNOUSERSITE=1        # ~/.local has a broken torch dist-info ahead of the env
# Teeing made stdout a pipe, and Python block-buffers on a pipe: without this the
# replay's per-request line goes silent for the whole arm.
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

# Every process the server spawns, not just the one exec'd: setproctitle renames the
# scheduler to `sglang::scheduler`, so it no longer matches "sglang.launch_server" --
# and it is the one holding the weights AND the KV pool. `[s]` stops the pattern
# matching whatever shell carries it.
SGLANG_PROCS='[s]glang::|[s]glang\.launch_server|[s]glang\.bench|[s]glang\.srt'

# TERM first and then wait: the host timers dump on SIGTERM, and killing the scheduler
# outright throws away the stage trace this run is measured with.
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

# The UUID of the card this run will actually get, not the node's GPU 0: Slurm hands out
# devices through CUDA_VISIBLE_DEVICES, and checking the wrong card both refuses good
# runs and waves through doomed ones. That variable may hold an index or a UUID.
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
  # `|| true` as in report_gpu: a guard on the empty value is unreachable if a non-zero
  # status aborts the assignment first. Failing to probe means "say nothing".
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
    # The likeliest cause: nothing scheduled this run onto a card, so it defaulted to
    # index 0 -- which on a shared box is where everyone else defaulted too.
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

# Say which physical card the server actually got. With no Slurm allocation and no
# CUDA_VISIBLE_DEVICES, sglang falls back to device 0 of the NODE -- on a shared box,
# the card everyone else without an allocation defaulted onto too.
report_gpu() {
  # Every value below is a pipeline inside a command substitution, so under `set -e -o
  # pipefail` a non-zero status ends the SCRIPT silently and the `[ -n ... ]` guards
  # never run. A diagnostic must not be able to kill the run it is describing.
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

# Callers set LOG, and optionally CAPTURE / TRACE / STAGE / SUBCTX_* / ROTATE_GPU.
# ROTATE_GPU=1 fills the summary's [GPU] row at the cost of an event pair per rotation,
# so take host overhead from a run without it.
launch() {
  kill_servers
  require_free_vram
  # `nohup ... &` outlives this shell, Ctrl-C and the salloc that owns the GPU included:
  # an abandoned toggle once left a scheduler holding 129 GB on a reclaimed card. Armed
  # here so only a run that started a server tears one down -- a REMOTE record must not.
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

# The trap is installed unconditionally: an interrupted run is the case that leaks.
LAUNCHED=0
TEE_PID=
on_exit() {
  if [ "$LAUNCHED" = 1 ]; then
    echo; echo "cleaning up the server this run started"
    kill_servers
  fi
  # Last, and only then: closing stdout is what lets `tee` see EOF and flush.
  exec 1>&- 2>&- || true
  [ -n "$TEE_PID" ] && wait "$TEE_PID" 2>/dev/null
  return 0
}
trap on_exit EXIT

# A `toggle` costs an hour and its result IS the console output. Appended, not
# truncated: a re-run on the same tag wants both, and the header says which is which.
CONSOLE=${CONSOLE-$OUT/${1:-run}$SUF.txt}
if [ -n "$CONSOLE" ]; then
  {
    echo "### $(date -Is)  $0 ${*:-}"
    echo "### ARM=${ARM:-} TAG=${TAG:-} ROTATE=${ROTATE:-} ACROSS=${ACROSS:-}" \
         "CONC=$CONC MEMFRAC=$MEMFRAC GEN_TOKENS=$GEN_TOKENS MODEL=$MODEL"
  } >> "$CONSOLE"
  exec > >(tee -a "$CONSOLE") 2>&1
  # `tee` outlives the shell's last write, so on_exit waits for it or the summary
  # tables are missing from the file.
  TEE_PID=$!
  echo "console -> $CONSOLE"
fi

case "${1:-}" in
  record)
    : "${TAG:?set TAG so this capture does not overwrite another combination}"
    # RESUME keeps both files so MASLab's reserve_unprocessed_queries can pick up where
    # a killed run stopped; the capture then holds both attempts.
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
      # The KV pool size decides whether the A/B measures the split or eviction.
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
    # A run whose server went away mid-flight still writes a full-length results file
    # with `None` where the calls failed, and evaluate.py then reports an accuracy over
    # only the rows it could score. Refuse to call that a recording.
    #
    # But an unparseable answer is not that: AgentVerse's parse_solver raises IndexError
    # on a reply that ran out of tokens mid-fence. The model answered and the method
    # could not use it -- a real failure every arm has some of, so keep it and score it.
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
    # A baseline that silently ran with the split on still looks like a valid comparison.
    grep -q "Sub-context split DISABLED" $OUT/server_off$SUF.log \
      || { echo "REFUSING: the split did not report itself disabled"; exit 1; }
    TRACE=$OUT/trace_base$SUF.jsonl STAGE=$OUT/stage_base$SUF \
      CLIENT=$OUT/client_base$SUF.json replay

    echo; echo "======== SPLIT ON ========"
    LOG=$OUT/server_on$SUF.log \
      TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF launch
    TRACE=$OUT/trace_sub$SUF.jsonl STAGE=$OUT/stage_sub$SUF \
      CLIENT=$OUT/client_sub$SUF.json replay

    # Only when asked, so the two-arm comparison stays what it was before rotation.
    if [ -n "${ROTATE:-}" ]; then
      echo; echo "======== SPLIT ON + ROTATE ========"
      SUBCTX_ROTATE=1 SUBCTX_ROTATE_ACROSS=${ACROSS:-} LOG=$OUT/server_rot$SUF.log \
        TRACE=$OUT/trace_rot$SUF.jsonl STAGE=$OUT/stage_rot$SUF launch
      grep -q "Sub-context KV rotation ENABLED" $OUT/server_rot$SUF.log \
        || { echo "REFUSING: rotation did not report itself enabled"; exit 1; }
      # ACROSS decides whether prefill is cut at block edges, worth ~4.5 pp of hit rate
      # and ~10% of prefill GPU time, so a run with no record of it cannot be read.
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

    # With rotation in the run the headline is baseline vs rotation; the plain split is
    # the control that separates the plumbing's cost from the rotation's benefit, and it
    # stays in the sections above. `--arms base,sub,rot` puts its column back.
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
      # Rotation fixes the position a block is reused at, not the context it was computed
      # under, so divergence is expected and is the number to read. pass@1 prices it.
      echo; echo "======== PARITY: rotate vs split (divergence EXPECTED) ========"
      python $REPO/subcontext_bench.py parity \
        $OUT/client_sub$SUF.json $OUT/client_rot$SUF.json || true
    fi
    ;;

  eval)
    # Separate from toggle on purpose: the replay pins the length with ignore_eos, so
    # its output is not a real attempt. Quality comes from the record run.
    : "${TAG:?set TAG to the run you want scored}"
    [ -s "$INFER" ] || { echo "no results at $INFER; run '$0 record' first"; exit 1; }
    echo ">> scoring $INFER${ARM:+ (arm $ARM)}"
    ( unset PYTHONPATH; cd $MASLAB && python evaluate.py \
        --eval_protocol code \
        --model_name $MAS_MODEL \
        --tested_dataset_name "$DATASET" \
        --tested_infer_path "$INFER" \
        --overwrite )
    # evaluate.py leaves eval_score None where the method produced nothing and reports
    # accuracy over the rest, giving each arm its own denominator -- off 152/160 vs on
    # 154/163 reverses over the whole set. Unanswered is a failure, so score every row.
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
    echo "usage: ARM=on|off|rot METHOD=.. DATASET=.. TAG=.. $0 {record|toggle|eval}"
    echo "  record  ARM=on METHOD=autogen MAS_CONFIG=config_code DATASET=humaneval TAG=ag_he $0 record"
    echo "  eval    ARM=on TAG=ag_he DATASET=humaneval $0 eval"
    echo "  toggle  REQUESTS=<capture> TAG=ag_he_ab $0 toggle   (no ARM: it runs all three)"
    echo "  CONC=N replays with N requests in flight; the summary's throughput row"
    echo "  is 1/latency at the default CONC=1"
    echo "  an arm's files live in ab_out/maslab/<arm>/ and are suffixed _<tag>_<arm>"
    exit 1;;
esac
