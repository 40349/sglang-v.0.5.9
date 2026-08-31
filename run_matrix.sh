#!/bin/bash
# The full sub-context experiment: 2 methods x 2 datasets, each run end to end on
# BOTH arms, plus a replay A/B on the captured request sequence.
#
# Both arms run end to end because the split changes model output (8/25 requests on
# the smoke run; three controls came back 0/25, so it is the reuse, not noise).
# Divergence is expected, so pass@1 has to be scored on both arms.
#
# Four stages per combination, each guarded by a marker file so a killed run resumes:
#   <tag>_on    split ON, end to end. Also records the sequence the A/B replays.
#   <tag>_off   split OFF, end to end. The accuracy control.
#   <tag>_ab    replay that sequence on both arms at a fixed generation length
#               -> GPU/host tables, hit rate, divergence count.
#   <tag>_eval  pass@1 on both arms' results.
#
# HumanEval combinations run first: a third of the work, so stopping early still
# leaves a complete result for both methods.
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=${OUT:-$REPO/ab_out}
# run_mas.sh writes captures and results under ab_out/maslab. The stage markers and
# these logs stay at the top level: they are this driver's own bookkeeping, and
# moving them would make a half-finished matrix look unstarted.
MAS_OUT=$OUT/maslab
LOGS=$OUT/matrix_logs
mkdir -p "$LOGS"

# tag | method | method_config | dataset
COMBOS=(
  "ag_he|autogen|config_code|humaneval"
  "av_he|agentverse||humaneval"
  "ag_mb|autogen|config_code|mbpp"
  "av_mb|agentverse||mbpp"
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/matrix.log"; }

for combo in "${COMBOS[@]}"; do
  IFS='|' read -r tag method cfg ds <<< "$combo"
  say "########## $tag ($method x $ds) ##########"

  if [ ! -f "$OUT/.done_${tag}_on" ]; then
    say "$tag: end-to-end, split ON"
    METHOD="$method" MAS_CONFIG="$cfg" DATASET="$ds" TAG="${tag}_on" \
      bash "$REPO/run_mas.sh" record > "$LOGS/${tag}_on.log" 2>&1
    touch "$OUT/.done_${tag}_on"
    say "$tag: ON done, $(wc -l < "$MAS_OUT/requests_${tag}_on.jsonl") requests captured"
  else say "$tag: ON already done, skipping"; fi

  if [ ! -f "$OUT/.done_${tag}_off" ]; then
    say "$tag: end-to-end, split OFF"
    SUBCTX_OFF=1 METHOD="$method" MAS_CONFIG="$cfg" DATASET="$ds" TAG="${tag}_off" \
      bash "$REPO/run_mas.sh" record > "$LOGS/${tag}_off.log" 2>&1
    touch "$OUT/.done_${tag}_off"
    say "$tag: OFF done"
  else say "$tag: OFF already done, skipping"; fi

  # The rotation arm is a third end-to-end run, not a replay arm, because its
  # pass@1 is the number that prices it: rotation fixes the position a block is
  # reused at, not the context it was computed under, so its output legitimately
  # differs and only a freely-generating run can be scored.
  if [ -n "${ROTATE:-}" ] && [ ! -f "$OUT/.done_${tag}_rot" ]; then
    say "$tag: end-to-end, split ON + rotation"
    SUBCTX_ROTATE=1 SUBCTX_ROTATE_ACROSS=${ACROSS:-1} \
      METHOD="$method" MAS_CONFIG="$cfg" DATASET="$ds" TAG="${tag}_rot" \
      bash "$REPO/run_mas.sh" record > "$LOGS/${tag}_rot.log" 2>&1
    touch "$OUT/.done_${tag}_rot"
    say "$tag: ROTATE done"
  fi

  if [ ! -f "$OUT/.done_${tag}_ab" ]; then
    say "$tag: replay A/B on the ON capture"
    REQUESTS="$MAS_OUT/requests_${tag}_on.jsonl" TAG="${tag}_ab" \
      ROTATE="${ROTATE:-}" ACROSS="${ACROSS:-1}" \
      bash "$REPO/run_mas.sh" toggle > "$LOGS/${tag}_ab.log" 2>&1
    touch "$OUT/.done_${tag}_ab"
    say "$tag: A/B done"
  else say "$tag: A/B already done, skipping"; fi

  if [ ! -f "$OUT/.done_${tag}_eval" ]; then
    arms="on off"
    [ -n "${ROTATE:-}" ] && arms="on off rot"
    for arm in $arms; do
      say "$tag: pass@1 ($arm)"
      TAG="${tag}_${arm}" DATASET="$ds" bash "$REPO/run_mas.sh" eval \
        > "$LOGS/${tag}_${arm}_eval.log" 2>&1
      say "$tag/$arm: $(grep -oE 'accuracy: [0-9.]+%' "$LOGS/${tag}_${arm}_eval.log" | head -1)"
    done
    touch "$OUT/.done_${tag}_eval"
  else say "$tag: eval already done, skipping"; fi
done

say "########## MATRIX COMPLETE ##########"
