#!/usr/bin/env bash
set -euo pipefail

EXP="alphaflow-meanflow-latentspace-B-2-uniform_sampling-analysis"

# ---- base ckpt path pattern ----
CKPT_BASE="/mnt/ssd1/eun/2026/meanflow/alphaflow/experiments/0041-imagenet_folder_dit_alphaflow-meanflow-latentspace-B-2-uniform_sampling-b0fef77-dirty/output"
CKPT_TEMPLATE="snapshot-%08d.pt"   # snapshot-00000001.pt etc.

# ---- logging: append to one file ----
LOG="./logs/log_meanflow_b2_uniform_sampling_analysis.txt"
mkdir -p "$(dirname "$LOG")"
touch "$LOG"

timestamp () { date "+%Y-%m-%d %H:%M:%S"; }

append_header () {
  {
    echo
    echo "================================================================================"
    echo "[$(timestamp)] $*"
    echo "================================================================================"
  } >> "$LOG"
}

run_with_append_log () {
  set +e
  yes y 2>/dev/null | python infra/launch.py "$@" >> "$LOG" 2>&1
  code="${PIPESTATUS[1]}"
  set -e
  return "$code"
}

# ------------------------------------------------------------------------------
# Usage:
#   scripts/run_meanflow_b2_uniform_sampling_analysis_sweep.sh \
#     "1 10000 20000 30000" \
#     "0.03 0.05 0.07"
#
# Loop order:
#   for each ckpt -> iterate over all float values (TMIN=DELTA) -> next ckpt
# ------------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<iter_list>\" \"<tmin_delta_list>\""
  echo "Example: $0 \"1 10000 20000 30000\" \"0.03 0.05 0.07\""
  exit 2
fi

ITER_LIST_STR="$1"
VAL_LIST_STR="$2"

read -r -a ITERS <<< "$ITER_LIST_STR"
read -r -a VALS  <<< "$VAL_LIST_STR"

append_header "START sweep | EXP=${EXP}"
append_header "CKPT_BASE=${CKPT_BASE}"
append_header "ITERS=(${ITERS[*]})"
append_header "VALS(TMIN=DELTA)=(${VALS[*]})"
append_header "LOG=${LOG}"

for iter in "${ITERS[@]}"; do
  ckpt_file=$(printf "$CKPT_TEMPLATE" "$iter")
  ckpt_path="${CKPT_BASE}/${ckpt_file}"

  append_header "CKPT iter=${iter} | path=${ckpt_path}"

  if [[ ! -f "$ckpt_path" ]]; then
    echo "[$(timestamp)] [SKIP] missing ckpt: ${ckpt_path}" >> "$LOG"
    continue
  fi

  for val in "${VALS[@]}"; do
    TMIN="$val"
    DELTA="$val"

    append_header "RUN | iter=${iter} | TMIN=${TMIN} | DELTA=${DELTA}"

    if run_with_append_log \
      "$EXP" \
      wandb.enabled=true \
      training.resume.on_start_ckpt_path="$ckpt_path" \
      loss.distrib_t_t_next_mf.time_sampling_mf_t.min="$TMIN" \
      loss.distrib_t_t_next_mf.time_sampling_mf_t_next.delta_size="$DELTA"
    then
      echo "[$(timestamp)] [OK] iter=${iter} TMIN=DELTA=${val}" >> "$LOG"
    else
      echo "[$(timestamp)] [FAIL] iter=${iter} TMIN=DELTA=${val}" >> "$LOG"
    fi
  done
done

append_header "END sweep | EXP=${EXP}"