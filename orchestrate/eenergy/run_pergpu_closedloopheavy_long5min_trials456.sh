#!/bin/bash
# Extends the 5-minute Heavy/Closed-Loop duration-extension battery from 3 trials to 6, to
# check whether the "no dominance among scored arms" result (vs. the short ~51-53s window's
# clean dominance) is a robust finding or 3-trial noise. Same 5 arms, same per-GPU-calibrated
# ceiling, same workload params (concurrency=32, num-convs=900, whale-frac=0.15,
# whale-min/max-chars=44000/50000) as run_pergpu_closedloopheavy_long5min.sh -- only the
# trial numbers (4,5,6) and output prefix are new-trial-specific; same PREFIX as before so
# scripts/eenergy/compare_closedloopheavy_duration.py's aggregate() picks up all 6 trials by
# passing trials=(1,2,3,4,5,6).
# 5 arms x 3 trials = 15 runs, sequential. Expected ~271s/trial (228s active + ~43s
# setup/teardown) x 15 =~ 68 minutes total.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
PREFIX=closedloopheavylongpergpu

run_common_setup () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  mkdir -p logs
}

wait_for_router () {
  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      return 0
    fi
    sleep 5
  done
}

teardown () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
}

run_trial () {
  local OUTNAME=$1
  local POLICY=$2
  local TRIAL=$3

  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running harness (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 900 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

ARMS=("drf_fixed drf" "drf_power_tiebreak_full drf_power_tiebreak_full" \
      "weighted_sum weighted_sum" "lmetric_power lmetric_power" \
      "round_robin round_robin")

for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 4 5 6; do
    run_trial "$OUTNAME" "$POLICY" "$TRIAL"
  done
done

echo "=== CLOSEDLOOPHEAVY 5-MINUTE DURATION-EXTENSION TRIALS 4-6 BATCH COMPLETE ==="
