#!/bin/bash
# Adds round_robin (vLLM-default condition, POLICY=round_robin) to the per-GPU-calibrated
# comparison. The 2026-09-04 4-arm x 6-condition re-validation (drf_fixed,
# drf_power_tiebreak_full, weighted_sum, lmetric_power) never included round_robin -- it
# predates that batch and only has OLD (uniform-450W/s-ceiling-era) data. round_robin's
# routing decision doesn't consult ramp_ceiling at all, but it's rerun fresh here anyway for
# apples-to-apples timing/hardware-state consistency with the other 4 arms' NEW runs, same
# as they were all rerun fresh relative to their own OLD data.
# 6 conditions x 3 trials = 18 runs, sequential (N_REPLICAS=6/GPU_OFFSET=2 saturates all 6
# target GPUs, no cross-run parallelism possible on this 8-GPU box).
# BurstGPT harness args copied verbatim from run_burstgpt_perGPU_safetriad.sh; the other 5
# conditions' args copied verbatim from run_pergpu_4arms_5conditions.sh.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=round_robin
OUTNAME=round_robin

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

# --- BurstGPT (trace-replay, no --dataset flag) ---
run_burstgpt_trial () {
  local TRIAL=$1
  local PREFIX=burstgptpergpu
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
    --trace-csv logs/burstgpt_heavy_trace.csv --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

# --- The other 5 conditions (--dataset-driven, PREFIX+pergpu suffix) ---
run_condition_trial () {
  local PREFIX=$1
  local DATASET=$2
  local TRIAL=$3
  shift 3
  local HARNESS_ARGS=("$@")
  local FULLPREFIX="${PREFIX}pergpu"

  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  run_common_setup
  rm -f logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${FULLPREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${FULLPREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running harness (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset "$DATASET" --request-timeout 180 \
    --output logs/${FULLPREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${FULLPREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for TRIAL in 1 2 3; do
  run_burstgpt_trial "$TRIAL"
done

for TRIAL in 1 2 3; do
  run_condition_trial openloopwhalelongoutmatched data/sharegpt_v3.json "$TRIAL" \
    --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_condition_trial cachehit data/sharegpt_v3.json "$TRIAL" \
    --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
    --concurrency 24 --num-convs 150 --max-tokens 128
done

for TRIAL in 1 2 3; do
  run_condition_trial closedloopheavy data/sharegpt_v3.json "$TRIAL" \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 150 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_condition_trial rampandroute data/sharegpt_v3.json "$TRIAL" \
    --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 150 --max-tokens 128 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_condition_trial wildchatnatural data/wildchat_v1.json "$TRIAL" \
    --min-turns 1 --max-turns 4 --rate 10.7 --num-convs 750 --max-tokens 1024
done

echo "=== FULL PER-GPU ROUND-ROBIN 6-CONDITION BATCH COMPLETE ==="
