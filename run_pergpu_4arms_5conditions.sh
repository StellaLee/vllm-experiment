#!/bin/bash
# Per-GPU-calibrated ramp ceiling re-validation across the remaining 5 conditions
# (Heavy/Matched, Light/Cachehit, Heavy/Closed-Loop, Ramp & Route, WildChat -- BurstGPT
# already done). 4 arms (drf_fixed, drf_power_tiebreak_full, weighted_sum, lmetric_power) x
# 5 conditions x 3 trials = 60 runs, sequential (GPU_OFFSET=2/N_REPLICAS=6 saturates all 6
# target GPUs, no cross-run parallelism possible on this 8-GPU box). Harness args for the 4
# synthetic conditions are copied verbatim from run_drf_power_tiebreak_full_all4.sh (the
# script that produced the OLD numbers being compared against); WildChat args copied from
# run_wildchat_natural_7arms.sh. RAMP_CEILING_PER_GPU is the only thing that changes.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
SUFFIX=pergpu

run_trial () {
  local PREFIX=$1
  local OUTNAME=$2
  local POLICY=$3
  local TRIAL=$4
  local DATASET=$5
  shift 5
  local HARNESS_ARGS=("$@")

  local FULLPREFIX="${PREFIX}${SUFFIX}"
  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${FULLPREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${FULLPREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"

  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      break
    fi
    sleep 5
  done

  echo "=== running harness (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset "$DATASET" --request-timeout 180 \
    --output logs/${FULLPREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${FULLPREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

# OUTNAME POLICY pairs -- drf_fixed's actual policy string is "drf" (the known pitfall,
# kept separated here from the start).
ARMS=("drf_fixed drf" "drf_power_tiebreak_full drf_power_tiebreak_full" \
      "weighted_sum weighted_sum" "lmetric_power lmetric_power")

run_condition () {
  local PREFIX=$1
  local DATASET=$2
  shift 2
  local HARNESS_ARGS=("$@")
  for PAIR in "${ARMS[@]}"; do
    set -- $PAIR
    OUTNAME=$1
    POLICY=$2
    for TRIAL in 1 2 3; do
      run_trial "$PREFIX" "$OUTNAME" "$POLICY" "$TRIAL" "$DATASET" "${HARNESS_ARGS[@]}"
    done
  done
}

run_condition openloopwhalelongoutmatched data/sharegpt_v3.json \
  --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000

run_condition cachehit data/sharegpt_v3.json \
  --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
  --concurrency 24 --num-convs 150 --max-tokens 128

run_condition closedloopheavy data/sharegpt_v3.json \
  --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 150 --max-tokens 1024 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000

run_condition rampandroute data/sharegpt_v3.json \
  --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 150 --max-tokens 128 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000

run_condition wildchatnatural data/wildchat_v1.json \
  --min-turns 1 --max-turns 4 --rate 10.7 --num-convs 750 --max-tokens 1024

echo "=== FULL PER-GPU 4-ARM 5-CONDITION BATCH COMPLETE ==="
