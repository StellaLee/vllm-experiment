#!/bin/bash
# Isolates the max_tokens confound flagged this session: Light/Cachehit's original condition
# uses --max-tokens 128 while Heavy/Closed-Loop, Heavy/Matched, and WildChat use 1024 --
# meaning cross-condition TBT comparisons aren't on the same scale (TBT_mean is each
# request's own MAX inter-token gap, and more decode steps = more chances to sample a bad
# one, independent of routing quality), and longer decode may shift which Share_* dimension
# binds D(c) most often. This changes ONLY --max-tokens (128->1024), keeping --num-convs at
# the ORIGINAL cachehit's 150 (not cachehitlong's 900) so this is a clean single-variable
# comparison against the existing short cachehitpergpu_ data -- duration effects are already
# being checked separately via cachehitlongpergpu_. Everything else (whale-frac 0.0,
# concurrency 24, min/max-turns 2/4, pad-chars 0) is identical to the original condition.
# 5 arms x 3 trials = 15 runs, sequential.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
PREFIX=cachehit1024pergpu

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
  timeout 1200 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
    --concurrency 24 --num-convs 150 --max-tokens 1024 \
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
  for TRIAL in 1 2 3; do
    run_trial "$OUTNAME" "$POLICY" "$TRIAL"
  done
done

echo "=== CACHEHIT1024 (max_tokens isolation) BATCH COMPLETE ==="
