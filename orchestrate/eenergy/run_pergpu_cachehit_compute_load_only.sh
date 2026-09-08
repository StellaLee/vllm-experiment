#!/bin/bash
# Tests whether compute_only's poor TTFT (worst arm on both Heavy/Closed-Loop-long and
# Heavy/Matched) is explained by those conditions' near-zero cache-hit rate (~0.0-0.1%,
# mean new/raw ~0.994-0.996 -- single-turn-only conditions with almost no real
# Share_compute signal to differentiate candidates, so compute_only degenerates toward
# blind rotation, unable to avoid an overloaded replica). Light/Cachehit has substantial
# partial cache reuse (mean new/raw ~0.69, real compute-share differentiation) -- if the
# hypothesis is right, compute_only's TTFT should look much more competitive here.
# Same params as the original cachehit condition (run_pergpu_4arms_5conditions.sh).
# 2 arms x 3 trials = 6 runs, sequential.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
PREFIX=cachehitpergpu

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
    --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
    --concurrency 24 --num-convs 150 --max-tokens 128 \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

ARMS=("compute_only compute_only" "load_only load_only")

for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 1 2 3; do
    run_trial "$OUTNAME" "$POLICY" "$TRIAL"
  done
done

echo "=== CACHEHIT COMPUTE_LOAD_ONLY BATCH COMPLETE ==="
