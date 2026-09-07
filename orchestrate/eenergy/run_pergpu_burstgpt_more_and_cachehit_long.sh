#!/bin/bash
# Extends the duration-sensitivity check beyond Heavy/Closed-Loop to two conditions that
# involve no synthetic whale injection at all:
#   1. BurstGPT (real trace replay, logs/burstgpt_heavy_trace.csv, 1000 requests, arrivals
#      spanning the trace's own 0-300s -- already ~316s active window, genuinely real, not
#      fabricated). No duration change possible/needed; adds trials 4-6 (existing has 1-3)
#      for the same reason we added trials 4-6 to closedloopheavy: more replication before
#      trusting a dominance verdict.
#   2. Light/Cachehit (data/sharegpt_v3.json, --whale-frac 0.0 already -- no whales, natural
#      cache-hit-heavy ShareGPT traffic). Original condition is ~51s (150 convs @
#      concurrency=24); this adds a "long" variant (num-convs 150->900, same concurrency,
#      same whale-frac 0.0, same everything else) targeting a ~300s active window, matching
#      Heavy/Closed-Loop's duration-extension approach but with zero whale-injection
#      parameters touched at any point.
# 5 arms x (3 BurstGPT trials + 3 Cachehit-long trials) = 30 runs, sequential
# (N_REPLICAS=6/GPU_OFFSET=2 saturates all 6 target GPUs).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"

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

run_burstgpt_trial () {
  local OUTNAME=$1
  local POLICY=$2
  local TRIAL=$3
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

run_cachehitlong_trial () {
  local OUTNAME=$1
  local POLICY=$2
  local TRIAL=$3
  local PREFIX=cachehitlongpergpu

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
    --concurrency 24 --num-convs 900 --max-tokens 128 \
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
    run_burstgpt_trial "$OUTNAME" "$POLICY" "$TRIAL"
  done
done

for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 1 2 3; do
    run_cachehitlong_trial "$OUTNAME" "$POLICY" "$TRIAL"
  done
done

echo "=== BURSTGPT TRIALS 4-6 + CACHEHIT-LONG 3-TRIAL BATCH COMPLETE ==="
