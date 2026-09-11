#!/bin/bash
# Follow-up to findings.md Part 24: Heavy/Matched at a lighter rate (4.0, below the current
# lightest tested 7.0), and WildChat/BurstGPT pushed to a genuinely more aggressive rate --
# the earlier rate=15.0/1.5x tests for WildChat/BurstGPT turned out to be far from saturated
# (p99 TTFT ~0.96s vs Heavy/Matched's own lightest-tested-rate p99 of 7.8s), so those
# conditions were never actually testing the mechanism under real pressure. WildChat has no
# whale injection (real conversational content only), so it needs proportionally more
# throughput than Heavy/Matched to reach comparable load; BurstGPT's trace-rate-scale is
# doubled from 1.5x to 3.0x for the same reason. Same 4 headline-adjacent arms, 3 trials each.
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

launch_and_wait () {
  local PREFIX=$1
  local OUTNAME=$2
  local POLICY=$3
  local TRIAL=$4
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log
  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router
}

ARMS=("coincidence_ceiling drf_power_tiebreak_full_coincidence_ceiling" \
      "round_robin round_robin" \
      "lmetric_power_coincidence_ceiling lmetric_power_coincidence_ceiling" \
      "weighted_sum_coincidence_ceiling weighted_sum_coincidence_ceiling")

# --- Stage 1: Heavy/Matched rate=4.0 (lighter than the current lightest, 7.0) ---
RATE=4.0
PREFIX="matchedrate${RATE}pergpu"
for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1; POLICY=$2
  for TRIAL in 1 2 3; do
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
    launch_and_wait "$PREFIX" "$OUTNAME" "$POLICY" "$TRIAL"
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/sharegpt_v3.json --request-timeout 180 \
      --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
      --min-turns 1 --max-turns 1 --rate $RATE --num-convs 750 --max-tokens 1024 \
      --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
      2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log
    teardown
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== MATCHED_RATE4_BATCH COMPLETE ==="

# --- Stage 2: WildChat rate=30.0 (2x the previous test point) ---
RATE=30.0
PREFIX="wildchatrate${RATE}pergpu"
for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1; POLICY=$2
  for TRIAL in 1 2 3; do
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
    launch_and_wait "$PREFIX" "$OUTNAME" "$POLICY" "$TRIAL"
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/wildchat_v1.json --request-timeout 180 \
      --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
      --min-turns 1 --max-turns 4 --rate $RATE --num-convs 750 --max-tokens 1024 \
      2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log
    teardown
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== WILDCHAT_RATE30_BATCH COMPLETE ==="

# --- Stage 3: BurstGPT trace-rate-scale=3.0 (2x the previous 1.5x) ---
SCALE=3.0
PREFIX="burstgptscale${SCALE}pergpu"
for PAIR in "${ARMS[@]}"; do
  set -- $PAIR
  OUTNAME=$1; POLICY=$2
  for TRIAL in 1 2 3; do
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
    launch_and_wait "$PREFIX" "$OUTNAME" "$POLICY" "$TRIAL"
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --trace-csv logs/burstgpt_heavy_trace.csv --trace-rate-scale $SCALE --request-timeout 180 \
      --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log
    teardown
    echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== BURSTGPT_SCALE3_BATCH COMPLETE ==="

echo "=== LOAD_EXTREMES_ALL_BATCH COMPLETE ==="
