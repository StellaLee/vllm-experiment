#!/bin/bash
# First hardware validation of the peak-shaving admission gate
# (docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md). Pairs the gate with
# drf_no_power (per the spec's recommendation -- the admission gate now enforces the power
# constraint, so routing doesn't need to double up on it). cap=2400W is the combination the
# offline spike (check_peak_shaving_admission_prototype.py) found near-free on both
# conditions: 0/901 deferrals on Heavy/CL-long at a 15s window, 3.2% deferral rate (~1.4s mean
# wait) on Heavy/Matched at a 30s window. n=3 trials each (quick-check convention) -- expand to
# n=6 if this looks promising.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=drf_no_power
OUTNAME=peak_shaving_validation
PEAK_CAP_W=2400

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
  local PREFIX=$1
  local PEAK_WINDOW_S=$2
  local TRIAL=$3
  shift 3
  local HARNESS_ARGS=("$@")
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL cap=${PEAK_CAP_W}W window=${PEAK_WINDOW_S}s ==="
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    PEAK_CAP_W="$PEAK_CAP_W" PEAK_WINDOW_S="$PEAK_WINDOW_S" \
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
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for TRIAL in 1 2 3; do
  run_trial closedloopheavylongpergpu 15.0 "$TRIAL" \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 900 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_trial openloopwhalelongoutmatchedpergpu 30.0 "$TRIAL" \
    --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

echo "=== PEAK_SHAVING_VALIDATION_BATCH COMPLETE ==="
