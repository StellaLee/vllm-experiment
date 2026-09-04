#!/bin/bash
# Re-run of BurstGPT (real trace, no injection) restricted to the safe triad (drf_fixed,
# drf_power_tiebreak_full, weighted_sum), now with RAMP_CEILING_PER_GPU instead of one
# shared 450 W/s constant -- see scripts/eenergy/README.md "Per-GPU ramp-ceiling
# calibration (2026-09-03)". BurstGPT was chosen because it's where the widest gap was
# found (drf_fixed dominated drf_power_tiebreak_full 6/6 metrics under the old uniform
# ceiling) -- if per-GPU calibration changes anything, it should show up here first.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

PREFIX=burstgptpergpu
RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"

run_trial () {
  local OUTNAME=$1
  local POLICY=$2
  local TRIAL=$3

  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

  echo "=== running harness (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --trace-csv logs/burstgpt_heavy_trace.csv --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for PAIR in "drf_fixed drf" "drf_power_tiebreak_full drf_power_tiebreak_full" "weighted_sum weighted_sum"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 1 2 3; do
    run_trial $OUTNAME $POLICY $TRIAL
  done
done

echo "=== FULL BURSTGPT-PERGPU SAFE-TRIAD BATCH COMPLETE ==="
