#!/bin/bash
# Hypothesis test (paired with the matched-condition run): does the self-calibrating ramp
# ceiling fix drf_power_tiebreak's own documented light-load regression (cache-hit workload,
# concurrency=24, real cache hits ~46-47%)? drf_power_tiebreak reference there: max_ramp
# 3533.9+-860.1 (worse and 10x noisier than drf_fixed's 2890.4+-80.4). Same params as
# run_drf_power_tiebreak_cachehit.sh.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

OUTNAME=drf_power_tiebreak_adaptive
POLICY=drf_power_tiebreak_adaptive
for TRIAL in 1 2 3; do
  echo "=== DRFADAPTIVECACHEHIT BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/cachehit_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
    POWER_TRACE=logs/cachehit_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/cachehit_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/cachehit_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  LAUNCH_PID=$!
  echo "launch script pid: $LAUNCH_PID"

  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      break
    fi
    sleep 5
  done

  echo "=== running harness (policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --pad-chars 0 --whale-frac 0.0 \
    --min-turns 2 --max-turns 4 --concurrency 24 --num-convs 150 --max-tokens 128 \
    --request-timeout 180 \
    --output logs/cachehit_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/cachehit_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== DRFADAPTIVECACHEHIT BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
done
echo "=== FULL DRFADAPTIVECACHEHIT BATCH COMPLETE ==="
