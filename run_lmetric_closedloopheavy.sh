#!/bin/bash
# Heavy load, CLOSED-LOOP variant: same whale-injection composition as the open-loop
# "matched" condition (whale_frac=0.15, 44-50k-char whales, single-turn) but
# concurrency-driven instead of rate-driven. concurrency=32 mirrors this project's original
# first-session 3-condition-run precedent (N=8, conc=32) -- now at N=6, so more per-replica
# pressure, deliberately heavy without re-hitting the full-saturation trap already diagnosed
# for --rate 25 (all arms collapsed to indistinguishable ~20s TTFT there).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

OUTNAME=lmetric
POLICY=lmetric
for TRIAL in 1 2 3; do
  echo "=== LMETRICCLOSEDLOOPHEAVY BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/closedloopheavy_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
    POWER_TRACE=logs/closedloopheavy_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/closedloopheavy_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/closedloopheavy_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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
    --dataset data/sharegpt_v3.json --min-turns 1 --max-turns 1 \
    --concurrency 32 --num-convs 150 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    --request-timeout 180 \
    --output logs/closedloopheavy_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/closedloopheavy_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== LMETRICCLOSEDLOOPHEAVY BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
done
echo "=== FULL LMETRICCLOSEDLOOPHEAVY BATCH COMPLETE ==="
