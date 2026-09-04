#!/bin/bash
# Light/cachehit condition (concurrency=24, plain multi-turn, no whale, max-tokens 128) with
# the REAL server-side prefix-cache hit rate snapshotted right before teardown each trial,
# to compare against the router's own cache_mirror.py estimate for this arm.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

OUTNAME=round_robin
POLICY=round_robin
REPLICA_HOSTPORTS="127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004,127.0.0.1:8005,127.0.0.1:8006"
MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct

for TRIAL in 1 2 3; do
  echo "=== ROUNDROBINCACHEHITREALCACHE BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/realcachecachehit_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=$MODEL \
    ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
    POWER_TRACE=logs/realcachecachehit_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/realcachecachehit_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/realcachecachehit_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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
    --host 127.0.0.1 --port 9100 --model $MODEL \
    --dataset data/sharegpt_v3.json --pad-chars 0 --whale-frac 0.0 \
    --min-turns 2 --max-turns 4 --concurrency 24 --num-convs 150 --max-tokens 128 \
    --request-timeout 180 \
    --output logs/realcachecachehit_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/realcachecachehit_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== snapshotting REAL server-side cache hit rate (policy=$OUTNAME trial=$TRIAL) ==="
  python3 scripts/eenergy/snapshot_cache_hit_rate.py \
    --replicas "$REPLICA_HOSTPORTS" --model "$MODEL" \
    --output logs/realcachecachehit_realcache_${OUTNAME}_t${TRIAL}.csv

  echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ROUNDROBINCACHEHITREALCACHE BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
done
echo "=== FULL ROUNDROBINCACHEHITREALCACHE BATCH COMPLETE ==="
