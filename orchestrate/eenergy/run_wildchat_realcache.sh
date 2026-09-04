#!/bin/bash
# Same wildchat_natural condition as run_wildchat_natural_7arms.sh (data/wildchat_v1.json, no
# whale injection, natural min-turns=1/max-turns=4, rate=10.7, max-tokens=1024), but snapshots
# the REAL server-side prefix-cache hit rate (vLLM's own vllm:prefix_cache_hits_total /
# queries_total counters) right before teardown -- validates cache_mirror.py's estimate against
# real telemetry on genuinely real, natural multi-turn content (not synthetic ShareGPT), a
# distinct check from the existing light/cachehit validation.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

OUTNAME=drf_power_tiebreak_full
POLICY=drf_power_tiebreak_full
REPLICA_HOSTPORTS="127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004,127.0.0.1:8005,127.0.0.1:8006"
MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct

for TRIAL in 1 2 3; do
  echo "=== WILDCHATREALCACHE BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/wildchatrealcache_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=$MODEL \
    ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
    POWER_TRACE=logs/wildchatrealcache_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/wildchatrealcache_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/wildchatrealcache_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

  echo "=== running harness (policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model $MODEL \
    --dataset data/wildchat_v1.json --min-turns 1 --max-turns 4 \
    --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --request-timeout 180 \
    --output logs/wildchatrealcache_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/wildchatrealcache_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== snapshotting REAL server-side cache hit rate (policy=$OUTNAME trial=$TRIAL) ==="
  python3 scripts/eenergy/snapshot_cache_hit_rate.py \
    --replicas "$REPLICA_HOSTPORTS" --model "$MODEL" \
    --output logs/wildchatrealcache_realcache_${OUTNAME}_t${TRIAL}.csv

  echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== WILDCHATREALCACHE BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
done
echo "=== FULL WILDCHATREALCACHE BATCH COMPLETE ==="
