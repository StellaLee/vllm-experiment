#!/bin/bash
# Live smoke test for two things at once:
# 1) BS-via-telemetry (ROUTER_BS_SOURCE=telemetry) actually works end-to-end against real
#    vLLM /metrics, not just unit-tested against a canned text blob.
# 2) With a real multi-turn, unpadded (plain) workload -- the one config that actually lets
#    prefix reuse happen -- audit whether the router's logged new_tokens (P-token) is honest:
#    does it show a real discount (< raw_tokens) exactly when a cache hit should occur?
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
sleep 3

mkdir -p logs
rm -f logs/live_smoke_launch.log logs/live_smoke_assignment2.csv logs/live_smoke_records2.jsonl

POLICY=lmetric N_REPLICAS=2 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
  ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
  ROUTER_BS_SOURCE=telemetry ROUTER_BS_POLL_INTERVAL_S=0.3 \
  ASSIGNMENT_LOG=logs/live_smoke_assignment2.csv \
  nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/live_smoke_launch.log 2>&1 &
echo "launch pid $!"

for i in $(seq 1 120); do
  if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
     -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
     >/dev/null 2>&1; then
    echo "router ready after ${i} checks"
    break
  fi
  sleep 5
done

echo "=== running plain multi-turn harness, SERIALIZED (concurrency=1) for unambiguous audit ==="
timeout 300 python3 src/replay_sharegpt.py \
  --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
  --dataset data/sharegpt_v3.json --min-turns 3 --max-turns 4 \
  --concurrency 1 --num-convs 15 --max-tokens 64 \
  --request-timeout 180 \
  --output logs/live_smoke_records2.jsonl \
  2>&1 | tee logs/live_smoke_harness2.log

echo "=== tearing down ==="
pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
sleep 3
echo "=== DONE ==="
