#!/bin/bash
# Re-run of the single failed trial: wildchatnaturalpergpu / lmetric_power / t3
# (original attempt hit a transient EADDRINUSE port-collision on one replica's
# distributed rendezvous, a teardown race from the prior trial -- not systematic).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
PREFIX=wildchatnaturalpergpu
OUTNAME=lmetric_power
POLICY=lmetric_power
TRIAL=3
DATASET=data/wildchat_v1.json

pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
pkill -f "power_logger.py" 2>/dev/null
sleep 5

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
  --dataset "$DATASET" --request-timeout 180 \
  --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
  --min-turns 1 --max-turns 4 --rate 10.7 --num-convs 750 --max-tokens 1024 \
  2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
pkill -f "power_logger.py" 2>/dev/null
sleep 3
echo "=== RERUN COMPLETE ==="
