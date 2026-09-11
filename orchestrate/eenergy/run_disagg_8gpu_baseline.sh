#!/bin/bash
# Uncapped 4:4 P/D-disaggregated baseline -- establishes this topology's own sustained
# mean/peak power per pool (not yet measured at 4:4 scale; the 1P1D calibration and the
# colocated 6-replica fleet are both different scales/topologies). Gate shim runs with both
# caps unset (disabled) so every request passes straight through -- this exercises the same
# code path Task 5's gated run will use, just with the admission check always a no-op, giving
# a true apples-to-apples comparison.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
NIXL_PROXY_PORT=8192
GATE_PORT=8190
OUTNAME=disagg_8gpu_baseline
PREFILL_PORTS=(8100 8101 8102 8103)
DECODE_PORTS=(8200 8201 8202 8203)
PREFILL_GPUS=(0 1 2 3)
DECODE_GPUS=(4 5 6 7)
IDLE_SETTLE_S=30
COOLDOWN_S=10

mkdir -p logs
pkill -f "nixl_toy_proxy_server.py" 2>/dev/null
pkill -f "run_disagg_gate.py" 2>/dev/null
for p in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
  pkill -f "vllm.entrypoints.*--port ${p}" 2>/dev/null
done
pkill -f "power_logger.py.*${OUTNAME}" 2>/dev/null
sleep 3

for i in "${!PREFILL_PORTS[@]}"; do
  PORT=${PREFILL_PORTS[$i]}
  GPU=${PREFILL_GPUS[$i]}
  echo "=== launching prefill instance $i (GPU${GPU}:${PORT}) ==="
  CUDA_VISIBLE_DEVICES=$GPU VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=$((20000 + i * 100)) \
    UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=$((5559 + i)) \
    vllm serve "$MODEL" --port $PORT --block-size 128 \
    --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    > logs/${OUTNAME}_prefill${i}.log 2>&1 &
done

for i in "${!DECODE_PORTS[@]}"; do
  PORT=${DECODE_PORTS[$i]}
  GPU=${DECODE_GPUS[$i]}
  echo "=== launching decode instance $i (GPU${GPU}:${PORT}) ==="
  CUDA_VISIBLE_DEVICES=$GPU VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=$((30000 + i * 100)) \
    UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=$((5659 + i)) \
    vllm serve "$MODEL" --port $PORT --block-size 128 \
    --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
    > logs/${OUTNAME}_decode${i}.log 2>&1 &
done

echo "waiting for all 8 instances to report healthy..."
for i in $(seq 1 240); do
  ALL_OK=1
  for p in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
    curl -sf http://127.0.0.1:${p}/health >/dev/null 2>&1 || ALL_OK=0
  done
  if [ "$ALL_OK" = "1" ]; then
    echo "all 8 instances healthy after ${i} checks"
    break
  fi
  sleep 5
done

echo "=== launching Nixl proxy on :${NIXL_PROXY_PORT} ==="
python3 scripts/eenergy/nixl_toy_proxy_server.py --port $NIXL_PROXY_PORT \
  --prefiller-hosts localhost localhost localhost localhost \
  --prefiller-ports "${PREFILL_PORTS[@]}" \
  --decoder-hosts localhost localhost localhost localhost \
  --decoder-ports "${DECODE_PORTS[@]}" \
  > logs/${OUTNAME}_nixl_proxy.log 2>&1 &
sleep 5

echo "=== launching gate shim on :${GATE_PORT} (both caps unset -- disabled) ==="
DISAGG_MODEL_NAME="$MODEL" DISAGG_GATE_PORT=$GATE_PORT \
  DISAGG_PROXY_URL="http://127.0.0.1:${NIXL_PROXY_PORT}" \
  DISAGG_PREFILL_GPU_INDICES="0,1,2,3" DISAGG_DECODE_GPU_INDICES="4,5,6,7" \
  python3 scripts/eenergy/run_disagg_gate.py > logs/${OUTNAME}_gate.log 2>&1 &
sleep 5

echo "=== starting per-pool power loggers ==="
python3 scripts/pesim/power_logger.py --gpus 0,1,2,3 --interval-ms 50 \
  --output logs/${OUTNAME}_power_prefill.csv &
POWER_PREFILL_PID=$!
python3 scripts/pesim/power_logger.py --gpus 4,5,6,7 --interval-ms 50 \
  --output logs/${OUTNAME}_power_decode.csv &
POWER_DECODE_PID=$!

echo "=== idle settle window (${IDLE_SETTLE_S}s) ==="
sleep $IDLE_SETTLE_S
BURST_START_EPOCH=$(date +%s.%N)
echo "burst_start_epoch=${BURST_START_EPOCH}" | tee logs/${OUTNAME}_timing.txt

echo "=== running workload against gate shim (port ${GATE_PORT}) ==="
timeout 1800 python3 src/replay_sharegpt.py \
  --host 127.0.0.1 --port $GATE_PORT --model "$MODEL" \
  --dataset data/sharegpt_v3.json --request-timeout 180 \
  --output logs/${OUTNAME}_records.jsonl \
  --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 600 --max-tokens 1024 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
  2>&1 | tee logs/${OUTNAME}_harness.log

BURST_END_EPOCH=$(date +%s.%N)
echo "burst_end_epoch=${BURST_END_EPOCH}" | tee -a logs/${OUTNAME}_timing.txt

echo "=== cooldown (${COOLDOWN_S}s) ==="
sleep $COOLDOWN_S

echo "=== tearing down ==="
kill $POWER_PREFILL_PID $POWER_DECODE_PID 2>/dev/null
pkill -f "nixl_toy_proxy_server.py" 2>/dev/null
pkill -f "run_disagg_gate.py" 2>/dev/null
for p in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
  pkill -f "vllm.entrypoints.*--port ${p}" 2>/dev/null
done
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null

echo "=== DISAGG_8GPU_BASELINE_BATCH COMPLETE ==="
