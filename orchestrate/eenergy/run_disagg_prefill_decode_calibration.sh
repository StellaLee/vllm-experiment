#!/bin/bash
# Power-metered prefill/decode energy calibration under REAL P/D disaggregation (NixlConnector,
# 1 prefill instance on GPU0 + 1 decode instance on GPU1 + toy_proxy_server.py), the original
# motivation for the whole disagg detour: does J/prefill-token or J/decode-token differ from
# the colocated calibration's values (0.068, 2.40 J/token respectively -- see
# findings/2026-09-11-eenergy-peak-shaving-admission-gate.md Part 3)?
#
# Unlike the colocated calibration, this does NOT need artificial pure-prefill/pure-decode
# isolation bursts -- disaggregation gives that isolation structurally for free: GPU0 only ever
# does prefill compute, GPU1 only ever does decode compute, for the SAME natural mixed workload,
# simultaneously. So this runs ONE realistic mixed burst (matching this project's standard
# whale-mix: whale_frac=0.15, 44-50k char whales, max_tokens=1024) and meters each GPU's power
# separately and continuously across idle-settle + burst + cooldown.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
OUTNAME=disagg_prefill_decode_calibration
IDLE_SETTLE_S=30
COOLDOWN_S=10

mkdir -p logs
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
pkill -f "power_logger.py.*${OUTNAME}" 2>/dev/null
sleep 3

echo "=== launching prefill (kv_producer) instance on GPU0:${PREFILL_PORT} ==="
CUDA_VISIBLE_DEVICES=0 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=20000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  vllm serve "$MODEL" --port $PREFILL_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  > logs/${OUTNAME}_prefill.log 2>&1 &

echo "=== launching decode (kv_consumer) instance on GPU1:${DECODE_PORT} ==="
CUDA_VISIBLE_DEVICES=1 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=30000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve "$MODEL" --port $DECODE_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  > logs/${OUTNAME}_decode.log 2>&1 &

echo "waiting for both instances to report healthy..."
for i in $(seq 1 180); do
  P_OK=0; D_OK=0
  curl -sf http://127.0.0.1:${PREFILL_PORT}/health >/dev/null 2>&1 && P_OK=1
  curl -sf http://127.0.0.1:${DECODE_PORT}/health >/dev/null 2>&1 && D_OK=1
  if [ "$P_OK" = "1" ] && [ "$D_OK" = "1" ]; then
    echo "both instances healthy after ${i} checks"
    break
  fi
  sleep 5
done

echo "=== launching proxy on :${PROXY_PORT} ==="
python3 scripts/eenergy/nixl_toy_proxy_server.py --port $PROXY_PORT \
  --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost --decoder-ports $DECODE_PORT \
  > logs/${OUTNAME}_proxy.log 2>&1 &
sleep 5

echo "=== starting per-GPU power loggers ==="
python3 scripts/pesim/power_logger.py --gpus 0 --interval-ms 50 \
  --output logs/${OUTNAME}_power_gpu0.csv &
POWER0_PID=$!
python3 scripts/pesim/power_logger.py --gpus 1 --interval-ms 50 \
  --output logs/${OUTNAME}_power_gpu1.csv &
POWER1_PID=$!

echo "=== idle settle window (${IDLE_SETTLE_S}s) -- captures idle baseline ==="
sleep $IDLE_SETTLE_S
BURST_START_EPOCH=$(date +%s.%N)
echo "burst_start_epoch=${BURST_START_EPOCH}" | tee logs/${OUTNAME}_timing.txt

echo "=== running mixed burst through proxy (port ${PROXY_PORT}) ==="
timeout 1200 python3 src/replay_sharegpt.py \
  --host 127.0.0.1 --port $PROXY_PORT --model "$MODEL" \
  --dataset data/sharegpt_v3.json --request-timeout 180 \
  --output logs/${OUTNAME}_records.jsonl \
  --min-turns 1 --max-turns 1 --concurrency 8 --num-convs 150 --max-tokens 1024 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
  2>&1 | tee logs/${OUTNAME}_harness.log

BURST_END_EPOCH=$(date +%s.%N)
echo "burst_end_epoch=${BURST_END_EPOCH}" | tee -a logs/${OUTNAME}_timing.txt

echo "=== cooldown (${COOLDOWN_S}s) -- flush trailing power samples ==="
sleep $COOLDOWN_S

echo "=== tearing down ==="
kill $POWER0_PID $POWER1_PID 2>/dev/null
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null

echo "=== DISAGG_PREFILL_DECODE_CALIBRATION_BATCH COMPLETE ==="
