#!/bin/bash
# Item 4 follow-up: does the disaggregated peak-shaving gate's core finding (prefill shaves,
# decode barely does) generalize past one dense 7B model? Qwen1.5-MoE-A2.7B-Chat (60 experts,
# top-4, ~27GB in bf16 -- already used successfully with NixlConnector's TP=2 combo untested in
# this project so far (every prior disagg script here uses --tensor-parallel-size 1, one GPU per
# instance) -- this is a calibration/feasibility probe, mirroring
# run_disagg_prefill_decode_calibration.sh's original 1P1D dense-model design, before committing
# to a full 4-instance gate battery. If this works cleanly, the next step is scaling to the full
# 2-prefill-instance/2-decode-instance (4 GPUs each, still a 4:4 GPU-pool split) gate battery
# analogous to run_disagg_8gpu_battery.sh.
#
# Topology: prefill instance on GPUs 0-1 (TP=2), decode instance on GPUs 2-3 (TP=2) -- GPUs 4-7
# left idle for this probe. --max-model-len 16384 and --gpu-memory-utilization 0.85 follow this
# project's existing Qwen1.5-MoE-A2.7B-Chat TP=2 convention (orchestrate/pesim/
# orchestrate_pesim_gate_tp2.sh); --block-size 128 and --enforce-eager follow this project's
# existing NixlConnector convention (run_disagg_prefill_decode_calibration.sh). Known risk: the
# VLLM_NIXL_SIDE_CHANNEL_PORT convention below (one base port per instance) is carried over
# unmodified from the TP=1 scripts -- if NixlConnector needs a distinct side-channel port per TP
# rank instead of one shared/auto-offset port per instance, this may fail at KV-transfer time
# even though both instances individually report healthy; check logs/*_prefill.log,
# logs/*_decode.log, and logs/*_proxy.log first if the sanity request hangs or errors.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen1.5-MoE-A2.7B-Chat
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
OUTNAME=disagg_moe_calibration
IDLE_SETTLE_S=30
COOLDOWN_S=10

mkdir -p logs
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
pkill -f "power_logger.py.*${OUTNAME}" 2>/dev/null
sleep 3

echo "=== launching prefill (kv_producer) instance on GPU0-1 (TP=2):${PREFILL_PORT} ==="
CUDA_VISIBLE_DEVICES=0,1 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=20000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  vllm serve "$MODEL" --port $PREFILL_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --max-model-len 16384 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  > logs/${OUTNAME}_prefill.log 2>&1 &

echo "=== launching decode (kv_consumer) instance on GPU2-3 (TP=2):${DECODE_PORT} ==="
CUDA_VISIBLE_DEVICES=2,3 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=30000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve "$MODEL" --port $DECODE_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 2 \
  --max-model-len 16384 --enforce-eager \
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

echo "=== sanity check: single request through the proxy ==="
curl -s http://127.0.0.1:${PROXY_PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 10, \"temperature\": 0}" \
  | tee logs/${OUTNAME}_sanity.json
echo

echo "=== starting per-GPU-pool power loggers ==="
python3 scripts/pesim/power_logger.py --gpus 0,1 --interval-ms 50 \
  --output logs/${OUTNAME}_power_prefill.csv &
POWER_PREFILL_PID=$!
python3 scripts/pesim/power_logger.py --gpus 2,3 --interval-ms 50 \
  --output logs/${OUTNAME}_power_decode.csv &
POWER_DECODE_PID=$!

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
kill $POWER_PREFILL_PID $POWER_DECODE_PID 2>/dev/null
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null

echo "=== DISAGG_MOE_CALIBRATION_COMPLETE ==="
