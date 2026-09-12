#!/bin/bash
# SPIKE (throwaway): is decode-pool power actually sensitive to concurrent decode batch size,
# or closer to a "GPU busy or not" step function? Motivated by the 8-GPU disaggregated gate
# result (findings/2026-09-11-eenergy-disagg-peak-shaving-gate.md): admission gating reduced
# prefill-pool peak power 21.3% but decode-pool peak power only 0.9%, despite closed-loop
# concurrency meaning deferred admissions really do reduce the number of concurrently
# in-flight decode streams. If decode power barely depends on concurrency, that's the real
# explanation (not just "gate can't touch in-flight streams") -- and it means NO admission-
# side mechanism (shedding, concurrency caps) would meaningfully shave decode power either;
# the only lever would be idling the decode instance entirely.
#
# Reuses the already-validated 1-prefill+1-decode NixlConnector pair
# (run_disagg_prefill_decode_calibration.sh's infra) rather than the full 4:4 topology --
# this question is about one decode GPU's own power-vs-concurrency relationship, which should
# generalize to the pool. Three concurrency phases run sequentially against the SAME running
# pair, one continuous power trace, phase boundaries timestamped for post-hoc slicing --
# short, non-whale prompts with a fixed 512-token cap so prefill cost stays small/consistent
# and decode dominates.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
OUTNAME=decode_concurrency_power_sweep
IDLE_SETTLE_S=20

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

echo "=== launching Nixl proxy on :${PROXY_PORT} ==="
python3 scripts/eenergy/nixl_toy_proxy_server.py --port $PROXY_PORT \
  --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost --decoder-ports $DECODE_PORT \
  > logs/${OUTNAME}_proxy.log 2>&1 &
sleep 5

echo "=== starting decode-GPU (GPU1) power logger -- one continuous trace across all phases ==="
python3 scripts/pesim/power_logger.py --gpus 1 --interval-ms 50 \
  --output logs/${OUTNAME}_power_decode.csv &
POWER_PID=$!

sleep $IDLE_SETTLE_S

run_phase () {
  local PHASE=$1
  local CONCURRENCY=$2
  local NUM_CONVS=$3
  echo "phase_${PHASE}_start_epoch=$(date +%s.%N)" | tee -a logs/${OUTNAME}_timing.txt
  echo "=== phase ${PHASE}: concurrency=${CONCURRENCY} num_convs=${NUM_CONVS} ==="
  timeout 600 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port $PROXY_PORT --model "$MODEL" \
    --dataset data/sharegpt_v3.json --request-timeout 120 \
    --output logs/${OUTNAME}_records_${PHASE}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency $CONCURRENCY --num-convs $NUM_CONVS \
    --max-tokens 512 --whale-frac 0.0 \
    2>&1 | tee logs/${OUTNAME}_harness_${PHASE}.log
  echo "phase_${PHASE}_end_epoch=$(date +%s.%N)" | tee -a logs/${OUTNAME}_timing.txt
  sleep 10  # brief gap so phases are cleanly separable in the power trace
}

run_phase c02 2 30
run_phase c08 8 80
run_phase c32 32 200

echo "=== tearing down ==="
kill $POWER_PID 2>/dev/null
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null

echo "=== DECODE_CONCURRENCY_POWER_SWEEP_COMPLETE ==="
