#!/bin/bash
# SPIKE (throwaway): P/D disaggregation energy probe, take 2 -- switches from
# P2pNcclConnector (undocumented/legacy, not mentioned anywhere in vLLM's own disagg_prefill.md
# docs; got a single request working via manual request_id address-tagging, but a second
# request reproducibly hung on a cache-key mismatch inside the connector's own bookkeeping,
# confirmed via live py-spy thread dumps on both sides) to NixlConnector -- the connector
# vLLM's own docs actually lead with, and which has a maintained CI reference
# (tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh + toy_proxy_server.py, vendored
# here as scripts/eenergy/nixl_toy_proxy_server.py) instead of a hand-rolled request-id
# convention. `nixl` installs cleanly via pip (prebuilt wheel, no separate UCX system install
# needed) -- see `python3 -m pip install nixl` in this venv.
#
# Config mirrors run_accuracy_test.sh's single-prefill/single-decode (1P1D) case, adapted to
# this project's model/GPUs: prefill on GPU0, decode on GPU1 (both idle regardless of the
# unrelated peak-shaving battery on GPUs 2-7), --enforce-eager, --gpu-memory-utilization 0.85
# (bumped up from the reference's 0.2, which was tuned for reference's tiny 0.6B smoke-test
# model -- our 7B model needs ~14.3 GiB per earlier logs).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
OUTNAME=disagg_energy_probe_nixl

mkdir -p logs
pkill -f "toy_proxy_server.py" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
sleep 3

echo "=== launching prefill (kv_producer) instance on GPU0:${PREFILL_PORT} ==="
CUDA_VISIBLE_DEVICES=0 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=20000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  vllm serve "$MODEL" --port $PREFILL_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  > logs/${OUTNAME}_prefill.log 2>&1 &
PREFILL_PID=$!

echo "=== launching decode (kv_consumer) instance on GPU1:${DECODE_PORT} ==="
CUDA_VISIBLE_DEVICES=1 VLLM_KV_CACHE_LAYOUT=HND VLLM_PORT=30000 UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve "$MODEL" --port $DECODE_PORT --block-size 128 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  > logs/${OUTNAME}_decode.log 2>&1 &
DECODE_PID=$!

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

echo "=== launching toy_proxy_server.py on :${PROXY_PORT} ==="
python3 scripts/eenergy/nixl_toy_proxy_server.py --port $PROXY_PORT \
  --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost --decoder-ports $DECODE_PORT \
  > logs/${OUTNAME}_proxy.log 2>&1 &
PROXY_PID=$!
sleep 5

echo "=== sanity check: single request through the proxy ==="
curl -s http://127.0.0.1:${PROXY_PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 10, \"temperature\": 0}" \
  | tee logs/${OUTNAME}_sanity.json
echo

echo "=== PIDs: prefill=$PREFILL_PID decode=$DECODE_PID proxy=$PROXY_PID ==="
echo "Leaving instances running for inspection -- tear down manually with:"
echo "  kill $PREFILL_PID $DECODE_PID $PROXY_PID"
