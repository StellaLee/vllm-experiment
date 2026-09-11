#!/bin/bash
# SPIKE (throwaway, not part of the peak-shaving gate's production path): probes whether
# J/prefill-token and J/decode-token differ under real P/D disaggregation vs the colocated
# isolation-burst calibration this session already ran (0.068 J/prefill-token, 2.40
# J/decode-token). Motivation: colocated calibration had to manufacture artificial
# pure-prefill/pure-decode bursts because both phases share the same GPU and batch;
# disaggregation gives that isolation structurally for free (GPU0 only ever does prefill
# compute, GPU1 only ever does decode compute, for the SAME natural concurrent workload,
# simultaneously) -- so a single realistic mixed burst, metered per-GPU, replaces the two
# artificial isolation bursts entirely.
#
# Uses vLLM 0.23.0's real P2pNcclConnector (1P1D) + the reference disagg_proxy_demo.py proxy
# (vendored from github.com/vllm-project/vllm main, examples/disaggregated/disaggregated_serving/
# -- GitHub is blocked from THIS box but not from the local machine, so it was fetched locally
# and scp'd up). Runs on GPUs 0/1, which are idle regardless of what's running on GPUs 2-7 (the
# peak-shaving multipolicy demo battery uses GPU_OFFSET=2).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
OUTNAME=disagg_energy_probe

mkdir -p logs
pkill -f "vllm.entrypoints.*--port ${PREFILL_PORT}" 2>/dev/null
pkill -f "vllm.entrypoints.*--port ${DECODE_PORT}" 2>/dev/null
pkill -f "disagg_proxy_demo.py" 2>/dev/null
pkill -f "power_logger.py.*${OUTNAME}" 2>/dev/null
sleep 3

echo "=== launching prefill (producer) instance on GPU0:${PREFILL_PORT} ==="
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PREFILL_PORT --dtype auto --enforce-eager \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_port":21001,"kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_port":"8100"}}' \
  > logs/${OUTNAME}_prefill.log 2>&1 &
PREFILL_PID=$!

echo "=== launching decode (consumer) instance on GPU1:${DECODE_PORT} ==="
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $DECODE_PORT --dtype auto --enforce-eager \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_port":21002,"kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_port":"8200"}}' \
  > logs/${OUTNAME}_decode.log 2>&1 &
DECODE_PID=$!

echo "waiting for both instances to report healthy..."
for i in $(seq 1 120); do
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
export OPENAI_API_KEY=dummy
python3 scripts/eenergy/disagg_p2p_proxy.py \
  --model "$MODEL" --prefill "localhost:${PREFILL_PORT}" --decode "localhost:${DECODE_PORT}" \
  --port $PROXY_PORT \
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
