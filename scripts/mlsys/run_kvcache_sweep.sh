#!/bin/bash
set -e

PYTHON=${PYTHON:-python3}
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost
MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
VLLM_LOG=$LOG_DIR/${DATE}-kvcache-sweep-vllm.log

NUM_CONVS=${NUM_CONVS:-100}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

DATASET_FULL=$EXPERIMENT_DIR/data/sharegpt_v3.json
DATASET_FALLBACK=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json
if [ -f "$DATASET_FULL" ]; then
  DATASET=$DATASET_FULL
  echo "Dataset: $DATASET_FULL"
elif [ -f "$DATASET_FALLBACK" ]; then
  DATASET=$DATASET_FALLBACK
  echo "WARNING: using BurstGPT preprocessed ShareGPT: $DATASET_FALLBACK"
else
  echo "ERROR: No ShareGPT dataset found. Run setup.sh first."
  exit 1
fi

VLLM_PID=""
MON_PID=""
cleanup() {
  [ -n "$MON_PID" ] && { kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
  [ -n "$VLLM_PID" ] && { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
}
trap cleanup EXIT

wait_for_vllm() {
  echo "Waiting for vLLM on $HOST:$PORT ..."
  local i
  for i in $(seq 1 90); do
    if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
      echo "vLLM is up!"
      return 0
    fi
    if [ "$i" -eq 90 ]; then
      echo "ERROR: vLLM not ready after 180s."
      exit 1
    fi
    printf "."
    sleep 2
  done
}

for UTIL in 0.3 0.5 0.7 0.9; do
  echo ""
  echo "=============================="
  echo "=== gpu-memory-utilization=$UTIL ==="
  echo "=============================="

  GPU_LOG=$LOG_DIR/${DATE}-kvcache-${UTIL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-kvcache-${UTIL}-sharegpt-detail.jsonl

  echo "--- [1] Starting vLLM (--gpu-memory-utilization $UTIL, --enable-prefix-caching) ---"
  $PYTHON -m vllm.entrypoints.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization "$UTIL" \
    --enable-prefix-caching \
    >> "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "vLLM PID: $VLLM_PID"

  wait_for_vllm

  echo "--- [2] Starting GPU monitor ---"
  $PYTHON $EXPERIMENT_DIR/src/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  echo "--- [3] Replaying ${NUM_CONVS} conversations (max ${MAX_TURNS} turns, max_tokens=${MAX_TOKENS}) ---"
  $PYTHON $EXPERIMENT_DIR/src/replay_sharegpt.py \
    --host "$HOST" --port "$PORT" \
    --dataset "$DATASET" \
    --num-convs "$NUM_CONVS" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --output "$DETAIL"

  echo "--- [4] Stopping GPU monitor ---"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  echo "--- [4b] Running per-utilization analysis ---"
  $PYTHON $EXPERIMENT_DIR/src/analyze.py \
    --gpu-log "$GPU_LOG" \
    --trace-log "$DETAIL" \
    --trace-type sharegpt \
    --output "$FINDINGS_DIR/${DATE}-kvcache-${UTIL}.md"

  echo "--- [5] Stopping vLLM ---"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
  echo "Waiting 5s for GPU memory to be released..."
  sleep 5

  echo "=== gpu-memory-utilization=$UTIL complete ==="
done

echo ""
echo "=== Running kvcache summary ==="
$PYTHON $EXPERIMENT_DIR/src/summarize.py --sweep kvcache --date "$DATE" \
  --log-dir "$LOG_DIR" --findings-dir "$FINDINGS_DIR"

echo ""
echo "=== KV cache sweep complete. Findings in $FINDINGS_DIR/ ==="
