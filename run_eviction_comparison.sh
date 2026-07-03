#!/bin/bash
# run_eviction_comparison.sh
# Compare LRU vs TDF eviction policies on ShareGPT multi-turn conversations.
# Uses Qwen2.5-Coder-7B-Instruct at gpu-memory-utilization=0.7 to trigger evictions.
# Both runs: concurrency=20, 50 conversations, 4 turns max.
set -e

PYTHON=/root/miniconda3/bin/python3
MODEL=/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct
PORT=8000
HOST=localhost
DATE=$(date +%Y-%m-%d)
DIR=/root/vllm-experiment
LOG_DIR=$DIR/logs
FINDINGS_DIR=$DIR/findings
DATASET=$DIR/data/sharegpt_v3.json

NUM_CONVS=${NUM_CONVS:-50}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
CONCURRENCY=${CONCURRENCY:-20}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
TDF_LAMBDA=${TDF_LAMBDA:-0.1}

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

LRU_LOG=$LOG_DIR/${DATE}-eviction-lru.jsonl
TDF_LOG=$LOG_DIR/${DATE}-eviction-tdf.jsonl
LRU_GPU=$LOG_DIR/${DATE}-eviction-lru-gpu.json
TDF_GPU=$LOG_DIR/${DATE}-eviction-tdf-gpu.json
FINDINGS=$FINDINGS_DIR/${DATE}-eviction-comparison.md

# ── helpers ───────────────────────────────────────────────────────────────
wait_for_vllm() {
  echo "  Waiting for vLLM on $HOST:$PORT ..."
  for i in $(seq 1 90); do
    if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
      echo "  vLLM is up!"
      return 0
    fi
    printf "."
    sleep 2
  done
  echo "ERROR: vLLM not ready after 180s"
  exit 1
}

start_vllm() {
  local policy=$1
  echo "=== Starting vLLM [policy=$policy] model=$MODEL gpu_util=$GPU_MEM_UTIL ==="
  TDF_EVICTION=$( [ "$policy" = "tdf" ] && echo "1" || echo "0" ) \
  TDF_LAMBDA=$TDF_LAMBDA \
  $PYTHON -m vllm.entrypoints.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    > "$LOG_DIR/${DATE}-vllm-${policy}.log" 2>&1 &
  VLLM_PID=$!
  echo "  vLLM PID: $VLLM_PID"
}

stop_vllm() {
  if [ -n "${VLLM_PID:-}" ]; then
    echo "  Stopping vLLM (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 3
  fi
}

run_policy() {
  local policy=$1
  local out_log=$2
  local gpu_log=$3

  start_vllm "$policy"
  wait_for_vllm

  echo "  Starting GPU monitor..."
  $PYTHON $DIR/monitor_gpu.py --output "$gpu_log" &
  MON_PID=$!
  sleep 1

  echo "  Replaying ShareGPT [policy=$policy concurrency=$CONCURRENCY convs=$NUM_CONVS turns=$MAX_TURNS]..."
  $PYTHON $DIR/replay_sharegpt.py \
    --host $HOST --port $PORT \
    --dataset "$DATASET" \
    --num-convs "$NUM_CONVS" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --output "$out_log"

  echo "  Stopping GPU monitor..."
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true

  stop_vllm
}

# ── Run LRU (baseline) ────────────────────────────────────────────────────
echo ""
echo "██████████ RUN 1/2: LRU (baseline) ██████████"
VLLM_PID=""
run_policy "lru" "$LRU_LOG" "$LRU_GPU"

echo ""
echo "██████████ RUN 2/2: TDF (lambda=$TDF_LAMBDA) ██████████"
VLLM_PID=""
run_policy "tdf" "$TDF_LOG" "$TDF_GPU"

# ── Compare ───────────────────────────────────────────────────────────────
echo ""
echo "=== Comparing eviction policies ==="
$PYTHON $DIR/compare_eviction.py \
  --lru "$LRU_LOG" \
  --tdf "$TDF_LOG" \
  --output "$FINDINGS" \
  --lambda-val "$TDF_LAMBDA" \
  --concurrency "$CONCURRENCY" \
  --num-convs "$NUM_CONVS"

echo ""
echo "=== Done! Findings: $FINDINGS ==="
cat "$FINDINGS"
