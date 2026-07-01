#!/bin/bash
set -e
PYTHON=/root/miniconda3/bin/python3
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost

# Number of conversations and turns (override via env)
NUM_CONVS=${NUM_CONVS:-50}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

GPU_LOG=$LOG_DIR/${DATE}-sharegpt-gpu.json
SHAREGPT_LOG=$LOG_DIR/${DATE}-sharegpt-detail.jsonl
FINDINGS=$FINDINGS_DIR/${DATE}-sharegpt-qwen2.5-0.5b.md

# Prefer the full multi-turn dataset; fall back to BurstGPT's preprocessed copy
DATASET_FULL=$EXPERIMENT_DIR/data/sharegpt_v3.json
DATASET_FALLBACK=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json

if [ -f "$DATASET_FULL" ]; then
  DATASET=$DATASET_FULL
  echo "Using full ShareGPT dataset: $DATASET_FULL"
elif [ -f "$DATASET_FALLBACK" ]; then
  DATASET=$DATASET_FALLBACK
  echo "WARNING: full dataset not found; using BurstGPT preprocessed ShareGPT: $DATASET_FALLBACK"
else
  echo "ERROR: No ShareGPT dataset found. Run setup.sh first."
  exit 1
fi

MON_PID=""
cleanup() {
  if [ -n "$MON_PID" ]; then
    echo "Stopping GPU monitor (PID $MON_PID)..."
    kill "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== [1/5] Waiting for vLLM on $HOST:$PORT ==="
for i in $(seq 1 60); do
  if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
    echo "vLLM is up!"
    break
  fi
  if [ $i -eq 60 ]; then echo "ERROR: vLLM not ready after 120s."; exit 1; fi
  printf "."
  sleep 2
done

echo "=== [2/5] Starting GPU monitor ==="
$PYTHON $EXPERIMENT_DIR/monitor_gpu.py --output "$GPU_LOG" &
MON_PID=$!
echo "Monitor PID: $MON_PID"
sleep 1

echo "=== [3/5] Replaying ShareGPT conversations (${NUM_CONVS} convs, max ${MAX_TURNS} turns) ==="
$PYTHON $EXPERIMENT_DIR/replay_sharegpt.py \
  --host $HOST --port $PORT \
  --dataset "$DATASET" \
  --num-convs $NUM_CONVS \
  --max-turns $MAX_TURNS \
  --max-tokens $MAX_TOKENS \
  --output "$SHAREGPT_LOG"

echo "=== [4/5] Stopping GPU monitor ==="
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
MON_PID=""

echo "=== [5/5] Running analysis ==="
$PYTHON $EXPERIMENT_DIR/analyze.py \
  --gpu-log "$GPU_LOG" \
  --trace-log "$SHAREGPT_LOG" \
  --trace-type sharegpt \
  --output "$FINDINGS"

echo ""
echo "=== Done! Findings: $FINDINGS ==="
cat "$FINDINGS"
