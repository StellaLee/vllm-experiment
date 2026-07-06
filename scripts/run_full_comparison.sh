#!/bin/bash
# run_full_comparison.sh
# Three-way eviction policy comparison (LRU / TDF / CF) on ShareGPT.
# Filters dataset to conversations with >=4 human turns so every conversation
# reaches turn 4, giving a robust sample for tail metrics.
set -e

PYTHON=${PYTHON:-python3}
MODEL=/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct
PORT=8000
DATE=$(date +%Y-%m-%d)
DIR=/root/vllm-experiment
LOG_DIR=$DIR/logs
FINDINGS_DIR=$DIR/findings
DATASET=$DIR/data/sharegpt_v3.json

NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
CONCURRENCY=${CONCURRENCY:-20}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
TDF_LAMBDA=${TDF_LAMBDA:-0.1}

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

TAG="${DATE}-full"
FINDINGS=$FINDINGS_DIR/${TAG}-eviction-comparison.md

wait_for_vllm() {
  echo "  Waiting for vLLM..."
  for i in $(seq 1 90); do
    if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://localhost:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then echo "  vLLM up"; return 0; fi
    printf "."; sleep 2
  done
  echo "ERROR: vLLM timeout"; exit 1
}

run_policy() {
  local policy=$1
  echo ""
  echo "████ POLICY: $policy ████"
  EVICTION_POLICY=$policy TDF_LAMBDA=$TDF_LAMBDA \
  $PYTHON -m vllm.entrypoints.api_server \
    --model "$MODEL" --port $PORT --dtype auto \
    --max-model-len 4096 --gpu-memory-utilization $GPU_MEM_UTIL \
    --enable-prefix-caching \
    > $LOG_DIR/${TAG}-vllm-${policy}.log 2>&1 &
  VLLM_PID=$!

  wait_for_vllm

  $PYTHON $DIR/src/monitor_gpu.py --output $LOG_DIR/${TAG}-${policy}-gpu.json &
  MON_PID=$!; sleep 1

  $PYTHON $DIR/src/replay_sharegpt.py \
    --host localhost --port $PORT \
    --dataset "$DATASET" \
    --num-convs $NUM_CONVS \
    --max-turns $MAX_TURNS \
    --min-turns $MAX_TURNS \
    --max-tokens $MAX_TOKENS \
    --concurrency $CONCURRENCY \
    --output $LOG_DIR/${TAG}-${policy}.jsonl

  kill $MON_PID 2>/dev/null || true; wait $MON_PID 2>/dev/null || true
  kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true
  sleep 3
}

run_policy lru
run_policy tdf
run_policy cf

echo ""
echo "=== Comparing ==="
$PYTHON $DIR/src/compare_eviction.py \
  --lru  $LOG_DIR/${TAG}-lru.jsonl \
  --tdf  $LOG_DIR/${TAG}-tdf.jsonl \
  --cf   $LOG_DIR/${TAG}-cf.jsonl \
  --output $FINDINGS \
  --lambda-val $TDF_LAMBDA \
  --concurrency $CONCURRENCY \
  --num-convs $NUM_CONVS

echo ""
echo "=== Done: $FINDINGS ==="
cat "$FINDINGS"
