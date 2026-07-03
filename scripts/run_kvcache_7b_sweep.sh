#!/bin/bash
set -e

PYTHON=/root/miniconda3/bin/python3
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost
MODEL=/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct
VLLM_LOG=$LOG_DIR/${DATE}-kvcache7b-sweep-vllm.log

NUM_CONVS=${NUM_CONVS:-100}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-256}
CONCURRENCY=${CONCURRENCY:-20}

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

DATASET_FULL=$EXPERIMENT_DIR/data/sharegpt_v3.json
DATASET_FALLBACK=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json
if [ -f "$DATASET_FULL" ]; then
  DATASET=$DATASET_FULL
elif [ -f "$DATASET_FALLBACK" ]; then
  DATASET=$DATASET_FALLBACK
else
  echo "ERROR: No ShareGPT dataset found."
  exit 1
fi

echo "Config: model=Coder-7B  concurrency=$CONCURRENCY  max_tokens=$MAX_TOKENS  max_turns=$MAX_TURNS  num_convs=$NUM_CONVS"

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
  for i in $(seq 1 120); do
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
    # Fail fast if vLLM process died
    if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM process $VLLM_PID died. Check $VLLM_LOG"
      tail -20 "$VLLM_LOG"
      exit 1
    fi
    if [ "$i" -eq 120 ]; then
      echo "ERROR: vLLM not ready after 240s."
      exit 1
    fi
    printf "."
    sleep 2
  done
}

check_oom() {
  if grep -q "CUDA out of memory\|OutOfMemoryError" "$VLLM_LOG" 2>/dev/null; then
    echo "EARLY STOP: OOM detected in vLLM log for util=$1"
    tail -10 "$VLLM_LOG"
    exit 1
  fi
}

for UTIL in 0.7 0.8 0.9; do
  echo ""
  echo "=============================="
  echo "=== gpu-memory-utilization=$UTIL ==="
  echo "=============================="

  GPU_LOG=$LOG_DIR/${DATE}-kvcache7b-${UTIL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-kvcache7b-${UTIL}-sharegpt-detail.jsonl

  echo "--- [1] Starting vLLM (--gpu-memory-utilization $UTIL) ---"
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
  check_oom "$UTIL"

  # Report how many KV blocks vLLM allocated
  GPU_BLOCKS=$(grep -o "# GPU blocks: [0-9]*" "$VLLM_LOG" | tail -1 || echo "unknown")
  echo "KV cache: $GPU_BLOCKS (16 tokens each)"

  echo "--- [2] Starting GPU monitor ---"
  $PYTHON $EXPERIMENT_DIR/src/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  echo "--- [3] Replaying ${NUM_CONVS} conversations (concurrency=${CONCURRENCY}, max_turns=${MAX_TURNS}, max_tokens=${MAX_TOKENS}) ---"
  $PYTHON $EXPERIMENT_DIR/src/replay_sharegpt.py \
    --host "$HOST" --port "$PORT" \
    --dataset "$DATASET" \
    --num-convs "$NUM_CONVS" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --output "$DETAIL"

  # Sanity check: did we get records?
  NREC=$(wc -l < "$DETAIL" 2>/dev/null || echo 0)
  echo "Records written: $NREC"
  if [ "$NREC" -lt 10 ]; then
    echo "EARLY STOP: too few records ($NREC) for util=$UTIL — possible crash or timeout."
    exit 1
  fi

  # Check for suspiciously high median TTFT (>10s = something broken)
  BAD=$($PYTHON -c "
import json, statistics
rows = [json.loads(l) for l in open('$DETAIL')]
ttfts = [r['ttft'] for r in rows if r.get('ttft') is not None]
if not ttfts: print('no_ttft'); exit()
med = statistics.median(ttfts)
print('ok' if med < 10 else f'HIGH:{med:.1f}s')
" 2>/dev/null || echo "parse_error")
  echo "TTFT check: $BAD"
  if [[ "$BAD" == HIGH* ]] || [[ "$BAD" == "no_ttft" ]] || [[ "$BAD" == "parse_error" ]]; then
    echo "EARLY STOP: suspicious TTFT ($BAD) at util=$UTIL"
    exit 1
  fi

  echo "--- [4] Stopping GPU monitor ---"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  echo "--- [5] Running per-utilization analysis ---"
  $PYTHON $EXPERIMENT_DIR/src/analyze.py \
    --gpu-log "$GPU_LOG" \
    --trace-log "$DETAIL" \
    --trace-type sharegpt \
    --output "$FINDINGS_DIR/${DATE}-kvcache7b-${UTIL}.md"
  cat "$FINDINGS_DIR/${DATE}-kvcache7b-${UTIL}.md"

  echo "--- [6] Stopping vLLM ---"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
  echo "Waiting 8s for GPU memory to be released..."
  sleep 8

  echo "=== gpu-memory-utilization=$UTIL complete ==="
done

echo ""
echo "=== All runs complete. ==="
