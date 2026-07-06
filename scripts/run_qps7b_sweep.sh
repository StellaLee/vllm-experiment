#!/bin/bash
set -e

PYTHON=${PYTHON:-python3}
BURSTGPT_BENCH=${BURSTGPT_BENCH:-burstgpt-bench}
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost
MODEL=/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct
SHAREGPT=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json
BURSTGPT_CSV=$EXPERIMENT_DIR/BurstGPT/data/BurstGPT_1.csv
BASE_SCALE=1.2344107085
NUM_REQUESTS=40
VLLM_LOG=$LOG_DIR/${DATE}-qps7b-vllm.log

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

VLLM_PID=""
MON_PID=""
cleanup() {
  [ -n "$MON_PID" ] && { kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
  [ -n "$VLLM_PID" ] && { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
}
trap cleanup EXIT

# ── Helper: read vllm:prefix_cache_hits from /metrics ──────────────────────
read_cache_hits() {
  $PYTHON -c "
import urllib.request, re
try:
    data = urllib.request.urlopen('http://localhost:${PORT}/metrics', timeout=5).read().decode()
    m = re.search(r'^vllm:prefix_cache_hits_total\S*\s+([\d.]+)', data, re.MULTILINE)
    print(m.group(1) if m else '0')
except Exception:
    print('0')
"
}

# ── Start vLLM once; keep it running across all scale runs ─────────────────
echo "=== Starting vLLM (Coder-7B, gpu-memory-utilization=0.9, prefix-caching) ==="
$PYTHON -m vllm.entrypoints.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM..."
for i in $(seq 1 90); do
  if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
    echo " vLLM ready!"
    break
  fi
  [ $i -eq 90 ] && { echo "ERROR: vLLM not ready after 180s"; exit 1; }
  printf "."; sleep 2
done

grep "# GPU blocks:" "$VLLM_LOG" | tail -1 || true
sleep 2

# ── QPS sweep: 1x → 2x → 4x → 8x (same vLLM instance, GPU stays hot) ──────
for entry in "1.0:1x" "2.0:2x" "4.0:4x" "8.0:8x"; do
  MULT=${entry%%:*}
  LABEL=${entry##*:}
  SCALE=$($PYTHON -c "print('%.6f' % ($BASE_SCALE * $MULT))")

  echo ""
  echo "========================================================"
  echo " QPS ${LABEL}  (--scale=${SCALE})"
  echo "========================================================"

  GPU_LOG=$LOG_DIR/${DATE}-qps7b-${LABEL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-qps7b-${LABEL}-burstgpt-detail.jsonl
  BURSLOG=$LOG_DIR/${DATE}-qps7b-${LABEL}-burstgpt.jsonl
  FINDINGS=$FINDINGS_DIR/${DATE}-qps7b-${LABEL}.md

  # Snapshot cumulative cache hits before this run
  HITS_BEFORE=$(read_cache_hits)

  echo "--- Starting GPU monitor ---"
  $PYTHON $EXPERIMENT_DIR/src/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  echo "--- Running BurstGPT (scale=${SCALE}, ${NUM_REQUESTS} requests) ---"
  $BURSTGPT_BENCH \
    --port=$PORT --host=$HOST \
    --temperature=0 --stream \
    --data_path=$SHAREGPT \
    --model_path=$MODEL \
    --surplus_prompts_num=$NUM_REQUESTS --prompt_num=$NUM_REQUESTS \
    --max_tokens=128 \
    --log_path=$BURSLOG \
    --detail_log_path=$DETAIL \
    --use_burstgpt --burstgpt_path=$BURSTGPT_CSV \
    --scale=$SCALE

  echo "--- Stopping GPU monitor ---"
  kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; MON_PID=""

  # Compute per-run cache hit delta and rate
  HITS_AFTER=$(read_cache_hits)
  NREC=$(wc -l < "$DETAIL" 2>/dev/null || echo 0)
  echo "--- KV cache hit rate ---"
  $PYTHON -c "
before = float('$HITS_BEFORE')
after  = float('$HITS_AFTER')
nrec   = int('$NREC')
delta  = after - before
rate   = (delta / nrec * 100) if nrec > 0 else 0.0
print(f'  prefix_cache_hits this run: {int(delta)}  /  {nrec} requests  →  hit rate: {rate:.1f}%')
"

  echo "--- Analyzing ${LABEL} ---"
  $PYTHON $EXPERIMENT_DIR/src/analyze.py \
    --gpu-log "$GPU_LOG" \
    --burstgpt-log "$DETAIL" \
    --output "$FINDINGS"
  cat "$FINDINGS"

  echo "=== ${LABEL} done. Sleeping 5s before next run... ==="
  sleep 5
done

echo ""
echo "=== All QPS 7B sweep runs complete. Findings in $FINDINGS_DIR/ ==="
