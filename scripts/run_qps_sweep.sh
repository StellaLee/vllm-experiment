#!/bin/bash
set -e

PYTHON=/root/miniconda3/bin/python3
BURSTGPT_BENCH=/root/miniconda3/bin/burstgpt-bench
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost
MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
SHAREGPT=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json
BURSTGPT_CSV=$EXPERIMENT_DIR/BurstGPT/data/BurstGPT_1.csv
BASE_SCALE=1.2344107085

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

# ===== Start vLLM once for all sweep runs =====
echo "=== Starting vLLM server (gpu-memory-utilization=0.9) ==="
$PYTHON -m vllm.entrypoints.api_server \
  --model $MODEL --port $PORT --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  > $LOG_DIR/${DATE}-qps-sweep-vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MON_PID=""
cleanup() {
  [ -n "$MON_PID" ] && { kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM..."
for i in $(seq 1 90); do
  if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
    echo " vLLM ready."
    break
  fi
  [ $i -eq 90 ] && { echo "ERROR: vLLM not ready after 180s"; exit 1; }
  printf "."; sleep 2
done
sleep 2  # let KV cache settle

# ===== Sweep: 0.5x, 2x, 4x relative to base scale =====
for MULT in 0.5 2.0 4.0; do
  SCALE=$($PYTHON -c "print('{:.6f}'.format($BASE_SCALE * $MULT))")
  # "2.0" → "2x", "0.5" → "0.5x", "4.0" → "4x"
  LABEL=$(echo "$MULT" | awk '{v=$1+0; if(v==int(v)) printf "%dx",int(v); else printf "%sx",v}')

  echo ""
  echo "========================================================"
  echo " QPS ${LABEL}  |  --scale=${SCALE}"
  echo "========================================================"

  GPU_LOG=$LOG_DIR/${DATE}-qps-${LABEL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt-detail.jsonl
  BURSLOG=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt.jsonl
  FINDINGS=$FINDINGS_DIR/${DATE}-qps-${LABEL}.md

  echo "--- Starting GPU monitor ---"
  $PYTHON $EXPERIMENT_DIR/src/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  echo "--- Running BurstGPT (scale=${SCALE}) ---"
  $BURSTGPT_BENCH \
    --port=$PORT --host=$HOST \
    --temperature=0 --stream \
    --data_path=$SHAREGPT \
    --model_path=$MODEL \
    --surplus_prompts_num=50 --prompt_num=50 \
    --max_tokens=128 \
    --log_path=$BURSLOG \
    --detail_log_path=$DETAIL \
    --use_burstgpt --burstgpt_path=$BURSTGPT_CSV \
    --scale=$SCALE

  echo "--- Stopping GPU monitor ---"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  echo "--- Analyzing ${LABEL} ---"
  $PYTHON $EXPERIMENT_DIR/src/analyze.py \
    --gpu-log "$GPU_LOG" \
    --burstgpt-log "$DETAIL" \
    --output "$FINDINGS"

  echo "=== ${LABEL} done ==="
  echo ""
  sleep 8  # cooldown between runs
done

# ===== Comparison Summary =====
echo "=== Generating comparison summary ==="
$PYTHON $EXPERIMENT_DIR/src/summarize.py --sweep qps --date "$DATE" \
  --log-dir "$LOG_DIR" --findings-dir "$FINDINGS_DIR"

echo ""
echo "=== QPS sweep complete! ==="
