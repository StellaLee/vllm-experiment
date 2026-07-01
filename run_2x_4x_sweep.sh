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
TARGET_REQUESTS=45

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

# ===== Start vLLM =====
echo "=== Starting vLLM server ==="
$PYTHON -m vllm.entrypoints.api_server \
  --model $MODEL --port $PORT --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  > $LOG_DIR/${DATE}-2x4x-vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MON_PID=""
BENCH_PID=""
cleanup() {
  [ -n "$BENCH_PID" ] && { kill "$BENCH_PID" 2>/dev/null || true; }
  [ -n "$MON_PID"   ] && { kill "$MON_PID"   2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
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
  [ $i -eq 90 ] && { echo "ERROR: vLLM not ready"; exit 1; }
  printf "."; sleep 2
done
sleep 2

# ===== Sweep 2x and 4x =====
for MULT in 2.0 4.0; do
  SCALE=$($PYTHON -c "print('{:.6f}'.format($BASE_SCALE * $MULT))")
  LABEL=$(echo "$MULT" | awk '{v=$1+0; if(v==int(v)) printf "%dx",int(v); else printf "%sx",v}')

  echo ""
  echo "========================================================"
  echo " QPS ${LABEL}  |  --scale=${SCALE}  |  stopping at ${TARGET_REQUESTS} requests"
  echo "========================================================"

  GPU_LOG=$LOG_DIR/${DATE}-qps-${LABEL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt-detail.jsonl
  BURSLOG=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt.jsonl
  FINDINGS=$FINDINGS_DIR/${DATE}-qps-${LABEL}.md

  # Start GPU monitor
  $PYTHON $EXPERIMENT_DIR/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  # Start BurstGPT in background
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
    --scale=$SCALE &
  BENCH_PID=$!
  echo "burstgpt-bench PID: $BENCH_PID"

  # Wait until 45 requests complete or bench finishes naturally
  echo "Waiting for ${TARGET_REQUESTS} requests..."
  for i in $(seq 1 600); do
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "burstgpt-bench finished naturally."
      BENCH_PID=""
      break
    fi
    count=$(wc -l < "$DETAIL" 2>/dev/null || echo 0)
    if [ "$count" -ge "$TARGET_REQUESTS" ]; then
      echo "${count} requests reached — killing burstgpt-bench"
      kill "$BENCH_PID" 2>/dev/null || true
      wait "$BENCH_PID" 2>/dev/null || true
      BENCH_PID=""
      break
    fi
    [ $((i % 10)) -eq 0 ] && echo "  ... ${count}/${TARGET_REQUESTS} requests"
    sleep 2
  done

  # Stop GPU monitor
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  # Analyze
  echo "--- Analyzing ${LABEL} ---"
  $PYTHON $EXPERIMENT_DIR/analyze.py \
    --gpu-log "$GPU_LOG" \
    --burstgpt-log "$DETAIL" \
    --output "$FINDINGS"

  echo "=== ${LABEL} done ==="
  sleep 5
done

# ===== Comparison Summary =====
echo "=== Generating comparison summary ==="
$PYTHON $EXPERIMENT_DIR/summarize.py --sweep qps --date "$DATE" \
  --log-dir "$LOG_DIR" --findings-dir "$FINDINGS_DIR"

echo "=== Sweep complete! ==="
