#!/bin/bash
set -e
PYTHON=/root/miniconda3/bin/python3
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/experiment
BURSTGPT_DIR=$EXPERIMENT_DIR/BurstGPT
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

GPU_LOG=$LOG_DIR/${DATE}-gpu.json
BURSTGPT_LOG=$LOG_DIR/${DATE}-burstgpt.json
BURSTGPT_DETAIL=$LOG_DIR/${DATE}-burstgpt-detail.json
FINDINGS=$FINDINGS_DIR/${DATE}-baseline-qwen2.5-0.5b.md

echo "=== [1/5] Waiting for vLLM on $HOST:$PORT ==="
for i in $(seq 1 60); do
  if wget -q --spider "http://$HOST:$PORT/health" 2>/dev/null; then
    echo "vLLM is up!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: vLLM not ready after 60s. Run start_server.sh in another terminal."
    exit 1
  fi
  printf "."
  sleep 2
done

echo "=== [2/5] Starting GPU monitor ==="
$PYTHON /root/experiment/monitor_gpu.py --output "$GPU_LOG" &
MON_PID=$!
echo "Monitor PID: $MON_PID"
sleep 1

echo "=== [3/5] Running BurstGPT profiler (50 requests, streaming) ==="
SHAREGPT=$BURSTGPT_DIR/example/preprocess_data/shareGPT.json
BURSTGPT_CSV=$BURSTGPT_DIR/data/BurstGPT_1.csv

FLAGS="--port=$PORT --host=$HOST --temperature=0 --stream"
FLAGS="$FLAGS --data_path=$SHAREGPT"
FLAGS="$FLAGS --model_path=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct"
FLAGS="$FLAGS --surplus_prompts_num=50 --prompt_num=50"
FLAGS="$FLAGS --max_tokens=128"
FLAGS="$FLAGS --log_path=$BURSTGPT_LOG"
FLAGS="$FLAGS --detail_log_path=$BURSTGPT_DETAIL"

if [ -f "$BURSTGPT_CSV" ]; then
  echo "Using BurstGPT trace: $BURSTGPT_CSV"
  FLAGS="$FLAGS --use_burstgpt --burstgpt_path=$BURSTGPT_CSV --scale=1.2344107085"
else
  echo "BurstGPT_1.csv not found. Using Poisson distribution at QPS=1."
  FLAGS="$FLAGS --qps=1.0"
fi

cd $BURSTGPT_DIR/example
$PYTHON profile_vllm_server.py $FLAGS
cd -

echo "=== [4/5] Stopping GPU monitor ==="
kill $MON_PID 2>/dev/null || true
wait $MON_PID 2>/dev/null || true

echo "=== [5/5] Running analysis ==="
$PYTHON /root/experiment/analyze.py \
  --gpu-log "$GPU_LOG" \
  --burstgpt-log "$BURSTGPT_DETAIL" \
  --output "$FINDINGS"

echo ""
echo "=== Done! Findings: $FINDINGS ==="
cat "$FINDINGS"
