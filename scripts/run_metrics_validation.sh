#!/bin/bash
# run_metrics_validation.sh
# Cross-check client-side measurements (replay_sharegpt.py) against vLLM /metrics.
# Fresh server so /metrics reflects only this run. rate=4 forces real queue time
# so TTFT/queue/prefill/decode are all exercised. Measurement validation only --
# absolute perf/noise is irrelevant here; we check client vs server agree.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
RATE=${RATE:-4}
NUM_CONVS=${NUM_CONVS:-100}
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_metricsval_pid
mkdir -p "$LOG_DIR"

echo "=== metrics validation: fresh server | rate=${RATE} | ${NUM_CONVS} convs ==="
PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --max-num-seqs 32 --max-num-batched-tokens 2048 \
  > "$LOG_DIR/${DATE}-metricsval-server.log" 2>&1 &
echo $! > "$PID_FILE"

echo -n "waiting for startup"
for i in $(seq 1 72); do
  sleep 5
  if grep -q "Application startup complete" "$LOG_DIR/${DATE}-metricsval-server.log" 2>/dev/null; then
    echo " ready"; break; fi
  echo -n "."
done

CLIENT_OUT="$LOG_DIR/${DATE}-metricsval-client.jsonl"
$PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
  --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
  --max-turns 4 --min-turns 4 --max-tokens 128 --rate "$RATE" --output "$CLIENT_OUT"

# Snapshot metrics BEFORE tearing down the server (box has no curl -> use python).
METRICS_OUT="$LOG_DIR/${DATE}-metricsval-metrics.txt"
$PYTHON -c "import urllib.request; open('$METRICS_OUT','w').write(urllib.request.urlopen('http://localhost:${PORT}/metrics', timeout=15).read().decode())"
echo "metrics snapshot -> $METRICS_OUT ($(wc -l < "$METRICS_OUT") lines)"

kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 6; rm -f "$PID_FILE"

echo ""
echo "=== client vs /metrics comparison ==="
$PYTHON scripts/compare_metrics.py "$CLIENT_OUT" "$METRICS_OUT"
