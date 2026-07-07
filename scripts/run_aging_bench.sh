#!/bin/bash
# run_aging_bench.sh
# Single condition: PREFIX_REORDER=1 DYNAMIC_CHUNK=1, configurable AGING_THRESHOLD_MS.
# BurstGPT at 8 req/s — tests aging mechanism against starvation at saturation.
# Compare results against base_r8_burstgpt and comb_r8_burstgpt from run_rate_sweep.sh.
#
# Usage:
#   bash scripts/run_aging_bench.sh                          # aging disabled (T=inf)
#   AGING_THRESHOLD_MS=2000 bash scripts/run_aging_bench.sh  # 2s threshold
#   RESULT_TAG=aging_2s_r8_burstgpt AGING_THRESHOLD_MS=2000 bash scripts/run_aging_bench.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_PROMPTS=${NUM_PROMPTS:-150}
AGING_THRESHOLD_MS=${AGING_THRESHOLD_MS:-inf}
RESULT_TAG=${RESULT_TAG:-aging_r8_burstgpt}
BURSTGPT=BurstGPT/data/BurstGPT_1.csv
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_aging_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Aging Benchmark: PREFIX_REORDER=1 DYNAMIC_CHUNK=1"
echo "  AGING_THRESHOLD_MS=${AGING_THRESHOLD_MS}  |  rate=8 req/s  |  BurstGPT"
echo "  tag=${RESULT_TAG}"
echo "======================================================="

echo "=== Starting server ==="
PREFIX_REORDER=1 DYNAMIC_CHUNK=1 AGING_THRESHOLD_MS="${AGING_THRESHOLD_MS}" \
  $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
    > "$LOG_DIR/${DATE}-aging-server.log" 2>&1 &
echo $! > "$PID_FILE"

echo -n "Waiting for startup"
for i in $(seq 1 72); do
    sleep 5
    if grep -q "Application startup complete" "$LOG_DIR/${DATE}-aging-server.log" 2>/dev/null; then
        echo " ready (${i}x5s)"
        break
    fi
    echo -n "."
done

trap 'kill "$(cat $PID_FILE)" 2>/dev/null || true' EXIT

echo "--- Benchmarking: tag=${RESULT_TAG}  rate=8 req/s ---"
$PYTHON -m vllm.entrypoints.cli.main bench serve \
    --host localhost --port "$PORT" \
    --model "$MODEL" \
    --dataset-name burstgpt \
    --dataset-path "$BURSTGPT" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate 8 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$LOG_DIR" \
    --metadata "tag=${RESULT_TAG}" \
    2>&1 | tee "$LOG_DIR/${DATE}-bench-${RESULT_TAG}.log"

echo "=== Done. Analyze with: python3 src/analyze_aging.py ==="
