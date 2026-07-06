#!/bin/bash
# run_chunk_sweep.sh
# Static chunk size sweep: run vLLM with fixed --max-num-batched-tokens at
# 256 / 512 / 1024 / 4096 tokens to map the TTFT/TPOT tradeoff curve.
#
# The 2048-token baseline (default) is reused from the existing chunk
# experiment results in logs/ (tags: baseline_burstgpt, baseline_sharegpt).
#
# Results saved to logs/ (gitignored). Analyze with:
#   python3 src/analyze_chunk_sweep.py
#
# Usage:
#   bash scripts/run_chunk_sweep.sh
#   NUM_PROMPTS=100 bash scripts/run_chunk_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_PROMPTS=${NUM_PROMPTS:-150}

BURSTGPT=BurstGPT/data/BurstGPT_1.csv
SHAREGPT=data/sharegpt_v3.json
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)

CHUNK_SIZES=(256 4096)

PID_FILE=/tmp/vllm_sweep_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Static Chunk Size Sweep"
echo "  Model    : $MODEL"
echo "  Prompts  : $NUM_PROMPTS  |  max-seqs: $MAX_SEQS"
echo "  Sizes    : ${CHUNK_SIZES[*]} + 2048 (reused from baseline)  [no 512, no 1024]"
echo "  Logs     : $LOG_DIR"
echo "======================================================="

start_server() {
    local chunk=$1
    echo ""
    echo "=== Starting server (max-num-batched-tokens=${chunk}) ==="
    DYNAMIC_CHUNK=0 \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$chunk" \
        > "$LOG_DIR/${DATE}-sweep-server-${chunk}.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-sweep-server-${chunk}.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-sweep-server-${chunk}.log"
    exit 1
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping server (pid=$(cat "$PID_FILE"))..."
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        sleep 8
        rm -f "$PID_FILE"
    fi
}

run_bench() {
    local dataset=$1
    local datapath=$2
    local tag=$3
    echo ""
    echo "--- Benchmarking: ${tag} ---"
    $PYTHON -m vllm.entrypoints.cli.main bench serve \
        --host localhost --port "$PORT" \
        --model "$MODEL" \
        --dataset-name "$dataset" \
        --dataset-path "$datapath" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$LOG_DIR" \
        --metadata "tag=${tag}" \
        2>&1 | tee "$LOG_DIR/${DATE}-bench-${tag}.log"
}

trap stop_server EXIT

for chunk in "${CHUNK_SIZES[@]}"; do
    start_server "$chunk"
    run_bench burstgpt "$BURSTGPT" "sweep_${chunk}_burstgpt"
    run_bench sharegpt "$SHAREGPT" "sweep_${chunk}_sharegpt"
    stop_server
done

echo ""
echo "=== Sweep complete. Results in $LOG_DIR ==="
echo "=== Run: python3 src/analyze_chunk_sweep.py ==="
