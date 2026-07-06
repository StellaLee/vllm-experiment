#!/bin/bash
# run_reorder_experiment.sh
# Compare baseline vs PREFIX_REORDER=1 on BurstGPT and ShareGPT.
# Tags: reorder_off_burstgpt, reorder_off_sharegpt,
#       reorder_on_burstgpt,  reorder_on_sharegpt

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
PID_FILE=/tmp/vllm_reorder_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Prefix-Aware Reordering Experiment"
echo "  Model   : $MODEL"
echo "  Prompts : $NUM_PROMPTS  |  max-seqs: $MAX_SEQS"
echo "======================================================="

start_server() {
    local reorder=$1
    echo ""
    echo "=== Starting server (PREFIX_REORDER=${reorder}) ==="
    PREFIX_REORDER=$reorder \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-reorder-server-${reorder}.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-reorder-server-${reorder}.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-reorder-server-${reorder}.log"
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

# Baseline: PREFIX_REORDER=0
start_server 0
run_bench burstgpt "$BURSTGPT" reorder_off_burstgpt
run_bench sharegpt "$SHAREGPT" reorder_off_sharegpt
stop_server

# Reordering: PREFIX_REORDER=1
start_server 1
run_bench burstgpt "$BURSTGPT" reorder_on_burstgpt
run_bench sharegpt "$SHAREGPT" reorder_on_sharegpt
stop_server

echo ""
echo "=== Done. Run: python3 src/analyze_reorder.py ==="
