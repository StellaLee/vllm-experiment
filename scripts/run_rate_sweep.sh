#!/bin/bash
# run_rate_sweep.sh
# Ablation A: arrival rate sweep for combined condition.
#
# Runs baseline (no reorder, static chunk) and combined (PREFIX_REORDER=1,
# DYNAMIC_CHUNK=1) at Poisson rates 1, 2, 4, 8 req/s on BurstGPT and ShareGPT.
#
# Tags: base_r{rate}_{dataset}  /  comb_r{rate}_{dataset}
# Analyze with: python3 src/analyze_rate_sweep.py

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
PID_FILE=/tmp/vllm_ratesweep_pid

RATES=(1 2 4 8)

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Ablation A: Arrival Rate Sweep"
echo "  Rates   : ${RATES[*]} req/s"
echo "  Model   : $MODEL"
echo "  Prompts : $NUM_PROMPTS  |  max-seqs: $MAX_SEQS"
echo "======================================================="

start_server() {
    local label=$1
    shift
    echo ""
    echo "=== Starting server: ${label} ==="
    "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-ratesweep-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-ratesweep-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-ratesweep-${label}-server.log"
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
    local rate=$4
    echo ""
    echo "--- Benchmarking: tag=${tag}  rate=${rate} req/s ---"
    $PYTHON -m vllm.entrypoints.cli.main bench serve \
        --host localhost --port "$PORT" \
        --model "$MODEL" \
        --dataset-name "$dataset" \
        --dataset-path "$datapath" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" \
        --percentile-metrics ttft,tpot,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$LOG_DIR" \
        --metadata "tag=${tag}" \
        2>&1 | tee "$LOG_DIR/${DATE}-bench-${tag}.log"
}

trap stop_server EXIT

# ── Condition 1: baseline (no reorder, no dynamic chunk) ──────────────────────
start_server "baseline" env PREFIX_REORDER=0 DYNAMIC_CHUNK=0

for rate in "${RATES[@]}"; do
    run_bench burstgpt "$BURSTGPT" "base_r${rate}_burstgpt" "$rate"
    run_bench sharegpt "$SHAREGPT"  "base_r${rate}_sharegpt"  "$rate"
done

stop_server

# ── Condition 2: combined (PREFIX_REORDER=1, DYNAMIC_CHUNK=1) ─────────────────
start_server "combined" env PREFIX_REORDER=1 DYNAMIC_CHUNK=1

for rate in "${RATES[@]}"; do
    run_bench burstgpt "$BURSTGPT" "comb_r${rate}_burstgpt" "$rate"
    run_bench sharegpt "$SHAREGPT"  "comb_r${rate}_sharegpt"  "$rate"
done

stop_server

echo ""
echo "=== Rate sweep done. Run: python3 src/analyze_rate_sweep.py ==="
