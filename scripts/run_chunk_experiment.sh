#!/bin/bash
# run_chunk_experiment.sh
# Compare static vs dynamic chunked-prefill sizing on BurstGPT and ShareGPT.
#
# Runs 4 benchmarks back-to-back:
#   baseline_burstgpt  — DYNAMIC_CHUNK=0, BurstGPT trace
#   baseline_sharegpt  — DYNAMIC_CHUNK=0, ShareGPT
#   dynamic_burstgpt   — DYNAMIC_CHUNK=1, BurstGPT trace
#   dynamic_sharegpt   — DYNAMIC_CHUNK=1, ShareGPT
#
# Results saved to findings/chunk_<timestamp>/ as JSON + markdown summary.
#
# Usage:
#   bash run_chunk_experiment.sh
#   NUM_PROMPTS=200 bash run_chunk_experiment.sh
set -euo pipefail

MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_PROMPTS=${NUM_PROMPTS:-150}
DYNAMIC_CHUNK_TARGET=${DYNAMIC_CHUNK_TARGET:-8}
DYNAMIC_CHUNK_MIN=${DYNAMIC_CHUNK_MIN:-256}

DATE=$(date +%Y%m%d_%H%M%S)
RESULTS=findings/chunk_${DATE}
BURSTGPT=BurstGPT/data/BurstGPT_1.csv
SHAREGPT=data/sharegpt_v3.json
VLLM=${PYTHON:-python3}
PID_FILE=/tmp/vllm_chunk_pid

mkdir -p "$RESULTS"
echo "======================================================="
echo "  Dynamic Chunk Size Experiment"
echo "  Model    : $MODEL"
echo "  Prompts  : $NUM_PROMPTS  |  max-seqs: $MAX_SEQS"
echo "  Results  : $RESULTS"
echo "======================================================="

start_server() {
    local dynamic=$1
    echo ""
    echo "=== Starting server (DYNAMIC_CHUNK=${dynamic}) ==="
    DYNAMIC_CHUNK=$dynamic \
    DYNAMIC_CHUNK_TARGET=$DYNAMIC_CHUNK_TARGET \
    DYNAMIC_CHUNK_MIN=$DYNAMIC_CHUNK_MIN \
        $VLLM -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --max-num-seqs $MAX_SEQS \
        > "$RESULTS/server_dynchunk${dynamic}.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$RESULTS/server_dynchunk${dynamic}.log" 2>/dev/null; then
            echo " ready (${i}x5s elapsed)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT after 6 min"
    tail -20 "$RESULTS/server_dynchunk${dynamic}.log"
    exit 1
}

stop_server() {
    echo "Stopping server (pid=$(cat $PID_FILE))..."
    kill "$(cat $PID_FILE)" 2>/dev/null || true
    sleep 8
}

run_bench() {
    local dataset=$1
    local datapath=$2
    local tag=$3
    echo ""
    echo "--- Benchmarking: ${tag} ---"
    $VLLM -m vllm.entrypoints.cli.main bench serve \
        --host localhost --port $PORT \
        --model "$MODEL" \
        --dataset-name "$dataset" \
        --dataset-path "$datapath" \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --percentile-metrics ttft,tpot,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$RESULTS" \
        --metadata "tag=${tag}" \
        2>&1 | tee "$RESULTS/bench_${tag}.log"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
start_server 0
run_bench burstgpt "$BURSTGPT" baseline_burstgpt
run_bench sharegpt "$SHAREGPT" baseline_sharegpt
stop_server

# ── Dynamic ───────────────────────────────────────────────────────────────────
start_server 1
run_bench burstgpt "$BURSTGPT" dynamic_burstgpt
run_bench sharegpt "$SHAREGPT" dynamic_sharegpt
stop_server

echo ""
echo "=== All runs complete. Running analysis... ==="
$VLLM analyze_chunk.py "$RESULTS"
