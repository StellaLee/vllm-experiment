#!/bin/bash
# run_near_sat_bench.sh
# Phase 1.3: near-saturation headline experiment with KV cache hit rate logging.
#
# Runs baseline (LRU|off|static) and combined (LRU|on|dynamic) at near-saturation
# arrival rates (~80-90% utilization) on both BurstGPT and ShareGPT.  This is the
# paper's primary evaluation point (rate=inf was a thundering-herd artifact).
#
# KV prefix-cache hit rate is scraped from the /metrics endpoint after each run
# and stored in the result JSON alongside the latency numbers.
#
# Tags produced:
#   ns_base_burstgpt,  ns_comb_burstgpt,  ns_aging_burstgpt
#   ns_base_sharegpt,  ns_comb_sharegpt,  ns_aging_sharegpt
#
# Usage:
#   bash scripts/run_near_sat_bench.sh
#   RATE_BURSTGPT=3 RATE_SHAREGPT=5 bash scripts/run_near_sat_bench.sh
#   AGING_THRESHOLD_MS=5000 bash scripts/run_near_sat_bench.sh  # change aging T
#
# Analyze with: python3 src/analyze_near_sat.py
#
# Prerequisites (remote server):
#   patches/reorder/scheduler.patch must be applied to the vLLM installation.
#   Run scripts/apply_patches.sh if not already done.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_PROMPTS=${NUM_PROMPTS:-150}
RATE_BURSTGPT=${RATE_BURSTGPT:-4}
RATE_SHAREGPT=${RATE_SHAREGPT:-4}
AGING_THRESHOLD_MS=${AGING_THRESHOLD_MS:-2000}   # T=2s gave best E2EL at saturation

BURSTGPT=BurstGPT/data/BurstGPT_1.csv
SHAREGPT=data/sharegpt_v3.json
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_nearsat_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Phase 1.3: Near-Saturation Headline Experiment"
echo "  BurstGPT: ${RATE_BURSTGPT} req/s  |  ShareGPT: ${RATE_SHAREGPT} req/s"
echo "  Prompts : ${NUM_PROMPTS}  |  max-seqs: ${MAX_SEQS}  |  aging T=${AGING_THRESHOLD_MS}ms"
echo "======================================================="

# ── helpers ───────────────────────────────────────────────────────────────────

start_server() {
    local label=$1
    shift
    echo ""
    echo "=== Starting server: ${label} ==="
    "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-nearsat-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" \
                "$LOG_DIR/${DATE}-nearsat-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-nearsat-${label}-server.log"
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

# Sum a Prometheus counter across all label combos; always exits 0.
scrape_counter() {
    curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
        | (grep -E "^${1}(\{|[[:space:]])" || true) \
        | awk '{s += $NF} END {print s+0}'
}

run_bench() {
    local dataset=$1 datapath=$2 tag=$3 rate=$4

    echo ""
    echo "--- Benchmarking: tag=${tag}  rate=${rate} req/s ---"

    # snapshot hit/query counters before run (server accumulates across runs)
    local hit_before query_before
    hit_before=$(scrape_counter "vllm:gpu_prefix_cache_hit_count_total")
    query_before=$(scrape_counter "vllm:gpu_prefix_cache_query_count_total")
    # fallback metric names (older vLLM versions)
    if [ "${hit_before}" = "0" ] && [ "${query_before}" = "0" ]; then
        hit_before=$(scrape_counter "vllm:gpu_prefix_cache_hits_total")
        query_before=$(scrape_counter "vllm:gpu_prefix_cache_queries_total")
    fi

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

    # snapshot after and write hit rate into result JSON
    local hit_after query_after
    hit_after=$(scrape_counter "vllm:gpu_prefix_cache_hit_count_total")
    query_after=$(scrape_counter "vllm:gpu_prefix_cache_query_count_total")
    if [ "${hit_after}" = "0" ] && [ "${query_after}" = "0" ]; then
        hit_after=$(scrape_counter "vllm:gpu_prefix_cache_hits_total")
        query_after=$(scrape_counter "vllm:gpu_prefix_cache_queries_total")
    fi

    $PYTHON src/augment_hit_rate.py \
        "$tag" "$LOG_DIR" \
        "$hit_before" "$query_before" \
        "$hit_after" "$query_after"
}

trap stop_server EXIT

# ── Condition 1: baseline (LRU | no reorder | static chunk) ───────────────────
start_server "baseline" env PREFIX_REORDER=0 DYNAMIC_CHUNK=0

run_bench burstgpt "$BURSTGPT" "ns_base_burstgpt" "$RATE_BURSTGPT"
run_bench sharegpt "$SHAREGPT"  "ns_base_sharegpt"  "$RATE_SHAREGPT"

stop_server

# ── Condition 2: combined (LRU | reorder | dynamic chunk) ─────────────────────
start_server "combined" env PREFIX_REORDER=1 DYNAMIC_CHUNK=1

run_bench burstgpt "$BURSTGPT" "ns_comb_burstgpt" "$RATE_BURSTGPT"
run_bench sharegpt "$SHAREGPT"  "ns_comb_sharegpt"  "$RATE_SHAREGPT"

stop_server

# ── Condition 3: combined + aging (LRU | reorder | dynamic chunk | aging T) ───
# At near-saturation (not 2x overload), aging may improve TTFT for cold requests
# without unbounded queue growth.  T=2s was best for E2EL at 8 req/s saturation.
start_server "aging" env PREFIX_REORDER=1 DYNAMIC_CHUNK=1 AGING_THRESHOLD_MS="${AGING_THRESHOLD_MS}"

run_bench burstgpt "$BURSTGPT" "ns_aging_burstgpt" "$RATE_BURSTGPT"
run_bench sharegpt "$SHAREGPT"  "ns_aging_sharegpt"  "$RATE_SHAREGPT"

stop_server

echo ""
echo "=== Near-saturation experiment done. ==="
echo "    Analyze with: python3 src/analyze_near_sat.py"
echo "    Expected runtime: ~18-22 min (3 server starts + 6 bench runs)"
