#!/bin/bash
# run_lru_factorial.sh
# Ablation B: 2x2 LRU factorial (reordering x chunk control) at near-saturation.
#
# Runs the 2 LRU conditions missing from run_near_sat_bench.sh:
#   - ns_dyn_*     (LRU | reorder=off | chunk=dynamic)
#   - ns_reorder_* (LRU | reorder=on  | chunk=static)
#
# Combined with ns_base_* and ns_comb_* from run_near_sat_bench.sh, this gives
# all 4 cells of {reorder: off/on} x {chunk: static/dynamic} at near-saturation.
#
# CF eviction conditions are intentionally excluded: the current TDF policy
# uses prefix_depth as a proxy for block importance, which is an insufficiently
# accurate heuristic.  CF eviction is deferred to Phase 2 pending a redesign
# that tracks subtree hit rate directly.  See Open Questions in paper-submission-plan.md.
#
# Tags produced (for both burstgpt and sharegpt):
#   ns_dyn_{dataset}     (LRU | reorder=off | chunk=dynamic)
#   ns_reorder_{dataset} (LRU | reorder=on  | chunk=static)
#
# Usage:
#   bash scripts/run_lru_factorial.sh
#   RATE=3 bash scripts/run_lru_factorial.sh   # match rate used in run_near_sat_bench.sh
#
# Analyze with: python3 src/analyze_factorial.py

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_PROMPTS=${NUM_PROMPTS:-150}
RATE=${RATE:-4}   # must match RATE_BURSTGPT / RATE_SHAREGPT from run_near_sat_bench.sh

BURSTGPT=BurstGPT/data/BurstGPT_1.csv
SHAREGPT=data/sharegpt_v3.json
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_factorial_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Ablation B: 2x2 LRU Factorial (missing 2 conditions)"
echo "  Rate    : ${RATE} req/s  |  Prompts: ${NUM_PROMPTS}"
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
        > "$LOG_DIR/${DATE}-factorial-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" \
                "$LOG_DIR/${DATE}-factorial-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-factorial-${label}-server.log"
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

scrape_counter() {
    curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
        | grep -E "^${1}(\{|[[:space:]])" \
        | awk '{s += $NF} END {print s+0}'
}

run_bench() {
    local dataset=$1 datapath=$2 tag=$3

    echo ""
    echo "--- Benchmarking: tag=${tag}  rate=${RATE} req/s ---"

    local hit_before query_before
    hit_before=$(scrape_counter "vllm:gpu_prefix_cache_hit_count_total")
    query_before=$(scrape_counter "vllm:gpu_prefix_cache_query_count_total")
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
        --request-rate "$RATE" \
        --percentile-metrics ttft,tpot,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$LOG_DIR" \
        --metadata "tag=${tag}" \
        2>&1 | tee "$LOG_DIR/${DATE}-bench-${tag}.log"

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

# ── LRU | reorder=off | chunk=dynamic ─────────────────────────────────────────
start_server "lru-dyn" env PREFIX_REORDER=0 DYNAMIC_CHUNK=1

run_bench burstgpt "$BURSTGPT" "ns_dyn_burstgpt"
run_bench sharegpt "$SHAREGPT"  "ns_dyn_sharegpt"

stop_server

# ── LRU | reorder=on | chunk=static ───────────────────────────────────────────
start_server "lru-reorder" env PREFIX_REORDER=1 DYNAMIC_CHUNK=0

run_bench burstgpt "$BURSTGPT" "ns_reorder_burstgpt"
run_bench sharegpt "$SHAREGPT"  "ns_reorder_sharegpt"

stop_server

echo ""
echo "=== LRU factorial done. ==="
echo "    Analyze with: python3 src/analyze_factorial.py"
echo "    Expected runtime: ~20 min (2 server starts + 4 bench runs)"
echo ""
echo "    Requires ns_base_* and ns_comb_* from run_near_sat_bench.sh for full table."
