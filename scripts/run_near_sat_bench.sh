#!/bin/bash
# run_near_sat_bench.sh
# Phase 1.3: near-saturation headline experiment with KV cache hit rate logging.
#
# Runs baseline / combined / aging at near-saturation arrival rates (~85% GPU util)
# on BurstGPT only. ShareGPT is excluded: the vLLM bench client randomly shuffles
# conversations, destroying prefix locality and making the dataset useless for
# testing prefix-aware scheduling. Use run_multiturn_bench.sh for ShareGPT.
#
# KV prefix-cache hit rate is scraped from the /metrics endpoint after each run
# and stored in the result JSON alongside the latency numbers.
#
# Tags produced:
#   ns_base_r<N>[_t<T>]_burstgpt
#   ns_comb_r<N>[_t<T>]_burstgpt
#   ns_aging_r<N>[_t<T>]_burstgpt
#
# Usage:
#   env RATE_BURSTGPT=3 bash scripts/run_near_sat_bench.sh
#   env RATE_BURSTGPT=3 TRIAL=1 bash scripts/run_near_sat_bench.sh  # repeated trial
#   env RATE_BURSTGPT=3 NUM_PROMPTS=500 bash scripts/run_near_sat_bench.sh
#   env AGING_THRESHOLD_MS=5000 bash scripts/run_near_sat_bench.sh
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
RATE_BURSTGPT=${RATE_BURSTGPT:-3}
AGING_THRESHOLD_MS=${AGING_THRESHOLD_MS:-2000}   # T=2s gave best E2EL at saturation
# Rate suffix for tags: dots replaced with 'p' (e.g. 2.5 -> r2p5, 3 -> r3)
# Optional TRIAL env var appends _t<N> for repeated trials (e.g. TRIAL=1 -> r3_t1)
RTAG_B="r$(echo "$RATE_BURSTGPT" | tr '.' 'p')${TRIAL:+_t${TRIAL}}"

BURSTGPT=BurstGPT/data/BurstGPT_1.csv
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_nearsat_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Phase 1.3: Near-Saturation Headline Experiment"
echo "  BurstGPT: ${RATE_BURSTGPT} req/s  |  trial: ${TRIAL:-<none>}"
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
# Each stage is wrapped in || true so pipefail never triggers.
scrape_counter() {
    (curl -s "http://localhost:${PORT}/metrics" 2>/dev/null || true) \
        | (grep -E "^${1}(\{|[[:space:]])" || true) \
        | awk '{s += $NF} END {print s+0}'
}

run_bench() {
    local dataset=$1 datapath=$2 tag=$3 rate=$4
    local gpu_csv="/tmp/gpu_mon_${tag}.csv"

    echo ""
    echo "--- Benchmarking: tag=${tag}  rate=${rate} req/s ---"

    # start GPU utilization monitor (1 sample/sec)
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits --loop=1 \
        > "$gpu_csv" 2>/dev/null &
    local gpu_mon_pid=$!

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

    # stop GPU monitor
    kill "$gpu_mon_pid" 2>/dev/null || true
    wait "$gpu_mon_pid" 2>/dev/null || true
    $PYTHON src/augment_gpu_util.py "$tag" "$LOG_DIR" "$gpu_csv"

    # scrape KV hit rate from live /metrics endpoint (Python, no curl needed)
    $PYTHON src/augment_hit_rate.py "$tag" "$LOG_DIR" "$PORT"
}

trap stop_server EXIT

# ── Condition 1: baseline (LRU | no reorder | static chunk) ───────────────────
start_server "baseline" env PREFIX_REORDER=0 DYNAMIC_CHUNK=0

run_bench burstgpt "$BURSTGPT" "ns_base_${RTAG_B}_burstgpt" "$RATE_BURSTGPT"

stop_server

# ── Condition 2: combined (LRU | reorder | dynamic chunk) ─────────────────────
start_server "combined" env PREFIX_REORDER=1 DYNAMIC_CHUNK=1

run_bench burstgpt "$BURSTGPT" "ns_comb_${RTAG_B}_burstgpt" "$RATE_BURSTGPT"

stop_server

# ── Condition 3: combined + aging (LRU | reorder | dynamic chunk | aging T) ───
start_server "aging" env PREFIX_REORDER=1 DYNAMIC_CHUNK=1 AGING_THRESHOLD_MS="${AGING_THRESHOLD_MS}"

run_bench burstgpt "$BURSTGPT" "ns_aging_${RTAG_B}_burstgpt" "$RATE_BURSTGPT"

stop_server

echo ""
echo "=== Near-saturation experiment done. ==="
echo "    Analyze with: python3 src/analyze_near_sat.py"
echo "    Expected runtime: ~10-12 min (3 server starts + 3 bench runs)"
