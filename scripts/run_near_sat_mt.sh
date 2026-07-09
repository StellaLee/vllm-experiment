#!/bin/bash
# run_near_sat_mt.sh
# Near-saturation open-loop multi-turn experiment.
#
# Runs baseline and combined (soft aging + hysteresis) at a Poisson arrival
# rate against the ShareGPT multi-turn replay harness. Unlike the vLLM bench
# client (which sends single-turn requests with near-zero prefix reuse),
# this replay sends accumulated conversation history so each turn shares the
# prefix of all prior turns -- the regime where our scheduler changes matter.
#
# RATE controls conversation arrivals/sec (Poisson). Start with 4 and adjust:
#   - GPU util < 60%: too low, increase RATE
#   - GPU util > 95%: too high, decrease RATE
# Target: ~80-90% GPU utilization.
#
# Tags: ns_mt_base_r<RATE>_t<TRIAL>, ns_mt_comb_r<RATE>_t<TRIAL>
#
# Usage:
#   RATE=4 bash scripts/run_near_sat_mt.sh          # single trial
#   RATE=4 TRIAL=2 bash scripts/run_near_sat_mt.sh  # second trial
#
# Analyze with: python3 src/analyze_ablation.py --avg-pairs ...

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
RATE=${RATE:-4}
TRIAL=${TRIAL:-1}

RTAG="r$(echo "$RATE" | tr '.' 'p')_t${TRIAL}"
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_nearsat_mt_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Near-Saturation Multi-Turn Experiment"
echo "  Rate: ${RATE} conv/s (open-loop Poisson) | trial: ${TRIAL}"
echo "  Convs: ${NUM_CONVS} | turns: ${MAX_TURNS} | max_tokens: ${MAX_TOKENS}"
echo "  Scheduler: PREFIX_REORDER + DYNAMIC_CHUNK + HOLD=3 + AGING_ALPHA=0.3"
echo "======================================================="

start_server() {
    local label=$1
    shift
    echo ""
    echo "=== Starting server: ${label} ==="
    env "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-nearsat-mt-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" \
                "$LOG_DIR/${DATE}-nearsat-mt-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-nearsat-mt-${label}-server.log"
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

run_replay() {
    local tag=$1
    local out="$LOG_DIR/${DATE}-mt-${tag}.jsonl"
    echo ""
    echo "--- Replaying: tag=${tag}  rate=${RATE} conv/s ---"
    $PYTHON src/replay_sharegpt.py \
        --host localhost --port "$PORT" \
        --model "$MODEL" \
        --dataset data/sharegpt_v3.json \
        --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" \
        --min-turns "$MAX_TURNS" \
        --max-tokens "$MAX_TOKENS" \
        --rate "$RATE" \
        --output "$out"
    echo "  -> $out"
}

trap stop_server EXIT

# ── Condition 1: baseline ──────────────────────────────────────────────────────
start_server "baseline" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "ns_mt_base_${RTAG}"
stop_server

# ── Condition 2: combined (soft aging + hysteresis) ───────────────────────────
start_server "combined" \
    PREFIX_REORDER=1 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3 AGING_ALPHA=0.3
run_replay "ns_mt_comb_${RTAG}"
stop_server

echo ""
echo "=== Near-saturation multi-turn done. ==="
echo ""
echo "Analyze with:"
echo "  python3 src/analyze_ablation.py \\"
echo "    'baseline:logs/${DATE}-mt-ns_mt_base_${RTAG}.jsonl' \\"
echo "    'combined:logs/${DATE}-mt-ns_mt_comb_${RTAG}.jsonl'"
