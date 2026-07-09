#!/bin/bash
# run_staggered_cl.sh
# Test whether the closed-loop chunk/reorder win (multi-turn c=15: T3 -45%,
# T4 -47% TTFT) is a THUNDERING-HERD ARTIFACT of the synchronized t=0 start.
#
# Runs baseline vs combined at concurrency=15. STAGGER controls the start:
#   STAGGER=0   -> herd (all 15 workers fire at t=0)  [original setup]
#   STAGGER=30  -> staggered (initial starts spread over 30s)
#
# Compare the combined-vs-baseline delta under herd vs under stagger:
#   delta survives staggering  -> real benefit
#   delta collapses toward 0    -> herd artifact
#
# NOTE: TPOT here uses the NEW real-token definition (include_usage), so absolute
# TPOT (~20ms) will NOT match the old word-proxy multi-turn findings (~41ms).
# TTFT/E2EL are unaffected and are the headline for the artifact question.
#
# Usage:
#   STAGGER=0  bash scripts/run_staggered_cl.sh    # herd reference
#   STAGGER=30 bash scripts/run_staggered_cl.sh    # staggered
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAX_BUDGET=${MAX_BUDGET:-2048}
CONC=${CONC:-15}
NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
STAGGER=${STAGGER:-0}
TRIAL=${TRIAL:-1}

if [ "$STAGGER" = "0" ]; then MODE=herd; STAG_ARG=(); else MODE=stag; STAG_ARG=(--stagger-window "$STAGGER"); fi
TAGSUF="${MODE}_t${TRIAL}"
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_stagcl_pid
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Staggered closed-loop | mode=${MODE} (STAGGER=${STAGGER}s) | c=${CONC}"
echo "  Convs: ${NUM_CONVS} | turns: ${MAX_TURNS} | max_tokens: ${MAX_TOKENS}"
echo "======================================================="

echo "=== applying scheduler patches (idempotent) ==="
$PYTHON scripts/patch_scheduler.py

start_server() {
    local label=$1; shift
    echo ""; echo "=== Starting server: ${label} ==="
    env "$@" $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$MAX_BUDGET" \
        > "$LOG_DIR/${DATE}-stagcl-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-stagcl-${label}-server.log" 2>/dev/null; then
            echo " ready"; return 0; fi
        echo -n "."
    done
    echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-stagcl-${label}-server.log"; exit 1
}
stop_server() {
    if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi
}
run_replay() {
    local tag=$1
    local out="$LOG_DIR/${DATE}-stagcl-${tag}.jsonl"
    echo ""; echo "--- Replaying: tag=${tag} (c=${CONC}, ${MODE}) ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" "${STAG_ARG[@]}" --output "$out"
    echo "  -> $out"
}
trap stop_server EXIT

# baseline
start_server "base_${MODE}" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "base_${TAGSUF}"
stop_server

# combined (reorder + depth-mode chunk hysteresis + aging) -- the original win config
start_server "comb_${MODE}" \
    PREFIX_REORDER=1 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3 AGING_ALPHA=0.3
run_replay "comb_${TAGSUF}"
stop_server

echo ""
echo "=== ${MODE} done. Analyze: ==="
echo "  $PYTHON src/analyze_ablation.py \\"
echo "    'baseline:logs/${DATE}-stagcl-base_${TAGSUF}.jsonl' \\"
echo "    'combined:logs/${DATE}-stagcl-comb_${TAGSUF}.jsonl'"
