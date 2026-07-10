#!/bin/bash
# run_staggered_slo.sh
# Staggered closed-loop, SLO-mode dynamic chunking + reorder + aging (the
# "corrected" strategy), at a configurable prefill-budget ceiling MAX_BUDGET.
#
# Purpose: sensitive-region probe. Raising MAX_BUDGET lets a big prefill chunk
# land on the decode batch and (maybe) stall TPOT; we then see whether the SLO
# strategy protects TPOT at a TTFT cost (the trade-off). Everything else is kept
# identical to run_staggered_cl.sh so this is a one-knob change.
#
# Combined arm = PREFIX_REORDER=1 + AGING_ALPHA (same as prior staggered run)
#              + DYNAMIC_CHUNK=1 CHUNK_MODE=slo (SLO controller, NOT depth-mode).
#
# Usage:
#   MAX_BUDGET=8192 STAGGER=30 TRIAL=slo8192 bash scripts/run_staggered_slo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAX_BUDGET=${MAX_BUDGET:-8192}         # regime knob: prefill-budget ceiling
CONC=${CONC:-15}
NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
STAGGER=${STAGGER:-30}
TRIAL=${TRIAL:-slo8192}
PAD_CHARS=${PAD_CHARS:-0}          # >0 forces large un-cacheable prefills (sensitive regime)
# SLO controller config (same as the open-loop probe) + aging (same as prior run)
SLO_MIN=${SLO_MIN:-512}
SLO_MS=${SLO_MS:-50}
AGING_ALPHA=${AGING_ALPHA:-0.3}

if [ "$STAGGER" = "0" ]; then MODE=herd; STAG_ARG=(); else MODE=stag; STAG_ARG=(--stagger-window "$STAGGER"); fi
TAGSUF="${MODE}_t${TRIAL}"
PREFIX=stagslo
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_stagslo_pid
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Staggered CL + SLO strategy | mode=${MODE} (STAGGER=${STAGGER}s) | c=${CONC}"
echo "  Prefill-budget ceiling: ${MAX_BUDGET} | SLO floor ${SLO_MIN}, target ${SLO_MS}ms"
echo "  combined = reorder + aging(${AGING_ALPHA}) + SLO chunk (${SLO_MIN}..${MAX_BUDGET})"
echo "  Convs: ${NUM_CONVS} | turns: ${MAX_TURNS} | max_tokens: ${MAX_TOKENS}"
echo "======================================================="

echo "=== applying scheduler patches + SLO controller (idempotent) ==="
$PYTHON scripts/patch_scheduler.py
$PYTHON scripts/hotpatch_slo_chunk.py

start_server() {
    local label=$1; shift
    echo ""; echo "=== Starting server: ${label} (budget=${MAX_BUDGET}) ==="
    env "$@" $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$MAX_BUDGET" \
        > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then
            echo " ready"; return 0; fi
        echo -n "."
    done
    echo " TIMEOUT"; tail -25 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1
}
stop_server() {
    if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi
}
run_replay() {
    local tag=$1
    local out="$LOG_DIR/${DATE}-${PREFIX}-${tag}.jsonl"
    echo ""; echo "--- Replaying: tag=${tag} (c=${CONC}, ${MODE}) ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" --pad-chars "$PAD_CHARS" "${STAG_ARG[@]}" --output "$out"
    echo "  -> $out"
}
trap stop_server EXIT

# baseline: static budget = MAX_BUDGET (no strategy)
start_server "base_${MODE}" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "base_${TAGSUF}"
stop_server

# combined: reorder + aging + SLO-mode dynamic chunk
start_server "comb_${MODE}" \
    PREFIX_REORDER=1 AGING_ALPHA="$AGING_ALPHA" \
    DYNAMIC_CHUNK=1 CHUNK_MODE=slo DYNAMIC_CHUNK_MIN="$SLO_MIN" DYNAMIC_CHUNK_SLO_MS="$SLO_MS"
run_replay "comb_${TAGSUF}"
stop_server

echo ""
echo "=== ${MODE} SLO run done. Analyze: ==="
echo "  $PYTHON src/analyze_ablation.py \\"
echo "    'baseline:logs/${DATE}-${PREFIX}-base_${TAGSUF}.jsonl' \\"
echo "    'combined:logs/${DATE}-${PREFIX}-comb_${TAGSUF}.jsonl'"
