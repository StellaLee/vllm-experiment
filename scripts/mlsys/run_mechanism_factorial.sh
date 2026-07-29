#!/bin/bash
# run_mechanism_factorial.sh
# Decompose the closed-loop herd win into REORDER vs CHUNK. Real ShareGPT multi-turn
# (prefix caching ON, NO padding -> warmth exists), original section 3/4 config.
# Factorial: {baseline, reorder, chunk, combined} x {herd, staggered}.
#   baseline  PREFIX_REORDER=0 DYNAMIC_CHUNK=0
#   reorder   PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3         (reorder-only)
#   chunk     PREFIX_REORDER=0 DYNAMIC_CHUNK=1 (depth-mode)            (chunk-only)
#   combined  PREFIX_REORDER=1 DYNAMIC_CHUNK=1 (depth) AGING_ALPHA=0.3 (reproduces the win)
# Answers: how much of the herd win is reorder vs chunk, and does reorder-only's win
# collapse under staggering (artifact claim for reordering specifically).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAX_BUDGET=${MAX_BUDGET:-2048}
CONC=${CONC:-15}
NUM_CONVS=${NUM_CONVS:-120}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
TRIAL=${TRIAL:-1}

PREFIX=mfact
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_mfact_pid
mkdir -p "$LOG_DIR"

echo "==== Mechanism factorial | c=${CONC} convs=${NUM_CONVS} turns=${MAX_TURNS} budget=${MAX_BUDGET} (real sharing, no pad) ===="

echo "=== patch vLLM (canonical apply_patches.sh) ==="
PYTHON="$PYTHON" bash scripts/apply_patches.sh

start_server() {
    local label=$1; shift
    echo ""; echo "=== server: ${label} ==="
    env "$@" $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$MAX_BUDGET" \
        > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "waiting"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then echo " ready"; return 0; fi
        echo -n "."
    done
    echo " TIMEOUT"; tail -25 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1
}
stop_server() { if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi; }
run_replay() {
    local tag=$1; local stagger=$2
    local out="$LOG_DIR/${DATE}-${PREFIX}-${tag}.jsonl"
    local stagarg=(); [ "$stagger" != "0" ] && stagarg=(--stagger-window "$stagger")
    echo "--- replay ${tag} (stagger=${stagger}s) ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" "${stagarg[@]}" --output "$out"
    echo "  -> $out"
}
trap stop_server EXIT

run_arm() {   # arm_label  stagger_seconds  env...
    local arm=$1; local stag=$2; shift 2
    local mode=herd; [ "$stag" != "0" ] && mode=stag
    start_server "${arm}_${mode}" "$@"
    run_replay "${arm}_${mode}_t${TRIAL}" "$stag"
    stop_server
}

for STAG in ${STAGS:-0 30}; do
    run_arm baseline "$STAG" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
    run_arm reorder  "$STAG" PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3
    run_arm chunk    "$STAG" PREFIX_REORDER=0 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3
    run_arm combined "$STAG" PREFIX_REORDER=1 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3 AGING_ALPHA=0.3
done

echo "=== mechanism factorial done ==="
echo "logs: ${DATE}-${PREFIX}-{baseline,reorder,chunk,combined}_{herd,stag}_t${TRIAL}.jsonl"
