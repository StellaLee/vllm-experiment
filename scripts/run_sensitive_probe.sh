#!/bin/bash
# run_sensitive_probe.sh
# Sensitive-regime chunking ISOLATION. Padded (large, un-cacheable) prefills,
# staggered closed-loop sub-saturation, reorder OFF, three chunk arms:
#   static<MAX>   fixed high budget (monolithic prefill)  -- baseline
#   static<MIN>   fixed small budget (chopped prefill)     -- pure chunking
#   slo           SLO-adaptive MIN..MAX                    -- controller
# Question: when decode IS stalled by prefill, does chopping the prefill protect
# TPOT (and at what TTFT cost)? static<MIN> is the decisive pure-chunking test;
# slo adds controller dynamics (budget logged at INFO).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAX_BUDGET=${MAX_BUDGET:-8192}
SARATHI_BUDGET=${SARATHI_BUDGET:-512}
SLO_MS=${SLO_MS:-50}
PAD_CHARS=${PAD_CHARS:-12000}
CONC=${CONC:-10}
STAGGER=${STAGGER:-30}
NUM_CONVS=${NUM_CONVS:-80}
MAX_TURNS=${MAX_TURNS:-3}
MAX_TOKENS=${MAX_TOKENS:-512}
TRIAL=${TRIAL:-sprobe}

PREFIX=sprobe
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_sprobe_pid
mkdir -p "$LOG_DIR"

echo "==== Sensitive chunking probe | staggered ${STAGGER}s c=${CONC} pad=${PAD_CHARS} out=${MAX_TOKENS} ===="
echo "  arms: static${MAX_BUDGET} / static${SARATHI_BUDGET} / slo(${SARATHI_BUDGET}..${MAX_BUDGET},${SLO_MS}ms) | reorder OFF"

echo "=== patches (idempotent) ==="
$PYTHON scripts/patch_scheduler.py
$PYTHON scripts/hotpatch_slo_chunk.py

start_server() {
    local label=$1; local budget=$2; shift 2
    echo ""; echo "=== server: ${label} (budget=${budget}) ==="
    env "$@" $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$budget" \
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
    local tag=$1
    local out="$LOG_DIR/${DATE}-${PREFIX}-${tag}.jsonl"
    echo "--- replay ${tag} ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" --stagger-window "$STAGGER" --pad-chars "$PAD_CHARS" --output "$out"
    echo "  -> $out"
}
trap stop_server EXIT

# arm 1: static high budget (monolithic prefill)
start_server "static${MAX_BUDGET}" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "static${MAX_BUDGET}_t${TRIAL}"
stop_server

# arm 2: static small budget (chopped prefill) -- pure chunking test
start_server "static${SARATHI_BUDGET}" "$SARATHI_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "static${SARATHI_BUDGET}_t${TRIAL}"
stop_server

# arm 3: SLO-adaptive controller (budget logged at INFO)
start_server "slo" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=1 CHUNK_MODE=slo \
    DYNAMIC_CHUNK_MIN="$SARATHI_BUDGET" DYNAMIC_CHUNK_SLO_MS="$SLO_MS"
run_replay "slo_t${TRIAL}"
stop_server

echo "=== sensitive probe done ==="
