#!/bin/bash
# run_probe_openloop.sh
# Decisive open-loop chunking probe.
#
# Answers ONE question: does any chunk-budget policy beat the static baseline
# in the open-loop (Poisson) regime? The answer decides whether we iterate the
# SLO controller or pivot to a characterization paper.
#
# Three arms, all PREFIX_REORDER=0 (chunk-only, to isolate chunking from the
# soft-aging starvation):
#   base2048   static high budget (DYNAMIC_CHUNK=0, max_batched=2048)  -- baseline
#   sarathi512 static stall-free  (DYNAMIC_CHUNK=0, max_batched=512)   -- fixed floor
#   slo        adaptive 512..2048 (CHUNK_MODE=slo)                     -- redesign
#
# Decision rule:
#   - If sarathi512 or slo beats base2048 on E2EL/TPOT p95, and slo does so
#     without wrecking TTFT -> a real open-loop win exists -> iterate SLO ctrl.
#   - If neither beats base2048 on any tail metric -> no open-loop chunk win ->
#     pivot to the characterization framing.
#   - Guardrail: slo TTFT p95 should stay near base2048 (NOT the old ~1452ms
#     depth-controller collapse). If so, the redesign is validated regardless.
#
# Usage:
#   bash scripts/run_probe_openloop.sh                 # rate=3, trial=1
#   RATE=3 TRIAL=2 bash scripts/run_probe_openloop.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
RATE=${RATE:-3}
TRIAL=${TRIAL:-1}

MAX_BUDGET=${MAX_BUDGET:-2048}       # ceiling for base2048 + slo
SARATHI_BUDGET=${SARATHI_BUDGET:-512} # fixed stall-free budget + slo floor
SLO_MS=${SLO_MS:-50}                  # target per-step (per-token) latency

RTAG="r$(echo "$RATE" | tr '.' 'p')_t${TRIAL}"
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_probe_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Open-Loop Chunking Probe"
echo "  Rate: ${RATE} conv/s (Poisson) | trial: ${TRIAL} | reorder OFF"
echo "  Convs: ${NUM_CONVS} | turns: ${MAX_TURNS} | max_tokens: ${MAX_TOKENS}"
echo "  Arms: base2048 / sarathi${SARATHI_BUDGET} / slo(${SARATHI_BUDGET}..${MAX_BUDGET}, SLO=${SLO_MS}ms)"
echo "======================================================="

# ── Ensure patches are present (idempotent) ─────────────────────────────────
echo ""
echo "=== Applying base scheduler patches + SLO controller (idempotent) ==="
$PYTHON scripts/patch_scheduler.py
$PYTHON scripts/hotpatch_slo_chunk.py

start_server() {
    local label=$1
    local maxtok=$2
    shift 2
    echo ""
    echo "=== Starting server: ${label} (max_batched=${maxtok}) ==="
    env "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" \
        --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$maxtok" \
        > "$LOG_DIR/${DATE}-probe-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" \
                "$LOG_DIR/${DATE}-probe-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-probe-${label}-server.log"
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
    local out="$LOG_DIR/${DATE}-probe-${tag}.jsonl"
    echo ""
    echo "--- Replaying: tag=${tag}  rate=${RATE} conv/s (reorder off) ---"
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

# ── Arm 1: static high budget (baseline) ────────────────────────────────────
start_server "base2048" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "base2048_${RTAG}"
stop_server

# ── Arm 2: static stall-free budget (Sarathi) ───────────────────────────────
start_server "sarathi512" "$SARATHI_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "sarathi512_${RTAG}"
stop_server

# ── Arm 3: SLO-feedback adaptive budget (redesign) ──────────────────────────
start_server "slo" "$MAX_BUDGET" \
    PREFIX_REORDER=0 DYNAMIC_CHUNK=1 CHUNK_MODE=slo \
    DYNAMIC_CHUNK_MIN="$SARATHI_BUDGET" DYNAMIC_CHUNK_SLO_MS="$SLO_MS"
run_replay "slo_${RTAG}"
stop_server

echo ""
echo "=== Open-loop probe done. ==="
echo ""
echo "Analyze with:"
echo "  $PYTHON src/analyze_ablation.py \\"
echo "    'base2048:logs/${DATE}-probe-base2048_${RTAG}.jsonl' \\"
echo "    'sarathi512:logs/${DATE}-probe-sarathi512_${RTAG}.jsonl' \\"
echo "    'slo:logs/${DATE}-probe-slo_${RTAG}.jsonl'"
