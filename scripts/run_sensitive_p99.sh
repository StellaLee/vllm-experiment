#!/bin/bash
# run_sensitive_p99.sh -- does keying the SLO controller off the TAIL (p99) of step
# latency instead of the MEAN (EMA) fix its blindness in the sensitive regime?
# Same sensitive setup as run_sensitive_probe.sh (uniform 12000-char padding, staggered
# closed-loop c=10, long decode). Four arms, reorder OFF:
#   static8192  fixed high budget (mono / run-to-completion)     -- reference
#   static512   fixed small budget (PS)                          -- the frontier to reach
#   slo         adaptive, EMA-of-MEAN step latency (blind)       -- reproduces pin-at-ceiling
#   slotail     adaptive, windowed p99 step latency (the fix)    -- should ENGAGE
# Key outputs: controller budget trajectory (INFO log 'chunk='), TPOT mean/p99, TTFT.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}; MAX_SEQS=${MAX_SEQS:-32}
MAX_BUDGET=${MAX_BUDGET:-8192}; MIN_BUDGET=${MIN_BUDGET:-512}; SLO_MS=${SLO_MS:-50}
PCTL=${PCTL:-99}
PAD_CHARS=${PAD_CHARS:-12000}; CONC=${CONC:-10}; STAGGER=${STAGGER:-30}
NUM_CONVS=${NUM_CONVS:-80}; MAX_TURNS=${MAX_TURNS:-3}; MAX_TOKENS=${MAX_TOKENS:-512}
TRIAL=${TRIAL:-1}
PREFIX=sp99; LOG_DIR=logs; DATE=$(date +%Y-%m-%d); PID_FILE=/tmp/vllm_sp99_pid
mkdir -p "$LOG_DIR"

echo "==== sensitive p99 controller | staggered ${STAGGER}s c=${CONC} pad=${PAD_CHARS} out=${MAX_TOKENS} | pctl=${PCTL} ===="
PYTHON="$PYTHON" bash scripts/apply_patches.sh

start_server() { local label=$1 budget=$2; shift 2
  echo ""; echo "=== server: ${label} (budget=${budget}) ==="
  env "$@" $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
      --max-num-batched-tokens "$budget" \
      > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
  echo $! > "$PID_FILE"; echo -n "waiting"
  for i in $(seq 1 72); do sleep 5
    if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then echo " ready"; return 0; fi
    echo -n "."; done
  echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1; }
stop_server() { if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi; }
run_replay() { local tag=$1
  local out="$LOG_DIR/${DATE}-${PREFIX}-${tag}.jsonl"
  echo "--- replay ${tag} ---"
  $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
      --dataset data/sharegpt_v3.json --num-convs "$NUM_CONVS" \
      --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
      --concurrency "$CONC" --stagger-window "$STAGGER" --pad-chars "$PAD_CHARS" --output "$out"
  echo "  -> $out"; }
trap stop_server EXIT

start_server "static${MAX_BUDGET}" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "static${MAX_BUDGET}_t${TRIAL}"; stop_server

start_server "static${MIN_BUDGET}" "$MIN_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=0
run_replay "static${MIN_BUDGET}_t${TRIAL}"; stop_server

start_server "slo" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=1 CHUNK_MODE=slo \
    DYNAMIC_CHUNK_MIN="$MIN_BUDGET" DYNAMIC_CHUNK_SLO_MS="$SLO_MS"
run_replay "slo_t${TRIAL}"; stop_server

start_server "slotail" "$MAX_BUDGET" PREFIX_REORDER=0 DYNAMIC_CHUNK=1 CHUNK_MODE=slotail \
    DYNAMIC_CHUNK_MIN="$MIN_BUDGET" DYNAMIC_CHUNK_SLO_MS="$SLO_MS" DYNAMIC_CHUNK_PCTL="$PCTL"
run_replay "slotail_t${TRIAL}"; stop_server

echo "=== sensitive p99 done ==="
