#!/bin/bash
# run_cs2_3arm.sh -- 3-arm controller comparison at ONE (mt, cv2) point:
#   mono (static MONO_BUDGET) vs chunk (static CHUNK_BUDGET) vs ours (slocvar adaptive
#   FLOOR..MONO_BUDGET). Paired: same per-trial pad seed across arms. Server per arm,
#   inner loop over trials. Tests: does the adaptive controller capture chunk's TPOT-tail
#   protection WITHOUT chunk's TTFT/throughput cost?
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}; MAX_SEQS=${MAX_SEQS:-32}; MAXLEN=${MAXLEN:-16384}
MONO_BUDGET=${MONO_BUDGET:-16384}; CHUNK_BUDGET=${CHUNK_BUDGET:-512}; GPU_UTIL=${GPU_UTIL:-0.90}
FLOOR=${FLOOR:-512}; SLO_MS=${SLO_MS:-50}; CVAR_PCTL=${CVAR_PCTL:-90}
TP=${TP:-1}
DATASET=${DATASET:-data/sharegpt_v3.json}
NUM_CONVS=${NUM_CONVS:-80}; MAX_TURNS=1; MAX_TOKENS=${MAX_TOKENS:-1024}; RATE=${RATE:-0.64}
PAD_MEAN=${PAD_MEAN:-8000}; PAD_MIN=${PAD_MIN:-100}; PAD_MAX=${PAD_MAX:-50000}
SEED_BASE=${SEED_BASE:-1000}; CV2=${CV2:-0}; TRIALS=${TRIALS:-"1 2 3"}
PREFIX=cs23arm
LOG_DIR=logs; DATE=$(date +%Y-%m-%d); PID_FILE=/tmp/vllm_cs23arm_pid
mkdir -p "$LOG_DIR"

echo "==== 3-arm | mono($MONO_BUDGET) vs chunk($CHUNK_BUDGET) vs ours(slocvar $FLOOR..$MONO_BUDGET slo${SLO_MS}ms cvar${CVAR_PCTL}) | mt=$MAX_TOKENS rate=$RATE cv2=$CV2 TP=$TP ===="
PYTHON="$PYTHON" bash scripts/apply_patches.sh

start_server() { local label=$1 budget=$2 mode=$3
  local EX
  if [ "$mode" = "slocvar" ]; then
    local TRACE="$LOG_DIR/${DATE}-${PREFIX}-${label}-chunktrace.csv"
    EX="DYNAMIC_CHUNK=1 CHUNK_MODE=slocvar DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_CVAR_PCTL=$CVAR_PCTL DYNAMIC_CHUNK_START=$CHUNK_BUDGET DYNAMIC_CHUNK_TRACE=$TRACE"
  else
    EX="DYNAMIC_CHUNK=0"
  fi
  echo ""; echo "=== server: ${label} (budget=${budget} mode=${mode}) ==="
  env PREFIX_REORDER=0 $EX $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
      --max-num-batched-tokens "$budget" --max-model-len "$MAXLEN" \
      --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
      > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
  echo $! > "$PID_FILE"; echo -n "waiting"
  for i in $(seq 1 96); do sleep 5
    if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then echo " ready"; return 0; fi
    echo -n "."; done
  echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1; }
stop_server() { if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi; }
trap stop_server EXIT

for spec in "ours ${MONO_BUDGET} slocvar" "mono ${MONO_BUDGET} static" "chunk ${CHUNK_BUDGET} static"; do
  set -- $spec; arm=$1; budget=$2; mode=$3
  start_server "$arm" "$budget" "$mode"
  for tr in $TRIALS; do
    seed=$((SEED_BASE + tr))
    out="$LOG_DIR/${DATE}-${PREFIX}-${arm}-cv0-t${tr}.jsonl"
    echo "--- ${arm} trial=${tr} seed=${seed} ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
      --dataset "$DATASET" --num-convs "$NUM_CONVS" --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" \
      --max-tokens "$MAX_TOKENS" --rate "$RATE" \
      --pad-mean-chars "$PAD_MEAN" --pad-cv2 "$CV2" --pad-min "$PAD_MIN" --pad-max "$PAD_MAX" --pad-seed "$seed" \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1
    echo "    done ($(grep -c . "$out" 2>/dev/null || echo 0) records)"
  done
  stop_server
done
echo "=== 3-arm done ==="
