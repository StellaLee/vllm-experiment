#!/bin/bash
# run_arm_unit.sh -- run ONE arm (mono|chunk|ours) of ONE Cs^2 point on a dedicated GPU pair
# + port, for N trials. Server/client flags are byte-identical to run_cs2_3arm.sh; the only
# difference is this drives a single arm so several units can run concurrently on different
# GPU pairs. Unique PORT + PID file per instance so parallel units never collide.
# Required env: GPUS (e.g. "0,1") PORT PREFIX ARM BUDGET MODE CV2 . Optional: TRIALS.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/data/pli/models/Qwen2.5-Coder-14B-Instruct}
MAX_SEQS=32; MAXLEN=16384; GPU_UTIL=0.90; TP=2
DATASET=data/sharegpt_v3.json
NUM_CONVS=80; MAX_TURNS=1; MAX_TOKENS=1024; RATE=0.64
PAD_MEAN=8000; PAD_MIN=100; PAD_MAX=50000; SEED_BASE=1000
FLOOR=512; SLO_MS=50; CVAR_PCTL=90
TRIALS=${TRIALS:-"1 2 3 4 5 6 7 8"}
: "${GPUS:?}"; : "${PORT:?}"; : "${PREFIX:?}"; : "${ARM:?}"; : "${BUDGET:?}"; : "${MODE:?}"; : "${CV2:?}"
LOG_DIR=logs; DATE=$(date +%Y-%m-%d); PID_FILE=/tmp/vllm_unit_${PORT}_pid
mkdir -p "$LOG_DIR"

EX="DYNAMIC_CHUNK=0"
if [ "$MODE" = "slocvar" ]; then
  # ours arm: BUDGET is the ceiling (16384); controller starts at the chunk floor (512) and grows.
  TRACE="$LOG_DIR/${DATE}-${PREFIX}-${ARM}-chunktrace.csv"
  EX="DYNAMIC_CHUNK=1 CHUNK_MODE=slocvar DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_CVAR_PCTL=$CVAR_PCTL DYNAMIC_CHUNK_START=$FLOOR DYNAMIC_CHUNK_TRACE=$TRACE"
fi

echo "[unit ${PREFIX}/${ARM}] gpus=$GPUS port=$PORT budget=$BUDGET mode=$MODE cv2=$CV2"
env CUDA_VISIBLE_DEVICES="$GPUS" PREFIX_REORDER=0 $EX $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$BUDGET" --max-model-len "$MAXLEN" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
    > "$LOG_DIR/${DATE}-${PREFIX}-${ARM}-server.log" 2>&1 &
echo $! > "$PID_FILE"
stop_server(){ [ -f "$PID_FILE" ] && { kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; }; }
trap stop_server EXIT

echo -n "[unit ${PREFIX}/${ARM}] waiting"
for i in $(seq 1 120); do sleep 5
  if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${ARM}-server.log" 2>/dev/null; then echo " ready"; break; fi
  echo -n "."
  [ "$i" = 120 ] && { echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-${PREFIX}-${ARM}-server.log"; exit 1; }
done

for tr in $TRIALS; do
  seed=$((SEED_BASE + tr))
  out="$LOG_DIR/${DATE}-${PREFIX}-${ARM}-cv0-t${tr}.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
    --dataset "$DATASET" --num-convs "$NUM_CONVS" --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" --rate "$RATE" \
    --pad-mean-chars "$PAD_MEAN" --pad-cv2 "$CV2" --pad-min "$PAD_MIN" --pad-max "$PAD_MAX" --pad-seed "$seed" \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1
  echo "[unit ${PREFIX}/${ARM}] trial $tr done ($(grep -c . "$out" 2>/dev/null || echo 0) recs)"
done
echo "[unit ${PREFIX}/${ARM}] ALL TRIALS DONE"
