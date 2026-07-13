#!/bin/bash
# run_cs2_ours.sh -- THIRD ARM for the Cs^2 sweep: our ADAPTIVE CHUNK CONTROLLER
# (SLO-adaptive budget floor..ceil), reorder OFF. Same grid/rate/pads as
# run_cs2_sweep.sh (same --pad-seed => identical prompts => paired with mono/chunk).
# Reorder is deliberately OFF: the padding is UNIQUE/uncacheable, so there are no warm
# prefixes for warmth reordering to act on -- it would be a pure no-op/confound here.
# This arm isolates one question: in the Cs^2>1 regime where a genuine PS win exists
# (static chunk beats mono), does the adaptive controller CAPTURE it (budget -> floor,
# tracks chunk) or stay BLIND (budget pinned at ceiling, tracks mono)? Budget logged INFO.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAXLEN=${MAXLEN:-16384}
CEIL=${CEIL:-16384}          # controller ceiling (= mono budget)
FLOOR=${FLOOR:-512}          # controller floor  (= chunk budget)
SLO_MS=${SLO_MS:-50}
GPU_UTIL=${GPU_UTIL:-0.90}
DATASET=${DATASET:-data/sharegpt_v3.json}
NUM_CONVS=${NUM_CONVS:-120}
MAX_TURNS=${MAX_TURNS:-3}
MAX_TOKENS=${MAX_TOKENS:-128}
RATE=${RATE:-3.0}
PAD_MEAN=${PAD_MEAN:-8000}
PAD_MIN=${PAD_MIN:-100}
PAD_MAX=${PAD_MAX:-50000}
PAD_SEED=${PAD_SEED:-12345}
CV2_GRID=${CV2_GRID:-"0 0.5 1 2 4 8"}
TRIAL=${TRIAL:-t1}
PREFIX=cs2sweep
LOG_DIR=logs; DATE=$(date +%Y-%m-%d); PID_FILE=/tmp/vllm_cs2ours_pid
mkdir -p "$LOG_DIR"

echo "==== Cs^2 THIRD ARM: ours (adaptive chunk slo ${FLOOR}..${CEIL}, ${SLO_MS}ms, reorder OFF) rate=${RATE} ===="
$PYTHON scripts/patch_scheduler.py
$PYTHON scripts/hotpatch_slo_chunk.py

echo "=== server: ours (adaptive chunk controller, reorder off) ==="
env PREFIX_REORDER=0 DYNAMIC_CHUNK=1 CHUNK_MODE=slo \
    DYNAMIC_CHUNK_MIN="$FLOOR" DYNAMIC_CHUNK_SLO_MS="$SLO_MS" DYNAMIC_CHUNK_EMA=0.3 \
    $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$CEIL" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    > "$LOG_DIR/${DATE}-${PREFIX}-ours-server.log" 2>&1 &
echo $! > "$PID_FILE"
trap 'kill "$(cat $PID_FILE)" 2>/dev/null || true' EXIT
echo -n "waiting"; for i in $(seq 1 72); do sleep 5
  if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-ours-server.log" 2>/dev/null; then echo " ready"; break; fi
  echo -n "."; done

for cv2 in $CV2_GRID; do
  cvtag=$(echo "$cv2" | tr '.' 'p')
  out="$LOG_DIR/${DATE}-${PREFIX}-ours-cv${cvtag}-${TRIAL}.jsonl"
  echo "--- replay ours cv2=${cv2} -> $(basename "$out") ---"
  $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
    --dataset "$DATASET" --num-convs "$NUM_CONVS" --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" --rate "$RATE" \
    --pad-mean-chars "$PAD_MEAN" --pad-cv2 "$cv2" --pad-min "$PAD_MIN" --pad-max "$PAD_MAX" --pad-seed "$PAD_SEED" \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1
  echo "    done ($(grep -c . "$out" 2>/dev/null || echo 0) records)"
done
echo "=== cs2 ours arm done ==="
