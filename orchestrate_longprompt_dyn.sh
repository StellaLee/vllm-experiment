#!/bin/bash
# orchestrate_longprompt_dyn.sh -- run the DYNAMIC chunk arm (slocvar controller) under the exact
# long-prompt whale setup used by orchestrate_longprompt.sh, with the SAME pad-seed (1001) so whale
# positions are paired with the existing static arms (logs/2026-07-21-longp-b{16384,2048,512}-t1.jsonl).
# Controller starts HIGH (16384, mono-like TTFT when idle) and must DROP to the 512 floor when a whale
# spikes step latency -- the test of whether reactive control catches a one-iteration whale.
# Outputs logs/<date>-longp-bours-t1.jsonl + chunktrace. Then analyzes vs all static arms.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpdyn_ALLDONE logs/longpdyn_FAILED
DATE=$(date +%Y-%m-%d)
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
FLOOR=${FLOOR:-512}; SLO_MS=${SLO_MS:-50}; CVAR_PCTL=${CVAR_PCTL:-90}; START=${START:-16384}
ARM=bours; PORT=8050
log "long-prompt DYNAMIC (slocvar) arm: conc=$CONC whale_frac=$WHALE_FRAC whale=[$WHALE_MIN,$WHALE_MAX] floor=$FLOOR start=$START slo=${SLO_MS}ms"

log "  [$ARM] server on GPUs 0,1 port=$PORT budget=16384(max) mode=slocvar start=$START"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=slocvar DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START \
    DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_CVAR_PCTL=$CVAR_PCTL \
    DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${ARM}-chunktrace.csv \
    $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens 16384 \
    --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
    > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
  [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longpdyn_FAILED; exit 1; }
done
out="logs/${DATE}-longp-${ARM}-t1.jsonl"
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
  --max-tokens $MAXTOK --concurrency $CONC \
  --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
  --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
  --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
  --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
log "  [$ARM] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${ARM}-server.log 2>/dev/null || echo 0)"
kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
log "analyzing (vs static arms)"
BUDGETS="16384 2048 512 ours" $PYTHON scripts/analyze_longprompt.py > logs/longpdyn_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longpdyn_ANALYSIS.txt
touch logs/longpdyn_ALLDONE
log "done -> logs/longpdyn_ANALYSIS.txt"
