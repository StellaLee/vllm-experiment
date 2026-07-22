#!/bin/bash
# orchestrate_longprompt_hslo_af.sh -- hslo with the ALPHA-FLOOR fix. Diagnosis of hslo400wf: the
# residual TBT-max ~2949ms leak was NOT the idle exception (359/361 ceiling steps had decode_depth>0)
# but the headroom formula RAILING to 16384 when the online alpha underestimated (~8x) on noisy
# small-prefill steps -> a whale prefilled whole while ~13 decoders were frozen. Fix: floor the
# online alpha at the offline hardware cost (DYNAMIC_CHUNK_ALPHA_MIN=0.18 ms/tok), which caps
# budget at ~(SLO-db)/alpha_hw ~ 2048 and makes predicted_iter <= SLO BY CONSTRUCTION (SLO becomes
# a hard TBT guarantee). Same SLO=400, warmup-at-floor (START=512), whale setup + pad-seed 1001.
# Prediction: TBT-max collapses from 2949 toward static-2048's ~1200ms while interior modulation
# survives -> hslo matches the ORACLE static point using only offline-computable knobs.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longphsloaf_ALLDONE logs/longphsloaf_FAILED
DATE=$(date +%Y-%m-%d)
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
FLOOR=${FLOOR:-512}; START=${START:-512}; SLO_MS=${SLO_MS:-400}; ALPHA_MIN=${ALPHA_MIN:-256}
ALPHA_HW_MS=${ALPHA_HW_MS:-0.18}
ARM=bhslo400af; PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH(base) FAILED"; touch logs/longphsloaf_FAILED; exit 1; }
$PYTHON scripts/hotpatch_hslo_alphafloor.py || { log "PATCH(alphafloor) FAILED"; touch logs/longphsloaf_FAILED; exit 1; }

log "long-prompt HSLO alpha-floor arm: slo=${SLO_MS}ms floor=$FLOOR START=$START alpha_hw=${ALPHA_HW_MS}ms/tok alpha_min_prefill=$ALPHA_MIN"
log "  [$ARM] server GPUs 0,1 port=$PORT mode=hslo slo=${SLO_MS}ms start=$START alpha_hw=${ALPHA_HW_MS}"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START \
    DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN \
    DYNAMIC_CHUNK_ALPHA_MIN=$ALPHA_HW_MS \
    DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${ARM}-chunktrace.csv \
    $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens 16384 \
    --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
    > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
  [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longphsloaf_FAILED; exit 1; }
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
log "analyzing (oracle statics + warmup-floor + alpha-floor)"
BUDGETS="16384 2048 512 hslo400wf hslo400af" $PYTHON scripts/analyze_longprompt.py > logs/longphsloaf_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longphsloaf_ANALYSIS.txt
touch logs/longphsloaf_ALLDONE
log "done -> logs/longphsloaf_ANALYSIS.txt"
