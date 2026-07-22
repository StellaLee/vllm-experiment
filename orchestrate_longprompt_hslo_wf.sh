#!/bin/bash
# orchestrate_longprompt_hslo_wf.sh -- the DECISIVE hslo run: SLO=400ms (the offline heuristic:
# ~wall-time of one isolated 2048-token prefill iteration => centers the controller at vLLM's
# default 2048 chunk) PLUS warmup-at-FLOOR (DYNAMIC_CHUNK_START=512). The sweep showed hslo finds
# the interior but every arm leaked TBT-max ~6s because warmup sat at start=16384 -> the first
# whale (before alpha is learned) prefilled fully = one 6s freeze. Starting at the floor caps that
# first whale at 512 while alpha learns, so the max should drop toward static-2048's ~1200ms while
# the median chunk still climbs to ~2048 under the 400ms SLO. If max bounds, hslo matches the
# ORACLE static-2048 point using only an offline-computable SLO -- the paper's result.
# Same whale setup + pad-seed 1001 (paired whales). Requires scripts/hotpatch_hslo.py (idempotent).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longphslowf_ALLDONE logs/longphslowf_FAILED
DATE=$(date +%Y-%m-%d)
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
FLOOR=${FLOOR:-512}; START=${START:-512}; SLO_MS=${SLO_MS:-400}; ALPHA_MIN=${ALPHA_MIN:-256}
ARM=bhslo400wf; PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH FAILED"; touch logs/longphslowf_FAILED; exit 1; }

log "long-prompt HSLO warmup-floor arm: conc=$CONC slo=${SLO_MS}ms floor=$FLOOR START=$START(=floor) alpha_min=$ALPHA_MIN"
log "  [$ARM] server GPUs 0,1 port=$PORT mode=hslo slo=${SLO_MS}ms start=$START"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START \
    DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN \
    DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${ARM}-chunktrace.csv \
    $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens 16384 \
    --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
    > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
  [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longphslowf_FAILED; exit 1; }
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
log "analyzing (oracle statics + hslo sweep + warmup-floor)"
BUDGETS="16384 2048 512 hslo200 hslo600 hslo1200 hslo400wf" $PYTHON scripts/analyze_longprompt.py > logs/longphslowf_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longphslowf_ANALYSIS.txt
touch logs/longphslowf_ALLDONE
log "done -> logs/longphslowf_ANALYSIS.txt"
