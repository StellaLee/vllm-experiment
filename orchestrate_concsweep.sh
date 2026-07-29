#!/bin/bash
# orchestrate_concsweep.sh -- stationary concurrency sweep, 4 arms x N concurrency levels.
# For each ARM (static budget or hslo) start ONE server, then run the client at each concurrency in
# CONCS (server reused across levels). Measures real outcome metrics (TBT tail/TTFT/goodput) per
# (arm,conc) so we can make defensible claims about whether a static budget is robust across load
# and whether hslo beats it. Whale workload (frac 0.15), single-turn, paired pad-seed.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/csweep_ALLDONE logs/csweep_FAILED
DATE=$(date +%Y-%m-%d)
MAX_SEQS=${MAX_SEQS:-64}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-160}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
CONCS=${CONCS:-"8 24 48"}
SLO_MS=${SLO_MS:-400}; FLOOR=${FLOOR:-512}; START=${START:-512}; ALPHA_MIN=${ALPHA_MIN:-256}; ALPHA_HW_MS=${ALPHA_HW_MS:-0.18}
ARMS=${ARMS:-"16384 2048 512 hslo400"}
PORT=8050

$PYTHON scripts/mlsys/hotpatch_hslo.py || { log "PATCH(base) FAILED"; touch logs/csweep_FAILED; exit 1; }
$PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py || { log "PATCH(alphafloor) FAILED"; touch logs/csweep_FAILED; exit 1; }

log "concsweep: ARMS='$ARMS' CONCS='$CONCS' whale_frac=$WHALE_FRAC max_seqs=$MAX_SEQS nconv=$NCONV"

run_arm(){ # $1=arm label (numeric budget, or hslo<slo>)
  local ARM=$1 MODE=static BUD=$1
  local EXTRA="DYNAMIC_CHUNK=0"
  local FB="logs/${DATE}-csweep-b${ARM}"
  if [[ "$ARM" == hslo* ]]; then
    MODE=hslo; BUD=16384
    EXTRA="DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN DYNAMIC_CHUNK_ALPHA_MIN=$ALPHA_HW_MS DYNAMIC_CHUNK_TRACE=${FB}-chunktrace.csv"
  fi
  log "  [$ARM] server GPUs 0,1 mode=$MODE budget=$BUD"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 $EXTRA \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUD \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours; touch logs/csweep_FAILED; exit 1; }
  done
  for C in $CONCS; do
    local out="${FB}-c${C}-t1.jsonl"
    log "  [$ARM] client concurrency=$C -> $out"
    $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --concurrency $C \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
      --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
    log "  [$ARM c$C] recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  done
  kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
}

for A in $ARMS; do run_arm "$A"; done

log "analyzing sweep"
if ! ARMS="$ARMS" CONCS="$CONCS" SLO_TBT_MS=500 SLO_TBT_MS2=1000 \
     $PYTHON scripts/mlsys/analyze_concsweep.py > logs/csweep_ANALYSIS.txt 2>&1; then
  log "ANALYSIS FAILED"; touch logs/csweep_FAILED; exit 1
fi
echo "[$(STAMP)] DONE" >> logs/csweep_ANALYSIS.txt
touch logs/csweep_ALLDONE
log "done -> logs/csweep_ANALYSIS.txt"
