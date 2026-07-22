#!/bin/bash
# orchestrate_longprompt_nonstationary.sh -- E2 (Option A): non-stationary concurrency phases.
# Whales present throughout; concurrency alternates LOW<->HIGH over wall-clock phases (SCHEDULE), so
# the SLO-correct prefill budget MOVES: static-2048 is clean in the low phase but violates the SLO in
# the high phase (db_high + 2048*alpha > SLO); static-512 pays TTFT in the low phase; hslo tracks
# both. Four arms, isolated sequential, paired workload (pad-seed 1001), per-phase analysis.
#
# DE-RISK (run BEFORE trusting phase params): confirm decode_baseline moves a meaningful fraction of
# the 400ms SLO across the concurrency range, else the optimum barely shifts. Quick check from the
# existing hslo trace:
#   /root/pli/venv-vllm023/bin/python scripts/plot_chunk_trace.py logs/2026-07-22-longp-bhslo400af-chunktrace.csv
# (inspect signal_ms = db vs depth). If db(conc40) is not a large fraction of the SLO, raise HIGH
# toward 48 and/or WHALE_FRAC before running. Default SCHEDULE below assumes the separation holds.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpns_ALLDONE logs/longpns_FAILED
DATE=$(date +%Y-%m-%d)
MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
SCHEDULE=${SCHEDULE:-"8@50,40@50"}; DURATION=${DURATION:-300}
FLOOR=${FLOOR:-512}; START=${START:-512}; SLO_MS=${SLO_MS:-400}; ALPHA_MIN=${ALPHA_MIN:-256}; ALPHA_HW_MS=${ALPHA_HW_MS:-0.18}
PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH(base) FAILED"; touch logs/longpns_FAILED; exit 1; }
$PYTHON scripts/hotpatch_hslo_alphafloor.py || { log "PATCH(alphafloor) FAILED"; touch logs/longpns_FAILED; exit 1; }

log "E2 non-stationary: SCHEDULE='$SCHEDULE' DURATION=${DURATION}s whale_frac=$WHALE_FRAC"

# arm spec: label | budget-or-hslo. For hslo we pass mode env; budget arms just set --max-num-batched-tokens.
run_arm(){ # $1=label $2=mode(static|hslo) $3=budget(for static)
  local ARM=$1 MODE=$2 BUD=$3
  log "  [$ARM] server GPUs 0,1 port=$PORT mode=$MODE budget=$BUD"
  local FB="logs/${DATE}-longp-b${ARM}"
  local EXTRA="DYNAMIC_CHUNK=0"
  if [ "$MODE" = "hslo" ]; then
    EXTRA="DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN DYNAMIC_CHUNK_ALPHA_MIN=$ALPHA_HW_MS DYNAMIC_CHUNK_TRACE=${FB}-chunktrace.csv"
    BUD=16384   # hslo controls the budget dynamically; server ceiling stays 16384
  fi
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 $EXTRA \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUD \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours; touch logs/longpns_FAILED; exit 1; }
  done
  local out="${FB}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency-schedule "$SCHEDULE" --duration $DURATION \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$ARM] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
}

run_arm 16384ns   static 16384
run_arm 2048ns    static 2048
run_arm 512ns     static 512
run_arm hslo400ns hslo   16384

log "analyzing (per-phase TBT/TTFT + hslo budget by phase)"
if ! SCHEDULE="$SCHEDULE" ARMS="16384ns 2048ns 512ns hslo400ns" \
     $PYTHON scripts/analyze_nonstationary.py > logs/longpns_ANALYSIS.txt 2>&1; then
  log "ANALYSIS FAILED"; touch logs/longpns_FAILED; exit 1
fi
echo "[$(STAMP)] DONE" >> logs/longpns_ANALYSIS.txt
touch logs/longpns_ALLDONE
log "done -> logs/longpns_ANALYSIS.txt"
