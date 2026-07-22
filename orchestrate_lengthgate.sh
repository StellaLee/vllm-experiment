#!/bin/bash
# orchestrate_lengthgate.sh -- length-gated dynamic chunking, four arms over a non-stationary
# whale-fraction schedule (open-loop, paired arrivals). Arms: mono 16384 / static 512 / static 2048
# / lengthgate (dynamic). Per-phase-type analysis (S: TTFT/throughput, W: tail/goodput). The SCHEDULE
# must be the pilot-calibrated one (Phase S prefill-bound, both phases sub-saturation).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/lgate_ALLDONE logs/lgate_FAILED
DATE=$(date +%Y-%m-%d)
SCHED=${SCHED:-"12:0.0@45,3:0.2@45"}; DUR=${DUR:-360}   # pilot-calibrated; >=2 full S<->W cycles
THRESH=${THRESH:-4096}; PROTECT=${PROTECT:-512}; BLAST=${BLAST:-16384}
PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH(hslo base) FAILED"; touch logs/lgate_FAILED; exit 1; }
$PYTHON scripts/hotpatch_lengthgate.py || { log "PATCH(lengthgate) FAILED"; touch logs/lgate_FAILED; exit 1; }

log "lengthgate: SCHED='$SCHED' DUR=${DUR}s thresh=$THRESH protect=$PROTECT blast=$BLAST"

run_arm(){ # $1=label $2=mode(static|lengthgate) $3=budget
  local ARM=$1 MODE=$2 BUD=$3 FB="logs/${DATE}-lgate-b$1"
  local EXTRA="DYNAMIC_CHUNK=0"
  if [ "$MODE" = "lengthgate" ]; then
    EXTRA="DYNAMIC_CHUNK=1 CHUNK_MODE=lengthgate DYNAMIC_CHUNK_MIN=$PROTECT DYNAMIC_CHUNK_START=$BLAST LENGTHGATE_THRESHOLD=$THRESH LENGTHGATE_PROTECT=$PROTECT LENGTHGATE_BLAST=$BLAST DYNAMIC_CHUNK_TRACE=${FB}-chunktrace.csv"
    BUD=16384
  fi
  log "  [$ARM] server mode=$MODE budget=$BUD"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 $EXTRA \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens $BUD --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_FAILED; exit 1; }; done
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_arm 16384      static     16384
run_arm 512        static     512
run_arm 2048       static     2048
run_arm lengthgate lengthgate 16384

log "analyzing per phase-type"
if ! SCHEDULE="$SCHED" ARMS="16384 512 2048 lengthgate" SLO_TBT_MS=500 \
     $PYTHON scripts/analyze_lengthgate.py > logs/lgate_ANALYSIS.txt 2>&1; then
  log "ANALYSIS FAILED"; touch logs/lgate_FAILED; exit 1
fi
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS.txt
touch logs/lgate_ALLDONE
log "done -> logs/lgate_ANALYSIS.txt"
