#!/bin/bash
# run_adaptive_lpt_arm.sh -- new arm: adaptive long-prefill-token-threshold (off=0 when calm,
# protect=512 when a long prompt is prefilling/waiting), step-wide budget static at mono (16384).
# Same phase-schedule workload as every other lgate arm. Re-analyzes all 9 arms into
# logs/lgate_ANALYSIS_v5.txt.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
DATE=$(date +%Y-%m-%d)
SCHED="6:0.0@60,1.0:0.2@60"; DUR=360
GATE=4096; PROTECT=512; OFF=0
PORT=8050
ARM=adaptivelpt
FB="logs/${DATE}-lgate-b${ARM}"

rm -f logs/lgate_adaptivelpt_ALLDONE logs/lgate_adaptivelpt_FAILED "${FB}-t1.jsonl" "${FB}-chunktrace.csv"

$PYTHON scripts/mlsys/hotpatch_adaptive_lpt.py || { log "PATCH FAILED"; touch logs/lgate_adaptivelpt_FAILED; exit 1; }

log "arm=$ARM budget=16384 gate=$GATE protect=$PROTECT off=$OFF SCHED='$SCHED' DUR=${DUR}s"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=$GATE ADAPTIVE_LPT_PROTECT=$PROTECT ADAPTIVE_LPT_OFF=$OFF \
  ADAPTIVE_LPT_TRACE="${FB}-chunktrace.csv" \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_adaptivelpt_FAILED; exit 1; }
done
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "[$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

log "re-analyzing all 9 arms"
SCHEDULE="$SCHED" \
  ARMS="16384 512 2048 lengthgate 16384lpt512 16384lpt256 16384lpt2048 2048lpt512 ${ARM}" \
  SLO_TBT_MS=500 $PYTHON scripts/mlsys/analyze_lengthgate.py > logs/lgate_ANALYSIS_v5.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS_v5.txt
touch logs/lgate_adaptivelpt_ALLDONE
log "done -> logs/lgate_ANALYSIS_v5.txt"
