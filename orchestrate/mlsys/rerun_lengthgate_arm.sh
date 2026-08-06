#!/bin/bash
# rerun_lengthgate_arm.sh -- re-run ONLY the lengthgate arm after the depth==0-blast fix, reusing
# the exact same schedule/params as the original 4-arm run so it stays comparable. Static arms
# (mono/512/2048) are unaffected by the controller fix and are NOT re-run.
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
THRESH=4096; PROTECT=512; BLAST=16384
PORT=8050

log "rerun lengthgate arm (post-fix): SCHED='$SCHED' DUR=${DUR}s thresh=$THRESH protect=$PROTECT blast=$BLAST"

ARM=lengthgate FB="logs/${DATE}-lgate-blengthgate"
rm -f logs/lgate_rerun_ALLDONE logs/lgate_rerun_FAILED "${FB}-chunktrace.csv" "${FB}-t1.jsonl"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
  DYNAMIC_CHUNK=1 CHUNK_MODE=lengthgate DYNAMIC_CHUNK_MIN=$PROTECT DYNAMIC_CHUNK_START=$BLAST \
  LENGTHGATE_THRESHOLD=$THRESH LENGTHGATE_PROTECT=$PROTECT LENGTHGATE_BLAST=$BLAST \
  DYNAMIC_CHUNK_TRACE=${FB}-chunktrace.csv \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_rerun_FAILED; exit 1; }
done
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

log "re-analyzing all 4 arms (3 static untouched + fixed lengthgate)"
SCHEDULE="$SCHED" ARMS="16384 512 2048 lengthgate" SLO_TBT_MS=500 \
  $PYTHON scripts/mlsys/analyze_lengthgate.py > logs/lgate_ANALYSIS_v2.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS_v2.txt
touch logs/lgate_rerun_ALLDONE
log "done -> logs/lgate_ANALYSIS_v2.txt"
