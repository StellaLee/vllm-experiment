#!/bin/bash
# rerun_lpt_grid.sh -- 3 more --long-prefill-token-threshold arms to complete the budget x
# threshold grid: mono+lpt256, mono+lpt2048 (threshold-value sweep at mono budget), and
# 2048+lpt512 (does per-request cap add anything on top of an already-shrunk step budget).
# Exact same phase-schedule workload/params as every other lgate arm (see rerun_lpt_arm.sh).
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
PORT=8050

rm -f logs/lgate_grid_ALLDONE logs/lgate_grid_FAILED

run_lpt(){ # $1=arm-label $2=budget $3=threshold
  local ARM=$1
  local BUD=$2
  local LPT=$3
  local FB="logs/${DATE}-lgate-b${ARM}"
  log "arm=$ARM budget=$BUD threshold=$LPT"
  rm -f "${FB}-t1.jsonl"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens $BUD --max-model-len 16384 \
    --long-prefill-token-threshold $LPT \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_grid_FAILED; exit 1; }
  done
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

run_lpt 16384lpt256  16384 256
run_lpt 16384lpt2048 16384 2048
run_lpt 2048lpt512   2048  512

log "re-analyzing full grid"
SCHEDULE="$SCHED" ARMS="16384 512 2048 lengthgate 16384lpt512 16384lpt256 16384lpt2048 2048lpt512" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > logs/lgate_ANALYSIS_v4.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS_v4.txt
touch logs/lgate_grid_ALLDONE
log "done -> logs/lgate_ANALYSIS_v4.txt"
