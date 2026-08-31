#!/bin/bash
# replicate_whale_aware_v2.sh -- 2 more same-seed trials of the whale-aware v2 controller
# (the paper's headline constructive result), bringing it up to this project's standard 3x
# replication bar. Trial 1 already exists as logs/2026-08-28-lgate-bnswhaleawarev2-*.
# Produces trials 2 and 3 (nswhaleawarev2b, nswhaleawarev2c), identical workload/seed,
# sequential (same GPU pair).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON=$(command -v python)
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
PORT=8050
WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
SCHED="1.0:0.0@120,1.0:0.3@120"
DUR=720
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

run_trial(){
  local TAG=$1
  local FB=logs/2026-08-28-lgate-bnswhaleawarev2${TAG}
  rm -f ${FB}-t1.jsonl ${FB}-whaletrace.csv
  cp ${SCHED_PY}.pristine ${SCHED_PY}
  $PYTHON scripts/hotpatch_whale_aware_budget_v2.py || { log "[$TAG] PATCH FAILED"; touch logs/replicate_wa_FAILED; return 1; }
  log "[$TAG] starting server"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    WHALE_AWARE_BUDGET=1 WHALE_AWARE_WHALE_TOK=4000 \
    WHALE_AWARE_LO_BUDGET=16384 WHALE_AWARE_HI_BUDGET=512 WHALE_AWARE_TRACE=${FB}-whaletrace.csv \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  local UP=0
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
  done
  if [ "$UP" = 0 ]; then
    log "[$TAG] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
    touch logs/replicate_wa_FAILED; return 1
  fi
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "[$TAG] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  cp ${SCHED_PY}.pristine ${SCHED_PY}
}

run_trial "b" || exit 1
run_trial "c" || exit 1

log "done -- both replicate trials complete"
touch logs/replicate_wa_ALLDONE
