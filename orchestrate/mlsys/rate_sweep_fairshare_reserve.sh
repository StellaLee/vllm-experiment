#!/bin/bash
# rate_sweep_fairshare_reserve.sh -- validate the reserve-corrected fair-share controller
# (fixes the monopolization bug found in fairshare_both: token_budget = reserve + share,
# strictly larger than the per-request threshold `share`, so no single request can consume an
# entire round). Reuses mono/static-512 baselines already collected under this exact workload
# (whale=[10000,50000] uniform, max-tokens=1024) from the earlier fairshare_budget sweep --
# only runs the new fsreserve arm at each rate, then re-analyzes combining old + new data.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
PORT=8050
WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

$PYTHON scripts/hotpatch_fairshare_reserve.py || { log "PATCH FAILED"; touch logs/ratesweep_fsr_FAILED; exit 1; }

run_static(){ # $1=arm-label $2=WRATE $3...=extra server CLI flags
  local ARM=$1; local WRATE=$2; shift 2
  local SCHED="6:0.0@60,${WRATE}:0.2@60"; local DUR=360
  local DATE=$(date +%Y-%m-%d)
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl"
  log "  [$ARM] starting server ($*)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-model-len 16384 "$@" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  local UP=0
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
  done
  if [ "$UP" = 0 ]; then
    log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
    touch logs/ratesweep_fsr_FAILED; return 1
  fi
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_rate(){ # $1=WRATE $2=TAG $3=OLD_ARMS
  local WRATE=$1 TAG=$2 OLD_ARMS=$3
  local SCHED="6:0.0@60,${WRATE}:0.2@60"; local DUR=360
  local DATE=$(date +%Y-%m-%d)
  local ARM="fsreserve${TAG}"
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "WRATE=$WRATE arm=$ARM: starting server (FAIRSHARE_RESERVE)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    FAIRSHARE_RESERVE=1 FAIRSHARE_RESERVE_TRACE="${FB}-chunktrace.csv" \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  local UP=0
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
  done
  if [ "$UP" = 0 ]; then
    log "WRATE=$WRATE: SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
    touch logs/ratesweep_fsr_FAILED; return 1
  fi
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

  $PYTHON -c "
import csv
rows = list(csv.reader(open('${FB}-chunktrace.csv')))
b = [int(r[4]) for r in rows if len(r) > 4]
s = [int(r[3]) for r in rows if len(r) > 4]
if b:
    sb = sorted(b); n = len(sb)
    ss = sorted(s)
    ceiling = sum(1 for x in b if x >= 16384)
    print(f'[$ARM] n_steps={n}  budget: min={sb[0]} p50={sb[n//2]} p90={sb[int(0.9*n)]} p99={sb[int(0.99*n)]} max={sb[-1]}')
    print(f'[$ARM] share: min={ss[0]} p50={ss[n//2]} p90={ss[int(0.9*n)]} max={ss[-1]}  at_budget_ceiling={ceiling} ({100*ceiling/n:.1f}%)')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_fsr_THRESHOLDS.txt

  log "re-analyzing WRATE=$WRATE: ${OLD_ARMS} ${ARM}"
  local OUT="logs/ratesweep_fsr_${TAG}_ANALYSIS.txt"
  SCHEDULE="$SCHED" ARMS="${OLD_ARMS} ${ARM}" SLO_TBT_MS=500 \
    $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1
  cat "$OUT"
}

run_rate 0.5 05 "16384fsb05 512fsb05"
run_rate 1.0 10 "16384fsb10 512fsb10"

# WRATE=2.0 baselines don't exist (earlier sweep was killed before reaching this rate) -- run fresh.
run_static "16384fsb20" 2.0 --max-num-batched-tokens 16384
run_static "512fsb20"   2.0 --max-num-batched-tokens 512
run_rate 2.0 20 "16384fsb20 512fsb20"

log "done -- see logs/ratesweep_fsr_{05,10,20}_ANALYSIS.txt and logs/ratesweep_fsr_THRESHOLDS.txt"
touch logs/ratesweep_fsr_ALLDONE
