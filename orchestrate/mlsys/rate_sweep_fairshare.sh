#!/bin/bash
# rate_sweep_fairshare.sh -- validate the fairshare-lpt controller (threshold = budget /
# (1 + running), zero calibrated constants) against the same 3 rate points already fully
# characterized this session (mono/static-512/16384lpt512/adaptivelpt2 all already have data on
# disk -- this script only runs the NEW fairshare arm at each rate, then re-analyzes combining
# old + new data). Also dumps a threshold-distribution summary from the trace at each rate, to
# check whether the formula is actually varying dynamically or just parked at its ceiling
# (16384, never engaging) or floor (127, always maxed out).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
PORT=8050
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

$PYTHON scripts/hotpatch_fairshare_lpt.py || { log "PATCH FAILED"; touch logs/ratesweep_fairshare_FAILED; exit 1; }

run_rate(){ # $1=WRATE $2=TAG(may be empty) $3=ARM_LABEL $4=OLD_ARMS(space-separated)
  local WRATE=$1 TAG=$2 ARM=$3 OLD_ARMS=$4
  local SCHED="6:0.0@60,${WRATE}:0.2@60"; local DUR=360
  local DATE=$(date +%Y-%m-%d)
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "WRATE=$WRATE arm=$ARM: starting server"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    FAIRSHARE_LPT=1 FAIRSHARE_LPT_TRACE="${FB}-chunktrace.csv" \
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
    touch logs/ratesweep_fairshare_FAILED; return 1
  fi
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

  # threshold distribution + ceiling/floor stall check
  $PYTHON -c "
import csv
rows = list(csv.reader(open('${FB}-chunktrace.csv')))
thr = [int(r[2]) for r in rows if len(r) > 2]
if thr:
    s = sorted(thr)
    n = len(s)
    ceiling = sum(1 for x in thr if x >= 16384)
    floor = sum(1 for x in thr if x <= 130)
    mid = n - ceiling - floor
    print(f'[$ARM] n_steps={n}  thr: min={s[0]} p50={s[n//2]} p90={s[int(0.9*n)]} p99={s[int(0.99*n)]} max={s[-1]}')
    print(f'[$ARM] at_ceiling(>=16384)={ceiling} ({100*ceiling/n:.1f}%)  at_floor(<=130)={floor} ({100*floor/n:.1f}%)  mid-range={mid} ({100*mid/n:.1f}%)')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_fairshare_THRESHOLDS.txt

  log "re-analyzing WRATE=$WRATE: ${OLD_ARMS} ${ARM}"
  local OUT="logs/ratesweep_fairshare_${TAG:-w10}_ANALYSIS.txt"
  SCHEDULE="$SCHED" ARMS="${OLD_ARMS} ${ARM}" SLO_TBT_MS=500 \
    $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1
  cat "$OUT"
  echo "[$(STAMP)] DONE ${TAG:-w10}" >> "$OUT"
}

run_rate 0.5 w05 fairsharew05 "16384w05 512w05 lpt512w05 adaptivelptw05"
run_rate 1.0 ""  fairshare    "16384 512 16384lpt512 adaptivelpt2"
run_rate 2.0 w20 fairsharew20 "16384w20 512w20 lpt512w20 adaptivelptw20"

log "done -- see logs/ratesweep_fairshare_{w05,w10,w20}_ANALYSIS.txt and logs/ratesweep_fairshare_THRESHOLDS.txt"
touch logs/ratesweep_fairshare_ALLDONE
