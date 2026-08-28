#!/bin/bash
# rate_sweep_drr.sh -- quick DRR-lite validation at WRATE=1.0, widened-whale/maxtok=1024
# workload. Reuses mono/static-512 baselines already collected under this exact workload
# (16384fsb10, 512fsb10) from the earlier fairshare_reserve sweep -- only runs the new drr
# arm, then re-analyzes combining old + new.
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

$PYTHON scripts/hotpatch_drr_lpt.py || { log "PATCH FAILED"; touch logs/ratesweep_drr_FAILED; exit 1; }

WRATE=1.0
TAG=10
SCHED="6:0.0@60,${WRATE}:0.2@60"
DUR=360
DATE=$(date +%Y-%m-%d)
ARM="drr${TAG}"
FB="logs/${DATE}-lgate-b${ARM}"
rm -f "${FB}-t1.jsonl" "${FB}-drrtrace.csv"

log "WRATE=$WRATE arm=$ARM: starting server (DRR_LPT)"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  DRR_LPT=1 DRR_QUANTUM=512 DRR_TRACE="${FB}-drrtrace.csv" \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "WRATE=$WRATE: SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/ratesweep_drr_FAILED; exit 1
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
rows = list(csv.reader(open('${FB}-drrtrace.csv')))
# wall_s, n_running, n_waiting, sum_deficit, max_deficit, n_at_bank_cap
if rows:
    n = len(rows)
    at_cap = [int(r[5]) for r in rows]
    max_def = [int(r[4]) for r in rows]
    n_running = [int(r[1]) for r in rows]
    print(f'[$ARM] n_rounds={n}')
    print(f'[$ARM] n_running: min={min(n_running)} p50={sorted(n_running)[n//2]} max={max(n_running)}')
    print(f'[$ARM] max_deficit_per_round: min={min(max_def)} p50={sorted(max_def)[n//2]} max={max(max_def)}')
    frac_any_at_cap = sum(1 for x in at_cap if x > 0) / n
    print(f'[$ARM] rounds with >=1 request banked at cap (2048): {100*frac_any_at_cap:.1f}%')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_drr_THRESHOLDS.txt

log "re-analyzing WRATE=$WRATE: 16384fsb10 512fsb10 ${ARM}"
OUT="logs/ratesweep_drr_${TAG}_ANALYSIS.txt"
SCHEDULE="$SCHED" ARMS="16384fsb10 512fsb10 ${ARM}" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1
cat "$OUT"

log "done -- see $OUT and logs/ratesweep_drr_THRESHOLDS.txt"
touch logs/ratesweep_drr_ALLDONE
