#!/bin/bash
# calibrate_congest_gate.sh -- one instrumented adaptivelpt pass per rate point
# (WRATE=0.5,1.0,2.0), ADAPTIVE_LPT_CONGEST_GATE=0 (feature off, trace-only), to find where
# len(self.running) actually sits during the known-fine (0.5/1.0) vs known-collapsed (2.0)
# regimes before picking a real ADAPTIVE_LPT_CONGEST_GATE/EXIT_GATE value. Per
# docs/superpowers/plans/2026-08-27-congestion-aware-adaptive-lpt.md Task 2 -- do not guess
# these numbers, use the trace.
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
DATE=$(date +%Y-%m-%d)
DUR=360

$PYTHON scripts/hotpatch_adaptive_lpt.py || { log "PATCH FAILED"; touch logs/calibrate_congest_FAILED; exit 1; }

for WRATE in 0.5 1.0 2.0; do
  TAG=$(echo $WRATE | tr -d '.')
  SCHED="6:0.0@60,${WRATE}:0.2@60"
  FB="logs/${DATE}-calibcongest-w${TAG}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "WRATE=$WRATE: starting server (CONGEST_GATE=0, trace-only)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=4096 ADAPTIVE_LPT_PROTECT=512 ADAPTIVE_LPT_OFF=0 \
    ADAPTIVE_LPT_CONGEST_GATE=0 ADAPTIVE_LPT_TRACE="${FB}-chunktrace.csv" \
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
    touch logs/calibrate_congest_FAILED; continue
  fi
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "WRATE=$WRATE: recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

  $PYTHON -c "
import csv
rows = list(csv.reader(open('${FB}-chunktrace.csv')))
running = [int(r[3]) for r in rows if len(r) > 3]
if running:
    s = sorted(running)
    n = len(s)
    print(f'WRATE=${WRATE}  n_steps={n}  running: min={s[0]} p50={s[n//2]} p90={s[int(0.9*n)]} p99={s[int(0.99*n)]} max={s[-1]}')
else:
    print(f'WRATE=${WRATE}  NO TRACE ROWS (check {\"${FB}\"}-chunktrace.csv format)')
" | tee -a logs/calibrate_congest_SUMMARY.txt
done

log "done -> logs/calibrate_congest_SUMMARY.txt"
cat logs/calibrate_congest_SUMMARY.txt
touch logs/calibrate_congest_ALLDONE
