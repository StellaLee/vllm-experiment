#!/bin/bash
# rate_sweep_whale_aware_calibrated.sh -- fixes a real calibration gap found by inspecting
# the whale-aware-v2 data: whale-min-chars=10000 (~3091 tok at 3.235 chars/tok) sits BELOW
# the controller's WHALE_AWARE_WHALE_TOK=4000 detection threshold, so ~22.5% of
# workload-generated "whales" (2554-3940 tok observed) were silently getting zero
# protection. Raises whale-min-chars to 18000 (~5563 tok, clears the threshold with real
# margin); max unchanged at 50000 (already validated safe under max-model-len=16384).
# Same mechanism, same seed, same everything else -- isolates the effect of closing this
# one calibration gap.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
PORT=8050
WHALE_MIN=18000
WHALE_MAX=50000
MAXTOK=1024
SCHED="1.0:0.0@120,1.0:0.3@120"
DUR=720
DATE=$(date +%Y-%m-%d)
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

cp ${SCHED_PY}.pristine ${SCHED_PY}
$PYTHON scripts/hotpatch_whale_aware_budget_v2.py || { log "PATCH FAILED"; touch logs/ratesweep_calib_FAILED; exit 1; }

ARM="nswhaleawarecalib"
FB="logs/${DATE}-lgate-b${ARM}"
FB_TRACE="${FB}-whaletrace.csv"
rm -f "${FB}-t1.jsonl" "$FB_TRACE"
log "[$ARM] starting server (WHALE_AWARE_BUDGET, calibrated whale-min-chars=$WHALE_MIN)"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  WHALE_AWARE_BUDGET=1 WHALE_AWARE_WHALE_TOK=4000 \
  WHALE_AWARE_LO_BUDGET=16384 WHALE_AWARE_HI_BUDGET=512 WHALE_AWARE_TRACE="$FB_TRACE" \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "[$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/ratesweep_calib_FAILED; exit 1
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
cp ${SCHED_PY}.pristine ${SCHED_PY}

log "checking whale-token distribution actually cleared the threshold this time"
$PYTHON -c "
import json
recs = [json.loads(l) for l in open('${FB}-t1.jsonl') if l.strip()]
cand = sorted(r['prompt_tokens_approx'] for r in recs if (r.get('prompt_tokens_approx') or 0) > 2500)
below = [w for w in cand if w <= 4000]
print(f'candidate-whale-range (>2500 tok) requests: {len(cand)}')
print(f'  still below WHALE_TOK=4000 threshold: {len(below)}  {below}')
if cand:
    print(f'  min={cand[0]} p50={cand[len(cand)//2]} max={cand[-1]}')
"

log "pooled TBT: original (uncalibrated) vs. calibrated whale distribution"
OUT="logs/ratesweep_calib_ANALYSIS.txt"
$PYTHON -c "
import json
def pooled(path, label):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    gaps = []
    for r in recs:
        gaps.extend(r.get('tbt_ms') or [])
    gaps.sort()
    n = len(gaps)
    def pctl(p): return gaps[min(n-1, int(p/100*n))]
    print(f'{label}: n={n} mean={sum(gaps)/n:.1f} p99={pctl(99):.1f} p99.9={pctl(99.9):.1f} max={gaps[-1]:.1f}')
for label, path in [
    ('whale-aware, uncalibrated (min=10000 chars)', 'logs/${DATE}-lgate-bnswhaleawarev2-t1.jsonl'),
    ('whale-aware, calibrated (min=18000 chars)   ', '${FB}-t1.jsonl'),
]:
    pooled(path, label)
" | tee "$OUT"

log "done -- see $OUT"
touch logs/ratesweep_calib_ALLDONE
