#!/bin/bash
# rate_sweep_static_reserve.sh -- decode-reserve reorder + a FIXED token_budget (512, never
# relaxed), same non-stationary schedule/workload as the whale-aware-v2 headline run. The
# ablation: does decode-reserve alone (restoring Sarathi-Serve's own decode-phase-priority
# design, which vLLM's actual scheduler drops) already match whale-aware-v2's tail
# protection, or does the dynamic whale-presence gating buy something beyond that?
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
PORT=8051
WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
HI_FRAC=0.3
RATE=1.0
DUR=720
SCHED="${RATE}:0.0@120,${RATE}:${HI_FRAC}@120"
DATE=$(date +%Y-%m-%d)
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

cp ${SCHED_PY}.pristine ${SCHED_PY}
$PYTHON scripts/hotpatch_static_reserve.py || { log "PATCH FAILED"; touch logs/ratesweep_staticreserve_FAILED; exit 1; }

ARM="nsstaticreserve"
FB="logs/${DATE}-lgate-b${ARM}"
FB_TRACE="${FB}-budgettrace.csv"
rm -f "${FB}-t1.jsonl" "$FB_TRACE"

log "[$ARM] starting server (STATIC_RESERVE, budget fixed at 512)"
env CUDA_VISIBLE_DEVICES=2,3 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  STATIC_RESERVE=1 STATIC_RESERVE_BUDGET=512 STATIC_RESERVE_TRACE="$FB_TRACE" \
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
  touch logs/ratesweep_staticreserve_FAILED; exit 1
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

$PYTHON -c "
import csv
rows = list(csv.reader(open('${FB_TRACE}')))
if rows:
    n = len(rows)
    b = set(int(r[1]) for r in rows)
    print(f'[$ARM] n_rounds={n} distinct_budget_values={sorted(b)} (should be exactly {{512}})')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_staticreserve_THRESHOLDS.txt

log "re-analyzing (phase-split): nsmono nsstatic512 ${ARM} nswhaleawarev2"
OUT="logs/ratesweep_staticreserve_ANALYSIS.txt"
SCHEDULE="$SCHED" ARMS="nsmono nsstatic512 ${ARM} nswhaleawarev2" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1 || true
cat "$OUT"

log "pooled (phase-agnostic) comparison"
$PYTHON -c "
import json, os
def pooled(path, label):
    if not os.path.exists(path):
        print(f'{label}: MISSING ({path})')
        return
    recs = [json.loads(l) for l in open(path) if l.strip()]
    gaps = []
    for r in recs:
        gaps.extend(r.get('tbt_ms') or [])
    gaps.sort()
    n = len(gaps)
    def pctl(p): return gaps[min(n-1, int(p/100*n))]
    print(f'{label}: n={n} mean={sum(gaps)/n:.1f} p99={pctl(99):.1f} p99.9={pctl(99.9):.1f} max={gaps[-1]:.1f}')
for label, path in [
    ('mono                  ', 'logs/${DATE}-lgate-bnsmono-t1.jsonl'),
    ('static512             ', 'logs/${DATE}-lgate-bnsstatic512-t1.jsonl'),
    ('static-reserve(ablation)', '${FB}-t1.jsonl'),
    ('whale-aware-v2(real)  ', 'logs/${DATE}-lgate-bnswhaleawarev2-t1.jsonl'),
]:
    pooled(path, label)
" | tee -a "$OUT"

log "done -- see $OUT"
touch logs/ratesweep_staticreserve_ALLDONE
