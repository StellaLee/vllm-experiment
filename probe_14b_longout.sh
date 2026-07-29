#!/bin/bash
# probe_14b_longout.sh -- does LONGER OUTPUT build a deep, sustained decode batch on the ORIGINAL
# 14B/TP2 setup (comparable to the concsweep)? Little's law: depth = decode_rate x decode_duration,
# so longer generation should keep more requests concurrently decoding. Competing effect: longer
# output = bigger per-seq KV = fewer fit. Measure the NET on depth + db.
# Same as prior whale experiments: 14B, TP2, util 0.90, max-model-len 16384, whale-frac 0.15,
# pad-seed 1001. Only max_tokens (and one conc point) change. max-num-seqs=128 so the seq cap
# doesn't bind at conc 96 (concsweep used 64; irrelevant at conc<=48, noted for the 96 point).
# Baseline row (maxtok 256, conc 48) should reproduce concsweep conc48: depth ~11-31, db ~30ms.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 5; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
rm -f logs/probe14lo_ALLDONE

run_cfg(){ # $1=max_tokens $2=concurrency $3=num_convs
  local MT=$1 C=$2 NC=$3
  local FB="logs/${DATE}-probe14lo-mt${MT}-c${C}"
  local TRACE="${FB}-chunktrace.csv"; rm -f "$TRACE"
  echo ">>> 14B  max_tokens=$MT  conc=$C  nconv=$NC"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
    DYNAMIC_CHUNK_SLO_MS=400 DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
    DYNAMIC_CHUNK_TRACE=$TRACE \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { echo "  SERVER TIMEOUT"; kill $SV 2>/dev/null; kill_ours; return; }; done
  echo -n "  KV pool: "; grep -iE "Available KV cache memory|GPU KV cache size" ${FB}-server.log | head -1 | sed 's/.*INFO[^]]*] //'
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NC --max-turns 1 --min-turns 1 --max-tokens $MT --concurrency $C \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  local PREE=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)
  kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
  TRACE=$TRACE JSONL=${FB}-t1.jsonl PREE=$PREE MT=$MT CONC=$C SLO=400 $PYTHON - <<'PY'
import csv, json, os
f=os.environ['TRACE']
rows=[r for r in csv.DictReader(open(f))] if os.path.exists(f) else []
dep=[int(float(r['depth'])) for r in rows if r.get('depth')]
db=[float(r['signal_ms']) for r in rows if r.get('signal_ms')]
dep=dep[len(dep)//10:]; db=db[len(db)//10:]
dbnz=[v for v in db if v>0]  # nonzero only (avoid estimator-0 artifact on prefill-heavy steps)
# TTFT from the request log = saturation guard
jf=os.environ['JSONL']
recs=[json.loads(l) for l in open(jf)] if os.path.exists(jf) else []
tt=[r['ttft']*1000 for r in recs if r.get('ttft') is not None]
def q(x,p):
    y=sorted(x); return y[min(len(y)-1,int(p/100*len(y)))] if y else 0
slo=float(os.environ['SLO']); dbm=q(dbnz,50)
ttm=sum(tt)/len(tt) if tt else 0
sat="SATURATED?" if ttm>12000 else "ok"
print(f"  mt={os.environ['MT']:>4} conc={os.environ['CONC']:>3} preempt={os.environ['PREE']}  "
      f"TTFT mean/p95={ttm/1000:.1f}/{q(tt,95)/1000:.1f}s [{sat}]  "
      f"depth med/p90/max={q(dep,50)}/{q(dep,90)}/{max(dep) if dep else 0}  "
      f"db(nz) med/p90={dbm:.1f}/{q(dbnz,90):.1f}ms  db/SLO={100*dbm/slo:.0f}%")
PY
}

run_cfg 256  24 300
run_cfg 1500 24 150
run_cfg 1500 36 150
touch logs/probe14lo_ALLDONE
echo "PROBE14LO DONE"
