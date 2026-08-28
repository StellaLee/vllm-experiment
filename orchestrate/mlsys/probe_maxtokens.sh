#!/bin/bash
# probe_maxtokens.sh -- how much does raising --max-tokens reduce output-length truncation?
# Single arm (mono, budget=16384, no threshold -- output length is a model-generation property,
# not a scheduling-policy one, so one arm is enough to characterize the distribution). Same
# phase-schedule workload/seed as the calibrated baseline. Runs on GPUs 2,3 / port 8060 so it
# doesn't contend with the rate-sweep on GPUs 0,1.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_probe(){ pkill -TERM -f "venv-vllm023.*api_server.*port 8060" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port 8060" && kill -9 $pid 2>/dev/null; done; sleep 6; }

MT=${MT:?"set MT (max-tokens to test), e.g. MT=512"}
SCHED="6:0.0@60,1.0:0.2@60"; DUR=360
PORT=8060
DATE=$(date +%Y-%m-%d)
FB="logs/${DATE}-mtprobe-mt${MT}"

log "max-tokens probe: MT=$MT SCHED='$SCHED' DUR=${DUR}s (GPUs 2,3, port $PORT)"
rm -f "${FB}-t1.jsonl"
env CUDA_VISIBLE_DEVICES=2,3 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_probe; touch logs/mtprobe_FAILED; exit 1; }
done
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MT \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_probe

$PYTHON -c "
import json, statistics as st
recs = [json.loads(l) for l in open('${FB}-t1.jsonl') if l.strip()]
ot = [r['output_tokens'] for r in recs]
cap = $MT
at_cap = sum(1 for x in ot if x >= cap)
print(f'MT=$MT  n={len(ot)}  mean={st.mean(ot):.1f}  median={st.median(ot):.0f}  p90={sorted(ot)[int(0.9*len(ot))]}  p99={sorted(ot)[int(0.99*len(ot))]}  capped>={cap}: {at_cap} ({100*at_cap/len(ot):.1f}%)')
" | tee logs/mtprobe_${MT}_SUMMARY.txt
touch logs/mtprobe_${MT}_ALLDONE
log "done -> logs/mtprobe_${MT}_SUMMARY.txt"
