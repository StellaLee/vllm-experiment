#!/bin/bash
# rerun_lpt_arm.sh -- test vLLM's NATIVE --long-prefill-token-threshold as a zero-code-change
# alternative to our custom lengthgate controller. Server budget stays at mono's 16384 (full
# throughput ceiling), but any SINGLE request whose own need exceeds the threshold gets capped to
# that threshold THIS STEP, regardless of the overall budget -- so a whale is naturally sliced
# without ever reducing the step's budget for anyone else (structurally avoids both bugs found in
# our custom lengthgate: no depth==0 loophole since the cap isn't gated on our logic, no
# backlog-then-catchup since other admissions were never starved).
# Same phase-schedule workload as every other lgate arm, so results are directly comparable.
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
LPT=${LPT:-512}
PORT=8050

log "rerun mono+LPT arm: SCHED='$SCHED' DUR=${DUR}s long-prefill-token-threshold=$LPT (budget stays 16384)"

ARM="16384lpt${LPT}"; FB="logs/${DATE}-lgate-b${ARM}"
rm -f logs/lgate_lpt_ALLDONE logs/lgate_lpt_FAILED "${FB}-t1.jsonl"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --long-prefill-token-threshold $LPT \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_lpt_FAILED; exit 1; }
done
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "[$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

log "re-analyzing: mono / static-512 / static-2048 / lengthgate / mono+LPT"
SCHEDULE="$SCHED" ARMS="16384 512 2048 lengthgate ${ARM}" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > logs/lgate_ANALYSIS_v3.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS_v3.txt
touch logs/lgate_lpt_ALLDONE
log "done -> logs/lgate_ANALYSIS_v3.txt"
