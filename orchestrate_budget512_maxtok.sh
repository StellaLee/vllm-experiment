#!/bin/bash
# orchestrate_budget512_maxtok.sh -- revisit the step-wide BUDGET sweep (mono=16384 vs
# chunk=512, long_prefill_token_threshold OFF) at wf=05/conc=20, crossed with MAXTOK
# {256,1024}. Mono/16384 baselines at both MAXTOK values already exist (thr=0 rows in
# orchestrate_cs2_whale_threshold_sweep.sh and its _maxtok variant) -- this run only adds
# the missing budget=512 arm, one server boot, client run twice (MAXTOK 256 then 1024).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs
DATE=$(date +%Y-%m-%d)
BUDGET=512
WF=0.05
CONC=20
MAX_SEQS=48
NCONV=200
WHALE_MIN=44000; WHALE_MAX=50000
MAX_PROMPT_CHARS=50000
PAD_MEAN=800; PAD_CV2=0.5
PAD_SEED=1001
MAXTOKS="256 1024"
rm -f logs/budget512mt_ALLDONE logs/budget512mt_FAILED
log "budget=512 vs mono(16384, already have) at wf=$WF conc=$CONC, maxtoks=[$MAXTOKS], threshold OFF"

port=8050
log "server on GPUs 0,1 port=$port budget=$BUDGET threshold=0(off) max_seqs=$MAX_SEQS"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUDGET \
    --long-prefill-token-threshold 0 \
    --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
    > logs/${DATE}-budget512mt-server.log 2>&1 &
sv=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" logs/${DATE}-budget512mt-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill "$sv" 2>/dev/null; touch logs/budget512mt_FAILED; exit 1; }
done

RC=0
for mt in $MAXTOKS; do
  out="logs/${DATE}-budget512mt-mt${mt}-w05-c20-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $mt --concurrency $CONC \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WF --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || RC=1
  log "  [maxtok=$mt] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) " \
      "preempt=$(grep -c -i preempt logs/${DATE}-budget512mt-server.log 2>/dev/null || echo 0)"
done

kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
kill_ours

log "analyzing"
DATE="$DATE" $PYTHON scripts/mlsys/analyze_budget512_maxtok.py > logs/budget512mt_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/budget512mt_ANALYSIS.txt
if [ "$RC" = 0 ]; then touch logs/budget512mt_ALLDONE; else touch logs/budget512mt_FAILED; fi
log "done -> logs/budget512mt_ANALYSIS.txt"
