#!/bin/bash
# orchestrate_longprompt_ff.sh -- run the two FEEDFORWARD dynamic strategies (v1 granted-budget basis,
# v2 actual-token basis) under the identical long-prompt whale setup (same pad-seed 1001 -> paired
# whale positions with the static + slocvar arms). Selected via CHUNK_FF_VARIANT (needs
# hotpatch_ff_variant.py applied). Outputs logs/<date>-longp-b{ffv1,ffv2}-t1.jsonl + chunktraces,
# then analyzes ALL arms together. Isolated sequential on GPUs 0,1.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpff_ALLDONE logs/longpff_FAILED
DATE=$(date +%Y-%m-%d)
CONC=20 MAX_SEQS=48 MAXTOK=256 NCONV=200
WHALE_FRAC=0.15 WHALE_MIN=44000 WHALE_MAX=50000 MAX_PROMPT_CHARS=50000 PAD_MEAN=800 PAD_CV2=0.5
FLOOR=512 SLO_MS=50 START=16384

run_ff(){ # arm variant
  local arm=$1 variant=$2 port=8050
  log "  [$arm] server GPUs 0,1 port=$port mode=feedforward variant=$variant start=$START floor=$FLOOR"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
      DYNAMIC_CHUNK=1 CHUNK_MODE=feedforward CHUNK_FF_VARIANT=$variant \
      DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START DYNAMIC_CHUNK_SLO_MS=$SLO_MS \
      DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${arm}-chunktrace.csv \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens 16384 \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local out="logs/${DATE}-longp-${arm}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency $CONC \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$arm] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${arm}-server.log 2>/dev/null || echo 0)"
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null; kill_ours
}

run_ff bffv1 v1
run_ff bffv2 v2
log "analyzing (all arms)"
BUDGETS="16384 2048 512 ours ffv1 ffv2" $PYTHON scripts/analyze_longprompt.py > logs/longpff_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longpff_ANALYSIS.txt
touch logs/longpff_ALLDONE
log "done -> logs/longpff_ANALYSIS.txt"
