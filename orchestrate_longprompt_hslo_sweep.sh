#!/bin/bash
# orchestrate_longprompt_hslo_sweep.sh -- sweep the hslo controller's per-iteration SLO target over
# {200,600,1200} ms under the identical long-prompt whale setup (pad-seed 1001, paired whales). The
# hslo@50ms arm railed to the 512 floor BY DESIGN: 50ms only buys ~59 prefill tokens after decode
# (headroom/alpha). The interior 2048 knee needs headroom ~ 2048*alpha ~ 760ms, i.e. SLO ~ 800ms.
# This sweep tests whether hslo lands in the interior once the target actually permits it -- and how
# TTFT tax / TBT-p99 trade off across the target. warmup stays at start=16384 (SLO is the only var),
# so the first-whale 6s max leak is EXPECTED to persist in every arm; the discriminator is p99 + the
# controller dwell (does it hold 1024-2048 for a mid SLO instead of railing?).
# Requires scripts/hotpatch_hslo.py (idempotent; reapplied here).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longphslosweep_ALLDONE logs/longphslosweep_FAILED
DATE=$(date +%Y-%m-%d)
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
FLOOR=${FLOOR:-512}; START=${START:-16384}; ALPHA_MIN=${ALPHA_MIN:-256}
SLOS=${SLOS:-"200 600 1200"}
PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH FAILED"; touch logs/longphslosweep_FAILED; exit 1; }

run_arm(){ # slo_ms
  local slo=$1 arm=bhslo$1
  log "  [$arm] server GPUs 0,1 port=$PORT mode=hslo slo=${slo}ms floor=$FLOOR start=$START alpha_min=$ALPHA_MIN"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
      DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START \
      DYNAMIC_CHUNK_SLO_MS=$slo DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN \
      DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${arm}-chunktrace.csv \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens 16384 \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local out="logs/${DATE}-longp-${arm}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency $CONC \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$arm] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${arm}-server.log 2>/dev/null || echo 0)"
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null; kill_ours
}

for slo in $SLOS; do run_arm "$slo" || { log "arm $slo FAILED"; touch logs/longphslosweep_FAILED; }; done
log "analyzing (baselines + hslo sweep)"
BUDGETS="16384 2048 512 hslo hslo200 hslo600 hslo1200" $PYTHON scripts/analyze_longprompt.py > logs/longphslosweep_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longphslosweep_ANALYSIS.txt
touch logs/longphslosweep_ALLDONE
log "done -> logs/longphslosweep_ANALYSIS.txt"
