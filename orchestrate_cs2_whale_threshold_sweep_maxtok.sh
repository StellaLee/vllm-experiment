#!/bin/bash
# orchestrate_cs2_whale_threshold_sweep_maxtok.sh -- same grid as
# orchestrate_cs2_whale_threshold_sweep.sh (whale/threshold/concurrency), but sweeps
# MAXTOK as a fourth axis to test whether the three-way TTFT/TPOT tradeoff found at
# MAXTOK=256 (findings/2026-07-23-whale-threshold-concurrency-tradeoff.md) is robust to
# longer output lengths -- longer decode phases mean more decode steps exposed to whale
# interference per request, and closed-loop concurrency replaces completed requests less
# often, changing effective load. Output files tagged with MAXTOK so the original
# MAXTOK=256 dataset is never touched.
# Tunables: THRESHOLDS WHALE_FRACS CONCS BUDGET MAX_SEQS MAXTOK NCONV WHALE_MIN WHALE_MAX.
# Markers: logs/cs2wtmt_${MAXTOK}_ALLDONE|FAILED.
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
BUDGET=${BUDGET:-16384}
THRESHOLDS=${THRESHOLDS:-"0 512"}
WHALE_FRACS=${WHALE_FRACS:-"0.05 0.15"}
CONCS=${CONCS:-"20 40"}
MAX_SEQS=${MAX_SEQS:-48}
MAXTOK=${MAXTOK:-1024}
NCONV=${NCONV:-200}
WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
PAD_SEED=${PAD_SEED:-1001}
rm -f "logs/cs2wtmt_${MAXTOK}_ALLDONE" "logs/cs2wtmt_${MAXTOK}_FAILED"
log "cs2 whale/threshold/concurrency/maxtok sweep: budget(fixed)=$BUDGET thresholds=[$THRESHOLDS] whale_fracs=[$WHALE_FRACS] concs=[$CONCS] max_seqs=$MAX_SEQS maxtok=$MAXTOK nconv=$NCONV whale=[$WHALE_MIN,$WHALE_MAX]chars"

run_threshold(){ # threshold
  local thr=$1 arm="t$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$BUDGET threshold=$thr max_seqs=$MAX_SEQS maxtok=$MAXTOK"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUDGET \
      --long-prefill-token-threshold $thr \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-cs2wtmt${MAXTOK}-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-cs2wtmt${MAXTOK}-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for wf in $WHALE_FRACS; do
    for conc in $CONCS; do
      local wtag=$($PYTHON -c "print(f'{round(float(\"$wf\")*100):02d}')")
      local out="logs/${DATE}-cs2wtmt${MAXTOK}-${arm}-w${wtag}-c${conc}-t1.jsonl"
      $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
        --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
        --max-tokens $MAXTOK --concurrency $conc \
        --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
        --whale-frac $wf --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
        --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
        --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
      log "  [$arm wf=${wf} conc=${conc}] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) " \
          "preempt=$(grep -c -i preempt logs/${DATE}-cs2wtmt${MAXTOK}-${arm}-server.log 2>/dev/null || echo 0)"
    done
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for t in $THRESHOLDS; do run_threshold "$t" || RC=1; done
log "analyzing"
OUT_ANALYSIS="logs/cs2wtmt_${MAXTOK}_ANALYSIS.txt"
THRESHOLDS="$THRESHOLDS" WHALE_FRACS="$WHALE_FRACS" CONCS="$CONCS" MAXTOK="$MAXTOK" DATE="$DATE" \
  $PYTHON scripts/mlsys/analyze_cs2_whale_threshold_maxtok_sweep.py > "$OUT_ANALYSIS" 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> "$OUT_ANALYSIS"
if [ "$RC" = 0 ]; then touch "logs/cs2wtmt_${MAXTOK}_ALLDONE"; else touch "logs/cs2wtmt_${MAXTOK}_FAILED"; fi
log "done -> $OUT_ANALYSIS"
