#!/bin/bash
# orchestrate_cs2_whale_threshold_sweep.sh -- synthesis of everything learned this session:
#   - the whale/bimodal workload is the ONLY design that's shown a clean, reproducible effect
#     so far (orchestrate_longprompt.sh's TBT win, the native-threshold grid's 16384lpt512) --
#     fall back to it instead of the continuous-lognormal open-loop design.
#   - budget-swapping (mono vs chunk=2048) isn't the mechanism-matched test of genuine PS: it
#     still lets one request monopolize whatever budget remains each round. long_prefill_token_
#     threshold caps each request's OWN slice independent of the aggregate budget -- the more
#     genuine per-request-sharing mechanism. So: budget FIXED at 16384, threshold swept.
#   - the open-loop Cs^2 sweep's threshold arm was uniformly WORSE at every Cs^2, with no trend
#     -- traced to utilization being too low (rate=0.05) for genuine admission-side competition
#     to ever occur. CONCURRENCY is the closed-loop knob that controls how much competition
#     exists, so it's swept alongside whale fraction, not held fixed.
# Same whale range (44-50k chars ~=12k tok) and short-population params as orchestrate_longprompt.sh
# (already validated safe, 0 preemptions at CONC=20/whale_frac=0.15/NCONV=200).
# Server boots ONCE per threshold (budget fixed) and serves every (whale_frac, conc) combo.
# Metrics: BOTH TTFT (admission-queue / genuine-PS test) and pooled TBT (decode-protection,
# the mechanism that's actually won before) -- report both, they are different mechanisms.
# Tunables: THRESHOLDS WHALE_FRACS CONCS BUDGET MAX_SEQS MAXTOK NCONV WHALE_MIN WHALE_MAX.
# Markers: logs/cs2wt_ALLDONE|FAILED.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/cs2wt_ALLDONE logs/cs2wt_FAILED
DATE=$(date +%Y-%m-%d)
BUDGET=${BUDGET:-16384}                     # FIXED, large -- never the bottleneck
THRESHOLDS=${THRESHOLDS:-"0 512"}           # first = mono-equivalent baseline (0 = off)
WHALE_FRACS=${WHALE_FRACS:-"0.05 0.15"}
CONCS=${CONCS:-"20 40"}
MAX_SEQS=${MAX_SEQS:-48}                    # matches orchestrate_longprompt.sh's validated-safe value
MAXTOK=${MAXTOK:-256}
NCONV=${NCONV:-200}
WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
PAD_SEED=${PAD_SEED:-1001}                  # matches orchestrate_longprompt.sh -- paired whales
log "cs2 whale/threshold/concurrency sweep: budget(fixed)=$BUDGET thresholds=[$THRESHOLDS] whale_fracs=[$WHALE_FRACS] concs=[$CONCS] max_seqs=$MAX_SEQS nconv=$NCONV whale=[$WHALE_MIN,$WHALE_MAX]chars"

run_threshold(){ # threshold
  local thr=$1 arm="t$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$BUDGET threshold=$thr max_seqs=$MAX_SEQS"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUDGET \
      --long-prefill-token-threshold $thr \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-cs2wt-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-cs2wt-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for wf in $WHALE_FRACS; do
    for conc in $CONCS; do
      local wtag=$($PYTHON -c "print(f'{round(float(\"$wf\")*100):02d}')")
      local out="logs/${DATE}-cs2wt-${arm}-w${wtag}-c${conc}-t1.jsonl"
      $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
        --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
        --max-tokens $MAXTOK --concurrency $conc \
        --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
        --whale-frac $wf --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
        --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
        --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
      log "  [$arm wf=${wf} conc=${conc}] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) " \
          "preempt=$(grep -c -i preempt logs/${DATE}-cs2wt-${arm}-server.log 2>/dev/null || echo 0)"
    done
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for t in $THRESHOLDS; do run_threshold "$t" || RC=1; done
log "analyzing"
THRESHOLDS="$THRESHOLDS" WHALE_FRACS="$WHALE_FRACS" CONCS="$CONCS" \
  $PYTHON scripts/mlsys/analyze_cs2_whale_threshold_sweep.py > logs/cs2wt_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/cs2wt_ANALYSIS.txt
if [ "$RC" = 0 ]; then touch logs/cs2wt_ALLDONE; else touch logs/cs2wt_FAILED; fi
log "done -> logs/cs2wt_ANALYSIS.txt"
