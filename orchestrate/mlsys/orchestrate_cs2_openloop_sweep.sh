#!/bin/bash
# orchestrate_cs2_openloop_sweep.sh -- genuine-term Cs^2 sweep, textbook-faithful design.
# Unlike the whale-fraction sweep (discrete bimodal mixture, non-monotonic Cs^2, closed-loop,
# TBT metric -- the wrong tool for testing Eq. genuine), this uses:
#   - a SINGLE continuous lognormal prompt-length distribution (--pad-mean-chars/--pad-cv2,
#     see sample_pad_len() docstring: Var/E^2 = pad_cv2 by construction) -- no whale mode,
#     so Cs^2 is a clean, monotonic, single-parameter dial;
#   - a large FIXED mean (~9-10k tokens) so E[S_prefill] stays roughly constant across the
#     sweep while only Cs^2 moves -- isolating the Cs^2 term in Eq. genuine;
#   - OPEN-LOOP Poisson arrivals (--rate), matching how Eq. genuine/P-K is actually derived
#     (closed-loop concurrency was part of why the whale sweep was unstable);
#   - TTFT (admission-queue wait) as the metric, not TBT -- TBT protection is a DIFFERENT,
#     absolute-chunk-size-driven mechanism, not the classical Cs^2 one.
# Server boots ONCE per budget and serves every cv2 point (pad-cv2 is client-side).
# Tunables: BUDGETS CV2_LIST RATE NCONV MAXTOK PAD_MEAN PAD_MIN PAD_MAX MAX_PROMPT_CHARS.
# Markers: logs/cs2ol_r${RATE}_n${NCONV}_ALLDONE|FAILED.
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
BUDGETS=${BUDGETS:-"16384 2048"}          # first = mono baseline
CV2_LIST=${CV2_LIST:-"0.1 0.5 1 2 4 8"}
RATE=${RATE:-0.05}                        # conv/s, open-loop Poisson
NCONV=${NCONV:-90}
MAXTOK=${MAXTOK:-256}
# Measured directly against the real tokenizer (filler text is denser than assumed --
# the unique per-repetition [req ci.turn] tag breaks up BPE merges): 3.235 chars/token,
# not ~3.8-4. pad_max=46000 chars -> ~14.2k tokens + 256 decode = ~14.5k, safe margin
# under max-model-len=16384 even with real ShareGPT content added on top.
PAD_MEAN=${PAD_MEAN:-20000}                # chars, ~6.2k tokens -- large-E[S_prefill] regime
PAD_MIN=${PAD_MIN:-200}
PAD_MAX=${PAD_MAX:-46000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-46000}
PAD_SEED=${PAD_SEED:-2001}                 # fixed across every (budget,cv2) -> paired comparison
rm -f "logs/cs2ol_r${RATE}_n${NCONV}_ALLDONE" "logs/cs2ol_r${RATE}_n${NCONV}_FAILED"
log "cs2 open-loop sweep: budgets=[$BUDGETS] cv2=[$CV2_LIST] rate=$RATE nconv=$NCONV mt=$MAXTOK pad_mean=$PAD_MEAN[$PAD_MIN,$PAD_MAX] seed=$PAD_SEED"

run_budget(){ # budget
  local budget=$1 arm="b$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs 48 --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-cs2ol-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-cs2ol-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for cv2 in $CV2_LIST; do
    local ctag=$($PYTHON -c "print(f'{round(float(\"$cv2\")*100):04d}')")
    local out="logs/${DATE}-cs2ol-${arm}-c${ctag}-r${RATE}-n${NCONV}-t1.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --rate $RATE \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $cv2 --pad-min $PAD_MIN --pad-max $PAD_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm cv2=$cv2] done recs=$(grep -c . "$out" 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for b in $BUDGETS; do run_budget "$b" || RC=1; done
log "analyzing"
OUT_ANALYSIS="logs/cs2ol_r${RATE}_n${NCONV}_ANALYSIS.txt"
BUDGETS="$BUDGETS" CV2_LIST="$CV2_LIST" RATE="$RATE" NCONV="$NCONV" \
  $PYTHON scripts/mlsys/analyze_cs2_openloop_sweep.py > "$OUT_ANALYSIS" 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> "$OUT_ANALYSIS"
if [ "$RC" = 0 ]; then touch "logs/cs2ol_r${RATE}_n${NCONV}_ALLDONE"; else touch "logs/cs2ol_r${RATE}_n${NCONV}_FAILED"; fi
log "done -> $OUT_ANALYSIS"
