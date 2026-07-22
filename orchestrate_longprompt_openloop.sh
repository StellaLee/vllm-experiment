#!/bin/bash
# orchestrate_longprompt_openloop.sh -- E1: open-loop Poisson version of the whale regime check.
# Identical bimodal-whale workload to orchestrate_longprompt_hslo_af.sh (whales 44k-50k chars, cap
# 50k, pad-seed 1001 -- NOT the older orchestrate_longprompt.sh 48k-60k defaults), but drives arrivals with --rate
# (open-loop Poisson) instead of --concurrency. RATE is derived from the realized throughput of the
# 07-21 closed-loop conc-20 chunk-2048 run (Little's law: same mean in-flight population that produced
# the original signal), unless overridden by env RATE. VALIDITY GATE: a clean-mono result (P99 TBT ~
# decode baseline) means "raise RATE and re-run", never a negative result -- see analysis footer.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpol_ALLDONE logs/longpol_FAILED
DATE=$(date +%Y-%m-%d)
MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
REF=${REF:-logs/2026-07-21-longp-b2048-t1.jsonl}
PORT=8050
BUDGETS_RUN="16384 2048 512"

# Derive RATE from the reference closed-loop run's realized throughput (conv/s) unless overridden.
if [ -z "${RATE:-}" ]; then
  RATE=$($PYTHON - "$REF" <<'PY'
import sys, json, os
sys.path.insert(0, "scripts"); import replay_timing as rt
p = sys.argv[1]
recs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
tp = rt.realized_throughput(recs)
print(f"{tp:.3f}" if tp > 0 else "1.5")
PY
)
fi
log "E1 open-loop: RATE=${RATE} conv/s (ref=$REF)  arms: $BUDGETS_RUN"

for B in $BUDGETS_RUN; do
  ARM=b${B}ol
  log "  [$ARM] server GPUs 0,1 port=$PORT budget=$B"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens $B \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
  SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longpol_FAILED; exit 1; }
  done
  out="logs/${DATE}-longp-${ARM}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --rate $RATE \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$ARM] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${ARM}-server.log 2>/dev/null || echo 0)"
  kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
done

log "analyzing (open-loop; mono baseline + realized concurrency validity)"
# core metric via the existing analyzer, over the -ol arm files
BUDGETS="16384ol 2048ol 512ol" $PYTHON scripts/analyze_longprompt.py > logs/longpol_ANALYSIS.txt 2>&1
# validity gate: realized mean/max concurrency per arm (must be non-trivial for a valid run)
$PYTHON - "$DATE" >> logs/longpol_ANALYSIS.txt 2>&1 <<'PY'
import sys, json, glob
sys.path.insert(0, "scripts"); import replay_timing as rt
date = sys.argv[1]
print("\n=== validity gate: realized concurrency (Little's law check) ===")
print(f"driven RATE derived from reference throughput; a live decode batch is required for signal.")
for arm in ("16384ol", "2048ol", "512ol"):
    recs = []
    for f in glob.glob(f"logs/{date}-longp-b{arm}-t1.jsonl"):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    mean, mx = rt.realized_concurrency(recs)
    tp = rt.realized_throughput(recs)
    print(f"  {arm:8s}: n={len(recs):4d}  realized_conc mean={mean:5.1f} max={mx:3d}  throughput={tp:.3f}/s")
print("\nVALIDITY: if the mono (16384ol) P99 TBT above is ~ its decode baseline (no multi-second tail),")
print("the rate was too low (no decoders to freeze) -> RAISE RATE and re-run; do NOT read as a null.")
PY
echo "[$(STAMP)] DONE" >> logs/longpol_ANALYSIS.txt
touch logs/longpol_ALLDONE
log "done -> logs/longpol_ANALYSIS.txt"
