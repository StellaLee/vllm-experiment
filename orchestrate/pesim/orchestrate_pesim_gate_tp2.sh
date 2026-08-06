#!/bin/bash
# orchestrate_pesim_gate_tp2.sh -- generalization check for the PES-IM gate finding:
# does "peak power flat, ramp-rate/duty-cycle shifts with chunk budget" survive a larger
# model AND real tensor parallelism (TP=2), instead of the single-GPU 7B main condition?
# Same experimental design as orchestrate_pesim_gate.sh (same bimodal short+whale traffic,
# same budgets/trials/warmup/clock-lock discipline) so results are directly comparable --
# only the model (14B) and GPU count (2, TP=2) change.
#
# Power logging: power_logger.py --gpus $GPUS --sum-gpus writes ONE summed row per sample
# (both GPUs' power/energy added together, since the facility-relevant quantity is total
# draw across the TP group, not per-chip) -- so analyze_pesim_gate.py and every downstream
# script work UNCHANGED on this trace, same as the single-GPU case.
#
# GPU pair: default 0,1 -- PIX-connected (single PCIe bridge), the best-available pairing on
# this 8x4090 box (no NVLink on these consumer cards; see docs/2026-07-20-tp-nccl-comms-caveat.md
# for why TP=2 here is host-staged NCCL, which if anything biases AGAINST seeing a clean result,
# not in its favor).
# Markers: logs/${NAME}_ALLDONE|FAILED.
# Tunables: GPUS BUDGETS CONC RATE MAX_SEQS MAXTOK NCONV WHALE_FRAC WHALE_MIN WHALE_MAX
#           PAD_MIN PAD_MAX MAX_PROMPT_CHARS NWARMUP LOCK_CLOCK TRIALS TAG.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
GPUS=${GPUS:-0,1}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TAG=${TAG:-}
NAME="pesim_gate_tp2_14b${TAG:+_$TAG}"
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/${NAME}_ALLDONE logs/${NAME}_FAILED
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 2048 512"}      # first = mono baseline
CONC=${CONC:-20}; RATE=${RATE:-}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
WHALE_PARETO_ALPHA=${WHALE_PARETO_ALPHA:-0}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}; PAD_MIN=${PAD_MIN:-100}; PAD_MAX=${PAD_MAX:-8000}
NWARMUP=${NWARMUP:-20}
LOCK_CLOCK=${LOCK_CLOCK:-3105}
TRIALS=${TRIALS:-"1"}
ARRIVAL_DESC="conc=$CONC (closed-loop)"; [ -n "$RATE" ] && ARRIVAL_DESC="rate=${RATE}/s (open-loop Poisson)"
log "PES-IM gate TP2/14B [$NAME]: gpus=$GPUS budgets=[$BUDGETS] $ARRIVAL_DESC mt=$MAXTOK nconv=$NCONV whale_frac=$WHALE_FRAC whale=[$WHALE_MIN,$WHALE_MAX]chars pad=[$PAD_MIN,$PAD_MAX]chars lock_clock=${LOCK_CLOCK}MHz trials=[$TRIALS]"

setup_gpu(){
  for g in "${GPU_ARR[@]}"; do
    nvidia-smi -i "$g" -pm 1 >&2
    nvidia-smi -i "$g" -lgc ${LOCK_CLOCK},${LOCK_CLOCK} >&2
  done
}
reset_gpu(){
  for g in "${GPU_ARR[@]}"; do nvidia-smi -i "$g" -rgc >&2; done
}
ipmi_snapshot(){ # tag
  ipmitool sensor list 2>/dev/null | grep -E 'Total_Power|GPU_Power' | while read -r line; do
    log "  [ipmi:$1] $line"
  done
}

run_arm(){ # budget
  local budget=$1 arm="b$1" port=${PORT:-8060}
  local prefix="logs/${DATE}-${NAME}-${arm}"
  log "  [$arm] server on GPUs $GPUS (TP=2) port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > "${prefix}-server.log" 2>&1 &
  local sv=$!
  for i in $(seq 1 180); do sleep 5
    grep -q "Application startup complete" "${prefix}-server.log" && break
    [ "$i" = 180 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done

  log "  [$arm] warmup ($NWARMUP reqs, discarded, always closed-loop)"
  $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NWARMUP --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency $CONC --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 \
    --pad-min $PAD_MIN --pad-max $PAD_MAX --pad-seed 99 \
    --output "${prefix}-warmup.jsonl" > /dev/null 2>&1
  for g in "${GPU_ARR[@]}"; do
    local t0=$(nvidia-smi -i "$g" --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    local c0=$(nvidia-smi -i "$g" --query-gpu=clocks.gr --format=csv,noheader,nounits)
    log "  [$arm] post-warmup gpu=$g temp=${t0}C clock=${c0}MHz"
  done
  ipmi_snapshot "${arm}-prerun"

  local arrival_args=(--concurrency "$CONC")
  [ -n "$RATE" ] && arrival_args=(--rate "$RATE")

  local rc=0
  for tr in $TRIALS; do
    local out="${prefix}-t${tr}.jsonl"
    local ptrace="${prefix}-t${tr}-power.csv"
    $PYTHON scripts/pesim/power_logger.py --gpus $GPUS --sum-gpus --interval-ms 50 --output "$ptrace" &
    local plog=$!
    sleep 1
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK "${arrival_args[@]}" \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min $PAD_MIN --pad-max $PAD_MAX \
      --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
      --whale-pareto-alpha $WHALE_PARETO_ALPHA \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $((1000+tr)) \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    kill -TERM "$plog" 2>/dev/null; wait "$plog" 2>/dev/null
    log "  [$arm] trial $tr done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt "${prefix}-server.log" 2>/dev/null || echo 0) power_rows=$(wc -l < "$ptrace" 2>/dev/null || echo 0)"
  done
  ipmi_snapshot "${arm}-postrun"
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

setup_gpu
RC=0
for b in $BUDGETS; do run_arm "$b" || RC=1; done
reset_gpu
log "analyzing"
NAME="$NAME" BUDGETS="$BUDGETS" DATE="$DATE" TRIALS="$TRIALS" $PYTHON scripts/pesim/analyze_pesim_gate.py > logs/${NAME}_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/${NAME}_ANALYSIS.txt
if [ "$RC" = 0 ]; then touch logs/${NAME}_ALLDONE; else touch logs/${NAME}_FAILED; fi
log "done -> logs/${NAME}_ANALYSIS.txt"
