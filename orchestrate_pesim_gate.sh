#!/bin/bash
# orchestrate_pesim_gate.sh -- PES-IM gate experiment: whale-prefill power spike (C3) +
# energy conservation (C1), carried by ONE experiment per the 5-page-limit plan
# (docs/2026-07-24-pes-im-plan.md). 7B, single GPU (no TP -- isolates the measurement from
# NCCL/TP power draw), mono(16384)/chunk(2048)/chunk(512), same bimodal short+whale mix as
# the 14B longprompt-tbt-win result. NVML power+energy side-car logger runs per-(arm,trial);
# GPU clock locked + persistence mode enabled to remove DVFS variance; short warmup burst
# discarded before each measured trial to reach thermal/clock steady-state; IPMI chassis
# sensor snapshotted before/after each arm as a coarse (~2.5s-latency) independent
# cross-check on GPU_Power/Total_Power, not a substitute for the NVML trace.
# Markers: logs/pesim_gate_ALLDONE|FAILED.
# Tunables: GPU BUDGETS CONC MAX_SEQS MAXTOK NCONV WHALE_FRAC WHALE_MIN WHALE_MAX
#           MAX_PROMPT_CHARS NWARMUP LOCK_CLOCK TRIALS.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
DATASET=data/sharegpt_v3.json
GPU=${GPU:-0}
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/pesim_gate_ALLDONE logs/pesim_gate_FAILED
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 2048 512"}      # first = mono baseline
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
NWARMUP=${NWARMUP:-20}
LOCK_CLOCK=${LOCK_CLOCK:-3105}
TRIALS=${TRIALS:-"1"}
log "PES-IM gate: gpu=$GPU budgets=[$BUDGETS] conc=$CONC mt=$MAXTOK nconv=$NCONV whale_frac=$WHALE_FRAC whale=[$WHALE_MIN,$WHALE_MAX]chars lock_clock=${LOCK_CLOCK}MHz trials=[$TRIALS]"

setup_gpu(){
  nvidia-smi -i $GPU -pm 1 >&2
  nvidia-smi -i $GPU -lgc ${LOCK_CLOCK},${LOCK_CLOCK} >&2
}
reset_gpu(){
  nvidia-smi -i $GPU -rgc >&2
}
ipmi_snapshot(){ # tag
  ipmitool sensor list 2>/dev/null | grep -E 'Total_Power|GPU_Power' | while read -r line; do
    log "  [ipmi:$1] $line"
  done
}

run_arm(){ # budget
  local budget=$1 arm="b$1" port=8060
  log "  [$arm] server on GPU $GPU port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-pesimgate-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-pesimgate-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done

  log "  [$arm] warmup ($NWARMUP reqs, discarded)"
  $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NWARMUP --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency $CONC --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 \
    --pad-min 100 --pad-max 8000 --pad-seed 99 \
    --output logs/${DATE}-pesimgate-${arm}-warmup.jsonl > /dev/null 2>&1
  local t0=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  local c0=$(nvidia-smi -i $GPU --query-gpu=clocks.gr --format=csv,noheader,nounits)
  log "  [$arm] post-warmup temp=${t0}C clock=${c0}MHz"
  ipmi_snapshot "${arm}-prerun"

  local rc=0
  for tr in $TRIALS; do
    local out="logs/${DATE}-pesimgate-${arm}-t${tr}.jsonl"
    local ptrace="logs/${DATE}-pesimgate-${arm}-t${tr}-power.csv"
    $PYTHON scripts/power_logger.py --gpus $GPU --interval-ms 50 --output "$ptrace" &
    local plog=$!
    sleep 1
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --concurrency $CONC \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
      --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $((1000+tr)) \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    kill -TERM "$plog" 2>/dev/null; wait "$plog" 2>/dev/null
    log "  [$arm] trial $tr done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-pesimgate-${arm}-server.log 2>/dev/null || echo 0) power_rows=$(wc -l < "$ptrace" 2>/dev/null || echo 0)"
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
BUDGETS="$BUDGETS" DATE="$DATE" TRIALS="$TRIALS" $PYTHON scripts/analyze_pesim_gate.py > logs/pesimgate_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/pesimgate_ANALYSIS.txt
if [ "$RC" = 0 ]; then touch logs/pesim_gate_ALLDONE; else touch logs/pesim_gate_FAILED; fi
log "done -> logs/pesimgate_ANALYSIS.txt"
