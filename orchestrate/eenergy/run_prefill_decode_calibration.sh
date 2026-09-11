#!/bin/bash
# Dedicated prefill/decode energy isolation calibration, replicated 3x within one continuous
# fleet session. See docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md Sec
# 4.2 and scripts/eenergy/check_prefill_decode_energy_regression.py (trial-level regression
# failed -- negative decode coefficient from idle-power/duration confounding).
#
# FIX (2nd revision, after the first replicated run showed j_per_prefill_token was NOT robust,
# CV=76.8%, monotonically drifting across trials, while j_per_decode_token was tight,
# CV=2.4%): the original 15s settle window never actually reached true idle -- measured
# post-burst power was 2-2.5x the true (once-measured, pre-experiment) idle baseline, and that
# residual was itself drifting down across trials. Subtracting one fixed, stale idle baseline
# systematically distorted the small prefill signal far more than the much larger decode
# signal. Fix: measure idle FRESH before every trial (45s, not once at the start), and use
# THAT trial's own fresh measurement to subtract from both its prefill and decode phases --
# removes the drift/staleness problem directly instead of just lengthening the old fixed
# window.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=round_robin
OUTNAME=prefill_decode_calibration
PREFIX=calibration
PHASES_LOG=logs/${PREFIX}_phases_${OUTNAME}.txt

pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
pkill -f "power_logger.py" 2>/dev/null
sleep 3
mkdir -p logs
rm -f "$PHASES_LOG"

POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
  ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
  POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}.csv \
  ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}.csv \
  nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}.log 2>&1 &
echo "launch script pid: $!"

for i in $(seq 1 120); do
  if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
     -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
     >/dev/null 2>&1; then
    echo "router ready after ${i} checks"
    break
  fi
  sleep 5
done

# One extra long settle right at the start too, so trial 1's fresh idle measurement isn't
# taken immediately after replica startup (which itself has residual activity).
echo "startup_settle_start $(date +%s.%N)" >> "$PHASES_LOG"
sleep 30
echo "startup_settle_end $(date +%s.%N)" >> "$PHASES_LOG"

for TRIAL in 1 2 3; do
  CONV_OFFSET=$(( (TRIAL - 1) * 24 ))

  echo "trial${TRIAL}_idle_start $(date +%s.%N)" >> "$PHASES_LOG"
  sleep 45
  echo "trial${TRIAL}_idle_end $(date +%s.%N)" >> "$PHASES_LOG"

  echo "trial${TRIAL}_prefill_start $(date +%s.%N)" >> "$PHASES_LOG"
  python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_prefill_burst_t${TRIAL}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 24 --conv-offset $CONV_OFFSET \
    --max-tokens 1 --whale-frac 1.0 --whale-min-chars 44000 --whale-max-chars 50000 \
    2>&1 | tee logs/${PREFIX}_harness_prefill_burst_t${TRIAL}.log
  echo "trial${TRIAL}_prefill_end $(date +%s.%N)" >> "$PHASES_LOG"

  echo "trial${TRIAL}_settle_start $(date +%s.%N)" >> "$PHASES_LOG"
  sleep 45
  echo "trial${TRIAL}_settle_end $(date +%s.%N)" >> "$PHASES_LOG"

  echo "trial${TRIAL}_decode_start $(date +%s.%N)" >> "$PHASES_LOG"
  python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_decode_burst_t${TRIAL}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 24 --conv-offset $CONV_OFFSET \
    --max-tokens 1024 --whale-frac 0.0 --pad-chars 0 \
    2>&1 | tee logs/${PREFIX}_harness_decode_burst_t${TRIAL}.log
  echo "trial${TRIAL}_decode_end $(date +%s.%N)" >> "$PHASES_LOG"
done

pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
pkill -f "power_logger.py" 2>/dev/null
sleep 3
echo "=== PREFILL_DECODE_CALIBRATION COMPLETE ==="
