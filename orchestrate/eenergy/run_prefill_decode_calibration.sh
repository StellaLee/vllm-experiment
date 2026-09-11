#!/bin/bash
# Dedicated prefill/decode energy isolation calibration. See
# docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md Sec 4.2 and
# scripts/eenergy/check_prefill_decode_energy_regression.py, which found that a trial-level
# aggregate regression cannot cleanly separate prefill vs decode energy cost -- idle power
# dominates and duration confounds with token counts, producing a physically-impossible
# negative decode coefficient despite R^2=0.99. This isolates each phase directly instead,
# the same way this project already calibrated the ramp ceiling (see the comment in
# launch_router_experiment.sh: "burst of 24 concurrent 45k-char prefills against one idle
# replica"): an idle baseline window, a pure-prefill burst (max_tokens=1, whale-sized
# prompts), and a decode-heavy burst (natural-length prompts, max_tokens=1024) -- ONE
# continuous power trace across all three phases, sliced by wall-clock phase boundaries for
# analysis (scripts/eenergy/check_prefill_decode_calibration_result.py).
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

echo "idle_start $(date +%s.%N)" >> "$PHASES_LOG"
sleep 30
echo "idle_end $(date +%s.%N)" >> "$PHASES_LOG"

echo "prefill_start $(date +%s.%N)" >> "$PHASES_LOG"
python3 src/replay_sharegpt.py \
  --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
  --dataset data/sharegpt_v3.json --request-timeout 180 \
  --output logs/${PREFIX}_records_prefill_burst.jsonl \
  --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 24 --max-tokens 1 \
  --whale-frac 1.0 --whale-min-chars 44000 --whale-max-chars 50000 \
  2>&1 | tee logs/${PREFIX}_harness_prefill_burst.log
echo "prefill_end $(date +%s.%N)" >> "$PHASES_LOG"

echo "settle_start $(date +%s.%N)" >> "$PHASES_LOG"
sleep 15
echo "settle_end $(date +%s.%N)" >> "$PHASES_LOG"

echo "decode_start $(date +%s.%N)" >> "$PHASES_LOG"
python3 src/replay_sharegpt.py \
  --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
  --dataset data/sharegpt_v3.json --request-timeout 180 \
  --output logs/${PREFIX}_records_decode_burst.jsonl \
  --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 24 --max-tokens 1024 \
  --whale-frac 0.0 --pad-chars 0 \
  2>&1 | tee logs/${PREFIX}_harness_decode_burst.log
echo "decode_end $(date +%s.%N)" >> "$PHASES_LOG"

pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
pkill -f "power_logger.py" 2>/dev/null
sleep 3
echo "=== PREFILL_DECODE_CALIBRATION COMPLETE ==="
