#!/bin/bash
# First hardware validation of drf_peak_power_tiebreak_full_coincidence_ceiling: retargets
# Share_power at instantaneous power LEVEL (share_power_level = power_w / power_level_ceiling_w)
# instead of ramp rate, in the full coincidence-ceiling rule (not just the plain tiebreak --
# drf_peak_power_tiebreak_full without coincidence-awareness was already validated on
# Heavy/CL-short, n=3). Motivated by the "does power-awareness help?" investigation
# (findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md): the ramp-based
# coincidence_ceiling is dominated outright by the power-BLIND drf_no_power on Light/Cachehit,
# and only clearly wins on Heavy/Closed-Loop's sustained-pressure conditions. Confirmed via
# lagged cross-correlation on existing Heavy/CL-long hardware traces
# (scripts/eenergy/check_compute_leads_ramp_lag_correlation.py) that ramp_rate is a noisy,
# LAGGING readout of compute (peak r~0.39 at +2-3s, near-zero at lag 0) -- levels should be
# structurally lower-noise than that derivative.
#
# Two conditions, n=3 each (quick check before deciding whether to expand to n=6):
#   - Heavy/Closed-Loop long: sanity check -- confirm the peak-power swap doesn't regress the
#     one condition where the ramp-based rule already clearly wins.
#   - Light/Cachehit: the sharpest test -- this is the condition where ramp-based
#     coincidence_ceiling was dominated OUTRIGHT by drf_no_power. If peak-power fixes it,
#     that's a clean, positive result.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=drf_peak_power_tiebreak_full_coincidence_ceiling
OUTNAME=peak_power_coincidence_ceiling

run_common_setup () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  mkdir -p logs
}

wait_for_router () {
  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      return 0
    fi
    sleep 5
  done
}

teardown () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
}

run_condition_trial () {
  local PREFIX=$1
  local DATASET=$2
  local TRIAL=$3
  shift 3
  local HARNESS_ARGS=("$@")
  local FULLPREFIX="${PREFIX}pergpu"

  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  run_common_setup
  rm -f logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${FULLPREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${FULLPREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${FULLPREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running harness (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset "$DATASET" --request-timeout 180 \
    --output logs/${FULLPREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${FULLPREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${FULLPREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${FULLPREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for TRIAL in 1 2 3; do
  run_condition_trial closedloopheavylong data/sharegpt_v3.json "$TRIAL" \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 900 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_condition_trial cachehit data/sharegpt_v3.json "$TRIAL" \
    --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
    --concurrency 24 --num-convs 150 --max-tokens 128
done

echo "=== PEAK_POWER_COINCIDENCE_CEILING 2-CONDITION BATCH COMPLETE ==="
