#!/bin/bash
# Targeted stress test for the admission gate's TOCTOU race, instead of relying on Matched's
# natural Poisson bursts to maybe produce one. Uses closed-loop "herd" concurrency (all N
# requests fired at t=0, no stagger -- see src/replay_sharegpt.py's own "(closed-loop, herd)"
# framing) with ALL-whale prompts, so many large-prefill requests genuinely compete for
# admission simultaneously. cap=2000W is deliberately tight relative to what a burst of ~24
# whale prefills would need, so the race condition is forced to matter, not left to chance.
#
# Two arms, 1 trial each (this is a fast, deterministic-shape mechanism check, not a
# statistical claim -- n=1 is fine here the way it wasn't for the noisy Matched TTFT numbers):
#   - no_gate: baseline burst, no admission control, to see how far over cap the burst alone
#     pushes the windowed average.
#   - gated: same burst, with the (reservation-ledger-fixed) admission gate active at
#     cap=2000W, to see whether it actually holds the windowed average under cap during a
#     deliberately-engineered worst case.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=drf_no_power
OUTNAME=peak_shaving_burst_stress
PREFIX=burstallwhalepergpu
PEAK_CAP_W=2000
PEAK_WINDOW_S=15.0

run_common_setup () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  mkdir -p logs
}

wait_for_router () {
  # GET /health, NOT a real completion -- a POST through /v1/completions here would exercise
  # the SAME decode_estimator real traffic uses, poisoning its cold-start EMA with an
  # unrepresentative tiny response before the trial even starts (the exact bug this stress
  # test's first run uncovered -- see proxy_server.py's make_app docstring comment on
  # handle_health).
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:9100/health >/dev/null 2>&1; then
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

run_arm () {
  local ARM=$1
  local EXTRA_ENV=$2
  echo "=== ${PREFIX} BATCH: arm=$ARM ==="
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_${ARM}.log

  env $EXTRA_ENV POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_${ARM}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_${ARM}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_${ARM}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running herd burst (arm=$ARM) ==="
  timeout 300 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_${ARM}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 24 --max-tokens 1024 \
    --whale-frac 1.0 --whale-min-chars 44000 --whale-max-chars 50000 \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_${ARM}.log

  echo "=== tearing down (arm=$ARM) ==="
  teardown
  echo "=== ${PREFIX} BATCH: arm=$ARM COMPLETE ==="
}

run_arm no_gate "PEAK_CAP_W="
run_arm gated "PEAK_CAP_W=$PEAK_CAP_W PEAK_WINDOW_S=$PEAK_WINDOW_S"

echo "=== PEAK_SHAVING_BURST_STRESS_BATCH COMPLETE ==="
