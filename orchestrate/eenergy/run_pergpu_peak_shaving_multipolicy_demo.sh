#!/bin/bash
# "Larger effect, multiple routing policies" demonstration battery, requested after the
# reservation-ledger-fixed gate only modestly reduced cap violations at cap=2400W (near the
# sustained mean, ~1945W on Heavy/CL-long). This battery deliberately picks a cap BELOW the
# sustained mean (1800W vs ~1945W mean) to force continuous throttling instead of only
# shaving rare spikes -- offline trace simulation (check_peak_shaving_admission_prototype.py)
# predicts ~88.6% of requests need deferral with mean wait ~104s, p95 ~188s on this trace.
# Framed explicitly as a latency-insensitive batch workload (not interactive serving), so a
# large TTFT cost is an accepted tradeoff -- justifying --request-timeout 300 (up from the
# usual 180) so the predicted p95/max waits resolve as slow completions, not client-side
# timeouts/failures.
#
# Cross product: 3 routing policies (round_robin, lmetric, drf_no_power) x 2 gate states
# (no_gate, gated) = 6 arms, n=1 trial each -- this is a first look at whether the gate's
# effect is consistent across routing policies, matching the deferral-fix policy independence
# already established (the gate sits strictly before router.route() and never touches routing
# logic). Expand to n=3+ per arm if the demo effect looks real and policy-independent.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
OUTNAME=peak_shaving_multipolicy_demo
PEAK_CAP_W=1800
PEAK_WINDOW_S=15.0
PREFIX=closedloopheavylongpergpu
REQUEST_TIMEOUT=300

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
  # unrepresentative tiny response before the trial even starts (a real bug found on an
  # earlier battery -- see proxy_server.py's make_app docstring comment on handle_health).
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
  local POLICY=$1
  local GATE=$2
  local EXTRA_ENV=$3
  local ARM="${POLICY}_${GATE}"
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

  echo "=== running harness (arm=$ARM) ==="
  timeout 1200 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout $REQUEST_TIMEOUT \
    --output logs/${PREFIX}_records_${OUTNAME}_${ARM}.jsonl \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 900 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_${ARM}.log

  echo "=== tearing down (arm=$ARM) ==="
  teardown
  echo "=== ${PREFIX} BATCH: arm=$ARM COMPLETE ==="
}

for POLICY in round_robin lmetric drf_no_power; do
  run_arm "$POLICY" no_gate "PEAK_CAP_W="
  run_arm "$POLICY" gated "PEAK_CAP_W=$PEAK_CAP_W PEAK_WINDOW_S=$PEAK_WINDOW_S"
done

echo "=== PEAK_SHAVING_MULTIPOLICY_DEMO_BATCH COMPLETE ==="
