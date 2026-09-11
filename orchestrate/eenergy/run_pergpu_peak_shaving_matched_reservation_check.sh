#!/bin/bash
# Re-validation of the peak-shaving admission gate on Heavy/Matched specifically, after
# adding the PowerBudget reservation ledger (reserve()/release()) to close a TOCTOU gap
# found in the prior validation: the gate only modestly reduced windowed-average cap
# violations there (2436W -> 2408W worst case, ~1%) and still let 2 of 3 trials exceed
# 2400W despite being active. This condition (bursty, open-loop) is the one that actually
# exercises the gap; Heavy/CL-long never got close to the cap either with or without the
# gate, so it isn't worth re-running here.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=drf_no_power
OUTNAME=peak_shaving_reservation_check
PEAK_CAP_W=2400
PEAK_WINDOW_S=30.0
PREFIX=openloopwhalelongoutmatchedpergpu

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
  # unrepresentative tiny response before the trial even starts (a real bug found on this
  # exact battery -- see proxy_server.py's make_app docstring comment on handle_health).
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

for TRIAL in 1 2 3; do
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL cap=${PEAK_CAP_W}W window=${PEAK_WINDOW_S}s ==="
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    PEAK_CAP_W="$PEAK_CAP_W" PEAK_WINDOW_S="$PEAK_WINDOW_S" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running harness (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
done

echo "=== PEAK_SHAVING_RESERVATION_CHECK_BATCH COMPLETE ==="
