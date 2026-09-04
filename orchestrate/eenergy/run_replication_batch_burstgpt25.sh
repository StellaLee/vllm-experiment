#!/bin/bash
# 7 arms x 3 trials on an even denser real BurstGPT trace slice (1500 real rows, same 60s
# span -> avg 25 req/s system-wide across N=6 replicas). BS_SOURCE=local this time (router's
# own dispatch/complete bookkeeping, not vllm:num_requests_running/waiting telemetry) --
# deliberately isolating the BS-source variable against the 15 req/s/telemetry run: if
# round_robin still overtakes the cache/load-aware arms here, that points to a genuine
# saturation effect independent of telemetry staleness; if the split disappears/reverts,
# that points to telemetry-driven herding as the real cause of the 15rps flip.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

TRACE=logs/burstgpt_arms_trace_25rps.csv
ARMS="round_robin lmetric drf_fixed p2c_whale whale_argmin constrained_lmetric pressure_switch"

for OUTNAME in $ARMS; do
  POLICY=$OUTNAME
  if [ "$OUTNAME" = "drf_fixed" ]; then POLICY=drf; fi

  for TRIAL in 1 2 3; do
    echo "=== BURSTGPT25 BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/burstgpt25_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      ROUTER_BS_SOURCE=local \
      POWER_TRACE=logs/burstgpt25_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/burstgpt25_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/burstgpt25_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
    LAUNCH_PID=$!
    echo "launch script pid: $LAUNCH_PID"

    for i in $(seq 1 120); do
      if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
         -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
         >/dev/null 2>&1; then
        echo "router ready after ${i} checks"
        break
      fi
      sleep 5
    done

    echo "=== running BurstGPT25 trace replay (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 300 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --trace-csv "$TRACE" --request-timeout 180 \
      --output logs/burstgpt25_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/burstgpt25_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== BURSTGPT25 BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL BURSTGPT25 REPLICATION BATCH COMPLETE ==="
