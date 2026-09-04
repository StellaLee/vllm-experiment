#!/bin/bash
# 7 arms x 3 trials on a denser real BurstGPT trace slice (900 real rows, same 60s span
# as the 5 req/s batch -> avg 15 req/s system-wide across N=6 replicas, real relative gap
# structure preserved -- densified by using MORE real consecutive rows, not by further
# compressing the timeline beyond the existing 3.0s gapcap). BS_SOURCE=telemetry this time:
# real vllm:num_requests_running/waiting off each replica's /metrics, not the router's own
# dispatch/complete bookkeeping -- this run is specifically meant to stress the fleet, which
# is exactly the regime where local bookkeeping can drift from ground truth (vLLM-side
# preemptions the router can't otherwise see).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

TRACE=logs/burstgpt_arms_trace_15rps.csv
ARMS="round_robin lmetric drf_fixed p2c_whale whale_argmin constrained_lmetric pressure_switch"

for OUTNAME in $ARMS; do
  POLICY=$OUTNAME
  if [ "$OUTNAME" = "drf_fixed" ]; then POLICY=drf; fi

  for TRIAL in 1 2 3; do
    echo "=== BURSTGPT15 BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/burstgpt15_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      ROUTER_BS_SOURCE=telemetry ROUTER_BS_POLL_INTERVAL_S=0.5 \
      POWER_TRACE=logs/burstgpt15_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/burstgpt15_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/burstgpt15_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running BurstGPT15 trace replay (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 300 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --trace-csv "$TRACE" --request-timeout 180 \
      --output logs/burstgpt15_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/burstgpt15_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== BURSTGPT15 BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL BURSTGPT15 REPLICATION BATCH COMPLETE ==="
