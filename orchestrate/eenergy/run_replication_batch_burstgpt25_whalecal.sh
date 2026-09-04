#!/bin/bash
# Recalibration check: only p2c_whale and whale_argmin consume is_whale (verified in
# scoring.py -- pick_lmetric/pick_drf/pick_constrained_lmetric/pick_pressure_switch never
# reference it), so round_robin/lmetric/drf_fixed/constrained_lmetric/pressure_switch results
# from the existing 25rps/local run are byte-for-byte unaffected by this threshold and don't
# need rerunning. Same trace/rate/bs_source as that run (logs/burstgpt_arms_trace_25rps.csv,
# ROUTER_BS_SOURCE=local) -- only ROUTER_WHALE_TOKEN_THRESHOLD changes, from the code default
# (4000, calibrated against the synthetic 13.6-15.5k-token whale-injection workload) to 1200
# (this trace's own p85, matching the project's whale-frac=0.15 convention).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

TRACE=logs/burstgpt_arms_trace_25rps.csv
ARMS="p2c_whale whale_argmin"

for OUTNAME in $ARMS; do
  for TRIAL in 1 2 3; do
    echo "=== BURSTGPT25WHALECAL BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/burstgpt25whalecal_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$OUTNAME N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      ROUTER_BS_SOURCE=local ROUTER_WHALE_TOKEN_THRESHOLD=1200 \
      POWER_TRACE=logs/burstgpt25whalecal_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/burstgpt25whalecal_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/burstgpt25whalecal_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running BurstGPT25whalecal trace replay (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 300 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --trace-csv "$TRACE" --request-timeout 180 \
      --output logs/burstgpt25whalecal_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/burstgpt25whalecal_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== BURSTGPT25WHALECAL BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL BURSTGPT25WHALECAL REPLICATION BATCH COMPLETE ==="
