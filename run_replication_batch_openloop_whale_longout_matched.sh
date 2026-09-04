#!/bin/bash
# Corrected density-matched isolating experiment, replacing the failed --rate 25 attempt
# (which saturated the fleet completely -- TTFT ~20s for every arm, uninformative). The
# --rate 25 attempt matched BurstGPT's raw REQUEST rate (25 req/s) but not its aggregate
# TOKEN throughput: BurstGPT's real 25rps trace demands ~18659 tokens/s (829631 prompt +
# 289928 response tokens over 60s, computed directly from logs/burstgpt_arms_trace_25rps.csv).
# This whale-injection workload's real per-request demand (measured from the successful
# --rate 8.5 run's actual records: mean prompt_tokens_approx=1403, mean output_tokens=337,
# i.e. actual generated length, not the 1024 cap) is ~1740 tokens/req -- 2.3x heavier per
# request than BurstGPT's real ~746 tokens/req average. Token-throughput-matched rate:
# 18659 / 1740 ~= 10.7 conv/s. num-convs scaled to 750 for a ~70s sustained window
# (comparable to BurstGPT's own 60s trace duration), vs the 1500 that produced a ~140s+
# window that also contributed to the earlier saturation.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

ARMS="round_robin lmetric drf_fixed p2c_whale whale_argmin constrained_lmetric pressure_switch"

for OUTNAME in $ARMS; do
  POLICY=$OUTNAME
  if [ "$OUTNAME" = "drf_fixed" ]; then POLICY=drf; fi

  for TRIAL in 1 2 3; do
    echo "=== OPENLOOPWHALELONGOUTMATCHED BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/openloopwhalelongoutmatched_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      POWER_TRACE=logs/openloopwhalelongoutmatched_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/openloopwhalelongoutmatched_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/openloopwhalelongoutmatched_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running harness open-loop longout matched-rate (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/sharegpt_v3.json --min-turns 1 --max-turns 1 \
      --rate 10.7 --num-convs 750 --max-tokens 1024 \
      --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
      --request-timeout 180 \
      --output logs/openloopwhalelongoutmatched_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/openloopwhalelongoutmatched_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== OPENLOOPWHALELONGOUTMATCHED BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL OPENLOOPWHALELONGOUTMATCHED REPLICATION BATCH COMPLETE ==="
