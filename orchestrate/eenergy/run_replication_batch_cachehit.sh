#!/bin/bash
# First-ever multi-arm comparison under a workload that actually produces KV$ hits:
# plain ShareGPT, no pad, no whale, multi-turn (min 2 / max 4 turns) -- every prior
# whale-injection comparison in this project ran with hit rate = 0 by construction
# (pad-in-front + single-turn), so every arm's Share_compute/P-token term was always
# just raw prompt length. This is the first measurement where that term can differ.
#
# 7 arms x 3 trials each, same replication discipline (fixed default pad-seed 12345,
# N=6/GPUs 2-7) as every other eenergy batch this project. Assignment log enabled on
# every run so real measured cache-hit-rate per arm is a first-class output, not assumed.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

ARMS="round_robin lmetric drf_fixed p2c_whale whale_argmin constrained_lmetric pressure_switch"

for OUTNAME in $ARMS; do
  POLICY=$OUTNAME
  if [ "$OUTNAME" = "drf_fixed" ]; then POLICY=drf; fi   # drf_fixed IS drf; the "_fixed"
                                                          # suffix is just this project's
                                                          # output-tagging convention
  for TRIAL in 1 2 3; do
    echo "=== CACHEHIT BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/cachehit_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      POWER_TRACE=logs/cachehit_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/cachehit_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/cachehit_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running harness (policy=$OUTNAME trial=$TRIAL, plain multi-turn, no pad, no whale) ==="
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/sharegpt_v3.json --min-turns 2 --max-turns 4 \
      --concurrency 24 --num-convs 150 --max-tokens 128 \
      --pad-chars 0 --whale-frac 0.0 \
      --request-timeout 180 \
      --output logs/cachehit_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/cachehit_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== CACHEHIT BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL CACHEHIT REPLICATION BATCH COMPLETE ==="
