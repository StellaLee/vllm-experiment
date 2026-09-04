#!/bin/bash
# Third isolating experiment: same long-output open-loop whale-injection workload
# (max-tokens=1024, same dataset/whale-params) as openloopwhalelongout, but --rate raised
# 8.5 -> 25 conv/s to match BurstGPT's 25 req/s density directly (this workload is
# single-turn, min-turns=max-turns=1, so conv/s == req/s -- directly comparable). Tests
# whether decode-duration variance + real density together produce the full round_robin
# reversal that decode-duration variance alone (at the lighter, closed-loop-matched 8.5
# conv/s) only partially did (gap narrowed 72%->53% but round_robin still didn't win).
# num-convs raised 150->1500 to match: at rate=25, 150 convs would all arrive within 6s (too
# short a burst for real steady-state congestion); 1500 convs over 25 conv/s ~= 60s sustained
# arrival window, matching BurstGPT's own 1500-row/60s trace scale. Wraps around and reuses
# the filtered conversation pool if it's smaller than 1500 (confirmed harness behavior) --
# fine here since pad/whale-classification is freshly seeded per loop index regardless.
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
    echo "=== OPENLOOPWHALELONGOUT25 BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/openloopwhalelongout25_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      POWER_TRACE=logs/openloopwhalelongout25_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/openloopwhalelongout25_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/openloopwhalelongout25_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running harness open-loop longout rate=25 (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 1800 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/sharegpt_v3.json --min-turns 1 --max-turns 1 \
      --rate 25 --num-convs 1500 --max-tokens 1024 \
      --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
      --request-timeout 180 \
      --output logs/openloopwhalelongout25_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/openloopwhalelongout25_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== OPENLOOPWHALELONGOUT25 BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL OPENLOOPWHALELONGOUT25 REPLICATION BATCH COMPLETE ==="
