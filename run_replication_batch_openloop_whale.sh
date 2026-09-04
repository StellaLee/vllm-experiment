#!/bin/bash
# Isolating experiment: SAME dataset/workload as the original closed-loop whale-injection
# comparison (findings.md's main table -- data/sharegpt_v3.json, single-turn, whale-frac 0.15,
# whale range 44-50k chars, num-convs 150, max-tokens 128), only the loop mode changes:
# --concurrency 24 (closed-loop) -> --rate 8.5 (open-loop Poisson). Rate chosen via Little's
# Law (L=lambda*W) to target the SAME average concurrency (~24) as the closed-loop run: the
# closed-loop run's own real per-request latency (logs/eenergy_records_lmetric.jsonl) has
# mean W=2.81s (n=150, 21 whales at 4.57s mean, 129 short at 2.53s mean) -> lambda = 24/2.81
# = 8.5 conv/s. Tests whether round_robin's closed-loop-clear-loss (TTFT mean 0.732 vs
# 0.492-0.549 for aware arms) survives, narrows, or reverses once the closed-loop backpressure
# that let bad routing decisions self-correct is removed -- same content, only arrival
# dynamics differ. NOTE: --rate mode's Poisson inter-arrival timing is NOT seeded by
# --pad-seed (only pad-length/whale-classification randomness is) -- each trial gets a
# genuinely different random arrival pattern, appropriate for this specific test.
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
    echo "=== OPENLOOPWHALE BATCH: policy=$OUTNAME trial=$TRIAL ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3

    mkdir -p logs
    rm -f logs/openloopwhale_launch_${OUTNAME}_t${TRIAL}.log

    POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
      ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
      POWER_TRACE=logs/openloopwhale_power_trace_${OUTNAME}_t${TRIAL}.csv \
      ASSIGNMENT_LOG=logs/openloopwhale_assignment_${OUTNAME}_t${TRIAL}.csv \
      nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/openloopwhale_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
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

    echo "=== running harness open-loop (policy=$OUTNAME trial=$TRIAL) ==="
    timeout 900 python3 src/replay_sharegpt.py \
      --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --dataset data/sharegpt_v3.json --min-turns 1 --max-turns 1 \
      --rate 8.5 --num-convs 150 --max-tokens 128 \
      --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
      --request-timeout 180 \
      --output logs/openloopwhale_records_${OUTNAME}_t${TRIAL}.jsonl \
      2>&1 | tee logs/openloopwhale_harness_${OUTNAME}_t${TRIAL}.log

    echo "=== tearing down (policy=$OUTNAME trial=$TRIAL) ==="
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
    pkill -f "power_logger.py" 2>/dev/null
    sleep 3
    echo "=== OPENLOOPWHALE BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
  done
done
echo "=== FULL OPENLOOPWHALE REPLICATION BATCH COMPLETE ==="
