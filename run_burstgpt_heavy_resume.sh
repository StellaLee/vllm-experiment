#!/bin/bash
# Real-trace validation: BurstGPT_1.csv, a whale-containing slice chosen by scanning for the
# densest 1000-row window of >4000-token requests (off=131000, found 44/1000 = 4.4% natural
# whale fraction -- far below our synthetic conditions' deliberately-injected 15%, so the
# named-rule advantage is expected to be weaker here, not stronger; that's the honest point of
# this condition, not a flaw in it). Compressed from its natural 2870s span to 300s
# (~3.3 req/s average) via scripts/mlsys/make_burstgpt_trace.py, same tool this project has
# used for BurstGPT replay before. 7 arms x 3 trials, sequential (GPU_OFFSET=2/N_REPLICAS=6
# saturates all 6 target GPUs per trial).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

PREFIX=burstgptheavy

run_trial () {
  local OUTNAME=$1
  local POLICY=$2
  local TRIAL=$3

  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3

  mkdir -p logs
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_W_PER_S=450.0 \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"

  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      break
    fi
    sleep 5
  done

  echo "=== running harness (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --trace-csv logs/burstgpt_heavy_trace.csv --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

# OUTNAME POLICY pairs -- drf_fixed's actual policy string is "drf" (bare "drf" is the
# sorted/Pareto-safe rule; "drf_fixed" is not a valid _POLICIES entry and crashes the router
# on startup, the exact known pitfall diagnosed twice earlier this session). Every other arm's
# outname equals its policy string.
# RESUME: round_robin already completed successfully (3/3 trials) before the drf_fixed
# policy-string bug was caught and fixed -- skip it here, start from drf_fixed.
for PAIR in "drf_fixed drf" "drf_power_tiebreak_full drf_power_tiebreak_full" \
            "drf_power_tiebreak_adaptive_isolated drf_power_tiebreak_adaptive_isolated" \
            "lmetric lmetric" "weighted_sum weighted_sum" "lmetric_power lmetric_power"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 1 2 3; do
    run_trial $OUTNAME $POLICY $TRIAL
  done
done

echo "=== FULL BURSTGPT-HEAVY 7-ARM BATCH COMPLETE ==="
