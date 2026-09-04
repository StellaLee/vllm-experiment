#!/bin/bash
# Real-content validation, NO synthetic whale injection: data/wildchat_v1.json (20k English
# WildChat-1M conversations, converted from HF parquet shard 0, toxic/redacted turns dropped).
# Natural long-tail only -- 1.6% of first-turns already exceed the whale-equivalent threshold
# (~4000 tok) with zero injection, vs. 0.39% in raw BurstGPT overall. Natural multi-turn
# (min-turns=1, max-turns=4) instead of forcing single-turn. Arrival rate/max-tokens matched to
# Heavy/Matched for a clean comparison point -- those are harness operation parameters, not
# content injection. 7 arms x 3 trials, sequential (GPU_OFFSET=2/N_REPLICAS=6 saturates all 6
# target GPUs per trial).
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

PREFIX=wildchatnatural

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
    --dataset data/wildchat_v1.json --min-turns 1 --max-turns 4 \
    --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

# drf_fixed's actual policy string is "drf" (the known pitfall, fixed here from the start).
for PAIR in "round_robin round_robin" "drf_fixed drf" "drf_power_tiebreak_full drf_power_tiebreak_full" \
            "drf_power_tiebreak_adaptive_isolated drf_power_tiebreak_adaptive_isolated" \
            "lmetric lmetric" "weighted_sum weighted_sum" "lmetric_power lmetric_power"; do
  set -- $PAIR
  OUTNAME=$1
  POLICY=$2
  for TRIAL in 1 2 3; do
    run_trial $OUTNAME $POLICY $TRIAL
  done
done

echo "=== FULL WILDCHAT-NATURAL 7-ARM BATCH COMPLETE ==="
