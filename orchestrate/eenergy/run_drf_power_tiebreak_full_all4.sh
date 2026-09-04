#!/bin/bash
# drf_power_tiebreak_full validation: same 4 conditions already run for every other DRF-family
# arm this session (Heavy/Matched, Light/Cachehit, Heavy/Closed-Loop, Ramp & Route), 3 trials
# each, sequentially (GPU_OFFSET=2/N_REPLICAS=6 saturates all 6 target GPUs per trial, so no
# cross-condition parallelism on this 8-GPU box). Tests the prediction that adding compute as
# a 4th tie-break coordinate (the Pareto-domination fix, verify_pareto_lemma_full.py) is
# behaviorally near-free in practice, since it only changes decisions in the rare residual
# full-tie case that (D, power, load) alone doesn't already resolve.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

OUTNAME=drf_power_tiebreak_full
POLICY=drf_power_tiebreak_full

run_trial () {
  local PREFIX=$1
  local TRIAL=$2
  shift 2
  local HARNESS_ARGS=("$@")

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
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for TRIAL in 1 2 3; do
  run_trial openloopwhalelongoutmatched $TRIAL \
    --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_trial cachehit $TRIAL \
    --pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4 \
    --concurrency 24 --num-convs 150 --max-tokens 128
done

for TRIAL in 1 2 3; do
  run_trial closedloopheavy $TRIAL \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 150 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_trial rampandroute $TRIAL \
    --min-turns 1 --max-turns 1 --concurrency 24 --num-convs 150 --max-tokens 128 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

echo "=== FULL DRF_POWER_TIEBREAK_FULL 4-CONDITION BATCH COMPLETE ==="
