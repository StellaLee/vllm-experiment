#!/bin/bash
# Launches N vLLM replicas (one GPU each), the power_logger.py sidecar (full trace for
# analysis), and the router (routing decisions for whichever POLICY is selected), then
# waits for the caller to run the benchmark harness against the router's port.
#
# Usage:
#   POLICY=drf N_REPLICAS=4 MODEL=/model/... RAMP_CEILING_W_PER_S=100.0 \
#     orchestrate/eenergy/launch_router_experiment.sh
#
# GPU_OFFSET (default 0): first GPU index to use; replicas take GPU_OFFSET..GPU_OFFSET+N-1.
# Set this to dodge GPUs another process already occupies on a shared box.
set -euo pipefail

POLICY=${POLICY:?set POLICY=round_robin|lmetric|drf|p2c_whale|whale_argmin|constrained_lmetric}
N_REPLICAS=${N_REPLICAS:-4}
GPU_OFFSET=${GPU_OFFSET:-0}
MODEL=${MODEL:?set MODEL=/path/to/model}
BASE_PORT=${BASE_PORT:-8001}
ROUTER_PORT=${ROUTER_PORT:-9000}
TOKEN_BUDGET=${TOKEN_BUDGET:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
# Calibrated 2026-08-31 on this 8x4090 box: burst of 24 concurrent 45k-char prefills against
# one idle replica, power sampled at the router's actual poll cadence (500ms). Steady-state
# ramp is near-zero (p50=0.1 W/s); the idle->loaded transition itself is the whole signal
# (p99=212 W/s, p99.9/max=433 W/s, observed 62W->235W->449W across two consecutive polls).
# 450 W/s ~= observed max with slight headroom. Re-calibrate if replica count, model size, or
# ROUTER_POWER_INTERVAL_S changes materially -- see scripts/eenergy/README.md.
RAMP_CEILING_W_PER_S=${RAMP_CEILING_W_PER_S:-450.0}
POWER_TRACE=${POWER_TRACE:-logs/eenergy_power_trace_${POLICY}.csv}
ASSIGNMENT_LOG=${ASSIGNMENT_LOG:-logs/eenergy_assignment_${POLICY}.csv}

REPLICA_SPECS=""
PIDS=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
  gpu=$((GPU_OFFSET + i))
  port=$((BASE_PORT + i))
  echo "Launching replica $i on GPU $gpu, port $port"
  CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$port" --dtype auto \
    --max-num-batched-tokens "$TOKEN_BUDGET" --max-num-seqs "$MAX_NUM_SEQS" &
  PIDS+=($!)
  if [ -z "$REPLICA_SPECS" ]; then
    REPLICA_SPECS="127.0.0.1:${port}:${gpu}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${RAMP_CEILING_W_PER_S}"
  else
    REPLICA_SPECS="${REPLICA_SPECS},127.0.0.1:${port}:${gpu}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${RAMP_CEILING_W_PER_S}"
  fi
done

echo "Waiting for replicas to come up..."
for i in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((BASE_PORT + i))
  until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do sleep 2; done
done

echo "Starting power_logger.py sidecar -> ${POWER_TRACE}"
GPU_LIST=$(seq -s, "$GPU_OFFSET" $((GPU_OFFSET + N_REPLICAS - 1)))
python3 scripts/pesim/power_logger.py --gpus "$GPU_LIST" --interval-ms 50 --output "$POWER_TRACE" &
POWER_PID=$!

echo "Starting router (policy=$POLICY) on port $ROUTER_PORT -> assignment log: ${ASSIGNMENT_LOG}"
ROUTER_POLICY="$POLICY" ROUTER_REPLICAS="$REPLICA_SPECS" ROUTER_MODEL_NAME="$MODEL" \
  ROUTER_PORT="$ROUTER_PORT" ROUTER_ASSIGNMENT_LOG="$ASSIGNMENT_LOG" \
  python3 scripts/eenergy/run_router.py &
ROUTER_PID=$!

echo "Router ready on port ${ROUTER_PORT}. Point the benchmark harness at it, e.g.:"
echo "  python3 src/replay_sharegpt.py --host 127.0.0.1 --port ${ROUTER_PORT} ..."
echo "Press Ctrl+C to tear down replicas, power logger, and router."

trap 'kill "${PIDS[@]}" "$POWER_PID" "$ROUTER_PID" 2>/dev/null' EXIT
wait "$ROUTER_PID"
