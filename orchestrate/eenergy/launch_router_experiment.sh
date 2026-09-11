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

POLICY=${POLICY:?set POLICY=round_robin|lmetric|drf|p2c_whale|whale_argmin|constrained_lmetric|pressure_switch}
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
# Optional per-GPU override, e.g. RAMP_CEILING_PER_GPU="2:378.5,3:405.2,6:505.5" -- any GPU
# not listed falls back to RAMP_CEILING_W_PER_S. Motivated by round_robin-trace evidence
# that per-GPU p99 ramp rate varies ~30% across nominally-identical 4090s (see
# scripts/eenergy/README.md, "Per-GPU ramp-ceiling calibration"). Empty by default, which
# reproduces the old single-constant behavior exactly.
RAMP_CEILING_PER_GPU=${RAMP_CEILING_PER_GPU:-}
POWER_TRACE=${POWER_TRACE:-logs/eenergy_power_trace_${POLICY}.csv}
ASSIGNMENT_LOG=${ASSIGNMENT_LOG:-logs/eenergy_assignment_${POLICY}.csv}
# Optional peak-shaving admission gate (unset PEAK_CAP_W = disabled, matching every other
# optional feature in this script). See
# docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md.
PEAK_CAP_W=${PEAK_CAP_W:-}
PEAK_WINDOW_S=${PEAK_WINDOW_S:-30.0}
PEAK_RECHECK_INTERVAL_S=${PEAK_RECHECK_INTERVAL_S:-1.0}
J_PER_PREFILL_TOKEN=${J_PER_PREFILL_TOKEN:-0.068}
J_PER_DECODE_TOKEN=${J_PER_DECODE_TOKEN:-2.40}
# 272.0 = real wire bytes/token (measured live against this project's vLLM streaming
# endpoint), NOT src/replay_sharegpt.py's 3.235 plain-text chars-per-token constant -- see
# proxy_server.py's make_app docstring comment for why that distinction matters (the
# original 3.235 default caused a self-reinforcing admission-gate lockup on first
# validation).
BYTES_PER_TOKEN=${BYTES_PER_TOKEN:-272.0}
# How long a reservation stays active before release -- must cover only the brief gap until
# the next real power_poll_loop sample, NOT a request's full processing time (holding it that
# long caused a ~10x TTFT regression on first validation of the reservation ledger). See
# proxy_server.py's make_app docstring comment.
RESERVATION_HOLD_S=${RESERVATION_HOLD_S:-1.0}

declare -A GPU_CEILING
if [ -n "$RAMP_CEILING_PER_GPU" ]; then
  IFS=',' read -ra _pairs <<< "$RAMP_CEILING_PER_GPU"
  for pair in "${_pairs[@]}"; do
    GPU_CEILING["${pair%%:*}"]="${pair##*:}"
  done
fi

REPLICA_SPECS=""
PIDS=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
  gpu=$((GPU_OFFSET + i))
  port=$((BASE_PORT + i))
  ceiling="${GPU_CEILING[$gpu]:-$RAMP_CEILING_W_PER_S}"
  echo "Launching replica $i on GPU $gpu, port $port, ramp_ceiling=${ceiling} W/s"
  CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$port" --dtype auto \
    --max-num-batched-tokens "$TOKEN_BUDGET" --max-num-seqs "$MAX_NUM_SEQS" &
  PIDS+=($!)
  if [ -z "$REPLICA_SPECS" ]; then
    REPLICA_SPECS="127.0.0.1:${port}:${gpu}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${ceiling}"
  else
    REPLICA_SPECS="${REPLICA_SPECS},127.0.0.1:${port}:${gpu}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${ceiling}"
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
  ROUTER_PEAK_CAP_W="$PEAK_CAP_W" ROUTER_PEAK_WINDOW_S="$PEAK_WINDOW_S" \
  ROUTER_PEAK_RECHECK_INTERVAL_S="$PEAK_RECHECK_INTERVAL_S" \
  ROUTER_J_PER_PREFILL_TOKEN="$J_PER_PREFILL_TOKEN" \
  ROUTER_J_PER_DECODE_TOKEN="$J_PER_DECODE_TOKEN" \
  ROUTER_BYTES_PER_TOKEN="$BYTES_PER_TOKEN" \
  ROUTER_RESERVATION_HOLD_S="$RESERVATION_HOLD_S" \
  python3 scripts/eenergy/run_router.py &
ROUTER_PID=$!

echo "Router ready on port ${ROUTER_PORT}. Point the benchmark harness at it, e.g.:"
echo "  python3 src/replay_sharegpt.py --host 127.0.0.1 --port ${ROUTER_PORT} ..."
echo "Press Ctrl+C to tear down replicas, power logger, and router."

trap 'kill "${PIDS[@]}" "$POWER_PID" "$ROUTER_PID" 2>/dev/null' EXIT
wait "$ROUTER_PID"
