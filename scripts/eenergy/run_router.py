#!/usr/bin/env python3
"""CLI entrypoint for the e-Energy router. Reads replica topology and policy from
environment variables so it composes cleanly with orchestrate/eenergy/*.sh, matching this
repo's existing env-var-driven convention (see scripts/mlsys/hotpatch_*.py).

Env:
  ROUTER_POLICY        round_robin | lmetric | drf | p2c_whale | whale_argmin |
                       constrained_lmetric | pressure_switch | drf_power_tiebreak |
                       lmetric_power | lmetric_power_convex | whale_argmin_power_switch
                       (required)
  ROUTER_REPLICAS       comma-separated host:port:gpu_index:token_budget:max_num_seqs:ramp_ceiling_w_per_s
                         e.g. "127.0.0.1:8001:0:16384:64:100.0,127.0.0.1:8002:1:16384:64:100.0"
  ROUTER_MODEL_NAME      HF model name/path for the tokenizer (required)
  ROUTER_HOST             default 0.0.0.0
  ROUTER_PORT              default 9000
  ROUTER_POWER_INTERVAL_S   default 0.5
  ROUTER_ASSIGNMENT_LOG     default unset (no logging). CSV path: wall_time,replica_id,
                            gpu_index -- one row per routed request, for post-hoc per-replica
                            power-pressure-window classification (sharper than the fleet-wide
                            any-GPU fallback).
  ROUTER_BS_SOURCE          local (default) | telemetry -- local uses the router's own
                            dispatch/complete bookkeeping (load_tracker.py); telemetry reads
                            each replica's real running+waiting count off its own vLLM
                            /metrics endpoint (bs_telemetry.py) via a background poller.
  ROUTER_BS_POLL_INTERVAL_S  default 0.5. Only used when ROUTER_BS_SOURCE=telemetry.
  ROUTER_WHALE_TOKEN_THRESHOLD  default 4000 (WHALE_TOKEN_THRESHOLD in router_core.py).
                            Admission-time prompt-token cutoff for p2c_whale/whale_argmin's
                            whale-aware routing. The default was calibrated against this
                            project's synthetic whale-injection workload (13.6-15.5k-token
                            whales); a workload with genuine but smaller size variance (e.g.
                            real BurstGPT traffic, max ~4k tokens) needs a lower value to make
                            that machinery engage at its own natural scale.
  ROUTER_PEAK_CAP_W          default unset (peak-shaving admission gate disabled). Fleet-
                            aggregate power cap in Watts -- see
                            docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md.
  ROUTER_PEAK_WINDOW_S       default 30.0. Rolling window (seconds) the cap is averaged over.
  ROUTER_PEAK_RECHECK_INTERVAL_S  default 1.0. How often a deferred request re-checks the budget.
  ROUTER_J_PER_PREFILL_TOKEN  default 0.068. Conservative-upper-bound prefill energy constant
                            (J/token) -- see docs/superpowers/specs/2026-09-11-peak-shaving-
                            admission-design.md Sec 4.2 for why this is deliberately NOT the
                            calibrated mean.
  ROUTER_J_PER_DECODE_TOKEN  default 2.40. Calibrated (robust, CV~3-8%) decode energy constant
                            (J/token).
  ROUTER_BYTES_PER_TOKEN     default 3.235. Reused from src/replay_sharegpt.py's existing
                            --chars-per-token calibration, for converting the live decode-
                            length byte EMA to a token count.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from proxy_server import run  # noqa: E402
from router_core import WHALE_TOKEN_THRESHOLD  # noqa: E402


def _parse_replicas(spec: str) -> list:
    specs = []
    for i, part in enumerate(spec.split(",")):
        host, port, gpu_index, token_budget, max_num_seqs, ramp_ceiling = part.split(":")
        specs.append(dict(
            replica_id=f"r{i}", host=host, port=int(port), gpu_index=int(gpu_index),
            token_budget=int(token_budget), max_num_seqs=int(max_num_seqs),
            ramp_ceiling_w_per_s=float(ramp_ceiling),
        ))
    return specs


def main() -> int:
    policy = os.environ["ROUTER_POLICY"]
    replica_specs = _parse_replicas(os.environ["ROUTER_REPLICAS"])
    model_name = os.environ["ROUTER_MODEL_NAME"]
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ROUTER_PORT", "9000"))
    power_interval_s = float(os.environ.get("ROUTER_POWER_INTERVAL_S", "0.5"))
    assignment_log_path = os.environ.get("ROUTER_ASSIGNMENT_LOG") or None
    bs_source = os.environ.get("ROUTER_BS_SOURCE", "local")
    bs_poll_interval_s = float(os.environ.get("ROUTER_BS_POLL_INTERVAL_S", "0.5"))
    whale_token_threshold = int(os.environ.get("ROUTER_WHALE_TOKEN_THRESHOLD",
                                                 str(WHALE_TOKEN_THRESHOLD)))
    peak_cap_w_raw = os.environ.get("ROUTER_PEAK_CAP_W") or None
    peak_cap_w = float(peak_cap_w_raw) if peak_cap_w_raw is not None else None
    peak_window_s = float(os.environ.get("ROUTER_PEAK_WINDOW_S", "30.0"))
    peak_recheck_interval_s = float(os.environ.get("ROUTER_PEAK_RECHECK_INTERVAL_S", "1.0"))
    j_per_prefill_token = float(os.environ.get("ROUTER_J_PER_PREFILL_TOKEN", "0.068"))
    j_per_decode_token = float(os.environ.get("ROUTER_J_PER_DECODE_TOKEN", "2.40"))
    bytes_per_token = float(os.environ.get("ROUTER_BYTES_PER_TOKEN", "3.235"))
    run(replica_specs, policy, model_name, host, port, power_interval_s, assignment_log_path,
        bs_source, bs_poll_interval_s, whale_token_threshold, peak_cap_w, peak_window_s,
        peak_recheck_interval_s, j_per_prefill_token, j_per_decode_token, bytes_per_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
