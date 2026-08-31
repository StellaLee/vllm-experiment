#!/usr/bin/env python3
"""CLI entrypoint for the e-Energy router. Reads replica topology and policy from
environment variables so it composes cleanly with orchestrate/eenergy/*.sh, matching this
repo's existing env-var-driven convention (see scripts/mlsys/hotpatch_*.py).

Env:
  ROUTER_POLICY        round_robin | lmetric | drf | p2c_whale | whale_argmin |
                       constrained_lmetric   (required)
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
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from proxy_server import run  # noqa: E402


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
    run(replica_specs, policy, model_name, host, port, power_interval_s, assignment_log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
