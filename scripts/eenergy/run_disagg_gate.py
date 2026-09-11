#!/usr/bin/env python3
"""CLI entrypoint for the disaggregated peak-shaving gate shim (see
docs/superpowers/specs/2026-09-11-disagg-peak-shaving-gate-design.md). Reads GPU pool
topology and gate config from environment variables, matching this repo's existing
env-var-driven convention (run_router.py).

Env:
  DISAGG_GATE_HOST                  default 0.0.0.0
  DISAGG_GATE_PORT                  default 8190
  DISAGG_PROXY_URL                  default http://127.0.0.1:8192 -- the (unmodified) Nixl
                                     toy proxy this shim forwards admitted requests to.
  DISAGG_MODEL_NAME                 HF model name/path for the tokenizer (required).
  DISAGG_PREFILL_GPU_INDICES        default 0,1,2,3
  DISAGG_DECODE_GPU_INDICES         default 4,5,6,7
  DISAGG_POWER_INTERVAL_S           default 0.5
  DISAGG_PEAK_CAP_PREFILL_W         default unset (prefill-pool budget disabled).
  DISAGG_PEAK_CAP_DECODE_W          default unset (decode-pool budget disabled,
                                     independently of the prefill cap).
  DISAGG_PEAK_WINDOW_S              default 30.0
  DISAGG_PEAK_RECHECK_INTERVAL_S    default 1.0
  DISAGG_J_PER_PREFILL_TOKEN        default 0.094 -- the REAL disaggregated-calibration
                                     value (findings/2026-09-11-eenergy-pd-disaggregation-
                                     energy-calibration.md), not the colocated gate's 0.068.
  DISAGG_J_PER_DECODE_TOKEN         default 0.78 -- ditto, not the colocated gate's 2.40.
  DISAGG_BYTES_PER_TOKEN            default 272.0 -- unchanged from the colocated gate
                                     (real wire bytes/token, topology-independent).
  DISAGG_RESERVATION_HOLD_S         default 1.0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from disagg_gate import run  # noqa: E402


def _parse_gpu_indices(spec: str) -> list:
    return [int(x) for x in spec.split(",")]


def main() -> int:
    model_name = os.environ["DISAGG_MODEL_NAME"]
    host = os.environ.get("DISAGG_GATE_HOST", "0.0.0.0")
    port = int(os.environ.get("DISAGG_GATE_PORT", "8190"))
    nixl_proxy_url = os.environ.get("DISAGG_PROXY_URL", "http://127.0.0.1:8192")
    prefill_gpu_indices = _parse_gpu_indices(
        os.environ.get("DISAGG_PREFILL_GPU_INDICES", "0,1,2,3"))
    decode_gpu_indices = _parse_gpu_indices(
        os.environ.get("DISAGG_DECODE_GPU_INDICES", "4,5,6,7"))
    power_interval_s = float(os.environ.get("DISAGG_POWER_INTERVAL_S", "0.5"))
    peak_cap_prefill_w_raw = os.environ.get("DISAGG_PEAK_CAP_PREFILL_W") or None
    peak_cap_prefill_w = float(peak_cap_prefill_w_raw) if peak_cap_prefill_w_raw is not None else None
    peak_cap_decode_w_raw = os.environ.get("DISAGG_PEAK_CAP_DECODE_W") or None
    peak_cap_decode_w = float(peak_cap_decode_w_raw) if peak_cap_decode_w_raw is not None else None
    peak_window_s = float(os.environ.get("DISAGG_PEAK_WINDOW_S", "30.0"))
    peak_recheck_interval_s = float(os.environ.get("DISAGG_PEAK_RECHECK_INTERVAL_S", "1.0"))
    j_per_prefill_token = float(os.environ.get("DISAGG_J_PER_PREFILL_TOKEN", "0.094"))
    j_per_decode_token = float(os.environ.get("DISAGG_J_PER_DECODE_TOKEN", "0.78"))
    bytes_per_token = float(os.environ.get("DISAGG_BYTES_PER_TOKEN", "272.0"))
    reservation_hold_s = float(os.environ.get("DISAGG_RESERVATION_HOLD_S", "1.0"))
    run(prefill_gpu_indices, decode_gpu_indices, nixl_proxy_url, model_name, host, port,
        power_interval_s, peak_cap_prefill_w, peak_cap_decode_w, peak_window_s,
        peak_recheck_interval_s, j_per_prefill_token, j_per_decode_token, bytes_per_token,
        reservation_hold_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
