import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy"))
from run_router import _parse_replicas  # noqa: E402


def test_parse_replicas_supports_heterogeneous_per_gpu_ramp_ceilings():
    """orchestrate/eenergy/launch_router_experiment.sh's RAMP_CEILING_PER_GPU override relies
    on this: each replica in ROUTER_REPLICAS carries its own ramp_ceiling_w_per_s, so a
    per-GPU-calibrated fleet (see scripts/eenergy/calibrate_ramp_ceiling.py) round-trips
    correctly rather than being silently collapsed to one shared value."""
    spec = "127.0.0.1:8001:2:16384:64:378.5,127.0.0.1:8002:6:16384:64:505.5"
    parsed = _parse_replicas(spec)
    assert len(parsed) == 2
    assert parsed[0]["gpu_index"] == 2
    assert parsed[0]["ramp_ceiling_w_per_s"] == 378.5
    assert parsed[1]["gpu_index"] == 6
    assert parsed[1]["ramp_ceiling_w_per_s"] == 505.5
    # a per-replica id is still assigned even though the ceiling now varies
    assert parsed[0]["replica_id"] == "r0"
    assert parsed[1]["replica_id"] == "r1"


def test_parse_replicas_uniform_ceiling_still_works():
    """Backward-compat: the old single-constant behavior (every replica sharing one
    RAMP_CEILING_W_PER_S) is just the degenerate case of the same parsing path."""
    spec = "127.0.0.1:8001:0:16384:64:450.0,127.0.0.1:8002:1:16384:64:450.0"
    parsed = _parse_replicas(spec)
    assert parsed[0]["ramp_ceiling_w_per_s"] == 450.0
    assert parsed[1]["ramp_ceiling_w_per_s"] == 450.0
