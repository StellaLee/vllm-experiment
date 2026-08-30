import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from ramp import compute_ramp_rate, update_ramp_state  # noqa: E402
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


def _state():
    cfg = ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                         token_budget=1, max_num_seqs=1, ramp_ceiling_w_per_s=1.0)
    return ReplicaState(config=cfg)


def test_compute_ramp_rate_basic():
    assert compute_ramp_rate(100.0, 0.0, 150.0, 2.0) == 25.0


def test_compute_ramp_rate_zero_or_negative_dt_returns_zero():
    assert compute_ramp_rate(100.0, 5.0, 150.0, 5.0) == 0.0
    assert compute_ramp_rate(100.0, 5.0, 150.0, 4.0) == 0.0


def test_update_ramp_state_first_sample_has_no_prior_so_ramp_is_zero():
    s = _state()
    update_ramp_state(s, power_w=200.0, ts=10.0)
    assert s.ramp_rate_w_per_s == 0.0
    assert s.last_power_w == 200.0
    assert s.last_power_ts == 10.0


def test_update_ramp_state_second_sample_computes_real_ramp():
    s = _state()
    update_ramp_state(s, power_w=200.0, ts=10.0)
    update_ramp_state(s, power_w=210.0, ts=11.0)
    assert s.ramp_rate_w_per_s == 10.0
