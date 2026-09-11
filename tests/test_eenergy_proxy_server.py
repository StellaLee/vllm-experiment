import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from proxy_server import build_replica_states, format_assignment_record, fleet_power_w  # noqa: E402


def test_build_replica_states_from_specs():
    specs = [
        dict(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
             token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0),
        dict(replica_id="r1", host="127.0.0.1", port=8002, gpu_index=1,
             token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0),
    ]
    states = build_replica_states(specs)
    assert len(states) == 2
    assert states[0].config.replica_id == "r0"
    assert states[1].config.port == 8002
    assert states[0].in_flight == 0


def test_format_assignment_record_is_csv_row():
    line = format_assignment_record(1788140425.123456, "r3", 5, new_tokens=1082, raw_tokens=1082)
    assert line == "1788140425.123456,r3,5,1082,1082,,,,,\n"


def test_format_assignment_record_shows_the_kv_hit_discount_when_present():
    line = format_assignment_record(1788140425.123456, "r0", 2, new_tokens=10, raw_tokens=1082)
    assert line == "1788140425.123456,r0,2,10,1082,,,,,\n"


def test_format_assignment_record_includes_share_compute_and_share_load_when_present():
    line = format_assignment_record(1788140425.123456, "r0", 2, new_tokens=10, raw_tokens=1082,
                                     share_compute=0.061035, share_load=0.34375)
    assert line == "1788140425.123456,r0,2,10,1082,0.061035,0.343750,,,\n"


def test_format_assignment_record_includes_threshold_violation_fields_when_present():
    line = format_assignment_record(1788140425.123456, "r0", 2, new_tokens=10, raw_tokens=1082,
                                     share_compute=0.06, share_load=0.34,
                                     d_chosen=2.0, min_d_available=1.0,
                                     avoidable_threshold_violation=True)
    assert line == "1788140425.123456,r0,2,10,1082,0.060000,0.340000,2.000000,1.000000,1\n"


def test_build_replica_states_default_telemetry_bs_is_zero():
    specs = [dict(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
                  token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0)]
    states = build_replica_states(specs)
    assert states[0].telemetry_bs == 0


def test_fleet_power_w_sums_across_replicas():
    specs = [dict(replica_id="r0", host="h", port=1, gpu_index=0, token_budget=100,
                   max_num_seqs=10, ramp_ceiling_w_per_s=50.0),
             dict(replica_id="r1", host="h", port=2, gpu_index=1, token_budget=100,
                   max_num_seqs=10, ramp_ceiling_w_per_s=50.0)]
    states = build_replica_states(specs)
    states[0].last_power_w = 300.0
    states[1].last_power_w = 250.0
    assert fleet_power_w(states) == 550.0


def test_fleet_power_w_zero_for_freshly_built_states():
    specs = [dict(replica_id="r0", host="h", port=1, gpu_index=0, token_budget=100,
                  max_num_seqs=10, ramp_ceiling_w_per_s=50.0)]
    states = build_replica_states(specs)
    assert fleet_power_w(states) == 0.0
