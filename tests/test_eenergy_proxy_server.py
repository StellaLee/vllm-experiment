import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from proxy_server import build_replica_states, format_assignment_record  # noqa: E402


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
    line = format_assignment_record(1788140425.123456, "r3", 5)
    assert line == "1788140425.123456,r3,5\n"


def test_build_replica_states_default_telemetry_bs_is_zero():
    specs = [dict(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
                  token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0)]
    states = build_replica_states(specs)
    assert states[0].telemetry_bs == 0
