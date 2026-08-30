import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


def test_replica_config_holds_fields():
    cfg = ReplicaConfig(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
                         token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0)
    assert cfg.replica_id == "r0"
    assert cfg.port == 8001
    assert cfg.ramp_ceiling_w_per_s == 100.0


def test_replica_state_defaults_and_independent_mutable_fields():
    cfg = ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                         token_budget=1, max_num_seqs=1, ramp_ceiling_w_per_s=1.0)
    s1 = ReplicaState(config=cfg)
    s2 = ReplicaState(config=cfg)
    assert s1.in_flight == 0
    assert s1.cached_block_hashes == set()
    s1.cached_block_hashes.add(123)
    assert s2.cached_block_hashes == set(), "default set must not be shared across instances"
