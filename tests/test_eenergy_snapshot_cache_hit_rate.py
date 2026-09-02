import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy"))
from snapshot_cache_hit_rate import parse_replica_hostports, summarize  # noqa: E402


def test_parse_replica_hostports_extracts_host_and_port_only():
    spec = "127.0.0.1:8001:2:16384:64:450.0,127.0.0.1:8002:3:16384:64:450.0"
    assert parse_replica_hostports(spec) == [("127.0.0.1", 8001), ("127.0.0.1", 8002)]


def test_parse_replica_hostports_single_replica():
    assert parse_replica_hostports("10.0.0.1:9001:0:1000:10:100.0") == [("10.0.0.1", 9001)]


def test_summarize_computes_fleet_wide_hit_rate():
    per_replica = [("h1", 8001, 50.0, 100.0), ("h2", 8002, 30.0, 100.0)]
    result = summarize(per_replica)
    assert result == dict(total_hits=80.0, total_queries=200.0, hit_rate=0.4)


def test_summarize_returns_zero_hit_rate_not_nan_when_no_queries_at_all():
    """An all-zero trial (e.g. every request uncacheable) is a real, reportable 0%, not a
    missing/undefined value."""
    per_replica = [("h1", 8001, 0.0, 0.0), ("h2", 8002, 0.0, 0.0)]
    result = summarize(per_replica)
    assert result == dict(total_hits=0.0, total_queries=0.0, hit_rate=0.0)


def test_summarize_handles_a_single_replica():
    per_replica = [("h1", 8001, 371.0, 371.0)]
    result = summarize(per_replica)
    assert result == dict(total_hits=371.0, total_queries=371.0, hit_rate=1.0)
