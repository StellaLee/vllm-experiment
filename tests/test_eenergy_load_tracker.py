import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from load_tracker import LoadTracker  # noqa: E402


def test_unknown_replica_starts_at_zero():
    t = LoadTracker()
    assert t.in_flight("r0") == 0


def test_dispatch_increments_and_complete_decrements():
    t = LoadTracker()
    t.on_dispatch("r0")
    t.on_dispatch("r0")
    assert t.in_flight("r0") == 2
    t.on_complete("r0")
    assert t.in_flight("r0") == 1


def test_complete_never_goes_negative():
    t = LoadTracker()
    t.on_complete("r0")
    assert t.in_flight("r0") == 0


def test_in_flight_if_dispatched_is_whatif_not_mutating():
    t = LoadTracker()
    t.on_dispatch("r0")
    assert t.in_flight_if_dispatched("r0") == 2
    assert t.in_flight("r0") == 1  # the what-if call must not mutate real state


def test_replicas_are_independent():
    t = LoadTracker()
    t.on_dispatch("r0")
    assert t.in_flight("r1") == 0
