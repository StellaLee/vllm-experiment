import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from whale_tracker import WhaleTracker  # noqa: E402


def test_starts_at_zero():
    t = WhaleTracker()
    assert t.active_whale_count("r0") == 0


def test_dispatch_whale_increments_only_whale_count():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=True)
    assert t.active_whale_count("r0") == 1


def test_dispatch_non_whale_does_not_increment():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=False)
    assert t.active_whale_count("r0") == 0


def test_complete_whale_decrements():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=True)
    t.on_dispatch("r0", is_whale=True)
    t.on_complete("r0", is_whale=True)
    assert t.active_whale_count("r0") == 1


def test_complete_does_not_go_negative():
    t = WhaleTracker()
    t.on_complete("r0", is_whale=True)
    assert t.active_whale_count("r0") == 0


def test_complete_non_whale_does_not_affect_whale_count():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=True)
    t.on_complete("r0", is_whale=False)
    assert t.active_whale_count("r0") == 1


def test_active_whale_count_if_dispatched_does_not_mutate():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=True)
    assert t.active_whale_count_if_dispatched("r0") == 2
    assert t.active_whale_count("r0") == 1  # unchanged


def test_tracks_replicas_independently():
    t = WhaleTracker()
    t.on_dispatch("r0", is_whale=True)
    assert t.active_whale_count("r1") == 0
