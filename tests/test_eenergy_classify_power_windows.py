import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy"))
from classify_power_windows import classify_windows, label_at  # noqa: E402


def test_flat_power_trace_is_all_normal():
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0), (3.0, 100.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    assert all(w.label == "normal" for w in windows)


def test_sharp_ramp_is_labeled_power_pressure():
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 300.0), (3.0, 300.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    labels = [w.label for w in windows]
    assert "power_pressure" in labels
    assert labels[0] == "normal"       # 100->100 over [0,1]
    assert labels[1] == "power_pressure"  # 100->300 over [1,2], ramp=200 > ceiling
    assert labels[2] == "normal"       # 300->300 over [2,3]


def test_adjacent_same_label_windows_are_merged():
    # two consecutive power_pressure intervals back to back should merge into one window
    samples = [(0.0, 0.0), (1.0, 100.0), (2.0, 200.0), (3.0, 200.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    pressure_windows = [w for w in windows if w.label == "power_pressure"]
    assert len(pressure_windows) == 1
    assert pressure_windows[0].start_ts == 0.0
    assert pressure_windows[0].end_ts == 2.0


def test_label_at_finds_covering_window_and_defaults_to_normal():
    samples = [(0.0, 100.0), (1.0, 300.0), (2.0, 300.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    assert label_at(windows, 0.5) == "power_pressure"
    assert label_at(windows, 1.5) == "normal"
    assert label_at(windows, 999.0) == "normal"  # outside any window


def test_fewer_than_two_samples_returns_empty():
    assert classify_windows([], ramp_ceiling_w_per_s=10.0) == []
    assert classify_windows([(0.0, 100.0)], ramp_ceiling_w_per_s=10.0) == []
