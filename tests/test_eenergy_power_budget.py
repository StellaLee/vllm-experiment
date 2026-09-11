import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from power_budget import PowerBudget  # noqa: E402


def test_would_exceed_false_with_fewer_than_two_samples():
    budget = PowerBudget(window_s=10.0)
    assert budget.would_exceed(cap_w=1000.0, marginal_j=100.0) is False
    budget.record(t=0.0, fleet_power_w=5000.0)  # single sample, way over any cap
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is False


def test_would_exceed_true_when_mean_power_plus_marginal_exceeds_cap():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=900.0)
    budget.record(t=1.0, fleet_power_w=900.0)
    # mean=900W, marginal 2000J over a 10s window = +200W -> projected 1100W > cap 1000W
    assert budget.would_exceed(cap_w=1000.0, marginal_j=2000.0) is True


def test_would_exceed_false_when_projection_stays_under_cap():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=500.0)
    budget.record(t=1.0, fleet_power_w=500.0)
    # mean=500W, marginal 1000J over a 10s window = +100W -> projected 600W <= cap 1000W
    assert budget.would_exceed(cap_w=1000.0, marginal_j=1000.0) is False


def test_record_evicts_samples_older_than_window():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=100.0)
    budget.record(t=5.0, fleet_power_w=100.0)
    budget.record(t=25.0, fleet_power_w=9000.0)  # window is now [15, 25]
    # both earlier samples (t=0, t=5) must be evicted -- only the t=25 sample remains, and
    # would_exceed must fail open (< 2 samples) rather than react to it alone
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is False


def test_record_keeps_samples_within_window():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=9000.0)
    budget.record(t=5.0, fleet_power_w=9000.0)
    budget.record(t=9.0, fleet_power_w=9000.0)  # all three within [t-10, t] = [-1, 9]
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is True
