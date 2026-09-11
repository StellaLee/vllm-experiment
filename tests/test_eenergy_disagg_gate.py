import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from disagg_gate import build_power_budgets, pool_power_w  # noqa: E402
from power_budget import PowerBudget  # noqa: E402


class FakeReader:
    """Duck-types NvmlPowerReader's .read(gpu_index) -> (power_w, ts) interface, so
    pool_power_w is testable without real pynvml/hardware -- matches this project's existing
    convention of keeping router-logic tests hardware-free."""

    def __init__(self, power_by_index: dict):
        self._power_by_index = power_by_index

    def read(self, gpu_index: int) -> tuple:
        return self._power_by_index[gpu_index], 12345.0


def test_pool_power_w_sums_across_the_given_gpu_indices():
    reader = FakeReader({0: 100.0, 1: 150.0, 2: 200.0, 3: 50.0})
    assert pool_power_w(reader, [0, 1, 2, 3]) == 500.0


def test_pool_power_w_ignores_gpus_outside_the_given_indices():
    reader = FakeReader({0: 100.0, 4: 900.0})
    assert pool_power_w(reader, [0]) == 100.0


def test_build_power_budgets_both_none_when_both_caps_unset():
    budget_prefill, budget_decode = build_power_budgets(None, None, peak_window_s=30.0)
    assert budget_prefill is None
    assert budget_decode is None


def test_build_power_budgets_only_prefill_enabled():
    budget_prefill, budget_decode = build_power_budgets(1800.0, None, peak_window_s=30.0)
    assert isinstance(budget_prefill, PowerBudget)
    assert budget_prefill.window_s == 30.0
    assert budget_decode is None


def test_build_power_budgets_both_enabled_independent_instances():
    budget_prefill, budget_decode = build_power_budgets(1800.0, 900.0, peak_window_s=15.0)
    assert isinstance(budget_prefill, PowerBudget)
    assert isinstance(budget_decode, PowerBudget)
    assert budget_prefill is not budget_decode
    assert budget_prefill.window_s == 15.0
    assert budget_decode.window_s == 15.0
