"""Peak-shaving admission gate shim for 8-GPU P/D-disaggregated serving (see
docs/superpowers/specs/2026-09-11-disagg-peak-shaving-gate-design.md). Sits in front of the
unmodified, vendored nixl_toy_proxy_server.py -- decides WHETHER/WHEN to admit a request
against two independent per-pool power budgets, never touches HOW requests are routed once
admitted (that's entirely the Nixl proxy's job, unchanged). Mirrors proxy_server.py's own
gate-in-front-of-routing separation, generalized from one combined fleet budget to two
independent pool budgets."""
from power_budget import PowerBudget


def build_power_budgets(peak_cap_prefill_w: float, peak_cap_decode_w: float,
                         peak_window_s: float = 30.0) -> tuple:
    """Returns (budget_prefill, budget_decode), each a fresh PowerBudget if its own cap is
    set (not None), else None -- independently, so the gate can run with only one pool capped
    (useful for isolating which pool actually drives peak power). Mirrors proxy_server.py's
    build_power_budget, generalized to two independent pools."""
    budget_prefill = PowerBudget(peak_window_s) if peak_cap_prefill_w is not None else None
    budget_decode = PowerBudget(peak_window_s) if peak_cap_decode_w is not None else None
    return budget_prefill, budget_decode


def pool_power_w(reader, gpu_indices: list) -> float:
    """Sum of a live NVML power read across every GPU in one pool -- the aggregate
    instantaneous power that pool's PowerBudget tracks. Unlike proxy_server.py's
    fleet_power_w (which sums a cached state.last_power_w kept current by a separate
    ramp-ceiling poller), this reads NVML directly at poll time: disagg_gate.py has no
    ReplicaState/ramp-ceiling machinery to piggyback on, and doesn't need any -- the gate
    only needs a fleet-aggregate power number, not per-replica ramp tracking."""
    return sum(reader.read(i)[0] for i in gpu_indices)
