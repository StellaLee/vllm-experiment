import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from power_budget import PowerBudget, estimate_marginal_energy_j, DecodeByteEstimator  # noqa: E402


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


def test_reserve_immediately_affects_would_exceed_without_a_new_sample():
    """The TOCTOU gap this ledger closes: two admission checks against the SAME stale power
    samples (no new record() call between them, simulating two requests arriving within one
    power_poll_loop interval) -- the second check must see the first request's reservation,
    not just the stale real-power reading."""
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=2000.0)
    budget.record(t=1.0, fleet_power_w=2000.0)
    # mean=2000W, cap=2400W -> 400W of headroom -> 4000J fits (400*10), 5000J doesn't
    assert budget.would_exceed(cap_w=2400.0, marginal_j=4000.0) is False
    budget.reserve(4000.0)
    # same stale samples, but now 4000J is already reserved -- a second, identical request
    # must now be rejected even though the real power reading hasn't changed at all
    assert budget.would_exceed(cap_w=2400.0, marginal_j=4000.0) is True


def test_release_removes_a_reservation():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=2000.0)
    budget.record(t=1.0, fleet_power_w=2000.0)
    budget.reserve(4000.0)
    assert budget.would_exceed(cap_w=2400.0, marginal_j=4000.0) is True
    budget.release(4000.0)
    assert budget.would_exceed(cap_w=2400.0, marginal_j=4000.0) is False


def test_release_does_not_go_negative():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=100.0)
    budget.record(t=1.0, fleet_power_w=100.0)
    budget.release(500.0)  # releasing more than was ever reserved
    # should not leave _reserved_j negative in a way that lets an oversized request through
    # unrealistically -- would_exceed's own math still holds for a fresh, large marginal_j
    assert budget.would_exceed(cap_w=100.0, marginal_j=100000.0) is True


def test_estimate_marginal_energy_j_combines_prefill_and_decode_terms():
    j = estimate_marginal_energy_j(prompt_tokens=100, expected_decode_tokens=50,
                                    j_per_prefill_token=0.068, j_per_decode_token=2.40)
    assert j == pytest.approx(100 * 0.068 + 50 * 2.40)


def test_estimate_marginal_energy_j_zero_tokens_is_zero():
    assert estimate_marginal_energy_j(0, 0, 0.068, 2.40) == 0.0


def test_decode_byte_estimator_cold_start_uses_fallback():
    est = DecodeByteEstimator()
    assert est.current_estimate_tokens(fallback_tokens=1024, bytes_per_token=3.235) == 1024


def test_decode_byte_estimator_first_completion_seeds_estimate():
    est = DecodeByteEstimator()
    est.record_completion(response_bytes=970.5)  # ~300 tokens at 3.235 bytes/token
    tokens = est.current_estimate_tokens(fallback_tokens=1024, bytes_per_token=3.235)
    assert tokens == pytest.approx(970.5 / 3.235)


def test_decode_byte_estimator_blends_subsequent_completions():
    est = DecodeByteEstimator(smoothing_alpha=0.5)
    est.record_completion(response_bytes=1000.0)
    est.record_completion(response_bytes=2000.0)
    # EMA: 0.5*2000 + 0.5*1000 = 1500
    tokens = est.current_estimate_tokens(fallback_tokens=99999, bytes_per_token=1.0)
    assert tokens == pytest.approx(1500.0)


def test_decode_byte_estimator_with_real_wire_bytes_per_token_gives_sane_token_estimate():
    """Regression test for the bug found on first live validation: response_bytes counts raw
    HTTP/SSE wire bytes (JSON event scaffolding included), NOT plain generated text -- real
    measurement against this project's vLLM streaming endpoint gave 272.2 and 272.1
    bytes/token across two independent samples (n_sse_events as the token-count proxy), not
    src/replay_sharegpt.py's 3.235 plain-text chars-per-token constant. Using the wrong
    (much smaller) constant here inflates the apparent token count by ~84x, which in
    production caused a self-reinforcing admission-gate lockup: the first real completion
    seeds the EMA with a wildly wrong estimate, which then blocks every subsequent request
    regardless of real power, so nothing completes to correct it. This test locks in that a
    realistic wire-byte response (272 bytes/token, ~300 real tokens) converts back to a
    plausible token count with the real constant -- not an order-of-magnitude-inflated one."""
    est = DecodeByteEstimator()
    real_bytes_per_token = 272.0
    realistic_response_bytes = 300 * real_bytes_per_token  # a ~300-token response, in wire bytes
    est.record_completion(response_bytes=realistic_response_bytes)
    tokens = est.current_estimate_tokens(fallback_tokens=1024, bytes_per_token=real_bytes_per_token)
    assert tokens == pytest.approx(300.0)
    # Sanity-check the bug this guards against: using the OLD plain-text constant against the
    # SAME real wire-byte value would have inflated the estimate by ~84x.
    wrong_bytes_per_token = 3.235
    inflated_tokens = realistic_response_bytes / wrong_bytes_per_token
    assert inflated_tokens > tokens * 50  # confirms the old constant was badly wrong, not just off
