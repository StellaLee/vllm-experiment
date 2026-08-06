#!/usr/bin/env python3
"""Regime-switching CONTINUOUS power model -- Option 2 fix to the two-state (flat P_b/P_max)
coincidence_factor_model.py, after the two-state assumption was shown to discard real signal:
per-5s-window stats on the real measured trace show power fluctuates continuously (min/max
spread of ~100W) even OUTSIDE whale windows, not sitting flat at a single baseline constant.
Root cause of the earlier flat-looking calibration: the "baseline" sample set was contaminated
by the first ~5s (cold-start ramp-up from idle, before the closed-loop concurrency=20 batch
fills) and last ~3-5s (drain-down as the finite trial's remaining requests complete) of each
trial -- genuine transients of a bounded-duration measurement, not steady-state decode power.

Fix: keep the SAME regime-switching TIMING model as before (whale windows are real,
alternating-renewal events -- reused unchanged from coincidence_factor_model.simulate_states),
but replace the flat P_b/P_max targets with a proper within-regime stochastic process: an
Ornstein-Uhlenbeck (OU) process whose MEAN switches between mu_b (baseline) and mu_max (whale)
following the same regime timing, and whose stationary variance (sigma_b^2 / sigma_max^2) is
calibrated from the TRIMMED empirical baseline/whale power distributions instead of collapsing
to a median/max point value:

    dP = kappa*(mu(t) - P)*dt + sqrt(2*kappa)*sigma(t)*dW,   kappa = 1/tau

which has stationary variance sigma(t)^2 if the regime is held indefinitely (standard OU
result Var = diffusion^2/(2*kappa) = (sqrt(2*kappa)*sigma)^2/(2*kappa) = sigma^2), so the
calibrated within-regime variance is preserved exactly, not just the mean.

kappa (equivalently tau) can no longer use the old closed-form tau=(P_max-P_b)/R plug-in,
because that derivation assumed a NOISE-FREE deterministic step response -- with OU noise now
also contributing to the simulated |dP/dt|, the same closed form would double-count. Instead
kappa is fit numerically (bisection) so the single-server simulated mean|dP/dt| matches the
REAL measured mean ramp rate (45.7 W/s mono, 30.8 W/s chunk=512 -- same reference numbers
already cited in the paper's Section V, so the fleet-scale comparison stays anchored to the
same ground truth as the two-state model).
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from coincidence_factor_model import (alternating_process, load_power, load_records,
                                       merge_windows, simulate_states, whale_windows)

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def calibrate_continuous(records_path, power_path, label, whale_thresh=15000, trim=5.0):
    """Like coincidence_factor_model.calibrate, but (a) trims `trim` seconds off both ends of
    the trace to exclude cold-start/drain-down transients, and (b) returns regime MEANS and
    STDS (mu_b, sigma_b, mu_max, sigma_max) instead of a single median/max point value, so the
    within-regime continuous variation is preserved rather than discarded."""
    recs = load_records(records_path)
    power = load_power(power_path)
    raw_windows = whale_windows(recs, whale_thresh=whale_thresh)
    windows = merge_windows(raw_windows)

    t0, t1 = power[0][0], power[-1][0]
    lo, hi = t0 + trim, t1 - trim
    T = hi - lo
    burst_durations = [min(e, hi) - max(s, lo) for s, e in windows if e > lo and s < hi]
    burst_durations = [d for d in burst_durations if d > 0]
    mean_burst = sum(burst_durations) / len(burst_durations)
    p_duty = sum(burst_durations) / T

    starts = sorted(s for s, e in windows if lo <= s <= hi)
    inter_arrivals = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    mean_interarrival = sum(inter_arrivals) / len(inter_arrivals) if len(starts) > 1 else T
    lam = 1.0 / mean_interarrival

    def in_any(t):
        return any(s <= t <= e for s, e in windows)

    baseline = np.array([p for t, p in power if lo <= t <= hi and not in_any(t)])
    whale = np.array([p for t, p in power if lo <= t <= hi and in_any(t)])
    mu_b, sigma_b = baseline.mean(), baseline.std()
    mu_max, sigma_max = whale.mean(), whale.std()

    print(f"[{label}] T={T:.1f}s (trimmed {trim}s/side) n_whale={len(windows)} "
          f"mean_burst={mean_burst:.2f}s p_duty={p_duty:.4f} "
          f"mu_b={mu_b:.1f}W sigma_b={sigma_b:.1f}W mu_max={mu_max:.1f}W sigma_max={sigma_max:.1f}W")
    return dict(lam=lam, mean_burst=mean_burst, p_duty=p_duty,
                mu_b=mu_b, sigma_b=sigma_b, mu_max=mu_max, sigma_max=sigma_max,
                P_b=mu_b, P_max=mu_max)  # P_b/P_max aliases: simulate_states only needs mean_burst/p_duty


def ou_continuous_aggregate(state, mu_b, mu_max, sigma_b, sigma_max, dt, kappa, rng):
    """Vectorized regime-switching OU simulation across N servers (rows of `state`). Returns
    the (N, n_steps) continuous per-server power matrix; caller sums across axis=0 for the
    fleet aggregate.

    Uses the EXACT OU discretization (piecewise-constant mu/sigma held over each dt step),
    not explicit Euler-Maruyama: P[i] = mu + (P[i-1]-mu)*exp(-kappa*dt) + sigma*sqrt(1-exp(-2*
    kappa*dt))*Z. This is unconditionally stable for any kappa (Euler-Maruyama blows up once
    kappa*dt is not small -- hit exactly this during kappa-fitting, since the search range
    needed to reach kappa~O(10) at dt=0.1 to match the real measured ramp rate). It is also
    exact, not an approximation, as long as mu/sigma are constant within each step (true here:
    dt=0.1s is far shorter than the ~1.6-3.7s regime-burst durations)."""
    N, n_steps = state.shape
    P = np.empty((N, n_steps), dtype=float)
    P[:, 0] = mu_b
    decay = np.exp(-kappa * dt)
    noise_scale = np.sqrt(1.0 - decay ** 2)
    for i in range(1, n_steps):
        on = state[:, i - 1]
        mu = np.where(on, mu_max, mu_b)
        sigma = np.where(on, sigma_max, sigma_b)
        Z = rng.standard_normal(N)
        P[:, i] = mu + (P[:, i - 1] - mu) * decay + sigma * noise_scale * Z
    return P


def single_server_mean_ramp(kappa, cal, dt, T_sim, warmup, n_mc, rng):
    """Simulate N=1 server for n_mc independent reps, return mean |dP/dt| over the
    post-warmup portion (same warmup-discard discipline as the two-state model's ramp-CF fix,
    to exclude the shared cold-start transient from the ramp statistic)."""
    i0 = int(warmup / dt)
    ramps = []
    for _ in range(n_mc):
        state = simulate_states(1, 0.0, cal, T_sim + warmup, dt, rng)
        P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                     cal["sigma_max"], dt, kappa, rng)[0]
        ramp = np.abs(np.diff(P[i0:])) / dt
        ramps.append(ramp.mean())
    return np.mean(ramps)


def fit_kappa(target_ramp, cal, dt, T_sim, warmup, n_mc, rng, lo=0.01, hi=50.0, tol=0.02, max_iter=25):
    """Bisection on kappa (=1/tau) so single_server_mean_ramp(kappa) matches target_ramp
    (the real measured mean ramp rate, W/s). mean|dP/dt| is monotonically increasing in kappa
    (faster mean-reversion -> both sharper regime transitions AND faster noise decorrelation
    -> larger instantaneous ramps), so bisection is well-posed."""
    r_lo = single_server_mean_ramp(lo, cal, dt, T_sim, warmup, n_mc, rng)
    r_hi = single_server_mean_ramp(hi, cal, dt, T_sim, warmup, n_mc, rng)
    assert r_lo <= target_ramp <= r_hi, f"target {target_ramp} outside bracket [{r_lo:.2f}, {r_hi:.2f}]"
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = single_server_mean_ramp(mid, cal, dt, T_sim, warmup, n_mc, rng)
        if abs(r_mid - target_ramp) / target_ramp < tol:
            return mid, r_mid
        if r_mid < target_ramp:
            lo = mid
        else:
            hi = mid
    return mid, r_mid


def fit_kappa_self_consistent(target_ramp, cal, dt, T_sim, n_mc, rng, safety=10.0, floor=50.0,
                               lo=0.01, hi=50.0, tol=0.02, max_iter=25, max_outer=6):
    """Fixed-point refinement of fit_kappa. fit_kappa's own bisection discards a fixed `warmup`
    internally -- but if the resulting kappa implies a slow OU relaxation (tau=1/kappa)
    comparable to or longer than that warmup, the "post-warmup" ramp measurement used DURING
    fitting is itself still contaminated by the cold-start transient (a systematic drift adds
    directly to the measured mean|dP/dt|), which biases the fit toward an even smaller kappa
    than warranted -- a self-reinforcing bug, not just an analysis-time nuisance. Confirmed by
    direct measurement: the real trace's own autocorrelation decorrelates in ~0.6s, while a
    naively-fitted kappa (50s warmup) implied tau up to ~48s for chunk=512 in some conditions --
    a ~50-80x mismatch, and the resulting aggregate (N=10,000) cold-start drift was found to be
    the same ORDER OF MAGNITUDE as the reported peak-ramp statistic itself (~20,000 W/s vs.
    ~13,000-23,000 W/s), not a rounding error.

    Fix: iteratively grow warmup to `safety`/kappa (default 10 tau, >99.99% converged) and refit
    kappa at that larger warmup, until warmup stops growing. Returns (kappa, ramp, warmup) --
    callers MUST reuse the returned warmup (not a hardcoded constant) for any subsequent
    simulation (single-server ramp checks and the N-server peak-ramp Monte Carlo alike), or the
    same contamination reappears downstream even with a correctly-fitted kappa."""
    warmup = floor
    kappa, r = fit_kappa(target_ramp, cal, dt, T_sim, warmup, n_mc, rng, lo, hi, tol, max_iter)
    for _ in range(max_outer):
        needed = max(floor, safety / kappa)
        if needed <= warmup * 1.05:
            break
        warmup = needed
        kappa, r = fit_kappa(target_ramp, cal, dt, T_sim, warmup, n_mc, rng, lo, hi, tol, max_iter)
    return kappa, r, warmup


def ramp_coincidence_factor_continuous(N, cal, kappa, target_ramp, T_window, dt, n_mc, rng, warmup=50.0):
    """Continuous-model analogue of coincidence_factor_model.ramp_coincidence_factor: same
    warmup-discard discipline (the earlier two-state model's own bug was a synchronized
    cold-start transient dominating the peak-ramp measurement; the fix generalizes directly
    here since all N servers still start at mu_b simultaneously).

    Denominator: N * target_ramp (the real measured single-server MEAN ramp rate this kappa
    was fit to reproduce), NOT kappa*(mu_max-mu_b) (the noiseless drift-only post-switch
    slope). The drift-only quantity undersells a single server's real ramp by ~20x here (2.3
    W/s vs the calibrated 45.7 W/s) because in this continuous model most of a real server's
    ramp comes from the OU noise term, not the regime-switch drift -- using it as the
    normalizer produced CF_ramp > 1, an unphysical result. target_ramp is what "a single
    server's own characteristic ramp" actually means once noise is part of the model."""
    single_max_ramp = target_ramp
    i0 = int(warmup / dt)
    peak_ramps = []
    for _ in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                     cal["sigma_max"], dt, kappa, rng)
        D = P.sum(axis=0)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps.append(ramp[i0:].max() if len(ramp) > i0 else 0.0)
    return np.mean(peak_ramps) / (N * single_max_ramp)


def main():
    dt = 0.1
    T_sim = 100.0
    warmup = 50.0
    rng = np.random.default_rng(20260728)

    print("=== Step 1: recalibrate regime stats (trimmed, mean+std not median/max) ===")
    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                                     "mono (b16384)")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv",
                                      "chunk=512")

    print()
    print("=== Step 2: fit kappa so single-server simulated mean|dP/dt| matches real measured rate ===")
    print("(targets: 45.7 W/s mono, 30.8 W/s chunk=512 -- same reference numbers cited in paper Sec V)")
    n_mc_fit = 30
    kappa_mono, r_mono = fit_kappa(45.7, cal_mono, dt, T_sim, warmup, n_mc_fit, rng)
    print(f"  mono:      kappa={kappa_mono:.4f} (tau={1/kappa_mono:.3f}s) -> simulated mean_ramp={r_mono:.2f} W/s")
    kappa_chunk, r_chunk = fit_kappa(30.8, cal_chunk, dt, T_sim, warmup, n_mc_fit, rng)
    print(f"  chunk=512: kappa={kappa_chunk:.4f} (tau={1/kappa_chunk:.3f}s) -> simulated mean_ramp={r_chunk:.2f} W/s")

    print()
    print("=== Step 3: fleet-scale ramp-CF under the continuous model, N-sweep ===")
    n_mc_cf = 30
    for N in [1000, 5000, 10000, 20000]:
        cf_mono = ramp_coincidence_factor_continuous(N, cal_mono, kappa_mono, 45.7, T_sim, dt, n_mc_cf, rng, warmup)
        cf_chunk = ramp_coincidence_factor_continuous(N, cal_chunk, kappa_chunk, 30.8, T_sim, dt, n_mc_cf, rng, warmup)
        print(f"  N={N:<6} CF_ramp mono={cf_mono:.4f}  chunk=512={cf_chunk:.4f}  "
              f"reduction={100*(1-cf_chunk/cf_mono):.1f}%")


if __name__ == "__main__":
    main()
