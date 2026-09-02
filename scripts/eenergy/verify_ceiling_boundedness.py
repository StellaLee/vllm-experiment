#!/usr/bin/env python3
"""Verify a claim about the round-filtered calibration (`observe_round`) that
`drf_power_tiebreak_adaptive_isolated` uses in place of the naive per-observation calibration
(`observe`) that plain `drf_power_tiebreak_adaptive` uses, before putting it in a paper.

BACKGROUND (adaptive_ceiling.py docstring, already in the codebase): naive `observe()`-based
calibration is self-defeating under sustained multi-replica pressure -- concentration (2+
replicas simultaneously elevated) is exactly the condition that fills the rolling window with
elevated values, so the ceiling inflates MOST during the episodes it's supposed to guard
against. `observe_round()` fixes this by skipping the WHOLE round whenever 2+ replicas are
simultaneously elevated above the floor.

INSENSITIVITY LEMMA (this script): the round-filtered ceiling is not merely "less sensitive"
to concentration-round magnitude -- it is COMPLETELY insensitive to it. Two fleet-ramp-reading
histories that agree on every round with <2 elevated replicas, and differ ARBITRARILY
(including unboundedly) on rounds with >=2 elevated replicas, produce IDENTICAL
`observe_round`-based ceiling trajectories at every timestep -- because `observe_round` returns
early on a concentrated round without touching the window at all, so no value from that round,
however extreme, can ever reach the percentile calculation. This is a real code property, not
an approximation, so it should verify with ZERO tolerance (exact equality), not "close enough."

CONTRAST: the same paired histories run through naive `observe()` DO diverge -- confirming this
is specifically what round-filtering fixes, not a property both calibration schemes already had.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from adaptive_ceiling import AdaptiveRampCeiling  # noqa: E402

FLOOR = 450.0
N_REPLICAS = 6


def gen_round(rng, concentrated: bool):
    """One round's ramp readings for N_REPLICAS candidates.
    concentrated=True  -> force >=2 replicas above FLOOR (a concentration event).
    concentrated=False -> force <=1 replica above FLOOR (isolated / benign variability)."""
    readings = [rng.uniform(0.0, FLOOR * 0.8) for _ in range(N_REPLICAS)]
    if concentrated:
        idx = rng.sample(range(N_REPLICAS), 2)
        for i in idx:
            readings[i] = rng.uniform(FLOOR * 1.1, FLOOR * 50.0)  # arbitrarily extreme
    else:
        if rng.random() < 0.5:
            i = rng.randrange(N_REPLICAS)
            readings[i] = rng.uniform(FLOOR * 1.1, FLOOR * 5.0)  # one isolated elevated reading
    return readings


def run_trial(rng, n_rounds, magnitude_multiplier):
    """Builds two paired round-schedules (same concentrated/isolated pattern, same isolated-
    round values) that differ ONLY in how extreme the concentrated-round values are (scaled by
    magnitude_multiplier). Runs both through observe_round AND through naive observe, returning
    the ceiling trajectories for each."""
    pattern = [rng.random() < 0.35 for _ in range(n_rounds)]  # True = concentrated round

    ceil_a_isolated = AdaptiveRampCeiling(floor_w_per_s=FLOOR)
    ceil_b_isolated = AdaptiveRampCeiling(floor_w_per_s=FLOOR)
    ceil_a_naive = AdaptiveRampCeiling(floor_w_per_s=FLOOR)
    ceil_b_naive = AdaptiveRampCeiling(floor_w_per_s=FLOOR)

    traj_a_isolated, traj_b_isolated = [], []
    traj_a_naive, traj_b_naive = [], []

    for concentrated in pattern:
        round_a = gen_round(rng, concentrated)
        if concentrated:
            # sequence B: identical round, but concentrated-round values scaled up further --
            # an arbitrarily more extreme concentration event.
            round_b = [v * magnitude_multiplier if v > FLOOR else v for v in round_a]
        else:
            round_b = round_a  # isolated rounds are IDENTICAL between the two sequences

        ceil_a_isolated.observe_round(round_a)
        ceil_b_isolated.observe_round(round_b)
        traj_a_isolated.append(ceil_a_isolated.ceiling())
        traj_b_isolated.append(ceil_b_isolated.ceiling())

        for v in round_a:
            ceil_a_naive.observe(v)
        for v in round_b:
            ceil_b_naive.observe(v)
        traj_a_naive.append(ceil_a_naive.ceiling())
        traj_b_naive.append(ceil_b_naive.ceiling())

    return traj_a_isolated, traj_b_isolated, traj_a_naive, traj_b_naive


# --- Claim: round-filtered ceiling is exactly insensitive to concentration-round magnitude ---
random.seed(0)
TRIALS = 2000
insensitivity_violations = 0
naive_diverged_count = 0
max_naive_divergence = 0.0

for trial in range(TRIALS):
    rng = random.Random(trial)
    n_rounds = rng.randint(20, 200)
    magnitude_multiplier = rng.uniform(1.5, 1000.0)
    traj_a_iso, traj_b_iso, traj_a_naive, traj_b_naive = run_trial(rng, n_rounds, magnitude_multiplier)

    if traj_a_iso != traj_b_iso:
        insensitivity_violations += 1
        print(f"INSENSITIVITY VIOLATED at trial {trial}: isolated ceilings diverged "
              f"despite identical isolated-round data.")
        for t, (a, b) in enumerate(zip(traj_a_iso, traj_b_iso)):
            if a != b:
                print(f"  round {t}: A={a} B={b}")
                break

    if traj_a_naive != traj_b_naive:
        naive_diverged_count += 1
        divergence = max(abs(a - b) for a, b in zip(traj_a_naive, traj_b_naive))
        max_naive_divergence = max(max_naive_divergence, divergence)

print(f"Insensitivity lemma (round-filtered ceiling unaffected by concentration-round "
      f"magnitude, however extreme): {TRIALS} random trials, {insensitivity_violations} "
      f"violations found.")
print(f"Contrast (naive observe()-based ceiling): diverged between the paired sequences in "
      f"{naive_diverged_count}/{TRIALS} trials (expected: most trials with at least one "
      f"concentrated round), max observed divergence in a single trial = "
      f"{max_naive_divergence:.1f} W/s.")

# --- Concrete illustrative trace, printed for the paper writeup ---
print("\n--- Illustrative trace ---")
rng = random.Random(42)
n_rounds = 40
magnitude_multiplier = 20.0
traj_a_iso, traj_b_iso, traj_a_naive, traj_b_naive = run_trial(rng, n_rounds, magnitude_multiplier)
print(f"Sequence A vs. B differ only in concentration-round magnitude (B's concentrated "
      f"readings are {magnitude_multiplier}x A's).")
print(f"Final round-filtered ceiling: A={traj_a_iso[-1]:.1f} W/s, B={traj_b_iso[-1]:.1f} W/s "
      f"(identical: {traj_a_iso[-1] == traj_b_iso[-1]})")
print(f"Final naive ceiling:          A={traj_a_naive[-1]:.1f} W/s, B={traj_b_naive[-1]:.1f} W/s "
      f"(divergence: {abs(traj_a_naive[-1] - traj_b_naive[-1]):.1f} W/s)")
