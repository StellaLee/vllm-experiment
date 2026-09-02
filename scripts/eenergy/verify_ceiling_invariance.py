#!/usr/bin/env python3
"""Verify two claims about drf_power_tiebreak_adaptive_isolated's use of a LIVE, fleet-history-
derived ramp ceiling in place of the static 450 W/s constant, before putting them in a paper.

Lemma 1 (verify_pareto_lemma.py, already established): argmin over the fully-SORTED
(descending) 3-tuple of (compute, load, share_power) shares is always Pareto-non-dominated
among the candidate set, when share_power = ramp/ceiling uses ONE static ceiling shared by
every candidate.

1. CEILING-INVARIANCE COROLLARY: Lemma 1 continues to hold verbatim when the ceiling is
   instead a SHARED, decision-time-constant scalar computed from anything at all -- static,
   adaptive, adversarial, whatever -- as long as it is the SAME scalar substituted into every
   candidate's power coordinate for that one decision. This is what the isolated design
   actually does: `AdaptiveRampCeiling` is instantiated ONCE per router
   (`self.adaptive_ceiling_isolated = AdaptiveRampCeiling()`), not once per replica, so
   `live_ceiling` is one number applied identically to all N candidates in
   `_build_candidates`. Brute-force falsification attempt: randomize the shared ceiling
   itself per trial (not just the raw shares) and confirm zero violations.

2. NECESSITY OF A SHARED (not per-candidate) CEILING: under a shared ceiling, share_power's
   ordering across candidates is a strictly monotonic function of raw ramp rate (dividing
   everyone by the same positive number preserves who-is-more-pressured). Under PER-CANDIDATE
   ceilings, this agreement can break: a candidate with a numerically higher raw ramp rate can
   score a LOWER share_power purely because its own personal ceiling happens to be set higher
   -- the sorted-vector rule then selects it as "safe" even though it is physically drawing
   more power-ramp than the alternative. This is not a Pareto-non-domination violation in
   share-space (Claim 1 still holds trivially there, since it treats the 3 coordinates as
   arbitrary reals) -- it is a demonstration that share-space non-domination only means what we
   want it to mean (protecting the physically-safer replica) when the ceiling is shared. Ties
   the paper's claim directly to the actual code structure: one AdaptiveRampCeiling per
   router, not one per replica.
"""
import random


def dominates(a, b):
    """a dominates b: a[i] <= b[i] for all i, strict for at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_nondominated(point, others):
    return not any(dominates(o, point) for o in others if o != point)


def sorted_desc(t):
    return tuple(sorted(t, reverse=True))


# --- Claim 1: ceiling-invariance, brute-force falsification attempt ---
random.seed(0)
violations = 0
TRIALS = 200000
for _ in range(TRIALS):
    n = random.randint(2, 5)
    kappa = random.uniform(0.05, 5.0)  # shared ceiling for this trial, randomized per trial
    raw = [(round(random.random(), 2), round(random.random(), 2),
            round(random.uniform(0.0, 2.0), 2)) for _ in range(n)]  # (compute, load, raw_ramp)
    cands = [(c, l, r / kappa) for (c, l, r) in raw]  # realized shares under the shared ceiling
    winner = min(cands, key=sorted_desc)
    if not pareto_nondominated(winner, cands):
        violations += 1
        print("CLAIM 1 VIOLATED:", cands, "kappa=", kappa, "winner=", winner)
        if violations > 5:
            break
print(f"Claim 1 (sorted-vector argmin stays Pareto-non-dominated under ANY shared, "
      f"per-trial-randomized ceiling): {TRIALS} random trials, {violations} violations found.")

# --- Claim 2: necessity of a SHARED (not per-candidate) ceiling ---
print("\nClaim 2 setup: two candidates, identical compute/load, different raw ramp AND "
      "different PER-CANDIDATE ceilings.")
compute, load = 0.3, 0.3
A_raw_ramp, A_kappa = 0.9, 10.0   # physically high ramp, but generously calibrated ceiling
B_raw_ramp, B_kappa = 0.1, 0.1    # physically low ramp, but tightly calibrated ceiling
A_share_power = A_raw_ramp / A_kappa
B_share_power = B_raw_ramp / B_kappa
A = (compute, load, A_share_power)
B = (compute, load, B_share_power)
print(f"A: raw_ramp={A_raw_ramp}, kappa={A_kappa} -> share_power={A_share_power:.3f}")
print(f"B: raw_ramp={B_raw_ramp}, kappa={B_kappa} -> share_power={B_share_power:.3f}")
print("Raw-ramp ordering: A has the HIGHER (worse) raw ramp:", A_raw_ramp > B_raw_ramp)
print("Share-space ordering: A has the LOWER (better) share_power:", A_share_power < B_share_power)
winner = min([A, B], key=sorted_desc)
picked_A = (winner == A)
print("Sorted-vector argmin picks:", "A" if picked_A else "B")
print("Is Claim 1 (share-space Pareto-non-domination) still satisfied?",
      pareto_nondominated(winner, [A, B]), "(expected True -- domination is judged IN "
      "share-space, and per-trial share-space still behaves per Claim 1)")
print("But the candidate selected as 'safe' has the physically HIGHER raw ramp rate:",
      picked_A and A_raw_ramp > B_raw_ramp,
      "-- share-space non-domination no longer implies physical-space non-domination once "
      "ceilings are candidate-specific. A shared ceiling (one AdaptiveRampCeiling per router, "
      "not per replica) is what keeps the two notions aligned.")
