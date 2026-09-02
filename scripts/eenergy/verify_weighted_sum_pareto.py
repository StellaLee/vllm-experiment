#!/usr/bin/env python3
"""Verify the theoretical claim discussed but not yet code-checked: a weighted sum with
every weight strictly positive respects Pareto domination -- unlike LMETRIC, which gives
Share_power an implicit weight of zero (it's not in the formula at all) and can therefore
select a dominated candidate (see the LMETRIC counterexample argument in conversation:
two candidates tied on compute*load with different power are indistinguishable to LMETRIC).

scoring.py's weighted_sum_score uses w_compute=w_load=w_power=0.33 (equal, all positive) --
this is the arm actually deployed as "weighted_sum" and run on hardware alongside the DRF
family. Brute-force falsification attempt, same style as verify_pareto_lemma.py's Claim 1.
"""
import random


def dominates(a, b):
    """a dominates b: a[i] <= b[i] for all i, strict for at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_nondominated(point, others):
    return not any(dominates(o, point) for o in others if o != point)


def weighted_sum(compute, load, power, w=(0.33, 0.33, 0.33)):
    return w[0] * compute + w[1] * load + w[2] * power


# --- Claim: positive-weighted argmin is always Pareto-non-dominated ---
random.seed(0)
violations = 0
TRIALS = 200000
for _ in range(TRIALS):
    n = random.randint(2, 5)
    cands = [tuple(round(random.random(), 2) for _ in range(3)) for _ in range(n)]  # (compute, load, power)
    winner = min(cands, key=lambda c: weighted_sum(*c))
    if not pareto_nondominated(winner, cands):
        violations += 1
        print("VIOLATED:", cands, "winner=", winner)
        if violations > 5:
            break
print(f"weighted_sum (0.33/0.33/0.33) argmin stays Pareto-non-dominated: "
      f"{TRIALS} random trials, {violations} violations found.")

# --- Contrast: LMETRIC's implicit zero-weight-on-power fails the same style of instance ---
# Two candidates tied on compute*load, differing only on power -- LMETRIC score is identical
# (power doesn't appear), so LMETRIC picks whichever is first in iteration order, which can
# be the power-dominated one.
print("\nContrast: LMETRIC-style product (compute*load only) vs. this weighted sum, on a "
      "pair tied on compute/load but differing on power:")
A = (0.5, 0.5, 0.1)  # (compute, load, power) -- dominates B
B = (0.5, 0.5, 0.9)  # dominated by A (equal compute/load, strictly higher power)
lmetric_style = lambda c: c[0] * c[1]  # ignores power entirely, like new_tokens*in_flight_after
print("A dominates B?", dominates(A, B))
winner_lmetric = min([B, A], key=lmetric_style)
winner_weighted = min([B, A], key=lambda c: weighted_sum(*c))
print(f"LMETRIC-style (ignores power) picks: {'B (DOMINATED)' if winner_lmetric == B else 'A'} "
      f"-- tied score, picks whichever is first in iteration order")
print(f"weighted_sum picks: {'A (dominator, correct)' if winner_weighted == A else 'B -- BROKEN'}")
assert winner_lmetric == B, "sanity check: LMETRIC-style comparator should fail this instance"
assert winner_weighted == A, "weighted_sum must resolve this instance correctly"
print("\nAll assertions passed.")
