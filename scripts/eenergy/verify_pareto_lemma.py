#!/usr/bin/env python3
"""Verify two claims before putting them in a paper:

1. LEMMA: argmin over the fully-SORTED (descending) 3-tuple of shares is always
   Pareto-non-dominated among the candidate set. (This is what drf_fixed's
   dominant_share_vector does.)

2. COUNTEREXAMPLE: argmin over the NAMED fixed-priority tuple
   (dominant_share, share_power, share_load) -- what drf_power_tiebreak does -- can
   select a candidate that IS Pareto-dominated by another candidate, specifically when
   dominant/power/load all tie but compute differs while compute is not the argmax
   coordinate for either candidate.

Brute-force random search for claim 1 (falsification attempt) and a concrete hand-built
instance for claim 2.
"""
import random
import itertools


def dominates(a, b):
    """a dominates b: a[i] <= b[i] for all i, strict for at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_nondominated(point, others):
    return not any(dominates(o, point) for o in others if o != point)


def sorted_desc(t):
    return tuple(sorted(t, reverse=True))


def named_tuple(compute, load, power):
    dominant = max(compute, load, power)
    return (dominant, power, load)


# --- Claim 1: brute-force falsification attempt ---
random.seed(0)
violations = 0
TRIALS = 200000
for _ in range(TRIALS):
    n = random.randint(2, 5)
    cands = [tuple(round(random.random(), 2) for _ in range(3)) for _ in range(n)]
    # argmin over sorted-desc tuple
    winner = min(cands, key=sorted_desc)
    if not pareto_nondominated(winner, cands):
        violations += 1
        print("CLAIM 1 VIOLATED:", cands, "winner=", winner)
        if violations > 5:
            break
print(f"Claim 1 (sorted-vector argmin is Pareto-non-dominated): "
      f"{TRIALS} random trials, {violations} violations found.")

# --- Claim 2: concrete counterexample ---
A = (0.3, 0.9, 0.9)  # (compute, load, power)
B = (0.5, 0.9, 0.9)
cands = [A, B]
print("\nClaim 2 setup: A =", A, " B =", B, " (compute, load, power)")
print("A dominates B?", dominates(A, B))
print("named_tuple(A) =", named_tuple(*A))
print("named_tuple(B) =", named_tuple(*B))
winner_named = min(cands, key=lambda c: named_tuple(*c))
print("argmin over NAMED tuple picks:", winner_named)
print("Is that the dominator (A)?", winner_named == A)

# Now force B to be picked by putting it first (min() keeps the first on an exact tie)
cands_reordered = [B, A]
winner_named2 = min(cands_reordered, key=lambda c: named_tuple(*c))
print("With B first in iteration order, argmin over NAMED tuple picks:", winner_named2)
print("Is the dominated point (B) selected?", winner_named2 == B)

winner_sorted = min(cands_reordered, key=lambda c: sorted_desc(c))
print("Meanwhile argmin over SORTED tuple (drf_fixed-style) picks:", winner_sorted,
      "(should be A, the dominator, regardless of order)")
