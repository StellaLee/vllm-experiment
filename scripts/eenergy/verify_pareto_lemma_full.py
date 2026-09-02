#!/usr/bin/env python3
"""Verify the fix for the Pareto-domination counterexample that verify_pareto_lemma.py's
Claim 2 establishes against the named rule (route via (D, share_power, share_load)):

CLAIM: appending share_compute as an explicit fourth tie-break coordinate --
(D, share_power, share_load, share_compute), what scoring.py calls
dominant_share_vector_power_priority_full -- restores Pareto-non-domination while keeping
the same primary criterion (D) and the same power-before-load priority as the plain named
rule. This is a strictly stronger claim than Lemma 1 (which only covers the SORTED rule):
here we keep the deliberate, asymmetric power-first tie-break structure the named rule was
built for, and still get the safety guarantee.

1. Brute-force falsification attempt, same style as verify_pareto_lemma.py's Claim 1: many
   random candidate sets, confirm argmin over the 4-tuple never selects a dominated point.
2. The exact counterexample instance that breaks the plain named rule (Claim 2 in
   verify_pareto_lemma.py): confirm the 4-tuple rule resolves it, regardless of iteration
   order, while the plain named rule still fails it (contrast, not just a fix in isolation).
"""
import random


def dominates(a, b):
    """a dominates b: a[i] <= b[i] for all i, strict for at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_nondominated(point, others):
    return not any(dominates(o, point) for o in others if o != point)


def named_tuple(compute, load, power):
    """Plain named rule (scoring.py's dominant_share_vector_power_priority): (D, power, load).
    Known-unsafe -- Claim 2 in verify_pareto_lemma.py."""
    dominant = max(compute, load, power)
    return (dominant, power, load)


def named_tuple_full(compute, load, power):
    """Fixed named rule (scoring.py's dominant_share_vector_power_priority_full):
    (D, power, load, compute) -- this script's subject."""
    dominant = max(compute, load, power)
    return (dominant, power, load, compute)


# --- Claim: fixed named rule's argmin is always Pareto-non-dominated ---
random.seed(0)
violations = 0
TRIALS = 200000
for _ in range(TRIALS):
    n = random.randint(2, 5)
    cands = [tuple(round(random.random(), 2) for _ in range(3)) for _ in range(n)]  # (compute, load, power)
    winner = min(cands, key=lambda c: named_tuple_full(*c))
    if not pareto_nondominated(winner, cands):
        violations += 1
        print("VIOLATED:", cands, "winner=", winner)
        if violations > 5:
            break
print(f"Fixed named rule (D, power, load, compute) argmin stays Pareto-non-dominated: "
      f"{TRIALS} random trials, {violations} violations found.")

# --- Contrast: the exact counterexample that breaks the plain named rule ---
A = (0.3, 0.9, 0.9)  # (compute, load, power) -- Pareto-dominates B
B = (0.5, 0.9, 0.9)  # dominated by A (strictly higher compute, everything else tied)
print("\nCounterexample setup: A =", A, " B =", B, " (compute, load, power)")
print("A dominates B?", dominates(A, B))

cands_b_first = [B, A]
winner_plain = min(cands_b_first, key=lambda c: named_tuple(*c))
winner_full = min(cands_b_first, key=lambda c: named_tuple_full(*c))
print(f"\nWith B first in iteration order:")
print(f"  Plain named rule (D, power, load) picks: {'B (DOMINATED -- bug)' if winner_plain == B else 'A'}")
print(f"  Fixed named rule (D, power, load, compute) picks: {'A (dominator, correct)' if winner_full == A else 'B -- STILL BROKEN'}")

cands_a_first = [A, B]
winner_full_2 = min(cands_a_first, key=lambda c: named_tuple_full(*c))
print(f"\nWith A first in iteration order (order should not matter for the fix):")
print(f"  Fixed named rule picks: {'A (dominator, correct)' if winner_full_2 == A else 'B -- STILL BROKEN'}")

assert winner_plain == B, "sanity check: the plain named rule should still fail this instance"
assert winner_full == A and winner_full_2 == A, "the fix must resolve the counterexample regardless of order"
print("\nAll assertions passed: fix resolves the counterexample, plain rule still fails it (as expected).")
