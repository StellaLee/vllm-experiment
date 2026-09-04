#!/usr/bin/env python3
"""Verify lmetric_power's Pareto-safety violation (the second `unsafe' arm used in
paper-eenergy/paper.md's theory section, alongside drf_power_tiebreak's counterexample --
see verify_pareto_lemma.py).

lmetric_power_score(c) = Share_compute(c) * Share_load(c) * (1 + Share_power(c))
(token_budget and max_num_seqs normalized away; see scripts/eenergy/router/scoring.py's
lmetric_power_score, which uses raw new_tokens/in_flight_after -- identical up to a
per-replica constant factor when token_budget/max_num_seqs are equal across candidates,
the common case this verification targets).

Unlike drf_power_tiebreak's violation (dense in the space -- found by pure continuous
random sampling in verify_pareto_lemma.py), lmetric_power's failure mode is a
MEASURE-ZERO event in a purely continuous share space: it only triggers on an exact tie
in Share_compute (most sharply, Share_compute == 0, i.e. a full prefix-cache hit). But
cache hits are not measure-zero in real traffic -- they are a common, discrete event
(this project's own Light/Cachehit condition specifically constructs a workload
dominated by them). So this script samples with a realistic cache-hit RATE (candidates'
compute independently zeroed with probability CACHE_HIT_RATE) rather than pure
continuous sampling, which would trivially find ~0 violations and misrepresent the risk.
"""
import random

CACHE_HIT_RATE = 0.4  # order-of-magnitude match to this project's Light/Cachehit condition


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_nondominated(point, others):
    return not any(dominates(o, point) for o in others if o != point)


def lmetric_power_key(c):
    compute, load, power = c
    return compute * load * (1.0 + power)


def sample_candidate():
    compute = 0.0 if random.random() < CACHE_HIT_RATE else round(random.uniform(0.01, 1.0), 3)
    load = round(random.uniform(0.0, 1.0), 3)
    power = round(random.uniform(0.0, 1.0), 3)
    return (compute, load, power)


random.seed(0)
violations = 0
TRIALS = 200000
examples = []
for _ in range(TRIALS):
    n = random.randint(2, 5)
    cands = [sample_candidate() for _ in range(n)]
    winner = min(cands, key=lmetric_power_key)
    if not pareto_nondominated(winner, cands):
        violations += 1
        if len(examples) < 3:
            examples.append((cands, winner))

print(f"lmetric_power Pareto-safety check: {TRIALS} trials "
      f"(cache-hit rate {CACHE_HIT_RATE}), {violations} violations "
      f"({100 * violations / TRIALS:.2f}%).")
for cands, winner in examples:
    print("  e.g. candidates=", cands, " winner=", winner)

# Minimal hand-built counterexample, same style as verify_pareto_lemma.py's claim 2
A = (0.0, 0.1, 0.1)  # (compute, load, power): cache hit, light load, light power
B = (0.0, 0.9, 0.9)  # cache hit, heavy load, heavy power -- Pareto-dominated by A
print("\nMinimal counterexample: A =", A, " B =", B)
print("A dominates B?", dominates(A, B))
print("lmetric_power_score(A) =", lmetric_power_key(A))
print("lmetric_power_score(B) =", lmetric_power_key(B))
cands = [B, A]  # B first -- min() keeps the first candidate on an exact tie
winner = min(cands, key=lmetric_power_key)
print("With B first in iteration order, lmetric_power picks:", winner)
print("Is the dominated point (B) selected?", winner == B)
