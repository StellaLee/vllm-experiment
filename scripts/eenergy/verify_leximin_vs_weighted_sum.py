#!/usr/bin/env python3
"""Verify the new §4.5 candidate claims (leximin vs. weighted_sum on the Pareto frontier)
before they go in the paper. Same style/thresholds as verify_pareto_lemma.py,
verify_weighted_sum_pareto.py, verify_lmetric_power_pareto.py: brute-force falsification
attempts over random instances, plus the hand-built minimal counterexamples, plus explicit
assertions so a script failure is loud rather than silent.

Four independent claims:

A. DIVERGENCE THEOREM. If D(X) < D(Y) and Sum(X) > Sum(Y), then X and Y are Pareto-
   incomparable, the sorted (leximin) rule picks X, and weighted_sum picks Y. Conversely,
   whenever one candidate actually dominates the other, the two rules always agree.

B. UNIVERSAL THRESHOLD-SAFETY. Whenever some candidate in the set has D(c) <= tau (for
   ANY tau > 0), the sorted rule's pick also has D <= tau -- i.e. it never causes an
   avoidable threshold violation, for every tau simultaneously.

C. IMPOSSIBILITY FOR ANY FIXED-WEIGHT RULE. For any positive weight vector w and any
   tau > 0, there exists a safe/unsafe candidate pair (S, U) where the w-weighted-sum rule
   picks the unsafe U over the safe S. Also: an empirical rate estimate (in the style of
   lmetric_power's 12.8%) of how often this happens on GENERIC random instances, not just
   the crafted pair, for the paper's actual weights (0.33, 0.33, 0.33).

D. PIGOU-DALTON TRANSFER. An equalizing transfer (moving mass from the max coordinate to a
   smaller one, without crossing) strictly improves the leximin ranking but leaves
   weighted_sum exactly unchanged.

E. PRICE OF EGALITARIANISM (tight bound). If D(X) < D(Y), the sorted rule's excess total
   burden over weighted_sum's pick, Sum(X) - Sum(Y), is strictly less than 2*D(X), and this
   bound is approached arbitrarily closely (X=(D0,D0,D0), Y=(D0+delta,0,0), delta -> 0+).
"""
import random

TRIALS = 200000
TAU = 1.0  # "at capacity" -- matches the paper's own share normalization convention


def dominates(a, b):
    """a dominates b: a[i] <= b[i] for all i, strict for at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def incomparable(a, b):
    return not dominates(a, b) and not dominates(b, a)


def D(c):
    return max(c)


def sorted_desc(c):
    return tuple(sorted(c, reverse=True))


def weighted_sum(c, w=(0.33, 0.33, 0.33)):
    return sum(wi * ci for wi, ci in zip(w, c))


def leximin_pick(cands):
    return min(cands, key=sorted_desc)


def weighted_pick(cands, w=(0.33, 0.33, 0.33)):
    return min(cands, key=lambda c: weighted_sum(c, w))


# ---------------------------------------------------------------------------
print("=" * 70)
print("A. DIVERGENCE THEOREM: D(X)<D(Y) and Sum(X)>Sum(Y) => incomparable,")
print("   leximin picks X, weighted_sum picks Y. Domination => rules agree.")
print("=" * 70)

random.seed(0)
divergence_eligible = 0
divergence_confirmed = 0
domination_disagreements = 0  # should stay exactly 0
n_pairs = 0
EPS = 1e-9  # margin so near-ties (measure-zero under full-precision floats, but easy to
            # hit by accident with coarse rounding) are never miscounted as "strict"
for _ in range(TRIALS):
    # NOTE: deliberately NOT rounded to a coarse grid (e.g. 3 decimals) -- rounding
    # created spurious exact sum-ties (e.g. 0.371+0.465+0.275 vs 0.756+0.168+0.187,
    # both "1.111" at 3dp but float-unequal by ~1e-16), which showed up as false
    # "violations" below. Full-precision continuous sampling makes true ties
    # measure-zero, matching the theorem's real-number premise.
    X = tuple(random.uniform(0, 1.5) for _ in range(3))
    Y = tuple(random.uniform(0, 1.5) for _ in range(3))
    n_pairs += 1

    # Invariant check: whenever one dominates the other, both rules must agree
    # (Geoffrion 1968 for weighted_sum; D-monotonicity for leximin -- already
    # used in the paper's own Corollary 1 proof).
    if dominates(X, Y) or dominates(Y, X):
        dominator = X if dominates(X, Y) else Y
        lp = leximin_pick([X, Y])
        wp = weighted_pick([X, Y])
        if lp != dominator or wp != dominator:
            domination_disagreements += 1
            print("  UNEXPECTED: domination case where a rule disagrees:", X, Y)

    # Divergence-eligible instances (try both orderings of the pair), require a
    # real margin on both the D-gap and the Sum-gap so this only fires on genuine
    # strict instances, not float dust.
    for A, B in ((X, Y), (Y, X)):
        if D(B) - D(A) > EPS and sum(A) - sum(B) > EPS:  # D(A) < D(B), Sum(A) > Sum(B)
            divergence_eligible += 1
            ok_incomparable = incomparable(A, B)
            ok_leximin = leximin_pick([A, B]) == A
            ok_weighted = weighted_pick([A, B]) == B
            if ok_incomparable and ok_leximin and ok_weighted:
                divergence_confirmed += 1
            else:
                print("  DIVERGENCE CLAIM VIOLATED:", A, B,
                      "incomparable=", ok_incomparable,
                      "leximin_picks_A=", ok_leximin,
                      "weighted_picks_B=", ok_weighted)

print(f"{n_pairs} random pairs checked.")
print(f"Domination cases where the two rules disagreed: {domination_disagreements} "
      f"(must be 0 -- Geoffrion 1968 + D-monotonicity guarantee agreement).")
print(f"Divergence-eligible orderings found: {divergence_eligible} "
      f"({100 * divergence_eligible / (2 * n_pairs):.2f}% of ordered pairs); "
      f"all confirmed as predicted: {divergence_confirmed}/{divergence_eligible}.")
assert domination_disagreements == 0
assert divergence_confirmed == divergence_eligible

# Minimal hand-built example for the paper text
X = (0.4, 0.4, 0.4)
Y = (0.9, 0.1, 0.1)
print(f"\nMinimal example: X={X} (D={D(X)}, Sum={sum(X):.2f}), "
      f"Y={Y} (D={D(Y)}, Sum={sum(Y):.2f})")
print("X dominates Y?", dominates(X, Y), " Y dominates X?", dominates(Y, X))
print("leximin picks:", leximin_pick([X, Y]), " weighted_sum picks:", weighted_pick([X, Y]))
assert leximin_pick([X, Y]) == X
assert weighted_pick([X, Y]) == Y

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("B. UNIVERSAL THRESHOLD-SAFETY: leximin never causes an avoidable")
print("   tau-violation, for random tau, whenever a safe candidate exists.")
print("=" * 70)

random.seed(1)
safety_violations = 0
safe_set_trials = 0
for _ in range(TRIALS):
    n = random.randint(2, 6)
    cands = [tuple(round(random.uniform(0, 2.0), 3) for _ in range(3)) for _ in range(n)]
    tau = round(random.uniform(0.05, 2.0), 3)
    if min(D(c) for c in cands) > tau:
        continue  # no safe candidate exists in this set -- not a testable instance
    safe_set_trials += 1
    pick = leximin_pick(cands)
    if D(pick) > tau:
        safety_violations += 1
        print("  SAFETY VIOLATED:", cands, "tau=", tau, "pick=", pick)

print(f"{safe_set_trials} trials had >=1 safe candidate for their random tau; "
      f"leximin caused an avoidable violation in {safety_violations} of them.")
assert safety_violations == 0

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("C. IMPOSSIBILITY FOR ANY FIXED-WEIGHT RULE (weighted_sum family):")
print("   S=(tau,tau,tau) vs U=(0,0,tau+M) construction, for random weights.")
print("=" * 70)

random.seed(2)
construction_failures = 0
CONSTRUCTION_TRIALS = 50000
for _ in range(CONSTRUCTION_TRIALS):
    w = tuple(random.uniform(0.01, 5.0) for _ in range(3))
    tau = random.uniform(0.05, 5.0)
    m_bound = tau * (w[0] + w[1]) / w[2]
    M = random.uniform(0.001 * m_bound, 0.999 * m_bound)  # inside the open interval (0, m_bound)
    S = (tau, tau, tau)
    U = (0.0, 0.0, tau + M)
    safe_ok = D(S) <= tau < D(U)
    picks_unsafe = weighted_sum(U, w) < weighted_sum(S, w)
    if not (safe_ok and picks_unsafe):
        construction_failures += 1
        print("  CONSTRUCTION FAILED:", "w=", w, "tau=", tau, "M=", M)

print(f"{CONSTRUCTION_TRIALS} random (weight, tau, M-in-range) triples: "
      f"{construction_failures} failures (must be 0 -- construction is a closed-form "
      f"identity, this just checks the algebra/implementation match).")
assert construction_failures == 0

print("\nConcrete instance with the paper's own weights (0.33, 0.33, 0.33), tau=1.0:")
tau = 1.0
w = (0.33, 0.33, 0.33)
M = 1.0  # inside (0, tau*(w0+w1)/w2) = (0, 2.0) for these weights
S = (tau, tau, tau)
U = (0.0, 0.0, tau + M)
print(f"  S={S}  D(S)={D(S)}  weighted_sum(S)={weighted_sum(S, w):.3f}")
print(f"  U={U}  D(U)={D(U)}  weighted_sum(U)={weighted_sum(U, w):.3f}")
print(f"  S is safe (D<=tau): {D(S) <= tau}   U is UNSAFE (D>tau): {D(U) > tau}")
print(f"  weighted_sum picks U over S: {weighted_pick([S, U], w) == U}")
print(f"  leximin picks S (correctly, the safe one): {leximin_pick([S, U]) == S}")
assert weighted_pick([S, U], w) == U
assert leximin_pick([S, U]) == S

print("\nEmpirical rate on GENERIC random instances (not the crafted pair), "
      "paper's weights, tau=1.0:")
random.seed(3)
generic_trials = 0
generic_avoidable_violations = 0
for _ in range(TRIALS):
    n = random.randint(2, 6)
    cands = [tuple(round(random.uniform(0, 1.5), 3) for _ in range(3)) for _ in range(n)]
    if min(D(c) for c in cands) > TAU:
        continue  # no safe candidate available -- not a fair test of "avoidable"
    generic_trials += 1
    pick = weighted_pick(cands)
    if D(pick) > TAU:
        generic_avoidable_violations += 1

rate = 100 * generic_avoidable_violations / generic_trials
print(f"  {generic_trials} instances had a safe candidate available; weighted_sum "
      f"(0.33/0.33/0.33) still picked an unsafe one in {generic_avoidable_violations} "
      f"of them ({rate:.2f}%).")

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("D. PIGOU-DALTON TRANSFER: leximin strictly rewards equalization,")
print("   weighted_sum is exactly indifferent to it.")
print("=" * 70)

random.seed(4)
pd_checked = 0
pd_failures = 0
for _ in range(20000):
    d = round(random.uniform(0, 1.0), 3)
    b = round(random.uniform(0, 1.0), 3)
    a = round(b + random.uniform(0.01, 1.0), 3)  # ensure a is the strict max, a > b
    if a <= max(b, d):
        continue
    eps = random.uniform(0.001, (a - b) / 2)
    s = (a, b, d)
    s_prime = (round(a - eps, 6), round(b + eps, 6), d)
    pd_checked += 1
    leximin_improves = sorted_desc(s_prime) < sorted_desc(s)  # strictly better (lower) sorted vector
    ws_unchanged = abs(weighted_sum(s_prime) - weighted_sum(s)) < 1e-9
    if not (leximin_improves and ws_unchanged):
        pd_failures += 1
        print("  PIGOU-DALTON CHECK FAILED:", s, "->", s_prime)

print(f"{pd_checked} random equalizing transfers checked: {pd_failures} failures.")
assert pd_failures == 0

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("E. PRICE OF EGALITARIANISM (tight bound): Sum(X)-Sum(Y) < 2*D(X),")
print("   approached arbitrarily closely as the D-gap shrinks to 0.")
print("=" * 70)

print(f"{'delta':>10} {'D(X)':>8} {'D(Y)':>8} {'Sum(X)':>8} {'Sum(Y)':>10} "
      f"{'waste':>10} {'waste/(2*D(X))':>16}")
D0 = 0.5
for delta in (0.1, 0.01, 0.001, 0.0001, 1e-6):
    X = (D0, D0, D0)             # Sum(X) maximal given D(X)=D0
    Y = (D0 + delta, 0.0, 0.0)   # Sum(Y) minimal given D(Y)=D0+delta
    waste = sum(X) - sum(Y)
    ratio = waste / (2 * D0)
    print(f"{delta:>10} {D(X):>8.5f} {D(Y):>8.5f} {sum(X):>8.5f} {sum(Y):>10.6f} "
          f"{waste:>10.6f} {ratio:>16.6f}")
    assert D(X) < D(Y) and sum(X) > sum(Y)
    assert waste < 2 * D0

random.seed(7)
PRICE_TRIALS = 300000
max_ratio = 0.0
n_eligible = 0
bound_violations = 0
for _ in range(PRICE_TRIALS):
    X = tuple(random.uniform(0, 1.5) for _ in range(3))
    Y = tuple(random.uniform(0, 1.5) for _ in range(3))
    for A, B in ((X, Y), (Y, X)):
        if D(B) - D(A) > EPS and sum(A) - sum(B) > EPS:  # D(A)<D(B), Sum(A)>Sum(B)
            n_eligible += 1
            waste = sum(A) - sum(B)
            bound = 2 * D(A)
            ratio = waste / bound
            if ratio > max_ratio:
                max_ratio = ratio
            if waste >= bound + EPS:
                bound_violations += 1
                print("  BOUND VIOLATED:", A, B, waste, bound)

print(f"\n{n_eligible} divergence-eligible pairs (of {2 * PRICE_TRIALS} ordered pairs): "
      f"{bound_violations} bound violations (must be 0); "
      f"max observed waste/(2*D(A)) = {max_ratio:.4f} (theoretical supremum is 1, "
      f"never attained).")
assert bound_violations == 0

print("\nAll claims verified.")
