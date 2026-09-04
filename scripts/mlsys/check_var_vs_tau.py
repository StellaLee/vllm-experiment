import json, statistics as st

ARMS = [
    ("512",   "logs/2026-07-24-cs2wt-t512-w15-c20-t1.jsonl"),
    ("2048",  "logs/2026-07-24-cs2wt-t2048-w15-c20-t1.jsonl"),
    ("4096",  "logs/2026-07-24-cs2wt-t4096-w15-c20-t1.jsonl"),
    ("8192",  "logs/2026-07-24-cs2wt-t8192-w15-c20-t1.jsonl"),
    ("16384", "logs/2026-07-24-cs2wt-t16384-w15-c20-t1.jsonl"),
]

print(f"{'tau':>8} {'n_gaps':>8} {'mean_ms':>10} {'var_ms2':>14} {'std_ms':>10} {'p99_ms':>10} {'max_ms':>10}")
results = []
for tau, path in ARMS:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    all_gaps = []
    for r in recs:
        all_gaps.extend(r.get("tbt_ms") or [])
    if not all_gaps:
        print(f"{tau:>8}: no gaps"); continue
    mean = st.mean(all_gaps)
    var = st.variance(all_gaps) if len(all_gaps) > 1 else float('nan')
    std = var ** 0.5
    sg = sorted(all_gaps)
    p99 = sg[int(0.99 * len(sg))]
    mx = sg[-1]
    results.append((int(tau), mean, var, std, p99, mx))
    print(f"{tau:>8} {len(all_gaps):>8} {mean:>10.2f} {var:>14.1f} {std:>10.2f} {p99:>10.1f} {mx:>10.1f}")

print("\n=== variance ratio checks (predicted: Var scales ~linearly with tau) ===")
base_tau, base_var = results[0][0], results[0][2]
for tau, mean, var, std, p99, mx in results:
    predicted_ratio = tau / base_tau
    observed_ratio = var / base_var
    print(f"  tau={tau:>6}  predicted_var_ratio(tau/{base_tau})={predicted_ratio:>7.2f}  observed_var_ratio={observed_ratio:>7.2f}")

# also: variance restricted to gaps above a threshold (isolate "interference events" from
# baseline decode noise, which shouldn't scale with tau at all)
print("\n=== restricted to gaps > 100ms (interference-only, excludes baseline decode noise) ===")
for tau, path in ARMS:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    all_gaps = []
    for r in recs:
        all_gaps.extend(r.get("tbt_ms") or [])
    big = [g for g in all_gaps if g > 100]
    if len(big) > 1:
        print(f"  tau={tau:>6}  n_big={len(big):>6} ({100*len(big)/len(all_gaps):.2f}%)  mean={st.mean(big):>8.1f}  var={st.variance(big):>12.1f}")
    else:
        print(f"  tau={tau:>6}  n_big={len(big)}")
