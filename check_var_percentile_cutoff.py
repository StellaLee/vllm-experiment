import json, statistics as st

ARMS = [
    ("512",   "logs/2026-07-24-cs2wt-t512-w15-c20-t1.jsonl"),
    ("2048",  "logs/2026-07-24-cs2wt-t2048-w15-c20-t1.jsonl"),
    ("4096",  "logs/2026-07-24-cs2wt-t4096-w15-c20-t1.jsonl"),
    ("8192",  "logs/2026-07-24-cs2wt-t8192-w15-c20-t1.jsonl"),
    ("16384", "logs/2026-07-24-cs2wt-t16384-w15-c20-t1.jsonl"),
]

# percentile-based split instead of a fixed-ms cutoff: top P% of gaps, same P at every tau
for P in [10, 5, 1]:
    print(f"\n=== top {P}% of gaps (percentile cutoff, not fixed-ms) ===")
    results = []
    for tau, path in ARMS:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        all_gaps = []
        for r in recs:
            all_gaps.extend(r.get("tbt_ms") or [])
        all_gaps.sort()
        n = len(all_gaps)
        k = max(2, int(n * P / 100))
        top = all_gaps[-k:]
        mean = st.mean(top)
        var = st.variance(top)
        results.append((int(tau), len(top), mean, var))
        print(f"  tau={tau:>6}  n={len(top):>6}  mean={mean:>9.1f}ms  var={var:>14.1f}")
    base_var = results[0][3]
    print(f"  ratios vs predicted (tau/512):")
    for tau, n, mean, var in results:
        pred = tau / 512
        obs = var / base_var
        print(f"    tau={tau:>6}  predicted={pred:>6.2f}x  observed={obs:>6.2f}x")
