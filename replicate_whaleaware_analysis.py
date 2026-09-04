import json, statistics as st

def pooled(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    gaps = []
    for r in recs:
        gaps.extend(r.get("tbt_ms") or [])
    gaps.sort()
    n = len(gaps)
    def pctl(p): return gaps[min(n-1, int(p/100*n))]
    return dict(n=n, mean=sum(gaps)/n, p99=pctl(99), p99_9=pctl(99.9), max=gaps[-1])

trials = {
    "trial1": "logs/2026-08-28-lgate-bnswhaleawarev2-t1.jsonl",
    "trial2(b)": "logs/2026-08-28-lgate-bnswhaleawarev2b-t1.jsonl",
    "trial3(c)": "logs/2026-08-28-lgate-bnswhaleawarev2c-t1.jsonl",
}

results = {}
print(f"{'trial':>10} {'n':>8} {'mean':>8} {'p99':>8} {'p99.9':>9} {'max':>9}")
for label, path in trials.items():
    r = pooled(path)
    results[label] = r
    print(f"{label:>10} {r['n']:>8} {r['mean']:>8.1f} {r['p99']:>8.1f} {r['p99_9']:>9.1f} {r['max']:>9.1f}")

means = [r['mean'] for r in results.values()]
p99s = [r['p99'] for r in results.values()]
p999s = [r['p99_9'] for r in results.values()]
maxs = [r['max'] for r in results.values()]

print(f"\n{'metric':>10} {'mean':>10} {'std':>10}")
print(f"{'mean':>10} {st.mean(means):>10.2f} {st.pstdev(means):>10.3f}")
print(f"{'p99':>10} {st.mean(p99s):>10.2f} {st.pstdev(p99s):>10.3f}")
print(f"{'p99.9':>10} {st.mean(p999s):>10.2f} {st.pstdev(p999s):>10.3f}")
print(f"{'max':>10} {st.mean(maxs):>10.2f} {st.pstdev(maxs):>10.3f}")
