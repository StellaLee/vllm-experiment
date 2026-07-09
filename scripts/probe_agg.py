import json, sys, glob, statistics as st

rate = sys.argv[1]  # e.g. "r4"
arms = ["base2048", "sarathi512", "slo"]

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float("nan")

def load(arm, trial):
    f = f"logs/2026-07-09-probe-{arm}_{rate}_{trial}.jsonl"
    try:
        return [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        return None

trials = [t for t in ["t1","t2","t3"] if load("base2048", t) is not None]
print(f"=== {rate}  trials={trials} ===\n")

for metric, key, scale in [("TTFT p95","ttft",1000), ("E2EL p95","latency",1), ("TPOT p95","tpot",1000)]:
    unit = "ms" if scale==1000 else "s"
    print(f"{metric} ({unit}):")
    print(f"{'arm':<12}" + "".join(f"{t:>10}" for t in trials) + f"{'mean':>10}{'vs base':>10}")
    means = {}
    for a in arms:
        vals = [pct([x[key] for x in load(a,t)], 0.95)*scale for t in trials]
        m = st.mean(vals)
        means[a] = m
        d = (m-means['base2048'])/means['base2048']*100 if a!='base2048' else 0
        cells = "".join(f"{v:>10.1f}" for v in vals)
        print(f"{a:<12}{cells}{m:>10.1f}{d:>+9.0f}%")
    print()
