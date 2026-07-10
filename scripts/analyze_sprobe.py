import json

logs = {
    "static8192 (monolithic)": "logs/2026-07-10-sprobe-static8192_tsprobe.jsonl",
    "static512  (chopped)":    "logs/2026-07-10-sprobe-static512_tsprobe.jsonl",
    "slo        (adaptive)":   "logs/2026-07-10-sprobe-slo_tsprobe.jsonl",
}

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))]

def load(path):
    tpot, ttft = [], []
    try: f = open(path)
    except FileNotFoundError: return None
    for line in f:
        r = json.loads(line)
        if r.get("tpot"): tpot.append(r["tpot"]*1000)
        if r.get("ttft"): ttft.append(r["ttft"]*1000)
    return tpot, ttft

print("{:<26}{:>7}{:>8}{:>8}{:>8}{:>8}{:>10}{:>6}".format(
    "arm","TPOTmean","p50","p95","p99","max","TTFTp95","n"))
print("-"*82)
for name, path in logs.items():
    r = load(path)
    if r is None:
        print("{:<26}{:>7}".format(name, "(pending)")); continue
    tpot, ttft = r
    print("{:<26}{:>7.1f}{:>8.1f}{:>8.1f}{:>8.1f}{:>8.1f}{:>10.1f}{:>6}".format(
        name, sum(tpot)/len(tpot), pct(tpot,.5), pct(tpot,.95),
        pct(tpot,.99), max(tpot), pct(ttft,.95), len(tpot)))
