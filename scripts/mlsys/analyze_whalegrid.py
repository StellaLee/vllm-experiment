import json, glob, statistics as st

BUDGETS = ["16384", "2048"]
SIZES = ["8k", "14k"]
FRACS = ["05", "15", "30"]


def load(budget, size, frac):
    files = glob.glob(f"logs/2026-07-24-whalegrid-b{budget}-s{size}-w{frac}-t1.jsonl")
    R = []
    for f in files:
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def pooled_tbt(rows):
    t = []
    for r in rows:
        t.extend(r.get("tbt_ms") or [])
    return t


print(f"{'size':>4} {'frac':>4} {'budget':>7} | {'n':>4} {'nwhale':>6} | "
      f"{'TTFTmean':>9} | {'TBTp99':>8} {'TBTmax':>8}")

results = {}
for size in SIZES:
    for frac in FRACS:
        for budget in BUDGETS:
            rows = load(budget, size, frac)
            if not rows:
                print(f"{size:>4} {frac:>4} {budget:>7} | NO DATA")
                continue
            nwhale = sum(1 for r in rows if (r.get("pad_chars") or 0) >= 20000)
            tt = [r["ttft"] * 1000 for r in rows if r.get("ttft") is not None]
            tbt = pooled_tbt(rows)
            ttm = st.mean(tt) if tt else float("nan")
            p99 = pctl(tbt, 99)
            mx = max(tbt) if tbt else float("nan")
            results[(size, frac, budget)] = (ttm, p99, mx, len(rows), nwhale)
            print(f"{size:>4} {frac:>4} {budget:>7} | {len(rows):>4} {nwhale:>6} | "
                  f"{ttm:9.0f} | {p99:8.1f} {mx:8.1f}")
    print()

print("\n=== Delta: chunk=2048 vs mono=16384, per (size,frac) ===\n")
print(f"{'size':>4} {'frac':>4} | {'dTTFT%':>8} | {'dTBTp99%':>9} {'dTBTmax%':>9}")
for size in SIZES:
    for frac in FRACS:
        mono = results.get((size, frac, "16384"))
        chunk = results.get((size, frac, "2048"))
        if not mono or not chunk:
            continue
        mttm, mp99, mmax, _, _ = mono
        cttm, cp99, cmax, _, _ = chunk
        d = lambda a, b: (b - a) / a * 100 if a else float("nan")
        print(f"{size:>4} {frac:>4} | {d(mttm,cttm):+8.1f} | {d(mp99,cp99):+9.1f} {d(mmax,cmax):+9.1f}")
