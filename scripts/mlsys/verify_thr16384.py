import json, statistics as st

def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]

def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")

def stats(rows, key):
    v = [r[key] * 1000 for r in rows if r.get(key) is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    return (st.mean(v), pctl(v, 95), pctl(v, 99), len(v))

SOURCES = {
    "0 (mono/off)": "logs/2026-07-24-cs2wt-t0-w15-c20-t1.jsonl",
    "16384": "logs/2026-07-24-cs2wt-t16384-w15-c20-t1.jsonl",
}

print(f"{'thr':>14} {'pop':>3} {'n':>4} | {'TTFTmean':>9} {'TTFTp95':>9} {'TTFTp99':>9} | "
      f"{'TPOTmean':>9} {'TPOTp95':>9} {'TPOTp99':>9}")

results = {}
for label, f in SOURCES.items():
    rows = load(f)
    whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
    short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
    for pl, pop in [("W", whale), ("S", short)]:
        tm, t95, t99, n = stats(pop, "ttft")
        pm, p95, p99, _ = stats(pop, "tpot")
        results[(label, pl)] = (tm, t95, t99, pm, p95, p99, n)
        print(f"{label:>14} {pl:>3} {n:>4} | {tm:9.0f} {t95:9.0f} {t99:9.0f} | {pm:9.1f} {p95:9.1f} {p99:9.1f}")

print("\n=== Delta: threshold=16384 vs threshold=0/off, S population ===\n")
m = results[("0 (mono/off)", "S")]
c = results[("16384", "S")]
mtm, mt95, mt99, mpm, mp95, mp99, _ = m
ctm, ct95, ct99, cpm, cp95, cp99, _ = c
d = lambda a, b: (b - a) / a * 100 if a else float("nan")
print(f"S:TTFT  mean={d(mtm,ctm):+.1f}%  p95={d(mt95,ct95):+.1f}%  p99={d(mt99,ct99):+.1f}%")
print(f"S:TPOT  mean={d(mpm,cpm):+.1f}%  p95={d(mp95,cp95):+.1f}%  p99={d(mp99,cp99):+.1f}%")

print("\n=== Same for whale population ===\n")
m = results[("0 (mono/off)", "W")]
c = results[("16384", "W")]
mtm, mt95, mt99, mpm, mp95, mp99, _ = m
ctm, ct95, ct99, cpm, cp95, cp99, _ = c
print(f"W:TTFT  mean={d(mtm,ctm):+.1f}%  p95={d(mt95,ct95):+.1f}%  p99={d(mt99,ct99):+.1f}%")
print(f"W:TPOT  mean={d(mpm,cpm):+.1f}%  p95={d(mp95,cp95):+.1f}%  p99={d(mp99,cp99):+.1f}%")
