import json, glob, statistics as st

THRS = ["0", "512", "2048", "4096"]


def load(thr):
    files = glob.glob(f"logs/2026-07-24-cs2wt-t{thr}-w15-c20-t1.jsonl")
    R = []
    for f in files:
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def stats(rows, key):
    v = [r[key] * 1000 for r in rows if r.get(key) is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    return (st.mean(v), pctl(v, 95), pctl(v, 99), len(v))


results = {}
print(f"{'thr':>6} {'pop':>3} {'n':>4} | {'TTFTmean':>9} {'TTFTp95':>9} {'TTFTp99':>9} | "
      f"{'TPOTmean':>9} {'TPOTp95':>9} {'TPOTp99':>9}")
for thr in THRS:
    rows = load(thr)
    whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
    short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
    for label, pop in [("W", whale), ("S", short)]:
        tm, t95, t99, n = stats(pop, "ttft")
        pm, p95, p99, _ = stats(pop, "tpot")
        results[(thr, label)] = (tm, t95, t99, pm, p95, p99, n)
        print(f"{thr:>6} {label:>3} {n:>4} | {tm:9.0f} {t95:9.0f} {t99:9.0f} | "
              f"{pm:9.1f} {p95:9.1f} {p99:9.1f}")

print("\n=== Delta vs mono (thr=0), S population ===\n")
mono = results[("0", "S")]
mtm, mt95, mt99, mpm, mp95, mp99, _ = mono
for thr in ("512", "2048", "4096"):
    ctm, ct95, ct99, cpm, cp95, cp99, _ = results[(thr, "S")]
    d = lambda a, b: (b - a) / a * 100 if a else float("nan")
    print(f"thr={thr:>5}: dTTFT mean={d(mtm,ctm):+.1f}% p95={d(mt95,ct95):+.1f}% p99={d(mt99,ct99):+.1f}% | "
          f"dTPOT mean={d(mpm,cpm):+.1f}% p95={d(mp95,cp95):+.1f}% p99={d(mp99,cp99):+.1f}%")
