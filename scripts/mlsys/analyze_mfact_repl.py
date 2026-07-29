#!/usr/bin/env python3
"""Aggregate replicated mechanism-factorial HERD arms: per-turn TTFT p95, % vs baseline
(paired within each trial), reported mean +- std across trials. Tests whether the headline
attribution (chunk drives the herd win, reorder does not) is robust to reseeding.
"""
import glob, json, os, re, statistics as st, sys

def load(fp): return [json.loads(l) for l in open(fp) if l.strip()]
def p95(xs):
    xs = sorted(xs); return xs[min(len(xs)-1, int(0.95*len(xs)))] if xs else None

def turn_p95(fp):
    by = {}
    for r in load(fp):
        if r.get("ttft") is not None:
            by.setdefault(r["turn"], []).append(r["ttft"]*1000)
    return {t: p95(v) for t, v in by.items()}

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "logs/*mfact-*_herd_t*.jsonl"
    rx = re.compile(r"mfact-(baseline|reorder|chunk|combined)_herd_t([A-Za-z0-9]+)")
    # (arm,trial) -> {turn: p95}
    data = {}
    for fp in sorted(glob.glob(pat)):
        m = rx.search(os.path.basename(fp))
        if not m: continue
        data[(m.group(1), m.group(2))] = turn_p95(fp)
    trials = sorted({t for (_, t) in data})
    turns = sorted({tn for d in data.values() for tn in d})
    print(f"Trials: {trials}   (per-turn TTFT p95, % vs baseline paired within trial)\n")
    for arm in ["reorder", "chunk", "combined"]:
        print(f"== {arm} vs baseline ==")
        for tn in turns:
            deltas = []
            for tr in trials:
                b = data.get(("baseline", tr), {}).get(tn)
                a = data.get((arm, tr), {}).get(tn)
                if b and a: deltas.append((a-b)/b*100)
            if not deltas: continue
            m = st.mean(deltas); s = st.stdev(deltas) if len(deltas) > 1 else 0.0
            wins = sum(1 for d in deltas if d < 0)
            perrun = "/".join(f"{d:+.0f}" for d in deltas)
            flag = "  <== WIN" if m < -10 else ("" if m < 10 else "  (worse)")
            print(f"  T{tn}: {m:+6.1f}% +- {s:4.1f}   [{perrun}]  win {wins}/{len(deltas)}{flag}")
        print()

if __name__ == "__main__":
    main()
