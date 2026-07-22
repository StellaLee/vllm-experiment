#!/usr/bin/env python3
"""Per-phase analysis for the non-stationary (concurrency-schedule) whale experiment.

Buckets pooled P99/max TBT and TTFT into concurrency phases (all cycles of a phase pooled), so we can
show: a static budget wins one phase and loses another, while hslo tracks the moving optimum. Phase
membership is reconstructed from the existing log fields (no per-token logging change).

Env:
  SCHEDULE  e.g. "8@50,40@50"   (required)
  ARMS      e.g. "16384 2048 512 hslo400ns"   (space-separated arm labels; first = mono baseline)
Reads logs/*-longp-b{arm}-t1.jsonl and, for hslo* arms, logs/*-longp-b{arm}-chunktrace.csv.
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_timing import bucket_by_phase, parse_schedule, phase_at


def pctl(x, p):
    if not x:
        return 0.0
    y = sorted(x)
    k = (len(y) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(y) - 1)
    return y[lo] + (y[hi] - y[lo]) * (k - lo)


def load(arm):
    recs = []
    for f in glob.glob(f"logs/*-longp-b{arm}-t1.jsonl"):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    return recs


def bucket_trace(rows, sched):
    """chunktrace rows (dicts with wall_s, depth, chunk) -> phase_idx -> [chunk...].
    t0 aligned to the first row with depth>0 (~ first request activity)."""
    parsed = []
    for r in rows:
        try:
            parsed.append((float(r["wall_s"]), int(float(r["depth"])), int(float(r["chunk"]))))
        except (KeyError, ValueError):
            continue
    active = [w for w, d, _ in parsed if d > 0]
    if not active:
        return {}
    t0 = min(active)
    out = {}
    for w, _, ch in parsed:
        if w < t0:
            continue
        idx, _, _ = phase_at(sched, w - t0)
        out.setdefault(idx, []).append(ch)
    return out


def main():
    SCHEDULE = os.environ["SCHEDULE"]
    ARMS = os.environ.get("ARMS", "16384 2048 512 hslo400ns").split()
    sched = parse_schedule(SCHEDULE)
    nphase = len(sched)
    print(f"schedule={SCHEDULE}  phases={[(n, s) for n, s in sched]}  arms={ARMS}\n")
    header = f"{'phase':>5} {'conc':>5} {'arm':>10} {'n_tok':>7} {'TTFT_ms':>8} {'TBTp99':>8} {'TBTmax':>8}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        recs = load(arm)
        b = bucket_by_phase(recs, sched)
        for idx in range(nphase):
            ph = b.get(idx, {"tbt": [], "ttft": [], "conc": sched[idx][0]})
            ttft_ms = 1000.0 * (sum(ph["ttft"]) / len(ph["ttft"])) if ph["ttft"] else 0.0
            print(f"{idx:>5} {ph['conc']:>5} {arm:>10} {len(ph['tbt']):>7} "
                  f"{ttft_ms:>8.0f} {pctl(ph['tbt'], 99):>8.1f} {(max(ph['tbt']) if ph['tbt'] else 0):>8.1f}")
        print()

    # hslo per-phase budget (evidence the controller tracks the phase)
    for arm in ARMS:
        if not arm.startswith("hslo"):
            continue
        for f in glob.glob(f"logs/*-longp-b{arm}-chunktrace.csv"):
            rows = list(csv.DictReader(open(f)))
            bt = bucket_trace(rows, sched)
            print(f"hslo budget by phase  [{os.path.basename(f)}]:")
            for idx in range(nphase):
                ch = bt.get(idx, [])
                if ch:
                    print(f"  phase {idx} (conc {sched[idx][0]:>3}): budget median={pctl(ch,50):.0f} "
                          f"p90={pctl(ch,90):.0f} min={min(ch)} max={max(ch)} steps={len(ch)}")


if __name__ == "__main__":
    main()
