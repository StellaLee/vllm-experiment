#!/usr/bin/env python3
"""Stationary concurrency-sweep analysis for the whale workload.
Reads logs/*-csweep-b{arm}-c{conc}-t1.jsonl for each (arm, conc) and reports outcome metrics:
  TTFTmean, TBT p50/p99/max (pooled per-token ITL, the sharp tail), and GOODPUT
  (fraction of inter-token gaps <= SLO threshold, token-level and request-level).
Then, per concurrency: the BEST STATIC arm on each axis and how hslo compares -- the evidence
needed to claim "a static budget is/ isn't robust across load" and "hslo does/doesn't beat it".

Env:
  ARMS   e.g. "16384 2048 512 hslo400"   (numeric = static budget; hslo* = controller)
  CONCS  e.g. "8 24 48"
  SLO_TBT_MS  goodput threshold (default 500); SLO_TBT_MS2 second threshold (default 1000)
"""
import json, glob, os, statistics as st

ARMS = os.environ.get("ARMS", "16384 2048 512 hslo400").split()
CONCS = [int(c) for c in os.environ.get("CONCS", "8 24 48").split()]
SLO1 = float(os.environ.get("SLO_TBT_MS", "500"))
SLO2 = float(os.environ.get("SLO_TBT_MS2", "1000"))
STATIC = [a for a in ARMS if a.isdigit() and a != "16384"]  # candidate static budgets (exclude mono)


def load(arm, conc):
    R = []
    for f in glob.glob(f"logs/*-csweep-b{arm}-c{conc}-t1.jsonl"):
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def pooled_tbt(R):
    t = []
    for r in R:
        t.extend(r.get("tbt_ms") or [])
    return t


def metrics(R):
    tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
    tbt = pooled_tbt(R)
    # request-level: fraction of requests whose worst inter-token gap <= threshold
    req_ok1 = sum(1 for r in R if (r.get("tbt_ms") and max(r["tbt_ms"]) <= SLO1)) if R else 0
    ncomp = sum(1 for r in R if r.get("tbt_ms"))
    return {
        "n": len(R), "ncomp": ncomp,
        "ttft": st.mean(tt) if tt else float("nan"),
        "p50": pctl(tbt, 50), "p99": pctl(tbt, 99), "max": max(tbt) if tbt else float("nan"),
        "gp_tok1": 100.0 * sum(1 for v in tbt if v <= SLO1) / len(tbt) if tbt else float("nan"),
        "gp_tok2": 100.0 * sum(1 for v in tbt if v <= SLO2) / len(tbt) if tbt else float("nan"),
        "gp_req1": 100.0 * req_ok1 / ncomp if ncomp else float("nan"),
    }


print(f"Concurrency sweep — whale workload, arms={ARMS}, concs={CONCS}")
print(f"goodput SLO thresholds: token/req <= {SLO1:.0f}ms  and token <= {SLO2:.0f}ms\n")
hdr = (f"{'conc':>5} {'arm':>8} {'n':>4} {'TTFTms':>7} {'TBTp50':>7} {'TBTp99':>8} {'TBTmax':>8} "
       f"{'gpTok%'+str(int(SLO1)):>8} {'gpReq%'+str(int(SLO1)):>8} {'gpTok%'+str(int(SLO2)):>9}")
print(hdr); print("-" * len(hdr))

table = {}  # (conc, arm) -> metrics
for conc in CONCS:
    for arm in ARMS:
        m = metrics(load(arm, conc))
        table[(conc, arm)] = m
        if not m["n"]:
            print(f"{conc:>5} {arm:>8} {'(no data)':>4}"); continue
        print(f"{conc:>5} {arm:>8} {m['n']:>4} {m['ttft']:>7.0f} {m['p50']:>7.1f} {m['p99']:>8.1f} "
              f"{m['max']:>8.1f} {m['gp_tok1']:>8.1f} {m['gp_req1']:>8.1f} {m['gp_tok2']:>9.1f}")
    print()

# Per-concurrency verdict: best static (on p99 TBT, and on TTFT), vs hslo.
hslo = next((a for a in ARMS if a.startswith("hslo")), None)
print("=== per-concurrency verdict: best static vs hslo ===")
for conc in CONCS:
    ok = [a for a in STATIC if table[(conc, a)]["n"]]
    if not ok:
        continue
    best_tail = min(ok, key=lambda a: table[(conc, a)]["p99"])
    best_ttft = min(ok, key=lambda a: table[(conc, a)]["ttft"])
    line = (f"conc {conc:>3}: best-tail static = {best_tail} (p99 {table[(conc,best_tail)]['p99']:.0f}ms), "
            f"best-TTFT static = {best_ttft} (TTFT {table[(conc,best_ttft)]['ttft']:.0f}ms)")
    if hslo and table[(conc, hslo)]["n"]:
        h = table[(conc, hslo)]; bt = table[(conc, best_tail)]
        dp99 = (h["p99"] - bt["p99"]) / bt["p99"] * 100
        dgp = h["gp_tok1"] - bt["gp_tok1"]
        line += (f"\n          hslo: p99 {h['p99']:.0f}ms ({dp99:+.0f}% vs best-tail static), "
                 f"goodput@{int(SLO1)} {h['gp_tok1']:.1f}% ({dgp:+.1f}pt)")
    print(line)

# Does the best static change with concurrency? (the crux of "no single static is right")
print("\n=== does the SLO-best static budget move with load? ===")
for axis, key, lo in (("tail (min p99)", "p99", True), ("TTFT (min)", "ttft", True),
                      (f"goodput@{int(SLO1)} (max)", "gp_tok1", False)):
    winners = []
    for conc in CONCS:
        ok = [a for a in STATIC if table[(conc, a)]["n"]]
        if not ok:
            winners.append("?"); continue
        w = (min if lo else max)(ok, key=lambda a: table[(conc, a)][key])
        winners.append(f"{conc}:{w}")
    moved = len(set(w.split(":")[1] for w in winners if ":" in w)) > 1
    print(f"  {axis:>22}: {'  '.join(winners)}   -> {'MOVES' if moved else 'stable'}")
