import json, sys

R = sys.argv[1] if len(sys.argv) > 1 else "r3_t1"
print(f"=== rate tag: {R} ===")
arms = {
    "base2048": f"logs/2026-07-09-probe-base2048_{R}.jsonl",
    "sarathi512": f"logs/2026-07-09-probe-sarathi512_{R}.jsonl",
    "slo": f"logs/2026-07-09-probe-slo_{R}.jsonl",
}

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q * len(xs)))]

data = {a: [json.loads(l) for l in open(f)] for a, f in arms.items()}
order = list(arms)

def hdr():
    return f"{'':<14}" + "".join(f"{a:>16}" for a in order)

def row(name, fn, ms=True):
    vals = {a: fn(data[a]) for a in order}
    b = vals["base2048"]
    cells = []
    for a in order:
        v = vals[a]
        d = (v - b) / b * 100 if b else 0
        s = f"{v*1000:.1f}ms" if ms else f"{v:.3f}s"
        cells.append(f"{s}({d:+.0f}%)")
    print(f"{name:<14}" + "".join(f"{c:>16}" for c in cells))

print(hdr())
row("TTFT p95", lambda r: pct([x["ttft"] for x in r], 0.95))
row("TTFT p99", lambda r: pct([x["ttft"] for x in r], 0.99))
row("TPOT p95", lambda r: pct([x["tpot"] for x in r], 0.95))
row("TPOT p50", lambda r: pct([x["tpot"] for x in r], 0.50))
row("E2EL p95", lambda r: pct([x["latency"] for x in r], 0.95), ms=False)
row("E2EL p99", lambda r: pct([x["latency"] for x in r], 0.99), ms=False)
row("E2EL p50", lambda r: pct([x["latency"] for x in r], 0.50), ms=False)

print("\nTPOT p95 by turn (ms):")
print(f"{'turn':<6}" + "".join(f"{a:>14}" for a in order))
for t in (1, 2, 3, 4):
    cells = [f"{pct([x['tpot'] for x in data[a] if x['turn']==t],0.95)*1000:.1f}" for a in order]
    print(f"T{t:<5}" + "".join(f"{c:>14}" for c in cells))

print("\nE2EL p95 by turn (s):")
print(f"{'turn':<6}" + "".join(f"{a:>14}" for a in order))
for t in (1, 2, 3, 4):
    cells = [f"{pct([x['latency'] for x in data[a] if x['turn']==t],0.95):.3f}" for a in order]
    print(f"T{t:<5}" + "".join(f"{c:>14}" for c in cells))
