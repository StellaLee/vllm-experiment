#!/usr/bin/env python3
"""Per-phase-type analysis for the length-gated dynamic-chunk experiment.
Splits each arm's records into phase types S (whale_frac==0, short burst) and W (whale present),
reconstructing phase from arrival wall-time vs the schedule (no new per-record logging). Reports:
  Phase S: TTFT mean/p95 + throughput      (large chunk should win -> mono ~ lengthgate < 512)
  Phase W: P99/max TBT + request-goodput   (small chunk should win -> 512 ~ lengthgate < 2048 << mono)
Win = lengthgate ~ mono in S AND ~ 512 in W, beating static-2048 in both.

Env: SCHEDULE (required), ARMS (default "16384 512 2048 lengthgate"), SLO_TBT_MS (default 500).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_timing import parse_phase_schedule, phase_type_at, token_times


def _ptype(sched, t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"


def bucket_lengthgate(recs, sched):
    starts = [r["ts"] - r["latency"] for r in recs
              if r.get("ts") is not None and r.get("latency") is not None]
    out = {"S": {"ttft": [], "tbt": [], "n": 0, "span": 0.0},
           "W": {"ttft": [], "tbt": [], "n": 0, "span": 0.0}}
    if not starts:
        return out
    t0 = min(starts)
    for r in recs:
        ts, lat, ttft = r.get("ts"), r.get("latency"), r.get("ttft")
        if ts is None or lat is None:
            continue
        start = ts - lat
        pt = _ptype(sched, start - t0)
        out[pt]["n"] += 1
        if ttft is not None:
            out[pt]["ttft"].append(ttft)
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            out[_ptype(sched, emit - t0)]["tbt"].append(ms)
    # per-type wall span (for throughput) = arrivals' start range within that type
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None:
            continue
        start = ts - lat
        pt = _ptype(sched, start - t0)
        out[pt]["span"] = max(out[pt]["span"], start - t0)
    return out


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def main():
    import statistics as st
    SCHEDULE = os.environ["SCHEDULE"]
    ARMS = os.environ.get("ARMS", "16384 512 2048 lengthgate").split()
    SLO = float(os.environ.get("SLO_TBT_MS", "500"))
    sched = parse_phase_schedule(SCHEDULE)
    print(f"length-gate experiment  schedule={SCHEDULE}  arms={ARMS}  SLO_TBT={SLO:.0f}ms\n")

    def load(arm):
        R = []
        for f in glob.glob(f"logs/*-lgate-b{arm}-t1.jsonl"):
            R += [json.loads(l) for l in open(f) if l.strip()]
        return R

    tbl = {}
    hdr = (f"{'arm':>10} | {'S:TTFTmean':>10} {'S:TTFTp95':>10} {'S:n':>5} | "
           f"{'W:TBTp99':>9} {'W:TBTmax':>9} {'W:gpReq%':>8} {'W:tbtN':>7}")
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        recs = load(arm)
        b = bucket_lengthgate(recs, sched)
        S, W = b["S"], b["W"]
        s_ttm = 1000.0 * st.mean(S["ttft"]) if S["ttft"] else float("nan")
        s_p95 = 1000.0 * pctl(S["ttft"], 95) if S["ttft"] else float("nan")
        w_p99 = pctl(W["tbt"], 99)
        w_max = max(W["tbt"]) if W["tbt"] else float("nan")
        # request-goodput in W: fraction of W-arriving requests whose worst gap <= SLO
        w_recs = [r for r in recs if r.get("tbt_ms") and
                  _ptype(sched, (r["ts"] - r["latency"]) - min(x["ts"] - x["latency"] for x in recs
                  if x.get("ts") is not None)) == "W"]
        w_ok = sum(1 for r in w_recs if max(r["tbt_ms"]) <= SLO)
        w_gp = 100.0 * w_ok / len(w_recs) if w_recs else float("nan")
        tbl[arm] = (s_ttm, w_p99, w_gp)
        print(f"{arm:>10} | {s_ttm:>10.0f} {s_p95:>10.0f} {S['n']:>5} | "
              f"{w_p99:>9.1f} {w_max:>9.1f} {w_gp:>8.1f} {len(W['tbt']):>7}")

    print("\n=== verdict (lengthgate should ~match mono in S and ~512 in W, beating 2048 in both) ===")
    if all(a in tbl for a in ("16384", "512", "2048", "lengthgate")):
        lg = tbl["lengthgate"]
        print(f"  Phase S TTFT:  mono={tbl['16384'][0]:.0f}  2048={tbl['2048'][0]:.0f}  "
              f"512={tbl['512'][0]:.0f}  lengthgate={lg[0]:.0f}ms")
        print(f"  Phase W P99TBT: mono={tbl['16384'][1]:.0f}  2048={tbl['2048'][1]:.0f}  "
              f"512={tbl['512'][1]:.0f}  lengthgate={lg[1]:.0f}ms")
        s_win = lg[0] <= tbl["2048"][0] * 1.10
        w_win = lg[1] <= tbl["2048"][1] * 1.10
        print(f"  lengthgate beats/ties 2048 in S (TTFT): {s_win};  in W (P99 TBT): {w_win}  "
              f"-> DYNAMIC WIN: {s_win and w_win}")


if __name__ == "__main__":
    main()
