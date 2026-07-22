#!/usr/bin/env python3
"""Pure timing/geometry helpers shared by the replay driver (src/replay_sharegpt.py) and the
non-stationary analyzer (scripts/analyze_nonstationary.py). Stdlib only, no side effects, so it is
unit-testable on any machine without the box (tests/test_replay_timing.py).

Per-token emission times are RECONSTRUCTED from the existing log fields (ts, latency, ttft, tbt_ms)
so no change to the replay's logging is needed:
    start_wall   = ts - latency
    token[1]     = start_wall + ttft
    token[k>=2]  = start_wall + ttft + sum(tbt_ms[:k-1]) / 1000
"""


def parse_schedule(s):
    """'8@50,40@50' -> [(8, 50.0), (40, 50.0)]  (concurrency, phase_seconds)."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        n, sec = part.split("@")
        out.append((int(n), float(sec)))
    if not out:
        raise ValueError(f"empty schedule: {s!r}")
    return out


def phase_at(sched, t):
    """Elapsed time t (s) -> (phase_idx, concurrency, cycle). Schedule cycles forever."""
    period = sum(sec for _, sec in sched)
    if period <= 0:
        raise ValueError("schedule period must be > 0")
    cycle = int(t // period)
    off = t - cycle * period
    for idx, (n, sec) in enumerate(sched):
        if off < sec:
            return idx, n, cycle
        off -= sec
    idx = len(sched) - 1           # floating-point edge -> last phase
    return idx, sched[idx][0], cycle


def token_times(rec):
    """Absolute wall-clock emission time of each output token (len == output token count)."""
    ts, lat, ttft = rec.get("ts"), rec.get("latency"), rec.get("ttft")
    tbt = rec.get("tbt_ms") or []
    if ts is None or lat is None or ttft is None:
        return []
    start = ts - lat
    times = [start + ttft]
    acc = ttft
    for ms in tbt:
        acc += ms / 1000.0
        times.append(start + acc)
    return times


def _intervals(recs):
    iv = []
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is not None and lat is not None:
            iv.append((ts - lat, ts))
    return iv


def realized_concurrency(recs):
    """(time-weighted mean, peak) count of overlapping [start, ts] request intervals."""
    iv = _intervals(recs)
    if not iv:
        return 0.0, 0
    events = []
    for s, e in iv:
        events.append((s, +1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))   # at a tie, opens (+1) before closes (-1) => peak
    cur = mx = 0
    area = 0.0
    prev_t = events[0][0]
    for t, d in events:
        area += cur * (t - prev_t)
        prev_t = t
        cur += d
        mx = max(mx, cur)
    span = events[-1][0] - events[0][0]
    mean = area / span if span > 0 else float(mx)
    return mean, mx


def realized_throughput(recs):
    """Completions per wall-second over the run span (conv/s)."""
    iv = _intervals(recs)
    if not iv:
        return 0.0
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    return len(iv) / span if span > 0 else 0.0


def bucket_by_phase(recs, sched):
    """phase_idx -> {'tbt': [ms...], 'ttft': [s...], 'conc': int}. t0 = earliest start_wall.
    Each request's TTFT is bucketed by its start_wall; each per-token TBT by that token's emission
    time. Phases with the same index across cycles aggregate together (all low phases pooled, etc.)."""
    starts = [ts - lat for ts, lat in
              ((r.get("ts"), r.get("latency")) for r in recs) if ts is not None and lat is not None]
    if not starts:
        return {}
    t0 = min(starts)
    out = {}

    def _bucket(idx, conc):
        return out.setdefault(idx, {"tbt": [], "ttft": [], "conc": conc})

    for r in recs:
        ts, lat, ttft = r.get("ts"), r.get("latency"), r.get("ttft")
        if ts is None or lat is None:
            continue
        start = ts - lat
        if ttft is not None:
            idx, conc, _ = phase_at(sched, start - t0)
            _bucket(idx, conc)["ttft"].append(ttft)
        tt = token_times(r)                     # tt[0] is token1 (at ttft); tt[j+1] is token j+2
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            idx, conc, _ = phase_at(sched, emit - t0)
            _bucket(idx, conc)["tbt"].append(ms)
    return out


def parse_phase_schedule(s):
    """'10:0.0@60,3:0.2@45' -> [(rate, whale_frac, seconds), ...] for the open-loop phase driver."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        rf, sec = part.split("@")
        rate, frac = rf.split(":")
        out.append((float(rate), float(frac), float(sec)))
    if not out:
        raise ValueError(f"empty phase schedule: {s!r}")
    return out


def phase_type_at(sched, t):
    """Elapsed t (s) -> (phase_idx, rate, whale_frac, cycle). Schedule cycles forever."""
    period = sum(sec for _, _, sec in sched)
    if period <= 0:
        raise ValueError("phase schedule period must be > 0")
    cycle = int(t // period)
    off = t - cycle * period
    for idx, (rate, frac, sec) in enumerate(sched):
        if off < sec:
            return idx, rate, frac, cycle
        off -= sec
    idx = len(sched) - 1
    return idx, sched[idx][0], sched[idx][1], cycle


def generate_phase_arrivals(sched, duration, seed):
    """Deterministic open-loop arrivals for a (rate, whale_frac, seconds) phase schedule.
    Piecewise-homogeneous Poisson: at time t use the current phase's rate for the next
    inter-arrival; the arrival's whale flag uses the phase's frac at its own time. Returns
    a sorted list of (arrival_s, is_whale, seq). Seeded so every arm replays identically."""
    import random as _random
    rng = _random.Random(f"phasearr-{seed}")
    out = []
    t = 0.0
    seq = 0
    while True:
        rate = phase_type_at(sched, t)[1]
        if rate <= 0:
            break
        t += rng.expovariate(rate)
        if t >= duration:
            break
        frac = phase_type_at(sched, t)[2]
        is_whale = rng.random() < frac
        out.append((t, is_whale, seq))
        seq += 1
    return out
