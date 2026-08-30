"""Classifies time windows in a power trace as 'power_pressure' (ramp rate exceeds the
configured ceiling) or 'normal', for the spec's conditional TTFT/TBT reporting
(docs/superpowers/specs/2026-08-30-eenergy-routing-design.md S3.2): the design predicts
that DRF-3way's latency cost relative to LMETRIC-alone should be concentrated in
power_pressure windows, not spread uniformly."""
from dataclasses import dataclass


@dataclass
class Window:
    start_ts: float
    end_ts: float
    label: str  # "power_pressure" or "normal"


def classify_windows(power_samples: list, ramp_ceiling_w_per_s: float,
                      merge_gap_s: float = 1.0) -> list:
    """power_samples: list of (timestamp, power_w) from a SINGLE replica's trace, sorted by
    timestamp. Computes the ramp rate between each consecutive pair; any interval whose
    |ramp rate| exceeds ramp_ceiling_w_per_s is labeled power_pressure. Adjacent
    same-label intervals within merge_gap_s of each other are merged into one window."""
    if len(power_samples) < 2:
        return []

    raw = []
    for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        ramp = abs(p1 - p0) / dt
        label = "power_pressure" if ramp > ramp_ceiling_w_per_s else "normal"
        raw.append(Window(t0, t1, label))

    merged = []
    for w in raw:
        if merged and merged[-1].label == w.label and w.start_ts - merged[-1].end_ts <= merge_gap_s:
            merged[-1] = Window(merged[-1].start_ts, w.end_ts, w.label)
        else:
            merged.append(w)
    return merged


def label_at(windows: list, ts: float) -> str:
    """Which window's label covers timestamp ts; 'normal' if none cover it (e.g. before the
    first sample or after the last)."""
    for w in windows:
        if w.start_ts <= ts <= w.end_ts:
            return w.label
    return "normal"
