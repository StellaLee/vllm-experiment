#!/usr/bin/env python3
"""Regenerate all data for fig_overview_schematic.png's three panels, self-contained (no
inheritance from older v1/v2 .npz files, which had drifted onto stale/mismatched datasets):

Panel (a): a real single-GPU power trace snippet around one whale request, mono vs.\ chunk=512,
    picked from the same widened-distribution, batch-interleaved CONC=20 library backing
    Table IV (the reserve-procurement headline's operating point).
Panel (b): a data-center-scale aggregate power profile (N=100 shown) built by the model-free
    bootstrap (real-trace resampling), same library as panel (a)/(c).
Panel (c): the N=10,000 reserve-procurement peak-ramp distributions (model-free bootstrap),
    same 100-trial library and same window convention as the real Table IV computation.

Usage:
    python scripts/pesim/gen_overview_v3_data.py   # writes overview_real_data_v3.npz
    python scripts/pesim/plot_paper_overview_schematic_v3.py
"""
import csv
import json
import sys

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
import numpy as np
from cf_tracebootstrap_model import build_library, sample_fleet

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
OUT_NPZ = "/Users/li/Documents/vllm-experiment/scripts/pesim/overview_real_data_v3.npz"
TAG = "pesim_gate_convdiverse_conc20_widened_interleaved_gpu0"

dt = 0.1
N_SHOWN = 100
N_RESERVE = 10000
SEED = 11


def find_path(budget, trial):
    import glob
    m = glob.glob(f"{RAW}/*-{TAG}-{budget}-t{trial}-power.csv")
    assert len(m) == 1, m
    return m[0]


def load_power(path):
    t, p = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["wall_time"]))
            p.append(float(row["power_w"]))
    return np.array(t), np.array(p)


def whale_snippet(budget, trial, pad_chars_target, pre=8.0, post=20.0):
    """Extract a power-trace window around the whale request with the given pad_chars,
    from `pre` seconds before its prefill starts to `post` seconds after."""
    jsonl_path = find_path(budget, trial).replace("-power.csv", ".jsonl")
    with open(jsonl_path) as f:
        recs = [json.loads(line) for line in f]
    whale = min(recs, key=lambda r: abs(r.get("pad_chars", 0) - pad_chars_target))
    prefill_start = whale["ts"] - whale["latency"]

    t, p = load_power(find_path(budget, trial))
    mask = (t >= prefill_start - pre) & (t <= prefill_start + post)
    t_win, p_win = t[mask], p[mask]
    return t_win - prefill_start, p_win


# --- Panel (a): real whale-event snippet, mono vs. chunk=512 ---
mono_t, mono_p = whale_snippet("b16384", 1, 49913)
chunk_t, chunk_p = whale_snippet("b512", 1, 49913)

# --- Panels (b) and (c): model-free bootstrap over the same 100-trial library used by Table IV's
# concurrency-20 operating point ---
mono_paths = [find_path("b16384", i) for i in range(1, 101)]
chunk_paths = [find_path("b512", i) for i in range(1, 101)]
lib_mono = build_library(mono_paths, dt)
lib_chunk = build_library(chunk_paths, dt)
shortest = min(lib_mono.shape[1], lib_chunk.shape[1]) * dt
n_steps = int(round(shortest * 0.5, -1) / dt)  # same window convention as Table IV
t_b = np.arange(n_steps) * dt

rng = np.random.default_rng(SEED)
b_mono = sample_fleet(lib_mono, N_SHOWN, n_steps, 0.0, rng)
b_chunk = sample_fleet(lib_chunk, N_SHOWN, n_steps, 0.0, rng)
print(f"panel b: window={n_steps*dt:.0f}s  mono mean-of-mean={b_mono.mean():.1f}W  "
      f"chunk mean-of-mean={b_chunk.mean():.1f}W")

n_mc = 150
c_mono_pool = np.empty(n_mc)
c_chunk_pool = np.empty(n_mc)
for k in range(n_mc):
    c_mono_pool[k] = np.abs(np.diff(sample_fleet(lib_mono, N_RESERVE, n_steps, 0.0, rng).sum(axis=0))).max() / dt
    c_chunk_pool[k] = np.abs(np.diff(sample_fleet(lib_chunk, N_RESERVE, n_steps, 0.0, rng).sum(axis=0))).max() / dt
# convert W/s -> MW/min, matching Table IV's own convention (ramp_reserve_procurement.py)
c_mono_pool *= 60.0 / 1e6
c_chunk_pool *= 60.0 / 1e6
c_r99_mono = np.percentile(c_mono_pool, 99)
c_r99_chunk = np.percentile(c_chunk_pool, 99)
print(f"panel c: R99 mono={c_r99_mono:.3f} MW/min  chunk={c_r99_chunk:.3f} MW/min  "
      f"reduction={100*(1-c_r99_chunk/c_r99_mono):.1f}%")

np.savez(
    OUT_NPZ,
    mono_t=mono_t, mono_p=mono_p,
    chunk_t=chunk_t, chunk_p=chunk_p,
    b_t=t_b, b_mono=b_mono, b_chunk=b_chunk, b_N=N_SHOWN,
    c_mono_pool=c_mono_pool, c_chunk_pool=c_chunk_pool,
    c_r99_mono=c_r99_mono, c_r99_chunk=c_r99_chunk,
)
print(f"wrote {OUT_NPZ}")
