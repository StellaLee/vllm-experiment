# QPS Sweep — Coder-7B (BurstGPT, 2026-07-02)

**Model:** Qwen2.5-Coder-7B-Instruct  
**GPU:** RTX 4090 (450W TDP)  
**Setup:** BurstGPT trace, 40 requests/run (38 completed), max_tokens=128, gpu-memory-utilization=0.9, --enable-prefix-caching  
**Design:** One vLLM instance shared across all scale runs (GPU stays hot between runs)

---

## Results

| Scale | Duration | Energy | Avg Power | Peak Power | Avg GPU Util | P50 Lat | P95 Lat | P50 TTFT | P95 TTFT | KV Hit Rate |
|-------|----------|--------|-----------|------------|--------------|---------|---------|----------|----------|-------------|
| 1×    | 981.5 s  | 13.31 Wh | 48.8 W  | 340.5 W    | 8.7%         | 3.98 s  | 5.82 s  | 45 ms    | 104 ms   | 0%          |
| 2×    | 491.0 s  | 8.35 Wh  | 61.2 W  | 340.2 W    | 13.2%        | 4.00 s  | 5.81 s  | 44 ms    | 53 ms    | 0%          |
| 4×    | 246.0 s  | 4.91 Wh  | 71.9 W  | 331.5 W    | 16.3%        | 3.95 s  | 5.89 s  | 48 ms    | 55 ms    | 0%          |
| 8×    | 124.0 s  | 3.10 Wh  | 89.9 W  | 328.8 W    | 22.6%        | 4.00 s  | 5.96 s  | 50 ms    | 55 ms    | 0%          |

---

## Interpretation

### Energy scales inversely with QPS — same as 0.5B
Energy roughly halves with each 2× QPS step: 13.31 → 8.35 → 4.91 → 3.10 Wh. Duration halves
proportionally (981 → 491 → 246 → 124 s). This confirms the BurstGPT trace's inter-arrival
structure — QPS scaling compresses idle gaps, not compute intensity.

### No GPU saturation at any scale
P50 latency is flat at ~4.0 s across all QPS levels; P95 is flat at 5.8–6.0 s. Peak power
(328–340 W) is the same across all scales — it spikes during each burst regardless of QPS.
Average GPU utilization climbs from 8.7% → 22.6%, but only because idle time shrinks, not
because the GPU is approaching its compute limit. RTX 4090 can handle Coder-7B at 8× BurstGPT
QPS without queuing pressure.

### TTFT: 1× has a cold-start P95 outlier
At 1×, P95 TTFT = 104 ms vs 45 ms median. The BurstGPT trace has an 810 s idle gap between
request 26 (arrival 234 s) and request 38 (arrival 1047 s). During this gap the vLLM scheduler
goes fully idle. When request 38 arrives cold, prefill involves re-loading weights into L2 cache,
causing the TTFT spike. At 2×–8×, the same gap compresses to 405 s / 202 s / 101 s — not
short enough to keep the GPU "truly warm" in a compute sense, but the cold-start overhead becomes
a smaller fraction of total TTFT. P95 drops to 53–55 ms at 2×–8× and stays flat there.

### KV cache hit rate: 0% (expected)
BurstGPT samples prompts from a flat ShareGPT pool with no conversation structure — every request
is a unique single-turn prompt with no shared prefix. The 0% hit rate confirms this and serves as
a control: the per-turn TTFT patterns observed here are driven entirely by compute load and idle
gaps, not by cache eviction.

### Avg power grows with QPS
48.8 → 61.2 → 71.9 → 89.9 W. Idle baseline is ~55–64 W. At 1× the GPU spends ~91% of time
idle, so average power is near idle. At 8× idle time drops to ~77%, pulling the average up.
Even at 8× the GPU is mostly idle (22.6% avg utilization) — it would take a much heavier model
or a much denser trace to approach TDP.

---

## Comparison with 0.5B QPS Sweep

| Metric | 0.5B @ 1× | 7B @ 1× | Ratio |
|--------|-----------|---------|-------|
| P50 Latency | 0.56 s | 3.98 s | 7.1× |
| P50 TTFT | 19 ms | 45 ms | 2.4× |
| Energy (1×) | 6.1 Wh | 13.3 Wh | 2.2× |
| Duration (1×) | 397 s | 981 s | 2.5× |
| GPU Util (1×) | 5.4% | 8.7% | 1.6× |

The 7B energy is 2.2× higher primarily because the longer decode time extends the observation
window (981 s vs 397 s), while peak power is similar. Neither model saturates the RTX 4090 on
BurstGPT — the trace's idle gaps dominate in both cases.

---

## Raw Files

| Scale | GPU log | Trace log | Findings |
|-------|---------|-----------|---------|
| 1× | `logs/2026-07-02-qps7b-1x-gpu.json` | `logs/2026-07-02-qps7b-1x-burstgpt-detail.jsonl` | `findings/2026-07-02-qps7b-1x.md` |
| 2× | `logs/2026-07-02-qps7b-2x-gpu.json` | `logs/2026-07-02-qps7b-2x-burstgpt-detail.jsonl` | `findings/2026-07-02-qps7b-2x.md` |
| 4× | `logs/2026-07-02-qps7b-4x-gpu.json` | `logs/2026-07-02-qps7b-4x-burstgpt-detail.jsonl` | `findings/2026-07-02-qps7b-4x.md` |
| 8× | `logs/2026-07-02-qps7b-8x-gpu.json` | `logs/2026-07-02-qps7b-8x-burstgpt-detail.jsonl` | `findings/2026-07-02-qps7b-8x.md` |
