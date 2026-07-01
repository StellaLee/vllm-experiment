# ShareGPT KV Cache Sweep — Qwen2.5-Coder-7B (2026-07-01)

**Model:** Qwen2.5-Coder-7B-Instruct  |  **Trace:** ShareGPT multi-turn, 100 conversations  
**Setup:** concurrency=20, max_turns=4, max_tokens=256  
**GPU:** RTX 4090 (24 GB)  |  **vLLM prefix caching:** enabled

---

## Aggregate Metrics

| gpu-memory-utilization | KV cache tokens | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy | Duration |
|---|---|---|---|---|---|---|---|---|
| **0.7** | 12,816 | 256 | 59ms | **887ms** | 4.27s | 5.91s | 4.85 Wh | 55.5s |
| 0.8 | ~58,000 | 256 | 59ms | 193ms | 4.21s | 5.34s | 4.69 Wh | 53.5s |
| 0.9 | ~103,000 | 256 | 59ms | 160ms | 4.28s | 5.38s | 4.73 Wh | 53.5s |

---

## Per-Turn TTFT — The Eviction Signal

| Turn | Avg Prompt Words | 0.7 util | 0.8 util | 0.9 util |
|---|---|---|---|---|
| 1 | 197 | 319ms | 101ms | 94ms |
| 2 | 157 | **88ms** | 81ms | 69ms |
| 3 | 187 | 157ms | 65ms | 68ms |
| 4 | 251 | **321ms** | 65ms | 68ms |

---

## Interpretation

### 0.8 and 0.9 util — prefix cache working throughout
TTFT falls from ~100ms at turn 1 to ~65–69ms by turn 3 and stays flat. Each
conversation's accumulated prefix blocks survive in cache until the next turn,
so only the new question tokens need prefilling.

### 0.7 util — U-shaped TTFT: eviction on later turns
- **Turn 1 (319ms):** 3× higher than 0.8/0.9 util. With only 12,816 tokens of KV
  budget and 20 concurrent conversations, the scheduler is under constant
  pressure — higher queue wait before each request is dispatched.
- **Turn 2 (88ms):** Cache hit. Turn-1 prefix blocks (50 tokens) are still fresh
  (LRU protects recently accessed blocks). Only 65 conversations remain active,
  reducing concurrent pressure.
- **Turn 3 (157ms):** Partial evictions begin. Accumulated prompts (~187 words)
  start competing with other concurrent conversations' KV blocks for the 12,816
  token budget.
- **Turn 4 (321ms):** Back to turn-1 level despite only 41 concurrent conversations.
  The longer accumulated prompts (~251 words ≈ 325 tokens) have had their prefix
  blocks evicted by other conversations' decode steps. Full re-prefill required —
  TTFT equals a cold-start prefill of the entire history.

### P95 TTFT as the tail-latency signal
At 0.7 util, P95 TTFT is **887ms** — 4.6× higher than 0.8 util (193ms) and 5.5×
higher than 0.9 util (160ms). The median (P50=59ms) stays the same across all
three, meaning the eviction effect is concentrated in the tail. Short conversations
and early turns are unaffected; users with longer multi-turn sessions bear the full
penalty.

### Energy
0.7 util costs **3.4% more energy** (4.85 vs 4.69 Wh) for the same 256 requests,
and takes 2 seconds longer (55.5 vs 53.5s). The overhead is from re-prefilling
evicted prefixes — redundant compute that contributes nothing to throughput.

---

## Phase 3 Baseline

This result establishes the **LRU eviction baseline** for the custom eviction
policy study. At 0.7 gpu-memory-utilization (12,816 token KV cache), turn-4 TTFT
under LRU is **321ms** — a 5× penalty vs the no-eviction baseline (65ms at 0.9 util).

The time-decayed frequency policy `Score(k) = (hit_count+1)·e^{-λ·age}` targets
this gap: by deprioritising blocks from conversations that have been idle relative
to actively messaging users, it should keep recent-turn prefixes in cache longer
and push the turn-4 TTFT back toward 65–70ms even within a 0.7 util budget.

## Per-utilization details
- [gpu-util=0.7](2026-07-01-kvcache7b-0.7.md)
- [gpu-util=0.8](2026-07-01-kvcache7b-0.8.md)
- [gpu-util=0.9](2026-07-01-kvcache7b-0.9.md)
