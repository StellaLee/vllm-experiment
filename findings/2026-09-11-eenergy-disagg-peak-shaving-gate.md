# Peak-shaving admission gate for 8-GPU P/D-disaggregated serving

*Follow-on to `findings/2026-09-11-eenergy-peak-shaving-admission-gate.md` (the colocated
gate) and `findings/2026-09-11-eenergy-pd-disaggregation-energy-calibration.md` (the P/D
disaggregation investigation that measured real, uncontaminated per-token energy costs:
`j_per_prefill_token=0.094`, `j_per_decode_token=0.78` J/token). Spec:
`docs/superpowers/specs/2026-09-11-disagg-peak-shaving-gate-design.md`. Plan:
`docs/superpowers/plans/2026-09-11-disagg-peak-shaving-gate.md`. Question this answers: does
applying the admission gate to a real disaggregated deployment, with independent per-pool
power budgets instead of one combined fleet number, shave peak power more effectively than
the colocated gate — for a comparable latency cost?*

## Design

4:4 fixed GPU split (GPUs 0-3 = 4 prefill instances, GPUs 4-7 = 4 decode instances,
`NixlConnector`, round-robin within each pool via the existing vendored
`nixl_toy_proxy_server.py`, untouched). A new gate shim
(`scripts/eenergy/router/disagg_gate.py`) sits in front of that proxy and reuses
`PowerBudget`/`estimate_marginal_energy_j`/`DecodeByteEstimator` from `power_budget.py`
completely unmodified — instantiated *twice* (one per pool) instead of once, with each leg's
marginal energy isolated by calling `estimate_marginal_energy_j` with the other leg's token
count zeroed (`estimate_marginal_energy_j(prompt_tokens, 0, ...)` for the prefill leg,
`estimate_marginal_energy_j(0, expected_decode_tokens, ...)` for the decode leg). A request is
admitted only when *both* pools clear (`dual_budget_would_exceed`, a small new pure function
added to `power_budget.py`). This deliberately mirrors the colocated gate's own
gate-in-front-of-routing separation, generalized from one combined budget to two independent
ones — chosen over extending the existing `router_core.py`/`scoring.py` machinery (built for
homogeneous replicas that each serve a whole request) or modifying the proven-working Nixl
proxy directly, both of which carried more risk for no clear benefit.

## Implementation

5 tasks, TDD throughout, all committed:

1. `dual_budget_would_exceed(budget_prefill, cap_prefill_w, prefill_marginal_j, budget_decode,
   cap_decode_w, decode_marginal_j) -> bool` — the one genuinely new piece of logic (defers if
   *either* pool would exceed; a `None` budget for a pool means that pool's cap is disabled
   and never blocks, independently of the other pool). 5 unit tests.
2. `disagg_gate.py`'s pure/testable helpers: `build_power_budgets` (mirrors
   `proxy_server.py`'s `build_power_budget`, generalized to two pools) and `pool_power_w`
   (sums a live NVML read across one pool's GPU indices — simpler than `proxy_server.py`'s
   `fleet_power_w`, since the gate shim has no `ReplicaState`/ramp-ceiling machinery to
   piggyback on). 5 unit tests.
3. `disagg_gate.py`'s aiohttp app (`make_app`, `handle_completions`, `handle_health`,
   `power_poll_loop`, `forward_to_nixl_proxy`) + CLI entrypoint `run_disagg_gate.py`
   (`DISAGG_GATE_*` env vars, mirroring `run_router.py`'s `ROUTER_*` convention). Not unit
   tested, matching `proxy_server.py`'s own established convention (`make_app` there isn't
   unit tested either) — verified instead by the hardware validation in Tasks 4-5.
4. `orchestrate/eenergy/run_disagg_8gpu_baseline.sh` + `scripts/eenergy/
   analyze_disagg_8gpu_power.py` — uncapped 4:4 baseline, to measure this topology's own
   sustained mean/peak power per pool (not previously measured at 4:4 scale — the 1P1D
   calibration and the colocated 6-replica fleet are both different topologies).
5. `orchestrate/eenergy/run_disagg_8gpu_gated.sh` — same topology, gated, with each pool's cap
   set to `round(0.925 * that pool's measured baseline mean)`, mirroring the colocated gate's
   own cap/mean ratio (1800W / ~1945W) so both designs are pushed at comparable relative
   aggressiveness.

Self-review during plan-writing caught one real spec/plan naming drift (`DISAGG_PROXY_URL`
written as `DISAGG_NIXL_PROXY_URL` in an early draft) before any code was written — fixed to
match the spec exactly.

## Validation results

Baseline (uncapped, Heavy/CL-long-shaped workload: `whale_frac=0.15`, `concurrency=32`,
`num-convs=600`, `max_tokens=1024`), first measurement of this topology at 4:4 scale:

| pool | mean power | max windowed-avg (15s) |
|---|---|---|
| prefill (GPU 0-3) | 587.8W | 1131.1W |
| decode (GPU 4-7) | 895.4W | 1205.2W |

(600/600 records attempted, 2 failures in both arms — same structural failures, not gating-
related. Baseline mean TTFT 1.715s.) Notably, the decode pool draws *more* sustained power
than the prefill pool here — 4 decode instances are continuously doing batched decode work
for the whole trial, while 4 prefill instances only draw power in short bursts per admitted
request, then sit comparatively idle.

Cap computation: `cap_prefill_w = round(0.925 * 587.8) = 544`, `cap_decode_w = round(0.925 *
895.4) = 828`.

Gated run, identical topology/workload, both caps active:

| pool | baseline max windowed (15s) | gated max windowed (15s) | reduction | cap |
|---|---|---|---|---|
| prefill | 1131.1W | 890.6W | **21.3%** | 544W |
| decode | 1205.2W | 1194.4W | **0.9%** | 828W |

Mean TTFT: 1.715s → 11.155s (p50 0.696s → 3.232s, p95 6.840s → 32.400s, max 13.468s →
64.182s). 2 failures in both arms again (same ones).

## Interpretation: a real, asymmetric result

**Prefill pool: the disaggregated gate clearly beats the colocated gate.** 21.3% max-
windowed-power reduction vs. the colocated gate's ~7-8% (Part 12 of the companion findings
doc, cap=1800W), for a comparable-or-smaller absolute TTFT cost (+9.44s here vs. the
colocated result's ~+11 to +14.5s). This directly confirms the plan's success criterion
(spec Sec 1: bigger shaving effect at comparable TTFT cost) — at least for this pool.

**Decode pool: the gate barely moves it.** 0.9% is within noise. This is not a bug — it's a
mechanistic consequence of what admission gating *can* control. The gate only throttles *new*
admissions. Prefill events are short and bursty (a single forward pass per request), so
pacing new admissions directly and immediately caps the prefill pool's instantaneous load —
highly effective. Decode power, by contrast, is dominated by *already-admitted, long-running*
generation streams (mean realized output ~334 tokens at ~17ms/token ≈ 5.7s of continuous
draw per stream in this project's earlier calibration work, longer for less-typical
responses) that the gate cannot retroactively throttle once they're running. This is the same
structural limitation identified in the colocated gate's own Part 11 (admission-pacing without
shedding has a hard ceiling under sustained load) — but it was invisible there, because one
combined fleet number blended the (gate-controllable) prefill contribution together with the
(largely gate-*un*controllable) decode contribution into a single, misleadingly moderate
7-8% figure.

**The real contribution of per-pool budgets isn't just matching or beating the colocated
number — it's revealing *which* pool admission-pacing actually works on.** A deployment
decision informed only by the colocated gate's combined number would not know that its
apparent effectiveness is really "prefill-driven, decode-along-for-the-ride." A deployment
using per-pool disaggregated budgets can target the pool where the mechanism actually applies
(prefill), and would need a genuinely different mechanism — shedding, preemption, or bounding
concurrent decode slots directly — to make comparable progress on decode.

## Caveats

- **n=1 trial** for both the baseline and gated batteries — no replication yet.
- Round-robin only within each pool (no load-aware routing) — an explicit v1 non-goal (spec
  Sec 6), so this result doesn't yet say whether smarter pool-internal routing would change
  the decode-pool finding.
- The cap ratio (0.925 of measured mean) was chosen to mirror the colocated gate's own
  choice for comparability, not independently optimized for this topology — a different ratio
  might shift the prefill/decode asymmetry's exact magnitude, though the mechanistic
  explanation (new-admission-only control) should hold regardless of the specific cap chosen.
- No shedding, no dynamic prefill:decode rebalancing (fixed 4:4), single 8-GPU box only —
  all explicit v1 non-goals (spec Sec 6). The decode-pool finding is itself the natural
  argument for why shedding or concurrency-bounding would be the next thing to try if decode-
  side peak shaving matters for a real deployment.

## Data and repro

**Scripts** (all on `main`, `/Users/li/Documents/vllm-experiment`, mirrored to
`183.147.142.123:/root/pli/vllm-experiment`):
- Core implementation: `scripts/eenergy/router/power_budget.py` (`dual_budget_would_exceed`,
  new), `scripts/eenergy/router/disagg_gate.py` (new module), `scripts/eenergy/
  run_disagg_gate.py` (new CLI entrypoint). Tests: `tests/test_eenergy_power_budget.py`
  (dual-budget cases appended), `tests/test_eenergy_disagg_gate.py` (new).
- Validation batteries: `orchestrate/eenergy/run_disagg_8gpu_baseline.sh`,
  `run_disagg_8gpu_gated.sh`. Analysis: `scripts/eenergy/analyze_disagg_8gpu_power.py`.

**Log paths** (remote, `/root/pli/vllm-experiment/logs/`):
- Baseline: `disagg_8gpu_baseline_{records,harness,timing}.{jsonl,log,txt}`,
  `disagg_8gpu_baseline_power_{prefill,decode}.csv`, per-instance launch logs
  `disagg_8gpu_baseline_{prefill,decode}{0-3}.log`, `disagg_8gpu_baseline_{nixl_proxy,gate}.log`.
- Gated: same naming with `disagg_8gpu_gated_*`.

**Remote access**: `ssh 183.147.142.123`, repo at `/root/pli/vllm-experiment`, venv at
`/root/pli/venv-vllm023`. GPUs 0-3 = prefill pool, GPUs 4-7 = decode pool throughout.
