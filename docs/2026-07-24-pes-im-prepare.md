# PES-IM paper — preparation notes (2026-07-24)

*Consolidates everything decided/found since `2026-07-24-pes-im-plan.md` was written: the
gate experiment's actual setup and results (including a real pivot in what C3 claims), the
theory to add, the literature landscape, and the tooling decision going forward. This is a
working reference, not paper prose — read `2026-07-24-pes-im-plan.md` for the task list and
CFP-fit reasoning.*

## 1. Setup — what the gate experiment actually ran

**Server** (one arm per budget, otherwise identical):
```
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
  --max-num-seqs 48 \
  --max-num-batched-tokens {16384 | 2048 | 512}   # mono / chunk / chunk
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```
Single GPU (no TP — isolates the measurement from NCCL power draw). GPU clock locked to
3105 MHz (max supported graphics clock) and persistence mode enabled for the whole run, to
remove DVFS/idle-ramp variance from the power trace; reset after.

**Traffic** (`replay_sharegpt.py`, same mechanism validated in the MLSys paper's §5.6):
```
--dataset data/sharegpt_v3.json --num-convs 200 --max-turns 1 --min-turns 1 \
--max-tokens 256 --concurrency 20 \
--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
--whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
--max-prompt-chars 50000
```
- 200 single-turn conversations, real ShareGPT text as the base prompt.
- Closed-loop concurrency 20: 20 persistent worker slots, next request fires the instant a
  slot frees — the GPU sees a continuously busy ~20-concurrent-decode pool for the whole run,
  not an idle-then-burst pattern. (This matters: see §3.)
- Every prompt gets a synthetic, unique, uncacheable pad prepended: 85% short population
  (Lognormal, mean 800 chars, cv²=0.5, clamped [100, 8000]); 15% whale population
  (Uniform[44000, 50000] chars ≈ 12-15k tokens).
- 3 trials (pad-seed 1001/1002/1003), 20 discarded warmup requests per arm before each
  measured trial (thermal/clock steady-state).

**Instrumentation:**
- `scripts/power_logger.py`: standalone NVML side-car, samples power draw + the hardware
  cumulative energy counter + temperature at 50ms resolution, wall-clock (`time.time()`)
  timestamped so it correlates against request records after the fact (`ts`/`latency`/`ttft`
  fields already in `replay_sharegpt.py`'s output — no harness changes needed).
- IPMI/BMC chassis sensors (genuine bonus find): the box's BMC (H3C rack chassis) exposes
  `GPU_Power` and `Total_Power` sensors — an independent hardware cross-check beyond NVML,
  which is exactly the kind of thing a power-systems reviewer would ask about. Caveat: each
  `ipmitool sensor list` query takes ~2.5s round-trip, so it's a steady-state/aggregate
  cross-check only, not a spike-resolution instrument. Snapshotted before/after each arm.
- `scripts/analyze_pesim_gate.py`: identifies whale prefill windows as
  `[ts - latency, ts - latency + ttft]` per whale request (`pad_chars >= 40000`), computes
  baseline power (median outside whale windows), peak power (max inside whale windows), and
  energy-per-output-token (from the cumulative energy counter delta).

## 2. Claims — original vs. what the data actually supports

- **C1 (energy conservation):** scheduling policy shouldn't change energy-per-token, only its
  temporal shape. **Holds well for chunk=2048** (+0.1% vs. mono), **costs a real +3.7% at
  chunk=512** — small but non-trivial, and directionally sensible (finer chunking = more
  scheduling overhead).
- **C3, as originally framed ("chunking flattens the power spike"): NOT supported.** See §3 —
  peak power is statistically flat across all three arms.
- **C3, revised and empirically supported: chunking reframes the power profile's *shape*, not
  its peak** — fewer, longer, smoother elevated-power episodes with a lower ramp rate, at a
  modest energy cost, while *increasing* total near-ceiling duty cycle. This is a genuine,
  mechanistically coherent, and arguably more power-systems-relevant finding than the original
  hypothesis (see §5, ramp-rate framing).

## 3. Gate experiment results (actual data, 3 arms × 3 trials, n=194-196/arm/trial)

**Peak power and energy (summary, mean over trials):**

| Budget | baseline (W) | peak (W) | energy (kJ / 1k out-tok) |
|---|---|---|---|
| 16384 (mono) | 376.0 | 458.6 | 937.57 |
| 2048 (chunk) | 379.7 | 463.6 | 938.71 |
| 512 (chunk) | 382.4 | 462.1 | 972.03 |

Go/no-go vs. mono: 2048 → peak +1.1%, energy/token +0.1%. 512 → peak +0.8%, energy/token
+3.7%. **Peak power does not flatten with chunking at this workload/concurrency** — all three
arms saturate near the GPU's ~450W power limit during whale windows.

**Power-shape statistics (pooled across trials, threshold = 420W):**

| Budget | % time above 420W | # distinct spike events | mean ramp rate (W/s) | max ramp rate (W/s) |
|---|---|---|---|---|
| 16384 (mono) | 47.8% | 53 | 41.1 | 4166.1 |
| 2048 (chunk) | 52.7% | 48 | 39.0 | 4570.3 |
| 512 (chunk) | 63.6% | 37 | 26.8 | 4301.0 |

As chunking gets more aggressive: fewer, longer, smoother episodes; mean ramp rate drops
~35% (mono→512); but *more* total near-ceiling time.

**Mechanistic read:** the GPU saturates its power ceiling whenever it's doing heavy compute,
whether that's one giant prefill or a rapid sequence of interleaved chunk+decode work —
chunking doesn't change *how much* compute happens, just how it's temporally sliced, so it
can't flatten peak magnitude. What it changes is the transition behavior: smoother ramps,
fewer discrete events, longer sustained engagement. This connects to the energy-proportional-
computing literature (§6D) — GPU power isn't proportional to temporal slicing, only to whether
the SMs are saturated at all, which is consistent with peak staying flat while shape shifts.

**Earlier smoke test for context** (mono/16384 only, n=20, whale-frac=0.3, less continuous
background load): baseline 305.3W, peak 450.4W, spike ratio 1.48× — a starker contrast than
the full run, plausibly because with fewer/shorter-running requests the "baseline" period
itself was less loaded. This is the open question in §4.

## 4. Open decision — not yet resolved

Whether the flat-baseline-vs-whale-spike contrast (peak-magnitude framing) is recoverable at
lower concurrency (a less continuously-saturated baseline), or whether we commit fully to the
ramp-rate/duty-cycle reframing (§2, §5) as the paper's actual empirical spine regardless of
concurrency. Three options were on the table and not yet decided:
1. Reframe around ramp rate (adopt what the data shows now).
2. Try a lower-concurrency rerun first to check if peak-flattening is recoverable.
3. Do both — build the theory now (it doesn't depend on which framing wins) and rerun at
   lower concurrency in parallel.

## 5. Theory to add (for "systems engineering" credibility with this audience)

Recommended, in priority order (full derivation owed, not yet written into paper prose):

1. **Two-state (ON/OFF) load model → coincidence/diversity factor** (core addition). Model
   each GPU as alternating baseline power $P_b$ / burst power $P_{max}$ with duty cycle
   $p = \lambda \cdot E[\tau]$ (renewal-reward). For $N$ independent GPUs, aggregate burst
   count $K(t) \sim \text{Binomial}(N,p)$; coincidence factor
   $CF(N) = E[\max_t D(t)]/(N \cdot P_{max})$. Independent arrivals: $CF(N) \to p$ as
   $N\to\infty$ (classical diversity-factor asymptote, matches EV-charging literature, §6C).
   Fully synchronized arrivals: $CF=1$ for all $N$ (no diversity benefit). General case with
   synchronization parameter $s\in[0,1]$: $CF(N,s) \approx s + (1-s)\cdot(p + O(1/\sqrt N))$ —
   this is the closed form behind the already-planned coincidence-factor grid-model sweep,
   giving it an analytic backbone instead of being purely a numeric sweep.
2. **Cold-load-pickup framing** (near-free, high payoff). Correlated whale arrivals across a
   fleet (synchronized batch job, viral prompt, shared cron trigger) collapsing $K(t)\to N$
   simultaneously is structurally identical to cold-load pickup — the well-studied
   power-reliability phenomenon where thermostatic loads (AC compressors, water heaters) all
   switch ON simultaneously after an outage ends. Naming this signals fluency in the venue's
   own vocabulary at the cost of a sentence or two.
3. **Ramp-rate limiting** (now the *primary* empirical hook, given §3's actual result). Many
   interconnection standards impose explicit ramp-rate limits (kW/s or kW/min) on large
   flexible loads (battery storage, EV fast-charging) to protect grid stability. Our measured
   $dP/dt$ already shows chunk=512's mean ramp rate ~35% lower than mono's — report this
   directly in the standards-relevant unit/vocabulary rather than only "flattens the peak"
   framing (which the data doesn't support).
4. *(Skip)* A formal queueing-theoretic link from the MLSys paper's Eq. genuine
   ($\rho/(1-\rho)$ term) to power-spike magnitude — intellectually available but CS-flavored
   math this audience can't independently evaluate; not worth the risk/space for a 5-page paper.

## 6. Literature landscape and our add-on value

Four distinct bodies of work, none of which do what we're proposing (verified via direct
search + abstract fetch, not recalled):

**A. Training-side power oscillation (closest mechanism, wrong workload).** Well-established,
active: Meta's LLaMA 3 paper reports 30MW fluctuations from synchronized collective
communication across a 24k-H100 cluster; Microsoft's "Power stabilization for AI training
datacenters" (Choukse et al., arXiv:2508.14318); an IEEE Trans. Power Systems paper on
wide-area oscillations from AI workloads; HPCA/resonance-safety-criterion papers on the same
phenomenon. All about *training* (synchronized allreduce/checkpoint steps), not inference
serving with a scheduling-policy lever.

**B. Inference-side power characterization — very recent (2026), closest neighbors.**
- "From Servers to Sites" (arXiv:2603.18383, Wilkins/Kazhamiaka/Rajagopal) — synthesizes power
  traces server→facility scale. **Confirmed via abstract fetch: does not study scheduling
  policy as a lever, no coincidence-factor formalism.**
- "Workload composition smooths aggregate power demand..." (arXiv:2604.10769) — batch/inference
  traffic *mix* changes aggregate ramping. **Confirmed via abstract fetch: no scheduling-policy
  lever, no request-length modeling, no diversity-factor derivation.**
- "Measurement of Generative AI Workload Power Profiles..." (arXiv:2604.07345) — adjacent,
  not yet read in full; check before citing precisely.
- "TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference" (arXiv:2501.02600) — appears
  to go the *other* direction (using power/thermal state as a scheduling input) rather than
  characterizing an existing latency-motivated policy's power side-effect; not yet read in
  full, check before citing.

**C. Grid-side diversity/coincidence-factor theory (mature, never applied to AI load).**
Multiple IEEE papers on EV-charging coincidence factors (e.g., coincidence factor drops below
25% past 50 EVs at 11kW charging); the general diversity-factor formalism for
feeder/transformer sizing. This is the math in §5.1 — established, just never pointed at an
LLM-serving load class.

**D. Energy-proportional computing (foundational, being revisited for GPUs).** Barroso &
Hölzle (2007); very recent "The Energy Cost of Execution-Idle in GPU Clusters"
(arXiv:2604.04745) showing GPU power scales sub-linearly with SM occupancy but
super-linearly with utilization — explains §3's result (peak resists chunking) as consistent
with known GPU non-proportionality, not an anomaly.

**Our add-on value:** first (as far as verified) to connect an already-deployed,
latency-motivated scheduling technique (chunked prefill) to a measured, request-level power/
ramp-rate consequence, and to bridge that single-GPU measurement to fleet-scale aggregate
demand using the established coincidence-factor formalism from power-distribution/EV-charging
engineering — empirically grounded in real measured traces (unlike B's synthetic-composition
approach), not just a characterization study (A doesn't apply to inference; B doesn't touch
scheduling policy or diversity math). Space B is moving fast (arXiv IDs from the last ~4
months) — real, if modest, incentive to keep the Aug 15 deadline.

## 7. Tooling decision — going forward

- **MLSys paper (all existing work + Phase 3):** keep `replay_sharegpt.py`. Every existing
  finding (§5.5-5.8, each replicated 3×) was produced by this harness; switching now buys
  nothing for an ML-systems audience and risks internal inconsistency within one paper's
  evidence base. It also has capabilities (phase-schedules, concurrency-schedules, controller
  hooks) a generic benchmarking tool isn't built for.
- **PES-IM gate experiment: switch to a hybrid.** Pre-generate a `custom`-format JSONL
  (`{"prompt": ..., "output_tokens": ...}`) using our validated whale-injection logic (same
  mechanism, same 44-50k char bounds), then replay it via
  `vllm bench serve --dataset-name custom --max-concurrency 20 --request-rate inf
  --save-result --save-detailed`. Confirmed via CLI/source inspection: `--max-concurrency N`
  + `--request-rate inf` gives genuine closed-loop semantics (next request fires the instant a
  slot frees) matching our current design; `--save-detailed` exports per-request
  `start_times`/`ttfts`/`output_lens`. One wrinkle: `start_time` uses `time.perf_counter()`
  (monotonic, not wall-clock epoch) — needs a captured perf_counter/time.time() offset at
  launch to correlate against the NVML power trace. Rationale: no existing lineage to break
  for this fresh experiment, and "measured with the standard vLLM benchmarking tool" reads
  better to a systems-engineering-literate, non-ML-native reviewer than a custom script.

## 8. Next steps

1. Resolve §4 (lower-concurrency rerun vs. commit to ramp-rate framing vs. both).
2. Build the hybrid bench-serve harness (§7) if/when a rerun is needed.
3. Write out the §5.1 coincidence-factor derivation formally and validate it against a
   Monte-Carlo simulation seeded with the real measured single-GPU trace (no GPU time needed
   for this piece — pure post-processing).
4. Read B's two not-yet-fully-checked adjacent papers (Measurement of Generative AI Workload
   Power Profiles; TAPAS) before finalizing related-work claims.


## 9. Literature review

**A. Training-side power oscillation (the closest analog, but wrong workload)**. This is a well-established, active area: Meta's LLaMA 3 paper reports 30MW power fluctuations from synchronized collective communication across a 24,000-H100 cluster; Microsoft's "Power stabilization for AI training datacenters" (Choukse et al., arXiv:2508.14318) addresses it with cross-stack software/hardware smoothing; there's an IEEE Transactions on Power Systems paper on wide-area oscillations from AI workloads, plus HPCA and resonance-safety-criterion papers on the same phenomenon. All of this is about training — power swings driven by synchronized allreduce/checkpoint steps across a fleet, not request-level heterogeneity. Our angle (inference serving, single-request prefill bursts, scheduling-policy-controllable) is a different mechanism entirely.

**B. Inference-side power characterization — very recent (2026), and the closest neighbors**. Two papers in particular:
- "From Servers to Sites" (arXiv:2603.18383, Wilkins/Kazhamiaka/Rajagopal, Stanford/Microsoft) — synthesizes power traces from servers up to facility scale, but doesn't study scheduling policy as a lever and doesn't use a coincidence-factor formalism.
- "Workload composition smooths aggregate power demand..." (arXiv:2604.10769) — shows batch/inference traffic mix changes aggregate ramping, but again no scheduling-policy lever, no request-length distribution modeling, no diversity-factor derivation.


**C. Grid-side diversity/coincidence-factor theory (well-established, but never applied to AI load)**. This is real, mature literature — multiple IEEE papers specifically on EV-charging coincidence factors (e.g., coincidence factor drops below 25% past 50 EVs at 11kW), the general diversity-factor formalism used for feeder/transformer sizing. This is the math I derived last turn — we're not inventing it, we're the first (as far as this search shows) to apply it to an LLM-serving load class.

**D. Energy-proportional computing (foundational, now being revisited for GPUs)**.lus very recent work ("The Energy Cost of Execution-Idle in GPU Clusters,"arXiv:2604.04745) showing GPU power scales sub-linearly with SM occupancy but super-linearly with utilization. This actually explains our own surprising result — peak power resisting chunking while ramp-rate/duty-cycle shifted is exactly what you'd expect if GPU power isn't proportional to how work is temporally sliced, only to whether the SMs are saturated at all. Useful for framing our finding as consistent with known GPU behavior rather than an anomaly.

**Our add-on value, stated precisely**: we're the first to connect an already-deployed, latency-motivated scheduling technique (chunked prefill — not a new mechanism we invented) to a measured, request-level power/ramp-rate consequence, and to bridge that single-GPU measurement to fleet-scale aggregate demand using the established coincidence-factor formalism from power-distribution/EV-charging engineering — empirically grounded (real measured traces, not synthetic assumptions like B's papers), not just a new characterization study (A doesn't
apply to inference; B doesn't touch scheduling policy or diversity math). Givee (arXiv IDs from the last ~4 months), this space is moving fast — real, ifmodest, incentive to keep the Aug 15 deadline rather than let it slip.

**Sources**:
- Power stabilization for AI training datacenters (Microsoft) (https://techcommunity.microsoft.com/blog/azurecompute/power-stabilization-for-ai-training-datacenters/4460937)
- AI Training Load Fluctuations at Gigawatt-scale (SemiAnalysis) (https://newsletter.semianalysis.com/p/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout)
- Operational Risks in Grid Integration of Large Data Center Loads (https://arxiv.org/html/2510.05437v2)
- Characterizing Power Management Opportunities for LLMs in the Cloud (Microso(https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf)
- Electricity Demand and Grid Impacts of AI Data Centers (https://arxiv.org/ht
- From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning (https://arxiv.org/pdf/2603.18383)
- Workload composition smooths aggregate power demand while sustaining short-horizon ramps in AI data centers (https://arxiv.org/html/2604.10769v1)
- Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning (https://arxiv.org/pdf/2604.07345)
- Coincidence Factors for Domestic EV Charging From Driving and Plug-In Behavi(https://www.researchgate.net/publication/352302299_Coincidence_Factors_for_Domestic_EV_Charging_from_Driving_and_Plug-in_Behavior)
- Diversity factor (Wikipedia) (https://en.wikipedia.org/wiki/Diversity_factor)
- The Energy Cost of Execution-Idle in GPU Clusters (https://arxiv.org/html/26
- The Case for Energy-Proportional Computing (Barroso & Hölzle) (https://www.researchgate.net/publication/2962080_The_Case_for_Energy-Proportional_Computing)
