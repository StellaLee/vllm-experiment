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
- **C3, revised — now the paper's adopted headline (RESOLVED 2026-07-24, see §4):**
  chunking reframes the power profile's *shape*, not its peak — fewer, longer, smoother
  elevated-power episodes with a lower ramp rate, at a modest energy cost, while *increasing*
  total near-ceiling duty cycle. **Confirmed to hold independently in all 3 trials** (not just
  pooled — see §3), which is why this, not the original peak-flattening hypothesis, is now the
  paper's central empirical claim. Arguably a *stronger* fit for Topic 4 than the original
  hypothesis: a non-obvious, measured tradeoff is more publishable to a rigor-focused audience
  than a simple "peak shaving" win, and it maps directly onto ramp-rate limits already used to
  regulate other flexible grid loads (see §5.3).

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
itself was less loaded. Superseded by §4's resolution — not pursued further.

**Per-trial monotonicity check (2026-07-24), the evidence that resolved §4:** computed
power-shape stats per trial (not pooled) to rule out a single-trial artifact driving the
pooled averages.

| Trial | mono ramp (W/s) | 2048 ramp | 512 ramp | mono frac≥420W | 2048 frac | 512 frac |
|---|---|---|---|---|---|---|
| 1 | 38.2 | 35.4 | 22.2 | 0.520 | 0.568 | 0.699 |
| 2 | 41.4 | 38.6 | 29.6 | 0.481 | 0.516 | 0.620 |
| 3 | 44.3 | 43.6 | 29.4 | 0.424 | 0.490 | 0.576 |

Mean ramp rate strictly decreases (mono > 2048 > 512) and near-ceiling duty cycle strictly
increases (mono < 2048 < 512) **in every trial independently** — this is a real, replicated
mechanism, not a pooling artifact.

**Whale-window disaggregation (2026-07-24), confirming the mechanism directly:** split every
power sample into "inside a whale prefill window" (± 1s pad) vs. "outside," per arm (mean
over 3 trials):

| Budget | frac≥420W, whale windows | frac≥420W, non-whale | mean ramp, whale windows |
|---|---|---|---|
| 16384 (mono) | 58.4% | 0.2% | 44.2 W/s |
| 2048 (chunk) | 63.1% | 0.1% | 41.1 W/s |
| 512 (chunk) | 72.7% | 2.7% | 25.9 W/s |

Near-ceiling time is essentially **zero outside whale windows for every arm** — confirming
the effect is entirely a whale-prefill phenomenon, not a diffuse artifact — and the same
monotonic ramp-rate trend holds *within* whale windows specifically. Bonus mechanistic
confirmation: whale windows contain progressively *more* power samples under chunking
(e.g. trial 1: mono n=1443 → chunk=512 n=1629) — chunking spreads the same whale prefill work
over more wall-clock time (interleaved with other decode work), which is exactly why total
near-ceiling *duration* increases even though instantaneous peak intensity doesn't. This
closes the loop on the mechanism: chunking doesn't change the compute, only how long it takes
to get through it while sharing the GPU with everything else.

## 4. Open decision — RESOLVED 2026-07-24: adopt the ramp-rate/duty-cycle reframing

Decided **against** chasing a lower-concurrency rerun to recover the original peak-flattening
framing, for two reasons: (1) it isn't guaranteed to reproduce, costing calendar time we don't
have much of before Aug 15; (2) artificially lowering concurrency to manufacture a peak
contrast is less representative of real continuous-load LLM serving, and could itself read as
"gaming" the experiment for a specific answer to a rigor-focused reviewer. The ramp-rate/
duty-cycle effect is already cleanly replicated (per-trial monotonicity above) and is
arguably a *better*, more standards-grounded claim than peak-flattening would have been. This
is now the paper's adopted headline (see the top-level headline statement below).

**Headline (adopted):** *Chunked prefill scheduling — already deployed in production for
latency reasons — doesn't reduce an LLM-serving GPU's peak power draw, but it reshapes its
temporal profile: ~35% lower mean ramp rate and fewer, longer, smoother power transitions, at
the cost of more sustained near-ceiling duty cycle and a modest (up to 3.7%) energy penalty at
the most aggressive setting. This tradeoff is invisible to latency-only evaluation and maps
directly onto ramp-rate limits already used to regulate other flexible grid loads (EV
fast-charging, battery storage).*

**Consequence for the coincidence-factor grid model (§5.1, §7):** unaffected in structure — it
takes any measured single-GPU trace as input regardless of which framing won — but its target
quantity shifts from aggregate *peak* to aggregate *ramp rate* ($dP/dt$), since that's where
chunking's real, replicated effect lives. The same diversity-factor math applies to $dP/dt$ as
to $P(t)$ itself (noted already in §5.1).

## 5. Theory to add (for "systems engineering" credibility with this audience)

Recommended, in priority order (full derivation owed, not yet written into paper prose):

1. **Two-state (ON/OFF) load model → coincidence/diversity factor** (core addition).
   **Validated by Monte Carlo (2026-07-25, `scripts/coincidence_factor_model.py`), and the
   validation process caught two real errors in the naive closed form below — both are now
   corrected and confirmed against simulation, not just asserted:**
   - **Original naive claim (WRONG as stated):** model each GPU as alternating baseline power
     $P_b$ / burst power $P_{max}$ with duty cycle $p=\lambda\cdot E[\tau]$; aggregate burst
     count $K(t)\sim\text{Binomial}(N,p)$; $CF(N,s)\approx s+(1-s)(p+O(1/\sqrt N))$, claiming
     $CF\to p$ as $N\to\infty$ under independent arrivals.
   - **Error 1 — missing baseline term.** The naive form implicitly assumes zero power when
     "off" (true for classical appliances like AC compressors, false for our GPUs: measured
     $P_b/P_{max}=0.81$, since continuous concurrency=20 decode load keeps the GPU far from
     idle even outside whale windows). **Corrected, validated endpoint:**
     $CF(N,s{=}0) \to P_b/P_{max} + (1-P_b/P_{max})\cdot p$ as $N\to\infty$ — predicted 0.9343,
     simulated 0.9423 at N=1000 (converging from above: 0.998→0.959→0.942 at N=10/100/1000).
     This is a substantive finding in its own right: LLM-serving GPUs under realistic
     continuous load look like "large baseline + small variable overlay," not a classical
     intermittent appliance — which structurally *limits* how much diversity-factor smoothing
     can ever help, independent of how well-diversified arrivals are.
   - **Error 2 — $s$ does not interpolate linearly.** Any nonzero probability of a
     fully-shared (synchronized) arrival event means such an event *will* eventually occur
     within a long-enough observation window, and when it does, every server bursts
     simultaneously ($K=N$ exactly) for its duration — so $CF$ jumps toward 1 abruptly once
     $s$ is large enough that a shared event is *likely* within the window, not gradually as
     $s$ increases. Demonstrated directly (N=200, 100s window): $s=0.05\to CF{\approx}0.98$
     (near the independent floor) but $s=0.10\to CF{\approx}0.99$ — a threshold effect, not a
     graded one. **This is arguably the more useful finding for the paper**: a *small*
     probability of correlated/synchronized whale arrival across a fleet is
     disproportionately dangerous for aggregate peak demand, regardless of how well the
     "normal" independent traffic is diversified — the same qualitative lesson as
     cold-load-pickup (item 2 below), now with a validated quantitative demonstration behind
     it rather than just an analogy.
   - **What to actually report in the paper:** the two validated endpoints (independent floor
     $\approx 0.93$; synchronized ceiling $=1$) plus the threshold-sensitivity finding, framed
     as "small correlation risk, large consequence" — not a smooth interpolating formula,
     which does not hold. A reference window must also be stated explicitly (fixed to our own
     measured trace's ~100s duration for the Monte Carlo) since coincidence factors are a
     property of the reference period as well as $N$ — matches real power-engineering
     practice of reporting different coincidence factors for 15-min/hourly/daily periods, and
     is why a percentile-based statistic (P99 of pooled aggregate demand) was used alongside
     the literal max — the max alone drifts upward as the window grows (an extreme-value
     effect, not a bug), while P99 is far more stable.
   - **Ramp-rate extension (2026-07-25, same script, `ramp_coincidence_factor`/
     `smoothed_aggregate`).** The level model above uses an instantaneous ON/OFF step; an
     earlier draft of a "ramp CF" built directly on that step was degenerate (its value scaled
     for free with $1/dt$, not tied to anything physical, so it was removed rather than
     reported). Replaced with a first-order-lag (RC) model per server, with time constant
     $\tau=(P_{max}-P_b)/\text{measured ramp rate}$ calibrated directly from the *real* §3
     numbers (mono 41.3 W/s, chunk=512 27.1 W/s) — so a single isolated simulated server's
     ramp matches what was actually measured on hardware, making $CF_{ramp}$ a meaningful
     ratio rather than a free parameter. **Two results, both checked at $n_{mc}=150$ for
     stability (not yet independently cross-validated the way the level-CF findings above
     were — treat as first-pass):**
     - **The single-GPU ramp-rate benefit attenuates, not disappears, at fleet scale.**
       Under fully independent arrivals ($s=0$), chunk=512's ~34.4% single-GPU ramp
       reduction becomes only a ~22-23% reduction in *realized aggregate fleet ramp rate*
       (stable at N=100 and N=1000: mono 1305/11920 W/s vs chunk=512 1002/9293 W/s). Reason:
       a slower individual transition means each server occupies its ramp phase for *longer*
       ($\tau$ scales inversely with rate), which raises the chance that an otherwise-
       independent neighboring server's ramp happens to overlap by pure coincidence — partly
       offsetting the per-server smoothing gain. This is a genuinely non-obvious systems point
       worth stating plainly: **individual-server smoothing does not translate 1:1 to
       fleet-level benefit; the translation is real but attenuated, and the attenuation
       mechanism (longer transition ⇒ larger overlap window) is itself an artifact of *how*
       chunking achieves its smoothing (a longer excursion), not a modeling error.**
     - **Ramp-rate coincidence does NOT show the same sharp threshold as level coincidence.**
       Sweeping $s$ finely (N=200): $CF_{ramp}$ rises smoothly (0.306→0.323→0.348→0.386→
       0.418→0.507→0.704 at $s=$0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5) — a graded curve, in clear
       contrast to the level CF's near-step jump from ~0.98 to ~0.99 between $s=0.05$ and
       $s=0.1$. Worth stating explicitly so the paper doesn't over-generalize "coincidence
       effects are always threshold-shaped" from the level result alone — the two quantities
       (instantaneous power level vs. its time-derivative) behave differently under the same
       correlation model, and only the level metric exhibits the cold-load-pickup-style cliff.
2. **Cold-load-pickup framing** (near-free, high payoff — **now quantitatively backed, not
   just an analogy**, per item 1's threshold-sensitivity result). Correlated whale arrivals
   across a fleet (synchronized batch job, viral prompt, shared cron trigger) collapsing
   $K(t)\to N$ simultaneously is structurally identical to cold-load pickup — the well-studied
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

1. ~~Resolve §4~~ — **DONE 2026-07-25**: committed to the ramp-rate/duty-cycle framing,
   confirmed via per-trial monotonicity check + whale-window disaggregation (§3).
2. Build the hybrid bench-serve harness (§7) — still open, only needed if/when a fresh
   experiment is run (not blocking the theory/writing work).
3. ~~Write out the §5.1 coincidence-factor derivation formally and validate it~~ —
   **DONE 2026-07-25** (`scripts/coincidence_factor_model.py`): validation caught two real
   errors in the naive closed form (missing baseline term, non-linear $s$-interpolation),
   both now corrected and confirmed against simulation. See §5 item 1 for the full result —
   this is genuinely stronger paper content than the original naive formula would have been.
4. Read B's two not-yet-fully-checked adjacent papers (Measurement of Generative AI Workload
   Power Profiles; TAPAS) before finalizing related-work claims. Still open.
5. ~~Separately re-derive/validate the ramp-rate ($dD/dt$) version of the coincidence-factor
   model~~ — **DONE 2026-07-25** (`scripts/coincidence_factor_model.py`,
   `ramp_coincidence_factor`/`smoothed_aggregate`): found two genuinely new, non-obvious
   results — (a) chunk=512's ~34.4% single-GPU ramp-rate benefit attenuates to only ~22-23% at
   fleet scale under independent arrivals (a longer individual transition raises the chance of
   coincidental overlap with neighboring servers' ramps, partly offsetting the per-server
   gain); (b) ramp-rate coincidence rises *smoothly* with the correlation parameter $s$, unlike
   level coincidence's sharp threshold jump — so "coincidence effects are threshold-shaped" is
   NOT a universal property, only a property of the level metric. See §5 item 1's new
   sub-bullet for the full numbers. Flagged as first-pass (not yet independently
   cross-validated the way the level-CF findings were).
6. Generate the two coincidence-factor figures (CF vs N at s=0; CF vs s at fixed N) — and now
   also candidate ramp-rate companion figures (fleet ramp reduction vs N; CF_ramp vs s
   contrasted with CF_max vs s) — for §6 of `paper.md`.
7. Start writing the empirical section (§2-3 numbers, now settled) and the theory section
   (§5, now validated for both level and ramp) — both are stable enough to write from; no need
   to wait on 2 or 4.


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
