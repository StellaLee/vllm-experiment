# IEEE PES IM 2027 (Beijing) — energy-reframe plan, grounded in the actual CFP

*Builds on `project_pes_im_energy_paper` memory (explored 2026-07-15, never started
executing) and the 2026-07-24 revision that introduced the whale-based power-spike gate.
This version replaces assumptions about venue/format with the actual call for papers
(https://www.pes-im.org/2027-beijing/program/call-for-papers/) and re-scopes accordingly —
the page limit in particular changes what's achievable in the time available.*

## The actual CFP, and what it changes

- **Theme:** "AI-Empowered Low-Carbon Energy Transition." **Dates:** Jan 10-13 2027, Beijing.
- **Deadlines:** full paper **2026-08-15**; provisional acceptance 2026-09-15; final paper
  2026-09-30; final acceptance 2026-10-07. There's a second checkpoint after the initial
  deadline — the Aug 15 submission needs to be accepted-worthy, not letter-perfect; more
  polish is possible before Sept 30.
- **Format: maximum 5 pages**, IEEE PES Authors Kit. This is workshop-length, not a
  9-11-page main-track paper — it changes the experiment plan as much as the writing plan.
- **9 topic areas**, and our fit is uneven across them:
  - Topics 1-2 ("AI-Empowered Power System Planning/Operation/Control," "Large Models,
    Foundation Models, and Intelligent Agents for Power Systems") are about **AI as a tool
    applied to power systems** — the opposite direction from our contribution. Do not try to
    force the paper into this framing; it's an easy mismatch for a reviewer to spot.
  - **Topic 4 ("Stability, Resilience, Reliability, Cybersecurity and Climate Adaptation") is
    the real fit.** New large, spiky computational loads affecting grid stability is a
    legitimate, active power-systems research area (the same category as EV-charging or
    cryptomining load studies) — this is where the paper should be submitted and positioned.
  - Topic 6 ("Digitalization, Data Infrastructure, Simulation Platforms and Standards") is a
    secondary, supporting fit for the coincidence-factor grid-impact model specifically.
  - **Positioning statement for the paper itself:** not "AI empowers the grid" but "the grid
    must plan for AI's growth" — AI/LLM-serving datacenters are a fast-growing, spiky
    electricity load, and this paper characterizes that load's power/energy behavior and
    gives grid planners a model for its aggregate impact. This is squarely a low-carbon
    energy-transition planning concern even though it isn't "AI-empowered" in the CFP's
    literal sense.

## Revised scope given the 5-page limit

A 5-page IEEE paper does not have room for a full E1-E4 experiment battery. The plan
collapses to one core experiment plus one modeling layer:

1. **One gate experiment carries both C1 and C3.** Log integrated energy *and* instantaneous
   power draw from the same whale-prefill run (mono vs. chunk=2048 vs. chunk=512, single GPU,
   7B model). **Update (2026-07-24) — ran, and C3 pivoted:** the power trace shows peak power
   does NOT flatten with chunking (all arms saturate ~460W near the GPU's ceiling), but a
   different, replicated (3 trials, holds per-trial not just pooled) effect does: chunking
   lowers mean ramp rate ~35% and duty-cycle-shifts toward fewer/longer/smoother episodes, at
   a modest energy cost (up to +3.7% at chunk=512). This ramp-rate/duty-cycle tradeoff, not
   peak-flattening, is now the paper's adopted headline — see
   `docs/2026-07-24-pes-im-prepare.md` §2-4 for the full data and reasoning. C1 still holds
   essentially for free from the same runs (energy-per-token +0.1% at chunk=2048, +3.7% at
   chunk=512 vs. mono).
2. **C2 (chunking is energy-dominated) and E2/E4 are dropped, not just deprioritized.** There
   is no page budget for a third empirical claim or a Pareto figure in 5 pages, and C2 was
   already flagged as "at-risk, fails safe" — cutting it loses nothing load-bearing.
3. **The coincidence-factor grid-impact model is the spine and gets the most page budget.**
   This is the one piece that makes the paper a Topic-4 contribution instead of a systems
   paper with a wattmeter bolted on.

## Task list against 2026-08-15

### Days 1-3: Phase 0 (setup) + the one gate experiment — **DONE 2026-07-24**

- [x] NVML power logger (`scripts/power_logger.py`, pynvml-based, samples power draw +
      cumulative energy counter + temperature at 50ms resolution, wall-clock timestamped) —
      standalone side-car rather than wired into the replay harness (correlates via existing
      per-request `ts`/`latency` fields instead, no harness changes needed).
- [x] Clock-lock (`nvidia-smi -lgc` to 3105MHz, max supported) + persistence mode + discarded
      warmup burst before each measured trial — kills DVFS variance.
- [x] **Wall-power/PDU access checked — found real IPMI/BMC chassis sensors**
      (`GPU_Power`, `Total_Power` on the H3C rack chassis), a genuine independent cross-check
      beyond NVML. Caveat: ~2.5s/query latency, so it's a steady-state cross-check only, not
      spike-resolution — snapshotted before/after each arm regardless.
- [x] **Gate experiment:** ran (`orchestrate_pesim_gate.sh`), 3 arms × 3 trials, n=194-196
      each. **Go/no-go outcome: pivoted, not failed.** Peak power did NOT flatten with
      chunking (all arms ~460W, near the GPU's power ceiling) — the original go/no-go
      criterion as literally stated was not met. But a different, cleanly replicated (holds
      per-trial, not just pooled) effect emerged: ramp rate and duty-cycle shape change
      monotonically with chunk size. Decided against the stated fallback (herd-based design)
      — this pivot is a *stronger* claim for a rigor-focused audience than the original
      peak-flattening hypothesis would have been. Full reasoning and data:
      `docs/2026-07-24-pes-im-prepare.md`.

### Days 2-5 (parallel with the above once a trace exists): coincidence-factor grid model

- [ ] No GPU time needed — can run alongside the gate experiment or the ongoing MLSys
      replication. Translate the measured single-GPU power trace $P(t)$ to datacenter scale
      via coincidence factor $CF = \text{peak(aggregate)} / \sum(\text{individual peaks})$,
      sweeping a synchronization parameter from independent arrivals (low CF, smooth) to full
      herd (CF→1, spike).
- [ ] Headline to aim for, in Topic-4 vocabulary: at fixed total energy (conservation),
      synchronization/scheduling changes aggregate peak demand and ramp rate by some multiple
      — a stability/resilience-relevant quantity, not just a latency number. Frame
      staggered/open-loop arrival and chunked scheduling explicitly as **demand-side
      management / demand-shaping levers** for a new load class, since that phrasing maps
      directly onto how this audience already thinks about EV charging and other flexible
      loads.
- [ ] State the model's caveats plainly (first-order model on measured single-GPU traces, not
      a measured datacenter; coincidence assumption is the key sensitivity — sweep it; PUE
      overhead and UPS/battery buffering attenuate real spikes, report GPU-level and
      PUE-scaled numbers separately).

### Days 5-11: writing (5 pages, start as soon as the gate trace exists — don't wait for everything to finish)

- [ ] Structure, roughly to the 5-page budget: motivation/grid relevance of AI-serving load
      growth (~0.75-1pp) — lean on the CFP's own "low-carbon energy transition" framing and
      the well-known, current concern about datacenter demand growth; brief methodology
      (~0.5pp); the conservation statement, C1 (~0.5pp); the ramp-rate/duty-cycle tradeoff,
      C3 revised (~1-1.25pp — chunking doesn't flatten peak power but lowers ramp rate ~35%
      and shifts duty cycle, at a modest energy cost — see
      `docs/2026-07-24-pes-im-prepare.md`); the coincidence-factor grid model (~1-1.5pp, the
      actual spine, now targeting aggregate ramp rate as well as peak); discussion tied
      explicitly to Topic 4's language (stability, resilience, climate adaptation,
      demand-side management, ramp-rate compliance) plus conclusion (~0.5-0.75pp).
- [ ] Related work needs to be power-systems literature (coincidence factor / load
      diversity-factor modeling, datacenter demand-response studies, EV-charging load
      studies as the nearest analogue), not the ML-systems literature the MLSys paper cites —
      essentially a from-scratch review, budget real time for it.
- [ ] Submit under **Topic 4**, explicitly. Don't hedge across Topics 1/2/4 — a reviewer
      reading a submission that claims a "AI-empowered" topic but delivers "AI as a grid load"
      content will read it as a mismatch, not a bonus.

### Days 11-15: buffer

- [ ] Revision pass against the actual 5-page limit (this is usually the hard part of a short
      paper — cutting, not adding). Remaining slack before Aug 15, plus the Sept 15/30
      checkpoint structure, means this doesn't all need to be finished to a polished state by
      the first deadline.

## Resource note

The current MLSys replication work is running on GPUs 0-1 (TP=2) as needed. The gate
experiment above only needs one GPU (7B, no TP), so it can run in parallel on a free GPU
without contending for the same resources.

## Honest fit caveat (sharpened, not softened, by reading the actual CFP)

The topic-list reality confirms rather than resolves the original "marginal fit" assessment:
none of the 9 topics are explicitly "grid impact of AI compute loads" — the closest is Topic
4's general stability/resilience framing. Submit here for a concrete strategic reason (a
second, energy-angle publication), with eyes open that this is a computer-systems
characterization wearing a power-systems hat, positioned as honestly as the framing allows
rather than oversold as a natural fit.
