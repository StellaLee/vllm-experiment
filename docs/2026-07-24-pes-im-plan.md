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
   7B model). The power trace gives C3 (the spike, and chunking flattening it); the
   integrated-energy totals from the same runs give C1 (energy-per-token roughly invariant
   across scheduling policy) essentially for free, without a separate sweep.
2. **C2 (chunking is energy-dominated) and E2/E4 are dropped, not just deprioritized.** There
   is no page budget for a third empirical claim or a Pareto figure in 5 pages, and C2 was
   already flagged as "at-risk, fails safe" — cutting it loses nothing load-bearing.
3. **The coincidence-factor grid-impact model is the spine and gets the most page budget.**
   This is the one piece that makes the paper a Topic-4 contribution instead of a systems
   paper with a wattmeter bolted on.

## Task list against 2026-08-15

### Days 1-3: Phase 0 (setup) + the one gate experiment

- [ ] NVML power logger (`nvidia-smi --query-gpu=power.draw --format=csv -lms 100`, plus
      `nvmlDeviceGetTotalEnergyConsumption` for integrated energy) wired into the replay
      harness so every request's window has both a power time series and a total-energy
      figure.
- [ ] Clock-lock (`nvidia-smi -lgc`) + thermal warmup wrapper — kills DVFS variance.
- [ ] **Check for wall-power/PDU access before committing to NVML alone.** A power-systems
      reviewer is likely to know NVML's limitations (board power, not full-system; sampling
      characteristics) and ask about it directly. Even one independent cross-check (IPMI
      sensors, a smart PDU reading) would matter more to this audience than to an MLSys one —
      worth a cheap early check.
- [ ] **Gate experiment:** Qwen2.5-Coder-7B, single GPU, bimodal short+whale mix (reuse
      `--whale-frac`/`--whale-min-chars`/`--whale-max-chars`, retuned for 7B since its
      prefill compute burst is smaller than 14B's — may need a larger whale or a different
      threshold to get a clean stall), mono vs. chunk=2048 vs. chunk=512, power + energy
      logged per run. **Go/no-go:** does mono show a clear, large power spike during the
      whale iteration that chunking measurably flattens, while integrated energy across the
      three arms stays roughly flat? If yes, this one experiment carries the whole empirical
      section. If it's noisy or small on 7B, fall back to the original herd-based power-spike
      design instead (more setup work, so decide this by day 3 at the latest to leave room for
      the rest of the schedule).

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
      (~0.5pp); the conservation statement, C1 (~0.5pp); the power-spike measurement, C3
      (~1-1.25pp); the coincidence-factor grid model (~1-1.5pp, the actual spine); discussion
      tied explicitly to Topic 4's language (stability, resilience, climate adaptation,
      demand-side management) plus conclusion (~0.5-0.75pp).
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
