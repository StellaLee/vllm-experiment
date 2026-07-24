# MLSys 2027 main-track plan (target: 2026-10-30)

*Supersedes `paper-submission-plan.md` (2026-07-07), which reflects an earlier paper framing
(pre-queuing-lens, pre-14B/TP2 work). This plan assumes the paper is already a solid workshop
submission as-is (§5.5–5.8 + the 7B artifact characterization) and lists what closes the gap
to a main-track submission. See the conversation/memory for the full reasoning behind each
item; this file is the checklist, not the argument.*

**Sequencing principle:** replicate before we ablate or extend scope — building on unreplicated
numbers risks redoing that work twice if a repeat trial moves a headline percentage. Phases
below are meant to run roughly in order, though phases 3–4 can overlap with phase 5 (no box
time needed there).

---

## Phase 1 — Replicate the three single-trial legs (Tier 1) — **DONE 2026-07-24**

- [x] §5.6 (`longprompt-tbt-win`): 2 more trials, same 3 arms. Caught and fixed a real
      context-overflow bug along the way (original whale bounds silently overflowed context
      at the true 3.235 chars/token ratio) — buggy runs archived, clean 3-trial set now on
      disk. TBT-max tight (80-85%/93-96%), TTFT interior-optimum confirmed and stronger than
      the original single trial suggested.
- [x] §5.7 (`adaptive-lpt`): 2 more trials of the same phase-scheduled workload. Tightest
      replication in the paper — S:TTFT std=0, W:max varies <1.1ms across all 3 trials.
      S:TTFT-ties-mono and W:max-matches-static-threshold both hold in every trial.
- [x] §5.8 (whale/threshold/concurrency tradeoff): 2 more trials at wf=05/conc=20 (clean) and
      wf=15/conc=20 (high-load), disaggregation built into the orchestrator as a first-class
      output. Both halves of the tradeoff (clean win at low load, broadening cost at high
      load) replicate; wf=15 shows visibly more trial-to-trial variance than wf=05, itself a
      coherent finding (closer to saturation).
- [x] Folded replicated numbers (mean ± std over 3 trials) back into `paper/paper.md` and all
      three tex sections (056/057/058), plus `08-limitations.tex` and the punch-list
      cross-check. Recompiled clean (no undefined refs). All three replications share one
      workload seed across trials — isolates serving/timing noise, not workload-draw
      variance. A fresh-seed replicate and the whale-fraction × whale-size grid (Phase 2)
      remain open across all three sections.

**Status: starting now.**

## Phase 2 — Quantitative validation of Eq. genuine (replaces the abandoned Cs² sweep) — **DONE 2026-07-24**

- [x] **Utilization ($\rho$) sweep**: wf=05 fixed, concurrency swept 20→38. The *absolute*
      S:TTFT reduction (threshold=512 vs. mono) grows monotonically with load (654→814→1247→
      2280ms) — the first clean quantitative validation of Eq. genuine's structure. Relative
      (%) deltas looked as noisy as the abandoned Cs² sweeps; the formula predicts an absolute
      term, and read that way the prediction holds. See `findings/2026-07-24-utilization-sweep.md`.
- [x] **Whale-fraction × whale-size grid**: 8k/14k tokens × 5%/15%/30% whale fraction, budget
      mechanism (mono vs. chunk=2048). TTFT and TBT-max wins are robust across the whole grid.
      Novel nuance found: TBT-p99 flips sign at wf=5% (mechanistically explained — mono's rare
      catastrophic events don't dominate the pooled top-1% cutoff at low whale frequency, chunk's
      more-frequent-smaller events do) — "chunking helps p99" is frequency-conditional even
      though "chunking bounds the max" is unconditional. See
      `findings/2026-07-24-whale-fraction-size-grid.md`.
- [x] **Cheap 3-value threshold check**: 512/2048/4096 at wf=15/conc=20. Larger threshold
      shrinks TTFT benefit toward zero without recovering TPOT cost (2048/4096 are as bad or
      worse on TPOT tail than 512) — **no headroom for a magnitude-adaptive LPT controller,
      closed, don't build one.** Extended with 8192/16384 arms: confirmed threshold=16384 ≈
      mono exactly, and located the capping→mono transition precisely between 4096 and 8192
      (well inside the whale's own size, not at it). See
      `findings/2026-07-24-threshold-value-check.md`.
- [x] **All three Phase 2 findings folded into the paper (2026-07-24)**: utilization sweep
      → new "first quantitative check" subsection in §5.5 (`055-queuing-lens.tex`); whale-size
      grid → new "mapping the frontier" paragraph in §5.6 (`056-genuine-term-demonstrated.tex`,
      including the TBT-p99 frequency-conditional sign-flip); threshold headroom closure →
      new "does a magnitude-adaptive version have headroom? No." paragraph in §5.7
      (`057-adaptive-cap.tex`). `paper.md` outline, tracker, and figure inventory updated to
      match. Recompiled clean (no undefined refs).

## Phase 3 — Broaden empirical scope

- [ ] Consider a third model-scale point beyond 7B/14B if box/model access allows (strengthens
      the model-scale axis from n=2 to n=3).
- [ ] Consider reviving the BurstGPT real-trace work (already exists from earlier in this
      project — see `project_burstgpt_experiment` memory) as external validity beyond synthetic
      whale injection, if it can be adapted to the whale/threshold framing.
- [ ] A100-class GPU generation: **not pursued** — no accessible path to this hardware. Keep as
      an explicit stated limitation rather than a task.

## Phase 4 — Ablations and robustness

- [ ] Ablate the adaptive-LPT controller's gate/protect values (4096/512) — currently carried
      over unmodified from the static-threshold grid, never independently tuned. Show they
      generalize (or characterize how performance degrades if they don't).
- [ ] Goodput / throughput-under-SLO framing: an open-loop rate sweep reporting max sustainable
      QPS under a P99-TBT SLO, matching Sarathi-Serve's own headline metric directly. Needs a
      fresh rate calibration given the overload lessons already learned this session (rate=0.05
      was the safe point for the 7B open-loop work; the 14B/TP2 whale workload will need its own
      calibration).

## Phase 5 — Positioning and writing (no box time; can run in parallel with Phases 2–4)

- [ ] Strawman-antagonist desk work (punch-list gating #2): reread how PRISM/Preble/Sarathi-Serve
      are characterized in `docs/related-work.md` and the tex positioning section; make sure
      we're not contesting a weakened version of the standard approach.
- [ ] Fresh literature check for anything published since the last review pass — the field moves
      fast; staleness here is an avoidable rejection reason.
- [ ] Re-examine the "why is this one paper" framing in the introduction now that the
      contribution list has grown to five items (already addressed once during the intro
      rewrite; worth a deliberate second look once Phases 1–4 land, since new results may shift
      the framing again).

## Phase 6 — Figures and camera-ready assembly

- [ ] Fig 3: stagger-window sweep (0/10/30/60s) — still needs the sweep, flagged since the
      original data-gaps list.
- [ ] Dedicated TBT-freeze figure for §5.6 (whale-aligned P99/max TBT, mono vs. chunk=2048 vs.
      chunk=512) — do not reuse `fig7_sensitive.png`, confirmed to be a different experiment.
- [ ] New 3-panel disaggregation figure for §5.8 (whale TTFT / short TTFT / short TPOT p95),
      mono vs. threshold=512, faceted by whale fraction.
- [ ] Full read-through pass once Phases 1–5 land, checking replicated numbers are consistently
      reflected everywhere (paper.md outline, tex sections, abstract, limitations, punch-list
      cross-check) — the same kind of staleness sweep already done twice this session
      (background.tex, introduction.tex) should be repeated once more before submission.

---

## Rough pacing against 2026-10-30

Not a hard schedule — box time is bursty and this runs alongside other priorities — but a
sanity check on scope: Phase 1 is roughly half a day of box time. Phase 2 is the largest
single chunk (utilization sweep + whale-size grid), likely the better part of a week of
box time plus analysis. Phases 3–4 are the most open-ended (model-scale/BurstGPT revival
could each be small or large depending on what's found). Phase 5 needs no box time and should
run continuously in the background of the others. Phase 6's figure work is quick once the
underlying data exists. Three months is enough room for all of this without rushing, provided
Phase 1 starts now rather than being deferred.
