# ACM e-Energy paper (routing / coincidence-factor)

Target: **ACM eEnergy 2027 Fall** (17th ACM International Conference on Future and
Sustainable Energy Systems). **Paper deadline: Sept 18, 2026** (user-confirmed 2026-09-02 —
~16 days out as of this writing). The CFP page itself is Cloudflare-blocked for every fetch
tool tried (both the 2026 and 2027 URLs, WebFetch and curl w/ browser UA, 403 each time);
deadline corroborated instead via the public HotCRP portal (`eenergy27fall.hotcrp.com`),
which shows a Sept 12, 2026 registration/abstract deadline ahead of the full-paper date.
Track TBC, likely "Systems and applied modeling" (carried over from the 2026 cycle via
search results, not 2027-specific).

**Format (via 2021/2022 CFPs, consistent both years, not yet 2027-confirmed): ACM sigconf
LaTeX (`acmart.cls`) with the `anonymous` option (double-blind review), full papers up to 10
pages 9-point ACM double-column excluding references/appendices.** Local toolchain ready
(`pdflatex`/`xelatex`, `acmart.cls` present). Double-blind means the submission PDF must not
name the author, institution, or any self-identifying repo URL.

**Central idea (pivoted a third time 2026-09-04 — see History below for earlier framings):**
lean theoretical, headline now leads with a SAFE result. Formalize power-aware LLM-serving
routing as a 3-resource (compute/load/power) egalitarian social welfare problem extending DRF
(Ghodsi et al. 2011) to a reactive power dimension. Prove the fully-sorted lexicographic
routing rule is Pareto-non-dominated at every decision (closed-form proof + 200,000-trial
brute-force verification). Prove two independently-motivated variants — a fixed-priority DRF
tie-break, and a power-extended LMETRIC score — each sacrifice that guarantee via distinct
mechanisms (explicit counterexamples, both verified in code; the LMETRIC-power variant's
violation rate is 12.8% of instances at a realistic cache-hit rate, not a rare corner case).
Validate on real 8×4090 hardware, under a per-GPU-calibrated ramp ceiling, that a Pareto-safe
DRF-family rule cleanly dominates the unsafe LMETRIC-power variant on every metric under
sustained fleet power pressure — the guarantee costs nothing there — and show the comparison
reverses under lighter, cache-hit-dominated traffic exactly where the LMETRIC-power variant's
own mechanism predicts it should. Full draft: `paper.md`. Source of record for all underlying
numbers: `../findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md`.

**Reason for the third pivot**: the prior headline (an unsafe variant "winning" empirically)
depended on numbers from a uniform, shared 450 W/s ramp-ceiling constant. A dedicated
per-GPU recalibration campaign (findings.md, Update 2026-09-04) found real per-GPU ceiling
variance of 34%, and re-running the full arm comparison under corrected calibration flipped
or erased apparent dominance relationships in 4 of 6 tested conditions — always in the
direction of making unsafe arms look better than they should. The specific arm that headlined
the prior draft (`drf_power_tiebreak`, the fixed-priority named rule) was never itself
re-validated under the fix. Rather than submit on an unvalidated number, we lead with the
comparison that is both provably safe and already validated under corrected calibration.

**Status:** theory (two proofs, two verified counterexamples) and the new headline
experimental result are both drafted with real content in `paper.md` §4-5. Sections 1/2/6/7
are structural, not fully polished prose, though contributions/discussion/conclusion have
been updated to match the new headline. Compiles cleanly to a 6-page PDF (`paper.tex` →
`paper.pdf`), well within the 10pp limit. Scope is deliberately narrow by design (one clean
theory-motivated winning condition plus an honest boundary characterization, not an
exhaustive empirical survey) — the project research log has substantially more data (other
load conditions, real BurstGPT traffic, 11 other routing policies) that is out of this
paper's current scope but available if the scope is revisited. Open: `figs/ramp_comparison.pdf`
still depicts the old Heavy/Matched sorted-vs-named comparison and needs regeneration for the
new Heavy/Closed-Loop 4-arm headline (or removal) — the current `paper.tex` build omits the
figure entirely rather than show a stale one.

### History: the original CF-decorrelation framing

The project started (2026-08-30) as a hardware-measured extension of the PES-IM
coincidence-factor (CF) theory — validated there only via Monte Carlo simulation up to
N=20000. The original idea: deliberately spread whale (long-prompt) requests across
replicas instead of letting them cluster, and measure whether this decorrelates per-GPU
power draw (lower CF) versus clustered/default routing, feeding into (not replacing) the
existing Monte Carlo extrapolation.

That framing is **not** what got directly tested — the whale-spreading arms
(`p2c_whale`/`whale_argmin`) were built and evaluated against the full Pareto-frontier
metric set (TTFT, TBT, ramp, coincidence, latency) rather than isolated as a clean
clustering-vs-spreading A/B, and both initially-promising results reversed under 3×
replication. Cross-GPU ramp coincidence survives as one of the frontier's tracked metrics,
but it is no longer the paper's organizing thesis.

### History, part 2: the Pareto-frontier-survey framing

The first pivot (above) reframed the paper around an empirical Pareto-frontier
characterization: 12 routing policies, no dominant one, the trade-off itself as the finding.
That framing surfaced a real problem during scoping — real BurstGPT traffic at 15-25 req/s
shows `round_robin` beating every power-aware policy tested, including the DRF-family
(mechanism still undiagnosed) — which would have made "validate broadly on real traffic" a
weak link in an empirical-survey-shaped paper. The second pivot (current, above) resolves
this by leaning theoretical instead: the paper's central claims are now a proof and a
verified counterexample, with experimental validation scoped narrowly to the one condition
that directly exercises the theory, rather than an exhaustive empirical survey that would
need to reckon with every condition tested. The real-BurstGPT finding is not discarded — it
remains fully documented in the research log — it is simply out of this narrower paper's
scope rather than a threat to its central claim.

**Open questions before design:**
- Confirm the actual 2027 deadline, track, and topics list from the live CFP page directly.
  Re-checked 2026-09-02: the 2027 CFP is not yet published (confirmed via search, not just
  the earlier 403 fetch failure — e-Energy 2026's own Winter-cycle rejects-welcome note
  implies the 2027 CFP follows after e-Energy 2026's cycles close). Nothing actionable here
  until ACM publishes it; re-check periodically rather than blocking on it.
- Router: minimal request router in front of N replicas. 7B/TP=1 → up to 8 independent
  replicas; 14B/TP=2 (the size used in the whale-injection work) → up to 4.
- Routing policies to compare: whale-clustering (status quo / no coordination) vs.
  deliberate whale-spreading.
- Physical caveat: 8×4090 in one chassis likely share upstream PDU/PSU, so this isn't 8
  independent grid circuits — measure per-GPU power independently, then feed into the
  existing Monte Carlo model for any data-center-scale claim rather than treating N=8 (or
  4) as the scale claim itself. Small N also means noisy CF estimates — plan for
  multi-trial replication from the start, not as an afterthought.

Sibling papers: `../paper-pes-im/` (IEEE PES IM, same CF theory, simulation-only),
`../paper-mlsys/` (SIGMETRICS, latency-focused, whale-aware budget controller — a different
knob, not directly reused here).
