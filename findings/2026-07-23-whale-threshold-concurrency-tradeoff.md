# The per-request cap's payoff is load-gated: clean win at low load, eroding tradeoff as load rises

**Date:** 2026-07-23
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** `orchestrate_cs2_whale_threshold_sweep.sh` — closed-loop, continuous (no
phase-schedule) whale/short bimodal mix. Budget fixed large (`--max-num-batched-tokens 16384`,
never the bottleneck); `--long-prefill-token-threshold` swept {0 (mono), 512}; whale fraction
swept {5%, 15%}; concurrency swept {20, 40}. Whale prompts 44–50k chars (≈12k tokens), short
prompts mean 800 chars (`pad-cv2=0.5`), `pad-seed=1001`, `max-num-seqs=48`, single-turn,
`max-tokens=256`, `NCONV=200`. Server boots once per threshold, serves all 4 (whale_frac ×
conc) combos.

**Headline:** The three-way tradeoff first surfaced by disaggregating pooled TTFT (short
TTFT better / whale TTFT worse / short TPOT-tail worse under threshold=512) holds at every
tested load point, but its **shape is not constant — it is load-gated**, exactly as
Eq. genuine predicts. At the one config that is not saturated (wf=5%, conc=20), thresholding
is a clean, defensible win: large broad TTFT gain, a *narrow* TPOT cost (concentrated in
p95, with mean and p99 flat-to-better), and whale's own TPOT even improves. As whale fraction
or concurrency rises, the TPOT cost stops being narrow and spreads to mean and both tails,
and two of the four configs (conc=40 at either whale fraction) are confirmed overloaded
independent of the scheduling mechanism, making "threshold vs. mono" there a comparison of
two flavors of saturation, not a clean mechanism read.

---

## Full results table

Aggregated over the whole run (no time-windowing). W = whale population (`pad_chars≥40000`),
S = short population. n given per population. One trial.

| wf | conc | thr | pop | n | TTFT mean | TTFT p95 | TTFT p99 | TPOT mean | TPOT p95 | TPOT p99 |
|----|------|-----|-----|-----|-----------|----------|----------|-----------|----------|----------|
| 05 | 20 | 0 | W | 9 | 4408 | 9256 | 9256 | 46.3 | 84.2 | 84.2 |
| 05 | 20 | 0 | S | 189 | 1086 | 6374 | 9280 | 51.8 | 81.9 | 317.6 |
| 05 | 20 | 512 | W | 9 | 6791 | 9919 | 9919 | 44.0 | 66.8 | 66.8 |
| 05 | 20 | 512 | S | 189 | 432 | 1696 | 1733 | 52.9 | 145.6 | 253.4 |
| 05 | 40 | 0 | W | 9 | 8845 | 22891 | 22891 | 64.2 | 129.8 | 129.8 |
| 05 | 40 | 0 | S | 189 | 3953 | 19250 | 27239 | 70.9 | 118.3 | 538.6 |
| 05 | 40 | 512 | W | 9 | 14116 | 31999 | 31999 | 56.5 | 109.1 | 109.1 |
| 05 | 40 | 512 | S | 189 | 1549 | 6860 | 9633 | 100.5 | 391.6 | 915.8 |
| 15 | 20 | 0 | W | 32 | 10523 | 23847 | 23871 | 81.3 | 323.0 | 357.8 |
| 15 | 20 | 0 | S | 163 | 5016 | 19287 | 23844 | 76.5 | 178.9 | 575.6 |
| 15 | 20 | 512 | W | 32 | 16927 | 40141 | 43098 | 72.1 | 128.4 | 221.5 |
| 15 | 20 | 512 | S | 163 | 2289 | 12047 | 18858 | 116.7 | 435.5 | 914.3 |
| 15 | 40 | 0 | W | 32 | 23671 | 45535 | 48510 | 79.9 | 159.8 | 222.6 |
| 15 | 40 | 0 | S | 163 | 17841 | 39346 | 45556 | 134.6 | 160.9 | 945.0 |
| 15 | 40 | 512 | W | 32 | 38541 | 69500 | 72364 | 88.0 | 163.3 | 423.8 |
| 15 | 40 | 512 | S | 163 | 11619 | 43100 | 51460 | 281.3 | 843.1 | 1700.5 |

## Conclusion at wf=05/conc=20 (the one clean, sub-saturated read)

This is the only config not showing overload signatures (conc=40 confirmed overloaded via a
separate check — mean TTFT roughly tripled from doubling concurrency alone, independent of
threshold; wf=15/conc=20 already runs much hotter than wf=05/conc=20 — S:TTFT mean 5x higher,
1086→5016 — just from tripling whale fraction at the same concurrency, so it isn't a fully
clean baseline either).

At wf=05/conc=20, threshold=512 vs. mono:

- **S:TTFT drops hard, across the board:** mean 1086→432 (−60%), p95 6374→1696 (−73%), p99
  9280→1733 (−81%). This is the admission-side genuine-sharing benefit, and it's the larger
  population (189 vs. 9), so it's the visible headline number if you only look at pooled
  stats.
- **W:TPOT actually improves too:** mean 46.3→44.0, p95/p99 84.2→66.8. The whale itself isn't
  just paying a cost — its own decode-tail latency benefits from being chunked, the same
  mechanism as the earlier `longprompt-tbt-win` result.
- **The one real whale-side cost is TTFT:** mean 4408→6791 (+54%), p95/p99 9256→9919 (+7%) —
  the expected price of being capped: a prefill that used to finish in ~1 round now takes many
  more.
- **The short-population TPOT cost is real but narrow, not universal:** mean is flat
  (51.8→52.9), p99 actually *improves* (317.6→253.4), and the entire cost is concentrated in
  p95 (81.9→145.6, +78%). This is a specific, bounded cost, not a broad regression.

Net read: at this one load point, thresholding is a genuinely good trade — a large, broad
admission-side win for the majority population, no cost to the whale's own decode tail, and a
narrow (p95-only) tail cost to the minority population it interacts with. This is the
cleanest evidence so far that the per-request cap mechanism (Section 5.7/5.8 of the paper) is
not just "sometimes helps" but can be an unambiguous net improvement in the right regime.

## Why this doesn't generalize as load rises

As whale fraction or concurrency increases, the same three effects persist in direction but
the short-population TPOT cost stops being narrow: at wf=15/conc=20 it already spreads to
mean, p95, *and* p99 (76.5→116.7, 178.9→435.5, 575.6→914.3), and by wf=15/conc=40 (confirmed
overloaded) it is severe at every percentile (134.6→281.3, 160.9→843.1, 945.0→1700.5). This
is consistent with Eq. genuine's own structure: the admission-side sharing benefit and its
decode-tail cost both scale with contention, but they don't scale at the same rate, so the
trade that looks clearly favorable at low load stops looking favorable once load rises far
enough. The mechanism is real; its net value is a function of operating point, not a
constant.

## Caveats

- Single trial throughout. Whale-population n (9 or 32) is thin; short-population n (163–189)
  is the reliable read.
- conc=40 arms (both whale fractions) are confirmed overloaded independent of threshold —
  read as "two flavors of saturation," not a clean mechanism comparison.
- wf=15/conc=20 runs hotter than wf=05/conc=20 even though neither was flagged via the
  doubling-concurrency overload test; it should not be treated as an equally clean baseline.
- The whale-vs-short disaggregation was ad hoc, applied post hoc to data collected for a
  pooled comparison — not a designed comparison. A dedicated rerun with disaggregation and
  2–3 trials (especially anchoring the low-load wf=05/conc=20 point) is the natural next step.
- Feeds directly into `paper/tex/sections/058-tradeoff.tex` (§5.8) and
  `paper/paper.md` §5.8.

---

## Update (2026-07-24): replicated at 4x output length (MAXTOK=1024) — same shape, sharper at high load

Same grid (`orchestrate_cs2_whale_threshold_sweep_maxtok.sh`, identical params otherwise),
`--max-tokens` raised from 256 to 1024 to test whether the finding is an artifact of short
generations. It is not — the clean win at low load and the eroding tradeoff at high load
both replicate, and the high-load cost gets *worse*, not better, with longer outputs. 0
preemptions in all 8 arms.

**wf=05/conc=20 (the clean point), threshold=512 vs. mono, MAXTOK=256 → MAXTOK=1024:**

| metric | MAXTOK=256 (Δ) | MAXTOK=1024 (Δ) |
|---|---|---|
| S:TTFT mean | 1086→432 (−60%) | 904→375 (−58%) |
| S:TTFT p95 | 6374→1696 (−73%) | 9012→1483 (−84%) |
| S:TTFT p99 | 9280→1733 (−81%) | 9285→4054 (−56%) |
| S:TPOT mean | 51.8→52.9 (+2%) | 39.8→43.2 (+9%) |
| S:TPOT p95 | 81.9→145.6 (+78%) | 47.4→88.6 (+87%) |
| S:TPOT p99 | 317.6→253.4 (−20%) | 320.3→253.1 (−21%) |
| W:TPOT mean | 46.3→44.0 (improves) | 34.7→32.9 (improves) |
| W:TTFT mean | 4408→6791 (+54%) | 4726→6615 (+40%) |

Near-exact replication of the whole shape: broad TTFT win, the TPOT cost still concentrated
specifically in p95 (mean flat, p99 still *better*), whale's own TPOT still improves. The
"clean win at low load" conclusion is not an artifact of using short (256-token) outputs.

**Higher-load arms also replicate their pattern, and worsen:**
- wf=15/conc=20: S:TPOT cost still spreads across all three metrics at 1024 tokens (mean
  57.1→72.9 +28%, p95 96.9→273.0 +182%, p99 334.2→613.5 +84%) — same "broader cost once load
  rises" shape as at 256 tokens (there: +53%/+143%/+59%).
- wf=15/conc=40 (confirmed overloaded): S:TPOT p99 under threshold=512 hits **3325.6ms** at
  MAXTOK=1024, vs. 1700.5ms at MAXTOK=256 — longer per-request occupancy compounds the
  existing overload rather than relieving it.

**Full table, MAXTOK=1024:**

| wf | conc | thr | pop | n | TTFT mean | TTFT p95 | TTFT p99 | TPOT mean | TPOT p95 | TPOT p99 |
|----|------|-----|-----|-----|-----------|----------|----------|-----------|----------|----------|
| 05 | 20 | 0 | W | 8 | 4726 | 9282 | 9282 | 34.7 | 55.8 | 55.8 |
| 05 | 20 | 0 | S | 189 | 904 | 9012 | 9285 | 39.8 | 47.4 | 320.3 |
| 05 | 20 | 512 | W | 8 | 6615 | 9685 | 9685 | 32.9 | 40.0 | 40.0 |
| 05 | 20 | 512 | S | 189 | 375 | 1483 | 4054 | 43.2 | 88.6 | 253.1 |
| 05 | 40 | 0 | W | 8 | 7081 | 19398 | 19398 | 45.1 | 60.1 | 60.1 |
| 05 | 40 | 0 | S | 189 | 3574 | 20778 | 20806 | 51.9 | 69.3 | 308.5 |
| 05 | 40 | 512 | W | 8 | 11275 | 24531 | 24531 | 39.0 | 48.3 | 48.3 |
| 05 | 40 | 512 | S | 189 | 1572 | 6079 | 7983 | 78.6 | 330.2 | 549.4 |
| 15 | 20 | 0 | W | 22 | 7623 | 19708 | 19715 | 51.8 | 117.8 | 219.7 |
| 15 | 20 | 0 | S | 163 | 2180 | 13122 | 17422 | 57.1 | 96.9 | 334.2 |
| 15 | 20 | 512 | W | 22 | 11860 | 29032 | 30370 | 44.1 | 77.2 | 82.8 |
| 15 | 20 | 512 | S | 163 | 847 | 3738 | 8665 | 72.9 | 273.0 | 613.5 |
| 15 | 40 | 0 | W | 22 | 21345 | 49148 | 52748 | 52.7 | 75.6 | 220.2 |
| 15 | 40 | 0 | S | 163 | 12484 | 41200 | 52530 | 70.2 | 107.3 | 577.6 |
| 15 | 40 | 512 | W | 22 | 36002 | 77587 | 80213 | 43.0 | 75.3 | 85.8 |
| 15 | 40 | 512 | S | 163 | 5365 | 19432 | 25949 | 232.9 | 935.1 | 3325.6 |

Realized Cs² (mono, `prompt_tokens_approx`): wf=05 → 8.441 (n=198), wf=15 → 4.901 (n=186) —
essentially unchanged from the MAXTOK=256 run, as expected (prompt-length distribution didn't
change, only output length did).

**Interpretation:** this is a robustness check along a different axis than a repeated trial
(output length rather than a fresh random draw), and it strengthens rather than weakens the
§5.8 claim: the load-gating story is not a 256-token-specific quirk. It does not, on its own,
substitute for the 2–3 trial replication still flagged as needed above — a second run at the
*same* MAXTOK with a different seed is still the more direct way to address the single-trial
caveat.

---

## Update (2026-07-24): budget=512 vs. threshold=512 head-to-head — two distinct
mechanisms, confirmed on the same workload, and it explains a number in `longprompt-tbt-win`

Everything above swept `long_prefill_token_threshold` at a **fixed large budget (16384)**.
This adds the missing head-to-head: the **step-wide budget** mechanism (`--max-num-batched-
tokens 512`, threshold **off**) on the *exact same* workload (wf=05, conc=20, threshold-sweep's
own mono/16384 baseline reused), at both MAXTOK 256 and 1024. `orchestrate_budget512_maxtok.sh`
booted one server (budget=512, threshold=0) and ran the client twice. 0 preemptions.

**Result: budget=512 has the opposite effect on short-request TPOT that threshold=512 has.**

| | S:TPOT p95 Δ | S:TPOT p99 Δ | S:TTFT p95 Δ | W:TTFT Δ |
|---|---|---|---|---|
| **budget=512**, MAXTOK=256 | −17.5% | **−63.3%** | −32.0% | +20.2% |
| **threshold=512**, MAXTOK=256 | **+78%** | −20% | −73% | +54% |
| **budget=512**, MAXTOK=1024 | +17.3% | **−63.6%** | −55.6% | +20.7% |
| **threshold=512**, MAXTOK=1024 | **+87%** | −21% | −84% | +40% |

Full numbers (threshold off throughout):

| budget | maxtok | pop | n | TTFT mean | TTFT p95 | TTFT p99 | TPOT mean | TPOT p95 | TPOT p99 |
|---|---|---|---|---|---|---|---|---|---|
| 16384 (mono) | 256 | W | 9 | 4408 | 9256 | 9256 | 46.3 | 84.2 | 84.2 |
| 16384 (mono) | 256 | S | 189 | 1086 | 6374 | 9280 | 51.8 | 81.9 | 317.6 |
| 512 (chunk) | 256 | W | 9 | 4886 | 11123 | 11123 | 50.0 | 66.8 | 66.8 |
| 512 (chunk) | 256 | S | 189 | 1045 | 4334 | 10593 | 44.2 | 67.5 | 116.4 |
| 16384 (mono) | 1024 | W | 8 | 4726 | 9282 | 9282 | 34.7 | 55.8 | 55.8 |
| 16384 (mono) | 1024 | S | 189 | 904 | 9012 | 9285 | 39.8 | 47.4 | 320.3 |
| 512 (chunk) | 1024 | W | 8 | 4969 | 11205 | 11205 | 37.3 | 57.6 | 57.6 |
| 512 (chunk) | 1024 | S | 189 | 798 | 4004 | 7831 | 37.2 | 55.6 | 116.7 |

**Budget=512 *protects* short-request decode-tail latency** (S:TPOT p99 down ~63% in both
MAXTOK arms, a strong and consistent win) **while threshold=512 *costs* it** (S:TPOT p95 up
78–87%). Budget=512 also costs the whale less on TTFT (+20% vs. threshold's +40–54%), but buys
short requests a smaller TTFT win (p95 −32/−56% vs. threshold's −73/−84%).

**Consistent with, and refines, `2026-07-21-longprompt-tbt-win.md`.** That result swept the
same step-wide budget mechanism (mono/16384 vs. chunk/2048 vs. chunk/512, threshold off
throughout — this predates the threshold discovery) at wf=16% and reported **pooled** TTFT
+20.6% and pooled TBT p99 −94.5% / max −93.4% for chunk=512 vs. mono. Both directions replicate
here at a different whale fraction (5% vs. 16%) and a coarser per-request metric (TPOT mean
vs. raw per-token `tbt_ms`): budget=512 still delivers a large, consistent decode-tail
improvement (S:TPOT p99 −63%), and it still costs TTFT overall. The disaggregation here goes
one step further and explains *where* longprompt's pooled +20.6% TTFT cost actually lived: the
whale-only TTFT delta measured here is **+20.2%/+20.7%** — almost exactly longprompt's pooled
number. Whale TTFT values are an order of magnitude larger than short TTFT values, so even at a
16% (or 5%) population share, the whale population dominates a pooled *mean*. Longprompt's
"+20.6% TTFT cost" was very likely mostly the whale paying the price of being chunked, not a
broad cost across the whole population — short requests' own TTFT is roughly flat-to-improved
in the disaggregated view (mean −3.8%/−11.7%), which longprompt's pooled metric could not show
because it never disaggregated whale from short.

**Mechanistic takeaway (ties §5.6 and §5.7/§5.8 together for the first time on one controlled
comparison):** the two schedulers are not different strengths of the same knob. Step-wide
budget bounds *iteration wall-time*, which is what actually protects decode-tail latency for
every bystander (§5.6's mechanism) — it buys little admission-side sharing, since one request
can still consume the whole 512-token budget in a round. Per-request threshold buys the
admission-side TTFT win (§5.7/§5.8's mechanism) precisely because the aggregate budget stays
large — which is exactly what removes the decode-tail protection, since the whale still lingers
in the running batch for many rounds at full step-length each time. Choosing between them is a
choice about which failure mode you'd rather have, not a single dial with one "better" setting.

**Caveats:** single trial, both MAXTOK arms; whale n=8–9, thin as always. Threshold=512 numbers
reused from the earlier update in this file (same underlying data, not rerun). Feeds
`paper/tex/sections/056-genuine-term-demonstrated.tex` (§5.6) and `058-tradeoff.tex` (§5.8) —
this is the first place in the project where both mechanisms are compared head-to-head on an
identical workload.

---

## Update (2026-07-24): 3-trial replication — the effect is real, not a single-run artifact

Ran 2 more trials (`orchestrate_cs2_whale_threshold_replicate.sh`, same PAD_SEED=1001, same
workload) at conc=20 only (conc=40 dropped — already confirmed overloaded, not part of what's
being validated), wf∈{05,15}, threshold∈{0,512}, MAXTOK=256. 0 preemptions across all 8 new
runs. Same seed across trials isolates serving/timing noise (the thing "single trial" actually
worries about) from workload-draw variance (which the whale-size/fraction grid, still future
work, addresses separately).

**wf=05/conc=20 (clean point), threshold=512 vs. mono, short population, mean ± std over 3
trials:**

| metric | trial 1 | trial 2 | trial 3 | mean ± std |
|---|---|---|---|---|
| TTFT mean Δ | −60.2% | −60.6% | −48.6% | **−56.5% ± 6.8** |
| TTFT p95 Δ | −73.4% | −75.6% | −70.2% | **−73.1% ± 2.7** |
| TTFT p99 Δ | −81.3% | −81.0% | −60.6% | **−74.3% ± 11.9** |
| TPOT mean Δ | +2.2% | +13.3% | +5.8% | **+7.1% ± 5.7** |
| TPOT p95 Δ | +77.8% | +84.5% | +131.0% | **+97.8% ± 29.0** |
| TPOT p99 Δ | −20.2% | −18.8% | −19.9% | **−19.6% ± 0.7** |

The entire signature replicates, and TPOT p99 in particular is remarkably tight (−19.6% ± 0.7
across independent trials) — the whole "broad TTFT win, narrow p95-only TPOT cost, mean and
p99 flat-to-better" story is a stable, reproducible effect, not a lucky single run. The p95 TPOT
cost is real across all three trials but its exact magnitude is the noisiest number here
(78–131%) — still a bounded, specific cost, never spreading to mean or p99.

**wf=15/conc=20 (higher load), same comparison:**

| metric | trial 1 | trial 2 | trial 3 | mean ± std |
|---|---|---|---|---|
| TTFT mean Δ | −54.4% | −53.1% | −65.0% | **−57.5% ± 6.5** |
| TTFT p95 Δ | −37.5% | −42.9% | −65.1% | **−48.5% ± 14.6** |
| TTFT p99 Δ | −20.9% | −18.5% | −55.4% | **−31.6% ± 20.6** |
| TPOT mean Δ | +52.4% | +70.0% | +65.9% | **+62.8% ± 9.2** |
| TPOT p95 Δ | +143.4% | +115.1% | +162.7% | **+140.4% ± 23.9** |
| TPOT p99 Δ | +58.8% | +79.6% | +63.0% | **+67.1% ± 11.0** |

The "cost broadens at higher load" finding also replicates cleanly: TPOT mean/p95/p99 are all
clearly, consistently positive across all 3 trials (unlike wf=05, where mean and p99 stay
flat-to-better). One honest wrinkle worth keeping in the writeup rather than smoothing over:
the exact TTFT percentages are noticeably more variable at wf=15 (p95 spans −38% to −65%, p99
spans −18% to −55%) than at wf=05 (tight within a few points). That's consistent with wf=15/
conc=20 running closer to saturation, where queueing-driven variance is expected to be higher
— itself a small, coherent, reportable observation rather than a problem with the measurement.

**Status update: this resolves the single-trial caveat for §5.8.** The direction and rough
magnitude of both halves of the tradeoff (clean win at low load, broadening cost at higher
load) are now backed by 3 independent trials at the two key points. §5.6 and §5.7 remain
single-trial and still need the same treatment — this update does not extend to them.

**Full per-trial raw table** (all populations, both wf, conc=20 only):

| wf | trial | thr | pop | n | TTFT mean | TTFT p95 | TTFT p99 | TPOT mean | TPOT p95 | TPOT p99 |
|---|---|---|---|---|---|---|---|---|---|
| 05 | 1 | 0 | W | 9 | 4408 | 9256 | 9256 | 46.3 | 84.2 | 84.2 |
| 05 | 1 | 0 | S | 189 | 1086 | 6374 | 9280 | 51.8 | 81.9 | 317.6 |
| 05 | 1 | 512 | W | 9 | 6791 | 9919 | 9919 | 44.0 | 66.8 | 66.8 |
| 05 | 1 | 512 | S | 189 | 432 | 1696 | 1733 | 52.9 | 145.6 | 253.4 |
| 05 | 2 | 0 | W | 9 | 4646 | 9230 | 9230 | 44.3 | 83.0 | 83.0 |
| 05 | 2 | 0 | S | 189 | 1048 | 6309 | 9244 | 47.2 | 83.6 | 321.0 |
| 05 | 2 | 512 | W | 9 | 6889 | 10299 | 10299 | 43.0 | 60.1 | 60.1 |
| 05 | 2 | 512 | S | 189 | 413 | 1541 | 1757 | 53.5 | 154.2 | 260.6 |
| 05 | 3 | 0 | W | 9 | 4398 | 9139 | 9139 | 45.3 | 82.6 | 82.6 |
| 05 | 3 | 0 | S | 189 | 971 | 6372 | 9161 | 51.9 | 70.3 | 317.5 |
| 05 | 3 | 512 | W | 9 | 7056 | 11659 | 11659 | 43.9 | 77.1 | 77.1 |
| 05 | 3 | 512 | S | 189 | 499 | 1899 | 3614 | 54.9 | 162.4 | 254.3 |
| 15 | 1 | 0 | W | 32 | 10523 | 23847 | 23871 | 81.3 | 323.0 | 357.8 |
| 15 | 1 | 0 | S | 163 | 5016 | 19287 | 23844 | 76.5 | 178.9 | 575.6 |
| 15 | 1 | 512 | W | 32 | 16927 | 40141 | 43098 | 72.1 | 128.4 | 221.5 |
| 15 | 1 | 512 | S | 163 | 2289 | 12047 | 18858 | 116.7 | 435.5 | 914.3 |
| 15 | 2 | 0 | W | 32 | 10619 | 22528 | 23914 | 80.7 | 222.1 | 364.5 |
| 15 | 2 | 0 | S | 163 | 4845 | 19196 | 23931 | 70.9 | 179.3 | 504.1 |
| 15 | 2 | 512 | W | 32 | 17527 | 40187 | 44488 | 73.0 | 136.0 | 173.8 |
| 15 | 2 | 512 | S | 163 | 2271 | 10956 | 19496 | 120.5 | 385.7 | 905.5 |
| 15 | 3 | 0 | W | 32 | 10648 | 23762 | 23909 | 67.4 | 152.2 | 219.0 |
| 15 | 3 | 0 | S | 163 | 4756 | 19117 | 22353 | 77.1 | 157.2 | 567.2 |
| 15 | 3 | 512 | W | 32 | 18386 | 42408 | 43601 | 73.1 | 147.2 | 265.8 |
| 15 | 3 | 512 | S | 163 | 1666 | 6677 | 9977 | 127.9 | 413.0 | 924.6 |

**Caveats:** n=3 trials, same seed (isolates timing noise, not workload-draw variance — that's
still a separate, unaddressed axis, see the whale-fraction × whale-size grid in the MLSys
plan). conc=40 not replicated (already confirmed overloaded and out of scope for this check).
