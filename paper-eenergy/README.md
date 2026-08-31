# ACM e-Energy paper (routing / coincidence-factor)

Target: **ACM e-Energy 2027**, due ~Jan 2027 (not yet confirmed from the live CFP — the
2027 and 2026 CFP pages both returned HTTP 403 on fetch 2026-08-30; scope/track inference
below is carried over from the 2026 cycle via search results, not read verbatim). Likely
track: "Systems and applied modeling" (system design/implementation/evaluation backed by
empirical data, or real-world energy/computing system modeling).

**Central idea (agreed 2026-08-30, nothing built yet):** extend the PES-IM
coincidence-factor (CF) theory — validated so far only via Monte Carlo simulation up to
N=20000 — with a real routing mechanism on the 8×4090 server. Deliberately spread whale
(long-prompt) requests across replicas instead of letting them cluster, and measure whether
this measurably decorrelates per-GPU power draw (lower CF) versus clustered/default
routing. This would be the first *hardware-measured* CF validation, feeding into (not
replacing) the existing Monte Carlo extrapolation to data-center scale.

**Status:** direction only. No client proxy/router exists yet — that's the first piece of
infra needed; see `../orchestrate/eenergy/`.

**Open questions before design:**
- Confirm the actual 2027 deadline, track, and topics list from the live CFP page directly.
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
