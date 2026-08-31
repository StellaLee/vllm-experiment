# [Working title] Routing-Induced Power Decorrelation Across a Multi-GPU LLM-Serving Fleet

*Target: ACM e-Energy 2027, track TBC (likely "Systems and applied modeling"). Deadline
TBC — see `README.md` for why it isn't confirmed yet.*

*Status: not started. No experiments run, no numbers exist. This file is a section
skeleton only, to be filled in once the router and first measurements exist.*

---

## Abstract

TBD once the routing mechanism is built and measured.

## 1. Introduction

Motivation: extend the coincidence-factor (CF) theory from the sibling PES-IM paper
(`../paper-pes-im/`) — so far validated only via Monte Carlo simulation — with a real,
hardware-measured routing intervention.

## 2. Background / Related Work

Carry over CF framing from `../paper-pes-im/paper.md` §2; add e-Energy-community related
work (data-center demand response, workload shifting for grid signals).

## 3. System / Mechanism

The routing policies under comparison and how they're implemented (router design TBD).

## 4. Experimental Setup

8×4090 server, replica count vs. model size tradeoff, workload (whale-injection, reused
from existing harness).

## 5. Results

TBD.

## 6. Discussion / Limitations

Small-N caveat, shared-PDU/PSU caveat, relationship to the Monte Carlo extrapolation model.

## 7. Conclusion

TBD.
