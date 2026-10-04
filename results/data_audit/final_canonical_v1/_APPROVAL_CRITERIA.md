# SOURCE CONTRACT APPROVAL CRITERIA — canonical_v1

Authority: user instruction, 2026-09-05. This file defines the gate for creating
`data/final_canonical/_APPROVED_SOURCE_CONTRACTS`.

**The approval decision belongs to the user. It is NOT to be self-approved by an agent or
assistant.** The reviewer presents `SOURCE_CONTRACTS_FOR_REVIEW.md`; the user decides.

## Scope of scrutiny

The four smaller corpora (MetaQA, 2Wiki, MuSiQue, SQuAD) are comparatively straightforward and
already built. The two definitions under scrutiny are:

**WebQSP**
- What exactly is the complete retrieval universe?
- Is it genuinely independent of WebQTest / WebQSP query subgraphs?
- (The legacy 781,485-doc substrate is a union of RoG **TEST-split** subgraphs, and all 1,578
  legacy questions are `WebQTest-*`. That is a query-conditioned substrate and is NOT the full
  corpus.)

**HotpotQA**
- Which Wikipedia snapshot / source defines FullWiki?
- Does the ~5.23M node set come from the complete frozen wiki corpus, rather than from
  distractor/fullwiki query-associated context extraction?
- (The legacy 507,494-doc substrate is the distractor-setting context union. That is NOT the
  full corpus.)

The contract must make this **provable from source paths and build code, not merely plausible
from counts.**

## The ten criteria — ALL must be TRUE for each dataset

1. Full source is identified and locally reproducible.
2. Source version / snapshot is pinned.
3. Source hashes exist.
4. Canonical node identity is deterministic.
5. Canonical corpus can be built **without reading the selected query list**.
6. Derived N comes from the source itself.
7. Every `QUALITY_LOCKED` gold resolves into that universe.
8. Rebuilding with different query lanes gives an identical corpus hash.
9. There is **no query-subgraph / context-union step** in corpus membership.
10. `WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE` is backed by code/source evidence.

**If any one of these fails, do not create the approval gate file.**

## Reference scale clues (validation clues, NOT targets)

```
MetaQA        43,234      confirmed
2Wiki        398,354      confirmed
MuSiQue      117,534      confirmed
SQuAD         20,233      confirmed
WebQSP     ~1,316,466     must confirm from source
HotpotQA   ~5,233,329     must confirm from source
```

The authoritative count is whatever the approved full source deterministically yields. A build
must never be bent to hit a reference count; a disagreement must be explained instead.

## Gate files

- `_APPROVED_SOURCE_CONTRACTS` — created only by the user's decision, after all ten criteria pass.
- `_REJECTED_SOURCE_CONTRACTS` — stops the build; corrections arrive as a fresh task.
- `_GATE_HOTPOT_HEAVY_OK` — separate, unrelated machine-capacity gate for the ~5.2M-doc hotpot
  streaming phase. Never created by the build agent.
