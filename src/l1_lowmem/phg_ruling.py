"""PHG_SMALL_DATASET_VALIDATION ruling (user, 2026-09-14) -- frozen as a record, write-once.

    python -u src/l1_lowmem/phg_ruling.py write   -> results/L1_LOWMEM/PHG_SMALL_DATASET_VALIDATION_RULING.json

The record carries the user's ruling verbatim, the structured decisions derived from it, and sha pins of every evidence record
the ruling rests on (PHG lane + PHG_NONEMPTY_REPAIR_V1 follow-up).  Nothing else is written; no frozen record is edited.
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, rj, wj, pin  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402

RULING_RECORD = "PHG_SMALL_DATASET_VALIDATION_RULING.json"
EXPERIMENTAL_PHG_CONTRACT = "%s + %s" % (P.ARM, PR.REPAIR)

RULING_TEXT_VERBATIM = """Yes — I would now rule that the small-dataset PHG validation passes.
The evidence is stronger than "partitions are similar":

* SQuAD: different PHG partition, but canonical L1 SAFE changes only `0.9930 → 0.9900`, with no significant paired loss.
* MetaQA: after a deterministic, H4-only repair, SAFE changes `0.6401 → 0.6712`, a significant gain.
* PHG's generic KM1 is worse than Mt-KaHyPar, but only by about 10–16%, versus FREIGHT's much larger gaps.
* Memory and runtime are dramatically lower.
* The H4_SK structure is proven exact.
* The MetaQA repair changed zero structural objective because the two moved nodes had no H4_SK incidence.

So this is exactly the conclusion we wanted to be able to make:
Mt-KaHyPar's exact partition is not required for L1. A substantially different globally valid partition can preserve—or in one observed case improve—the downstream retrieval behavior.
I would freeze the following ruling:

```text
PHG_SMALL_DATASET_VALIDATION = PASS

status:
    LEADING_LOW_MEMORY_PARTITIONER_CANDIDATE

evidence:
    squad:
        L1 = NO_SIGNIFICANT_PAIRED_LOSS
    metaqa:
        L1 = SIGNIFICANT_PAIRED_GAIN_OBSERVED

important:
    MetaQA gain is an observation,
    NOT a general PHG-superiority claim.

canonical decision criterion:
    downstream L1 retrieval
    > exact partition reproduction
    > generic KM1 alone
```

The repair is acceptable
I would also accept `PHG_NONEMPTY_REPAIR_V1` as part of the experimental PHG contract.
It has all the properties we need:

```text
deterministic                  ✓
dataset-independent rule       ✓
uses H4_SK only                ✓
doesn't inspect queries        ✓
doesn't inspect answers        ✓
doesn't inspect L1 metrics     ✓
minimal number of moves        ✓
preserves size upper bound     ✓
```

And in this particular MetaQA case it is almost the ideal proof case:

```text
empty blocks = 2
moves        = 2

ΔKM1 move 1 = 0
ΔKM1 move 2 = 0

STRUCT cut unchanged
KNN cut unchanged
```

The fact that the resulting two blocks are singletons does not bother me. Our contract requires non-empty parts; it never established a minimum occupancy. Inventing a minimum-size rule now would be the post-hoc change.
Do not chase the MetaQA gain yet
The `+3.1` SAFE points are very interesting, especially the hop-3 `+7.1` points, but I agree with how Claude recorded it.
Do not now tune PHG to reproduce or maximize that phenomenon.
There is a plausible hypothesis:

```text
PHG block geometry
    ↓
slightly broader/more uniform P50 scope
    ↓
better structural reach
    ↓
better multi-hop recall
```

but that's a new research question, not something established by this experiment.
It may actually become relevant later for L1 research, but first we need to establish PHG as the scale-enabling partitioner.
I would now move to scale
The order I recommend is:

```text
MuSiQue
   ↓
WebQSP
   ↓
HotpotQA
   ↓
2Wiki
```

But the interpretation changes depending on dataset.
MuSiQue = bridge experiment
Run PHG on MuSiQue next because:

```text
117,534 nodes
Mt-KaHyPar local = RESOURCE_INFEASIBLE_LOCAL
```

This is the first dataset where PHG can demonstrate that it actually removes a blocker.
We don't have a valid Mt-KaHyPar paired partition reference there, so don't claim PHG equivalence.
Classify this run as:

```text
PHG_SCALE_BRIDGE
```

and ask:

```text
Does exact H4_SK
→ PHG
→ NONEMPTY_REPAIR_V1 if needed
→ P50
→ canonical L1

complete locally with sane partition/L1 behavior?
```

Record absolute L1 numbers, but no comparative quality claim against Mt-KaHyPar.
If MuSiQue completes comfortably, that's meaningful.
Then WebQSP is the real scale breakthrough
WebQSP is the next sensible target:

```text
~2.59M nodes
~29.6M canonical SK pins
```

It is much larger than MuSiQue but substantially smaller than Hotpot/2Wiki.
Use exactly the frozen contract:

```text
H4_SK
PHG
NP = frozen rank count unless technically impossible
CONNECTIVITY
1.03
EDGE_SIZE_THRESHOLD=1.0
deterministic mode
NONEMPTY_REPAIR_V1
P50
canonical L1
```

No parameter tuning between datasets.
If 4 ranks no longer fit, don't silently change NP. Record the resource failure and create a separate explicit rank-count scale ruling, because MPI rank count is part of PHG's algorithmic contract and may change the partition.
What should be the real success criterion now?
For datasets with a proper Mt-KaHyPar reference:

```text
PRIMARY:
canonical SAFE recall
paired on exactly identical query IDs
```

For datasets without one:

```text
PRIMARY:
valid exact H4_SK partition
+ successful canonical L1 execution
+ sane absolute retrieval behavior

NO claim of equivalence
```

Always retain:

```text
KM1
STRUCT cut
KNN cut
block distribution
P50 scope
candidate coverage
hop-wise results
memory
runtime
```

as diagnostics.
One more thing I would freeze now
The SQuAD result taught us an important methodological lesson:

```text
better KM1 ≠ necessarily better L1
```

And MetaQA made that even clearer.
So I would explicitly add to the L1 methodology:
Partition-objective quality is a diagnostic proxy. Downstream canonical L1 retrieval on a fixed evaluation population is the primary criterion for partitioner substitution.
That protects us later from rejecting a good scale partitioner solely because its KM1 is 8–15% worse than Mt-KaHyPar.
I'd tell Claude to freeze the current PHG validation and move to MuSiQue only next, with no additional PHG tuning. If MuSiQue works, then we authorize WebQSP separately."""

EVIDENCE_RECORDS = ["PHG_PREREG.json", "PHG_BUILD.json", "PHG_PACKAGES.json", "PHG_RUNS_squad.json", "PHG_RUNS_metaqa.json", "PHG_REPAIR_PREREG.json",
                    "PHG_REPAIR_RUNS_squad.json", "PHG_REPAIR_RUNS_metaqa.json", "PHG_REPORT.json", "PHG_REPORT.md", "L1_DOWNSTREAM_squad__phg.json",
                    "L1_DOWNSTREAM_metaqa__phg_repair1.json", "L1_REPLAY_squad__LOWMEM__PHG_con.json", "L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
                    "MUSIQUE_MTKAHYPAR_RETRY.json", "L1_DOWNSTREAM_musique.json", "FREIGHT_RUNS_musique.json"]


def write():
    rp = os.path.join(OUT, RULING_RECORD)
    if os.path.exists(rp):
        raise RuntimeError("ruling record already exists -- written once; supersede, never edit")
    runs_s, runs_m = rj(os.path.join(OUT, "PHG_RUNS_squad.json")), rj(os.path.join(OUT, "PHG_RUNS_metaqa.json"))
    rep_m = rj(os.path.join(OUT, "PHG_REPAIR_RUNS_metaqa.json"))
    rep_s = rj(os.path.join(OUT, "PHG_REPAIR_RUNS_squad.json"))
    report = rj(os.path.join(OUT, "PHG_REPORT.json"))
    if runs_s["DECISION"] != PR.NEW_LABEL or rep_m["DECISION"] != PR.NEW_LABEL or rep_s["DECISION"] != "REPAIR_NOOP_IDENTICAL" or report["STATUS"] != "STOP_FOR_REVIEW":
        raise RuntimeError("evidence records do not carry the decisions the ruling rests on")
    s_pr, m_pr = runs_s["downstream"]["paired"], rep_m["downstream"]["paired"]
    pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    rec = {
        "RECORD": "PHG_SMALL_DATASET_VALIDATION_RULING", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ruled_by": "user, chat message of 2026-09-14 after the STOP_FOR_REVIEW of PHG_REPORT.json version 2", "ruling_text_verbatim": RULING_TEXT_VERBATIM,
        "PHG_SMALL_DATASET_VALIDATION": "PASS", "status": "LEADING_LOW_MEMORY_PARTITIONER_CANDIDATE",
        "evidence": {
            "squad": {"L1": "NO_SIGNIFICANT_PAIRED_LOSS", "SAFE_paired": s_pr["SAFE_ALL_P50"], "BASE_paired": s_pr["BASE_ALL_P50"],
                      "km1_ratio_vs_mtkahypar": runs_s["diagnostic_vs_mtkahypar"]["km1_ratio"], "DECISION_record": runs_s["DECISION"]},
            "metaqa": {"L1": "SIGNIFICANT_PAIRED_GAIN_OBSERVED", "SAFE_paired": m_pr["SAFE_ALL_P50"], "BASE_paired": m_pr["BASE_ALL_P50"],
                       "km1_ratio_vs_mtkahypar": rep_m["km1_ratio_vs_mtkahypar"], "DECISION_record": rep_m["DECISION"],
                       "repair": {"empty_blocks": rep_m["empty_blocks"], "moves": [{k: mv[k] for k in ("empty_block", "node_position", "donor_block", "delta_km1_weighted", "incident_nets")} for mv in rep_m["moves"]],
                                  "km1_before_after": [rep_m["before"]["km1_weighted"], rep_m["after"]["km1_weighted"]],
                                  "struct_cut_delta": rep_m["after"]["struct_cut_delta"], "knn_cut_delta": rep_m["after"]["knn_cut_delta"]}},
            "structure": "H4_SK proven exact on both datasets (structure gate: digest from the Zoltan query functions == ORIGINAL; weight reception exact)",
            "resources": {ds: {"peak_rss_mb_max_rank": round(R["stages"]["3_partition"]["memory"]["max_rank_peak_kb"] / 1024.0, 1),
                               "partition_wall_seconds": R["stages"]["3_partition"]["timing"]["partition_wall_seconds"],
                               "mtkahypar_peak_rss_mb": (R.get("diagnostic_vs_mtkahypar") or {}).get("mtkahypar_peak_rss_mb"),
                               "mtkahypar_wall_seconds": (R.get("diagnostic_vs_mtkahypar") or {}).get("mtkahypar_wall_seconds")} for ds, R in (("squad", runs_s), ("metaqa", runs_m))},
            "records": {fn: pin(os.path.join(OUT, fn)) for fn in EVIDENCE_RECORDS if os.path.exists(os.path.join(OUT, fn))},
            "superseded_originals_in_history": {"PHG_RUNS_squad.json": runs_s.get("supersedes"), "PHG_REPAIR_PREREG.json": (rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json")) or {}).get("supersedes"),
                                                "PHG_REPORT": report.get("supersedes")}},
        "important": "the MetaQA gain is an observation, NOT a general PHG-superiority claim; PHG is not tuned to reproduce or maximize it; the hypothesis "
                     "(PHG block geometry -> slightly broader / more uniform P50 scope -> better structural reach -> better multi-hop recall) is a NEW research "
                     "question, not established by this experiment, and is deferred until PHG is established as the scale-enabling partitioner",
        "canonical_decision_criterion": ["downstream L1 retrieval", "> exact partition reproduction", "> generic KM1 alone"],
        "methodology_addition_frozen": {"lesson": "better KM1 != necessarily better L1 (SQuAD; MetaQA made it clearer)",
                                        "rule": "Partition-objective quality is a diagnostic proxy. Downstream canonical L1 retrieval on a fixed evaluation population "
                                                "is the primary criterion for partitioner substitution.",
                                        "protects_against": "rejecting a good scale partitioner solely because its KM1 is 8-15 % worse than Mt-KaHyPar"},
        "repair_accepted": {"adapter": PR.REPAIR, "part_of": "the experimental PHG contract",
                            "properties": ["deterministic", "dataset-independent rule", "uses H4_SK only", "does not inspect queries", "does not inspect answers",
                                           "does not inspect L1 metrics", "minimal number of moves (one per empty block)", "preserves the size upper bound"],
                            "no_minimum_occupancy_rule": "the contract requires non-empty parts and never established a minimum occupancy; the two MetaQA singleton "
                                                         "blocks are allowed; inventing a minimum-size rule now would be the post-hoc change",
                            "algorithm_record": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json"))},
        "experimental_PHG_contract": {"name": EXPERIMENTAL_PHG_CONTRACT, "input": "exact canonical H4_SK (H4_SPLIT_PRESERVE, STRUCT + KNN, unit vertex weights, H4 net weights), "
                                                                                    "frozen H4_SK_STREAM_V1 shards as the representation",
                                      "partitioner": "Zoltan-PHG as pinned in PHG_PREREG.json / PHG_BUILD.json (driver binary %s)" % pre["build"]["driver_binary_sha256"][:16],
                                      "parameters": "PHG_PREREG.json phg_parameters_in_order, unchanged; no parameter tuning between datasets",
                                      "fixed_elements": ["CONNECTIVITY objective", "IMBALANCE_TOL 1.03", "PHG_EDGE_SIZE_THRESHOLD 1.0", "DETERMINISTIC 1 / PHG_RANDOMIZE_INPUT 0",
                                                         "NUM_GLOBAL_PARTS = k = max(1, N // 100)", "validity = canonical bound ceil(1.03 N/k), no empty block"],
                                      "mpi_ranks": P.NP,
                                      "rank_count_rule": "NP = the frozen rank count unless technically impossible; if %d ranks no longer fit, NP is NOT silently changed: "
                                                         "the resource failure is recorded and a separate explicit rank-count scale ruling is required (the rank count is "
                                                         "part of PHG's algorithmic contract and may change the partition)" % P.NP,
                                      "post_partition": "%s (no-op when no block is empty)" % PR.REPAIR, "downstream": "P50 -> unchanged canonical L1 (replay cache -> l1_eval numerics)"},
        "success_criteria": {"with_mtkahypar_reference": {"PRIMARY": "canonical SAFE recall paired on exactly identical query ids (exact two-sided McNemar; "
                                                                      "SIG_LOSS iff delta < 0 and p < 0.05, else NO_SIGNIFICANT_PAIRED_LOSS; no non-inferiority margin)"},
                             "without_mtkahypar_reference": {"PRIMARY": ["valid exact H4_SK partition", "successful canonical L1 execution", "sane absolute retrieval behaviour"],
                                                             "claim": "NO claim of equivalence; absolute L1 numbers recorded, no comparative quality claim against Mt-KaHyPar"},
                             "always_retained_diagnostics": ["KM1", "STRUCT cut", "KNN cut", "block distribution", "P50 scope", "candidate coverage", "hop-wise results",
                                                             "memory", "runtime"]},
        "next": {"authorized_now": {"dataset": "musique", "classification": "PHG_SCALE_BRIDGE",
                                    "why": "117,534 nodes; Mt-KaHyPar local = RESOURCE_INFEASIBLE_LOCAL (MUSIQUE_MTKAHYPAR_RETRY.json) -- the first dataset where PHG can "
                                           "demonstrate that it removes a blocker; no valid Mt-KaHyPar paired reference exists there, so no equivalence claim",
                                    "question": "does exact H4_SK -> PHG -> NONEMPTY_REPAIR_V1 if needed -> P50 -> canonical L1 complete locally with sane partition / L1 behaviour?"},
                 "requires_separate_authorization": ["webqsp (~2.59M nodes, ~29.6M canonical SK pins): authorized separately only if MuSiQue works", "hotpotqa", "2wiki"],
                 "order": ["musique", "webqsp", "hotpotqa", "2wiki"], "no_additional_phg_tuning": True},
        "not_done_by_this_ruling": "no promotion into the canonical partition contract; PHG is a CANDIDATE; Mt-KaHyPar remains the canonical partitioner of record; "
                                   "nothing frozen is edited (records are superseded, never edited)"}
    wj(rp, rec)
    log("ruling frozen ->", os.path.relpath(rp, REPO), "| PHG_SMALL_DATASET_VALIDATION =", rec["PHG_SMALL_DATASET_VALIDATION"], "| status", rec["status"])
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "write":
        write()
    else:
        print(__doc__)
