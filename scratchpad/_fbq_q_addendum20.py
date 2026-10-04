"""FBX_SCALE declaration addendum 20 (write-once): the QUESTION-LEVEL Freebase measurement is DEFINED here, before any final partition exists, and is NOT SCORED.  It is a definition record, not a host stage
(the scoring job's reservation gets its own addendum, measured, when the prerequisites below hold).  It implements the user's rulings of 2026-10-04 on FBQ_SET v2: official dev is evaluation-eligible for the new sources, C = 400, the
leak-clean view stays, GrailQA's unusable rows are a measured substrate-coverage fact (reported, never counted as retrieval failures), CWQ-train is pool-only and never evaluation.
It changes no gate and no earlier record, reads no gold label (counts only come from the question-set metadata columns), and makes no recall claim.
Usage: python scratchpad/_fbq_q_addendum20.py   (refuses to overwrite)"""
import hashlib
import io
import json
import os
import time

import pandas as pd

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_20.json"
EXT = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%d.json" % i for i in (17, 18, 19)]
POOL, BAL, AUD = FB + "/FBQ_SET__v2__POOL.parquet", FB + "/FBQ_SET__v2__BALANCED.parquet", FB + "/FBQ_SET__v2__DEDUPE_AUDIT.parquet"
SETJ, SETMD, PROV = FB + "/FBQ_SET__v2.json", FB + "/FBQ_SET__v2.md", FB + "/FBQ_DOWNLOAD_PROVENANCE__v1.json"
CODE = ["scratchpad/_fbq_build_set_v2.py", "scratchpad/_fbq_download_v2.py", "scratchpad/_fbq_q_addendum20.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dist(df, *cols):
    return {str(k): int(v) for k, v in df.groupby(list(cols)).size().items()}


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    pool = pd.read_parquet(POOL)
    sj = json.load(io.open(SETJ, encoding="utf-8"))
    assert len(pool) == 85999 and sha(POOL) == sj["outputs"]["pool"]["sha256"], "the pool is not the pinned FBQ_SET v2 pool"
    # ---- the evaluation population E (user rulings): official dev of every source, mappable, leak-clean; CWQ-train and every train row of the new sources never
    elig = pool.eval_eligible | (pool.eval_flag == "NEWSRC_DEV_UNRULED")
    E = elig & pool.fully_mappable & pool.leak_clean
    assert not (E & (pool.source == "nsm_cwq") & (pool.split == "train")).any(), "CWQ-train must never be evaluation"
    assert not (E & pool.source.isin(["freebaseqa", "grailqa"]) & (pool.split != "dev")).any(), "a new-source train row entered E"
    e = pool[E].copy()
    e = e.sort_values(["family_id", "rank"], kind="stable")
    e["_k"] = e.groupby("family_id").cumcount()
    capped = e[e._k < 2]                                           # the pool's family cap (2), applied inside E
    strict = pool[E & pool.leak_clean_strict]
    pool_only = pool[~E]
    unusable_new = pool[elig & ~pool.fully_mappable]
    rec = {
        "stage": "FBX_SCALE / addendum 20: QUESTION-LEVEL Freebase measurement -- DEFINITION ONLY (not scored; not a host stage)",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "DEFINITION_FROZEN_UNSCORED",
        "rulings_implemented": {
            "official_dev": "evaluation-eligible for the new sources (FreebaseQA dev, GrailQA dev; CWQ dev and WebQSP dev/DEV_L3W keep their existing eligibility)",
            "C": 400, "leak_clean": "kept: leak_clean (user rule, FBQ_SET v2 builder) is REQUIRED for every evaluation row; leak_clean_strict is reported as the sensitivity view",
            "grailqa_unusable": "a measured substrate-coverage fact: report raw population, usable population, unusable fraction and reasons; every metric is conditional on usable questions; an unusable question is NEVER a retrieval failure",
            "cwq_train": "pool-only, never evaluation (asserted by this script on the pinned pool)",
            "scoring": "NOT scored here; the definition is frozen before the final partition exists so it cannot drift after the partition is seen"},
        "extends": {p: sha(p) for p in EXT},
        "pinned_inputs": {"pool": {"path": POOL, "sha256": sha(POOL)}, "balanced_C400_set": {"path": BAL, "sha256": sha(BAL)}, "dedupe_audit": {"path": AUD, "sha256": sha(AUD)},
                          "set_record_json": {"path": SETJ, "sha256": sha(SETJ)}, "set_record_md": {"path": SETMD, "sha256": sha(SETMD)}, "download_provenance": {"path": PROV, "sha256": sha(PROV)},
                          "builder_seed": sj["seed"], "builder_family_cap": sj["family_cap"], "builder_min_cell_for_merge": sj["min_cell_for_merge"]},
        "populations_at_declaration": {
            "pool": len(pool), "balanced_C400_rows": int(pool.in_balanced_C400.sum()),
            "E_evaluation_population": {"rule": "(eval_eligible OR eval_flag == NEWSRC_DEV_UNRULED) AND fully_mappable AND leak_clean", "rows": int(E.sum()), "by_source_split": dist(pool[E], "source", "split"),
                                        "by_hop_bucket": dist(pool[E], "hop_bucket"), "by_qtype": dist(pool[E], "qtype"), "by_source_hop_bucket": dist(pool[E], "source", "hop_bucket"),
                                        "after_family_cap_2": {"rows": len(capped), "by_source": dist(capped, "source"), "by_hop_bucket": dist(capped, "hop_bucket")},
                                        "leak_clean_strict_sensitivity_rows": len(strict)},
            "pool_only_not_evaluation": {"rows": len(pool_only), "by_source_split": dist(pool_only, "source", "split"), "label": "POOL_ONLY (training / calibration / diagnostics; never evaluation)"},
            "eligible_but_unusable_in_pool": {"rows": len(unusable_new), "by_source": dist(unusable_new, "source"), "note": "reported, never scored as failures"},
            "balanced_C400_rows_that_are_also_in_E": int((pool.in_balanced_C400 & E).sum()),
            "note": "E is the evaluation population; the balanced C=400 set is the hop x type balanced POOL (it mixes train-pool rows) and is not itself an evaluation set; the macro-average below re-balances E over the same hop x type cells"},
        "grailqa_substrate_coverage_raw": {k: sj["resolution"][k] for k in ("grailqa/train", "grailqa/dev", "grailqa/ALL")},
        "grailqa_note": "unusable = not fully mappable: questions with zero topic entities, with a literal answer, or with an unresolved MID (the categories overlap); counts above are the builder's measured census (raw source rows, before dedupe)",
        "measurement": {
            "name": "ORACLE-TOPIC CO-LOCATION CURVE of a partition (front-end-free; a property of the partition on real questions, NOT a retrieval recall)",
            "unit": "one question q in E; T_q = the topic positions and A_q = the answer positions in the frozen 302M-node tree (pool columns topic_pos / answer_pos; fully_mappable => every MID resolved and no literal answer)",
            "inputs_fixed_at_scoring": {"partition_map": "a Freebase map M over all N nodes with K in {100, 250, 500} from the frozen multilevel out-of-core pipeline, validity gate PASS (max block <= ceil(1.03 N / K)); its hash is pinned in the scoring addendum, NOT here",
                                        "adjacency": "the frozen tree's STRUCT CSR, both directions, all relations, uncapped (degree appears only as the 1/deg normaliser below)"},
            "block_mass": "mass_q(P) = sum over t in T_q of [ 1{M(t) = P} + |{v in N(t) : M(v) = P}| / max(1, deg(t)) ]  (the topic's own block plus the share of its neighbours in P; one propagation step, no walk)",
            "block_rank": "blocks ordered by mass_q desc, ties by block id asc; rank_q(P) in 0..K-1",
            "curve": "for b in {1, 2, 4, 8, 16, 32, 64} (b <= K; the x axis is also reported as b / K): ALL_q(b) = 1 iff every block of every answer has rank < b; ANY_q(b) = 1 iff at least one does; per-answer share = mean over a in A_q of 1{rank_q(M(a)) < b}",
            "feasibility_ceiling": "n_ans_blocks_q = |{M(a) : a in A_q}|; ALL_q(b) <= 1{n_ans_blocks_q <= b}",
            "null": "N0 = size-matched random relabelling (a seeded permutation of the label vector, which preserves every block size), seeds 1, 2, 3, mean; uplift = observed minus N0 on the same questions",
            "reference_on_the_known_substrate": "the identical definition on the WebQSP evaluation rows (the 786-row population) over the seven frozen WebQSP PHG_SK maps (K 100 / 250 / 500), so the Freebase curve is read against a known one at equal b / K",
            "what_it_is_not": "no dense / SPLADE front end exists at Freebase scale until the encode is finished and verified; this measures how well the PARTITION co-locates gold answers with a gold-topic seed's one-step neighbourhood, and says nothing about FLAT retrieval or end-to-end recall",
            "gold_labels_read_at_scoring": "gold topic and answer positions of the dev rows of E only; no test split anywhere; pool-only rows are never scored"},
        "reporting": {
            "strata": ["source", "hop_bucket {1, 2, 3, 4+}", "qtype (v2 merged cells)", "ans_bucket", "leak_clean_strict (sensitivity)"],
            "macro_average": "equal weight over the v2 hop x type cells with at least min_cell_for_merge (50) rows in E after the v2 merging rule; cells below 50 are listed with their n and excluded from the macro",
            "coverage_on_every_line": "raw N, usable N, unusable fraction and reasons next to every number; metrics are conditional on usable questions",
            "uncertainty": "bootstrap resampling whole FAMILIES (pool column family_id; CWQ children share a parent), 2,000 resamples, seed 20261004, percentile 95 % intervals; paired differences against N0 use the same resamples",
            "claims_allowed": "descriptive co-location of gold answers in the partition; no recall claim at Freebase scale; no comparison with an end-to-end system"},
        "prerequisites_before_scoring": ["E6 level-0 V-cycle done and the E7 report validity PASS (addendum 19 job)",
                                         "the K in {100, 250, 500} Freebase maps built and validity-gated (their own addendum; hashes pinned in the scoring addendum)",
                                         "the user's explicit go",
                                         "a scoring-stage addendum with a MEASURED reservation (host, RAM <= 70 GB, disk <= 55 GB, under scratchpad/_host_yield.py)"],
        "stop_rules": ["any change to E, the quantities, the null or the strata after a partition map is seen requires a new addendum (supersession), never an edit",
                       "a question that is unusable is dropped with a reason, never scored; no row of POOL_ONLY is scored",
                       "RAM > 70 GB or disk > 55 GB in the scoring stage: stop and report"],
        "not_done_here": ["scoring of any kind", "building the K maps", "any claim", "the dense / SPLADE front end at Freebase scale"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))
    print(json.dumps(rec["populations_at_declaration"], indent=1)[:2500])


if __name__ == "__main__":
    main()
