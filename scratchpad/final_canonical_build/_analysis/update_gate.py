"""Fold the derived Phase 4/5/7/8 numbers and the source recommendation into
data/final_canonical/_WEBQSP_ACCEPTANCE_GATE.json.  Read-modify-write of that one file only.
"""
import json
import time

P = "data/final_canonical/_WEBQSP_ACCEPTANCE_GATE.json"
g = json.load(open(P, encoding="utf-8"))

g["MID_CLASSIFICATION_REQUIRED"]["_ANSWERED_2026_09_05"] = {
    "STATUS": "DONE -- answered WITHOUT a Freebase download",
    "method": (
        "The released RoG graph is itself a LABELLED OBSERVATION of Freebase name coverage: RoG rendered an "
        "endpoint as a name where Freebase had type.object.name and left the bare MID where it did not. Per "
        "Freebase TYPE (relation id minus its last component, e.g. people.marriage.spouse -> people.marriage) "
        "we therefore MEASURE the named rate. Two independent query-independent signals per node: pass 1 = the "
        "types the node asserts as a head; pass 2 = the RANGE of the relations pointing at it. No question "
        "text, answers, gold paths, topic annotations or neighbour names were used."),
    "scripts": ["scratchpad/final_canonical_build/_analysis/webqsp_mid_cvt_classification.py",
                "scratchpad/final_canonical_build/_analysis/webqsp_mid_pass2_inrelation.py"],
    "raw": ["scratchpad/final_canonical_build/_audit/phase45_mid_cvt_classification.json",
            "scratchpad/final_canonical_build/_audit/phase45_mid_pass2_inrelation.json"],
    "TOTAL_BARE_MIDS": 1652618,
    "NAMELESS_MEDIATOR_MIDS": 1588085,
    "NAME_RESOLVABLE_MIDS": 62375,
    "UNKNOWN_MIDS": 0,
    "_percentages_of_bare_mids": {"NAMELESS_MEDIATOR": "96.10%", "NAME_RESOLVABLE": "3.77%",
                                  "MIXED_UNDECIDED": "0.13%", "UNKNOWN": "0.00%"},
    "NAME_RESOLVABLE_UNCERTAINTY_BAND": {
        "both_signals_agree": 19566, "either_signal": 62375,
        "_reading": ("the population a Freebase metadata pass would actually rescue is 0.75%-2.41% of union "
                     "entities. Reported as a band rather than collapsed to a point.")},
    "EVIDENCE_THE_SPLIT_IS_REAL_NOT_A_THRESHOLD_ARTEFACT": {
        "bimodality": ("over 3,119 distinct types: 556 types have named-rate < 0.01, 2,531 have > 0.9, only "
                       "~20 lie in between. The relation-range histogram has the same shape (1,460 relations "
                       "< 0.01, 5,514 > 0.9)."),
        "threshold_sensitivity": {"tau_0.005": 964780, "tau_0.01": 966364, "tau_0.02": 967122,
                                  "tau_0.05": 969037, "tau_0.10": 969160,
                                  "_reading": "a 20x change in tau moves the answer by 0.45%. tau=0.02 used throughout."},
        "control_false_positive_rate": ("of 533,758 endpoints RoG DID name (so they demonstrably have "
                                        "type.object.name) and which have outgoing edges, only 238 (0.045%) "
                                        "are false-positively called mediator-like."),
        "cross_validation": ("pass 2 re-labels, from incoming relations ONLY, the 1,014,462 bare MIDs pass 1 "
                             "could label from outgoing types only. 1,011,820 carry an in-relation signal; "
                             "agreement 95.75%."),
        "domain_validation": ("the types called mediators are the textbook Freebase CVTs: film.performance "
                              "(198,122 nodes, rate 0.0002), common.webpage (121,320), award.award_nomination "
                              "(63,096), film.film_crew_gig (56,716), education.education (34,469), "
                              "award.award_honor (34,368), tv.tv_guest_role (28,798), "
                              "measurement_unit.dated_integer (26,953, exactly 0), sports.sports_team_roster "
                              "(19,581), music.track_contribution (18,573). The types called ordinary are "
                              "people.person (0.99997), film.film (0.9982), location.location (0.9602), "
                              "book.written_work (0.9990).")},
    "VERDICT_ON_THE_PRIOR_BLANKET_CLAIM": ("CONFIRMED at 96.10%, and its residual quantified for the first "
                                           "time. The claim was directionally right and quantitatively incomplete."),
    "freebase_decision_ANSWER": (
        "A meaningful population does NOT remain unresolved. 96.10% of bare MIDs are genuinely nameless "
        "mediators; the resolvable remainder is 0.75%-2.41% of union entities and is overwhelmingly late 'g.' "
        "machine ids that plausibly never had type.object.name. A full Freebase GRAPH download is therefore "
        "NOT justified by this number. A targeted METADATA pass is separately justified -- see "
        "SOURCE_RECOMMENDATION_2026_09_05 -- because it upgrades all 925,761 already-named endpoints with "
        "aliases and types, not because of the 62,375."),
}

g["CVT_POLICY_ANSWERED_2026_09_05"] = {
    "RECOMMENDATION": ("CVTs are STRUCTURE-ONLY nodes: give them a graph identity and NO Dense/SPLADE text. "
                       "If retrieval text is ever required, use the deterministic schema string "
                       "'CVT | types: <sorted types> | relations: <sorted relations>' -- never a neighbour name."),
    "evidence": [
        ("Prior KGQA code treats CVTs as structure, never as text. NSM's util/deal_cvt.py::is_cvt is a pure "
         "boolean membership test used only to steer traversal: get_2hop_subgraph.py uses it to decide whether "
         "to take an extra hop, preprocess_step1.py uses it to drop a dangling CVT triple as meaningless. NSM "
         "never reads or assigns a CVT label."),
        ("They carry almost no distinguishing text: under the deterministic recipe the 967,122 pass-1 CVTs "
         "collapse to only 2,155 DISTINCT strings (449x compression). The single most common string covers "
         "129,982 nodes ('CVT | types: film.performance | relations: film.performance.film'); the top eight "
         "cover ~343k. A vector shared by 130,000 nodes cannot discriminate between them."),
        ("The alternative is the documented catastrophe at src/pipeline/loader_webqsp.py:52-60,68 -- verified "
         "directly in Phase 7, not cited from the prior audit.")],
    "encoder_cost_consequence": ("If CVTs are structure-only, 1,588,085 nodes need no vector at all and "
                                 "DENSE_REENCODE_N falls from 2,070,021 to 481,936 -- a 77% cut. If CVT text is "
                                 "kept, cost is bounded by DISTINCT texts (2,155), not node count. Either way "
                                 "the real work is the ~1.0M ordinary topics."),
    "both_variants_retained": ("the schema recipe is kept as a documented option so structure-only vs "
                               "schema-text can be A/B'd at L1 without any corpus change. Under either variant "
                               "the CVT keeps its own MID as identity and never borrows a name."),
}

g["LEGACY_FABRICATION_BUG_VERIFIED_2026_09_05"] = {
    "_directive": ("'Explicitly verify the legacy neighbour-name assignment bug rather than citing the prior "
                   "audit.' Done: deduce() was re-implemented from loader_webqsp.py:52-60 and run against the "
                   "same TEST parquet shards the loader read, then compared with the stored titles in "
                   "data/processed/master_nodes_webqsp.json (READ-ONLY)."),
    "MID_nodes": 459417,
    "MID_with_no_named_neighbour_title_left_blank": 545,
    "FABRICATED_NAMES": 458872,
    "fabricated_title_is_SOME_named_neighbour_of_that_MID": 458872,
    "fabricated_title_is_NOT_any_named_neighbour": 0,
    "fabricated_title_equals_the_neighbour_THIS_RUN_picks": 373462,
    "BUG_VERIFIED": True,
    "NON_DETERMINISTIC": True,
    "_the_stronger_finding": (
        "Every stored title is provably a stolen neighbour name (458,872/458,872 = 100.000%), but 85,410 of "
        "them differ from what a re-run produces, because loader_webqsp.py builds its adjacency by iterating a "
        "Python SET of triples, whose order is hash-seed dependent. The legacy substrate is therefore not only "
        "wrong, it is UNREPRODUCIBLE across processes."),
    "examples": {"g.112yf9tt7": "Wassily Kandinsky", "g.112yfc45p": "Colombia", "g.112yfd_tp": "Dunkirk"},
    "commit_ba6bd71_assessment": ("'resolve all raw Freebase MIDs to names' did not resolve anything; it made "
                                  "the fabrication total. Its claim 'ZERO raw IDs in any content/title' is true "
                                  "and is precisely the defect."),
}

g["PHASE7_FOUR_WAY_COMPARISON_2026_09_05"] = {
    "legacy_781k": {"N": 781485, "stored": 783113, "identity": "webqsp_doc_<i> positional",
                    "MID_endpoints": 459417, "named": 322068, "fabricated_names": 458872,
                    "MID_recoverable": "only by re-deriving sorted(entities); not stored",
                    "source": "RoG-webqsp TEST shards only, 2,277,228 triples"},
    "phaseC_1316466": {"N": 1316466, "identity": "webqsp_ent_<sha256(text)[:16]>", "MID_endpoints": 776788,
                       "named": 539678, "fabricated_names": 0,
                       "MID_recoverable": "for the 59.0% stored AS MIDs", "source": "all 5 WebQSP shards",
                       "gold_conditioned": "YES -- 16,937 rows"},
    "released_RoG_union": {"N": 2592894, "R": 7058, "E": 8309195,
                           "identity": "released endpoint string (mixed)", "MID_endpoints": 1652618,
                           "named": 925761, "fabricated_names": 0,
                           "MID_recoverable": "NO for the 925,761 named", "source": "WebQSP u CWQ, 29 shards"},
    "reconstructed_MID_graph": "NOT BUILT -- see PHASE3_NOT_EXECUTED",
    "overlap": {"legacy781k_subset_of_union": 781485, "legacy781k_outside_union": 0,
                "legacy781k_subset_of_phaseC": 781485, "legacy781k_outside_phaseC": 0,
                "phaseC_intersect_union": 1299529, "phaseC_minus_union": 16937,
                "union_minus_phaseC": 1293365},
    "_correction_to_prior_audit": ("phaseC_intersect_union = 1,299,529 corrects the prior audit's 1,299,527 by "
                                   "+2: that run joined through a non-invertible backslash escape which "
                                   "corrupted 5 of 2,592,894 strings. This run reads the lossless "
                                   "union_entities.jsonl."),
    "phaseC_id_rule_confirmed": ("canonical_doc_id == 'webqsp_ent_' + sha256(text)[:16], verified against the "
                                 "stored text_sha256 field (sha256('Jamaica')[:16] = 42c298ba17fed5da). "
                                 "text == original_source_id in 1,316,466 / 1,316,466 rows. The store is clean "
                                 "but bare."),
}

g["PHASE3_NOT_EXECUTED"] = {
    "STATUS": "NOT BUILT -- reported as a failure, not worked around",
    "reason": ("No benchmark Freebase source exists on this machine (Phase 2 scanned 1,996,229 files across "
               "195,273 directories in 343.7s: zero hits for FastRDFStore-data.zip, fb_en.txt, cvtnodes.bin, "
               "Freebase-Setup, Virtuoso, object_names/object_types, id2name_parts, FACC1, surface maps). "
               "Acquiring it was explicitly out of scope for this task."),
    "capacity_finding": ("FastRDFStore-data.zip is LIVE (HTTP 200) at 68,549,649,154 B = 63.84 GiB against "
                         "112.9 GiB free on a single volume. The zip alone is 57% of free space, and fb_en.txt "
                         "must then be extracted AND rewritten full-size by manual_filter_rel.py before hop "
                         "extraction starts. Infeasible as configured."),
    "NO_FALLBACK": ("There was NO silent fallback to the released endpoint union and nothing was fabricated. "
                    "The released union remains a MEASURED DESCRIPTION of the published artifact and is NOT "
                    "promoted to canonical identity."),
    "_phase1_key_result": (
        "The RoG repo (HEAD ccf8ec847bf61005a1b27cc9e5aff5c8ead7a24b) contains NO subgraph-extraction code, NO "
        "Freebase loader and NO MID->name resolver -- verified against the 50-blob repository tree, not taken "
        "on trust. NSM's released pipeline (HEAD 2e20915956a69b6f832143ded9d54442ccbe698e) emits MIDs "
        "verbatim: preprocess_step1.py serialises {'kb_id': head, 'text': head}. The MID->surface projection "
        "that produced the mixed-endpoint artifact is therefore an UNRELEASED RoG step, and the released graph "
        "is not byte-reproducible from released code."),
}

g["PHASE8_REUSE_PROJECTION_2026_09_05"] = {
    "_conditional": ("CONDITIONAL and labelled as such: canonical human-readable text is NOT frozen, because "
                     "Phase 4 metadata resolution could not run. Projected is the best text derivable today "
                     "under MID-level identity."),
    "_method": ("token-ID equality under the frozen tokenizers (Alibaba-NLP/gte-Qwen2-1.5B-instruct, "
                "naver/splade-cocondenser-ensembledistil), never raw-text equality. NOTHING WAS ENCODED."),
    "baseline_released_endpoint_text": {"DENSE_REUSE_N": 1299535, "DENSE_PCT": "50.12%",
                                        "SPLADE_REUSE_N": 1302766, "SPLADE_PCT": "50.24%"},
    "under_MID_level_canonical_text": {"DENSE_REUSE_N": 522873, "DENSE_PCT": "20.17%",
                                       "DENSE_REENCODE_N": 2070021, "SPLADE_REUSE_N": 526096,
                                       "SPLADE_PCT": "20.29%", "SPLADE_REENCODE_N": 2066798},
    "DELTA": {"DENSE": -776662, "SPLADE": -776670},
    "decomposition": ("of the 1,316,466 Phase-C rows, 742,946 become CVT schema text and 33,718 are bare MIDs "
                      "blocked on Freebase metadata; 522,865 keep byte-identical text. The 940,276 union "
                      "endpoints whose text is unchanged are the 925,761 surfaces plus 14,515 literals."),
    "_reported_plainly": ("Per the directive, the drop is reported rather than optimised around. Two "
                          "mitigations: (a) structure-only CVTs cut DENSE_REENCODE_N to 481,936; (b) if CVT "
                          "text is kept, encoder cost is bounded by 2,155 distinct strings."),
    "_will_get_worse_correctly": ("Once real Freebase metadata lands, the 925,761 named endpoints gain "
                                  "'; aliases: ...; types: ...' and their text changes too, driving dense reuse "
                                  "toward ~0. That is the right trade: correct representation over preserved "
                                  "embeddings."),
}

g["SOURCE_RECOMMENDATION_2026_09_05"] = {
    "HEADLINE": ("FastRDFStore-data.zip is the only correct GRAPH source and is not viable on this machine as "
                 "configured. Do not substitute anything for it."),
    "1_reject_third_party_reconstruction": ("camazlucas/Freebase-WebQSP-CWQ-Subgraph is 50,304 triples and 16 "
                                            "relations SHORT of the published graph (2,574,900 / 7,042 / "
                                            "8,258,891) -- measurably worse than our own union on the two "
                                            "figures that can be checked. Diagnostic only."),
    "2_reject_full_rdf_mirror": ("CleverThis/freebase (964 parquet shards, ~300 GB extracted, ~3B triples) is "
                                 "the wrong tool for the graph -- it is all of Freebase, not the benchmark "
                                 "subgraph -- and it does not fit."),
    "3_PREFERRED_IF_ONE_DOWNLOAD_IS_EVER_AUTHORISED": {
        "resource": "idirlab/freebases, Zenodo 10.5281/zenodo.7909511",
        "size_bytes": 14148416296, "size": "13.18 GiB", "md5": "170689b7aad9f029566a4deb36605b01",
        "contains": ["object_names", "object_types", "entities_id_label", "properties_id_label"],
        "why": ("the METADATA problem is far cheaper than the GRAPH problem and is separable. 13.18 GiB "
                "one-volume, versus 63.84 GiB plus two full-size derived files. It upgrades all 925,761 named "
                "endpoints with aliases and types and closes the 62,375-node band, WITHOUT touching topology.")},
    "4_sequencing_if_graph_reconstruction_is_later_authorised": (
        "free >= 200 GB or use an external volume; run NSM Steps 0-3 against fb_en.txt + cvtnodes.bin with the "
        "WebQSP+CWQ topic-entity population; report entity/relation/triple counts against BOTH the paper "
        "(2,566,291 / 7,058 / 8,309,195) AND the released union (2,592,894 / 7,058 / 8,309,195) with "
        "differences EXPLAINED, never tuned; then attach metadata from step 3."),
    "5_is_cvt_does_not_block": ("cvtnodes.bin is not separately distributed -- it exists only inside the 63.84 "
                                "GiB zip. It is not a prerequisite: the structural classifier reproduces the "
                                "CVT partition at 96.10% of bare MIDs with 0.045% control false-positives, "
                                "95.75% cross-signal agreement and 0.45% threshold sensitivity. When "
                                "cvtnodes.bin is eventually obtained it becomes a VALIDATION SET for this "
                                "classifier."),
    "6_what_must_not_happen": ["promoting the released mixed-endpoint union to canonical identity",
                               "re-running the legacy deduce() heuristic in any form",
                               "naming a CVT"],
    "NOTHING_WAS_DOWNLOADED": True,
}

g["_status"] = ("PHASES 1,2,4,5,6,7,8 COMPLETE + 2WIKI FIX APPLIED AT SOURCE. PHASE 3 NOT EXECUTED (no "
                "Freebase source; reported as a failure per directive). No canonical WebQSP write, no encoder, "
                "no kNN, no H4.")
g["_deliverable"] = "data/final_canonical/webqsp/WEBQSP_FREEBASE_PIPELINE_AUDIT.md"
g["_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

json.dump(g, open(P, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("gate updated; top-level keys now:", len(g))
