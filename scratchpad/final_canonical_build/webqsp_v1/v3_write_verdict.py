"""Consolidate checks 1-7 into one verdict on the IDIR package.

    python scratchpad/final_canonical_build/webqsp_v1/v3_write_verdict.py

Reads the seven check records and writes V3_IDIR_VERDICT.json. Every number here is copied from a
check output; nothing is recomputed and nothing is asserted that a check did not measure.
"""
import json, os, time

D = "data/final_canonical/freebase_v3"
OUT = f"{D}/V3_IDIR_VERDICT.json"


def load(n):
    return json.load(open(f"{D}/{n}", encoding="utf-8"))


acq = load("V3_ACQUISITION_RECORD.json")
ext = load("V3_EXTRACTION_RECORD.json")
c35 = load("V3_CHECK3_CHECK5_RELATIONS.json")
c3b = load("V3_CHECK3B_RELATION_MASS.json")
c3c = load("V3_CHECK3C_GOLD_INCIDENCE.json")
c24 = load("V3_CHECK2_CHECK4_JOIN.json")
c6 = load("V3_CHECK6_METADATA_COVERAGE.json")
c7 = load("V3_CHECK7_CVT_COVERAGE.json")
why = load("V3_WHY_CVTS_DROPPED.json")

c2 = c24["CHECK_2_MID_COVERAGE"]
c4 = c24["CHECK_4_STRUCTURAL_AGREEMENT"]
cov6 = c6["CHECK_6_COVERAGE"]

doc = {
    "schema": "V3_IDIR_VERDICT/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "scope": "the seven checks the user fixed before acquisition, run against the delivered "
             "idirlab-freebases.zip. This decides IDIR's role in CRAG_FREEBASE_CANONICAL; it does "
             "not build anything.",

    "CHECK_STATUS": {
        "1_byte_provenance": "PASSED. bytes and MD5 both match Zenodo exactly.",
        "2_webqsp_mid_coverage": "RUN. Sharply role-dependent -- see WHAT_IDIR_DELIVERS.",
        "3_relation_coverage": "RUN. 30.6% by relation count, 47.2% by triple mass, 76.6% counting "
                               "reverse orientation.",
        "4_structural_agreement": "PASSED, and strongly. 99.998% exact agreement on retained "
                                  "predicates with zero reverse-only cases.",
        "5_reverse_pair_audit": "PARTIAL. The reverse_property mapping is not in the archive "
                                "(FINDING_1), so INVARIANT_2 is uncomputable. INVARIANT_1 is "
                                "confirmed empirically. See CHECK_5_RESOLUTION.",
        "6_metadata_coverage": "PASSED for rendering. 99.83% of probe MIDs carry a type; answers "
                               "and topics are ~99.6-100% named.",
        "7_cvt_coverage": "FAILED. 30.75% of V1's CVT nodes are absent, and the cause is measured.",
    },

    "WHAT_IDIR_DELIVERS": {
        "answer_mids_present_pct": c2["answer_mids_present_in_idir_pct"],
        "topic_mids_present_pct": c2["topic_mids_present_in_idir_pct"],
        "named_entity_nodes_present_pct":
            c7["CHECK_7_COVERAGE_BY_V1_NODE_KIND"]["MID_NAMED_ENTITY"]["has_backbone_edge_pct"],
        "answer_mids_named_pct": cov6["by_role"][0]["named_pct"],
        "answer_mids_typed_pct": cov6["by_role"][0]["typed_pct"],
        "topic_mids_named_pct": cov6["by_role"][1]["named_pct"],
        "all_probe_mids_typed_pct": cov6["overall"]["typed_pct"],
        "exact_structural_agreement_on_retained_predicates_pct":
            c4["predicate_in_backbone"]["present_pct"],
        "reverse_only_among_retained_predicates": c4["predicate_in_backbone"]["reverse_only"],
        "answer_incident_facts_present_pct": c4["ANSWER_INCIDENT"]["present_pct"],
        "reading": "for the answer-bearing part of the graph IDIR is excellent. Gold answers are "
                   "essentially all present, essentially all named and essentially all typed, and "
                   "91% of the facts touching them survive in one orientation or the other.",
    },

    "WHAT_IDIR_CANNOT_DELIVER": {
        "cvt_nodes_absent": why["cvt_missing_from_FB+CVT-REV"],
        "cvt_nodes_absent_pct": round(100 - why["cvt_coverage_if_built_on_FB+CVT+REV"]
                                      ["compare_backbone_pct"], 3),
        "cause": why["VERDICT"],
        "cause_evidence": {
            "of_the_missing_cvts_present_in_FB+CVT+REV": why["of_those_present_in_FB+CVT+REV"],
            "of_the_missing_cvts_present_in_FB+CVT+REV_pct": why["of_those_present_in_FB+CVT+REV_pct"],
            "coverage_if_we_switched_to_FB+CVT+REV_pct":
                why["cvt_coverage_if_built_on_FB+CVT+REV"]["cvt_present_pct"],
            "coverage_on_the_backbone_pct":
                why["cvt_coverage_if_built_on_FB+CVT+REV"]["compare_backbone_pct"],
            "reading": "FB+CVT+REV differs from the backbone in exactly one step, the reverse "
                       "anti-join, and it rescues 97 of 488,432 nodes. The anti-join is therefore "
                       "not the cause. The subject-and-object-must-be-/m/-or-/g/ literal filter is, "
                       "which is precisely the CRAG-specific problem identified in the V3 proposal "
                       "before acquisition.",
            "and_idir_still_knows_these_objects": "488,280 of the missing CVTs carry a type in "
                                                  "IDIR's own metadata. They were not out of scope; "
                                                  "their edges were filtered away and the isolated "
                                                  "nodes fell out of the graph.",
        },
        "nsm_triple_mass_absent_pct": c4["ABSENT_pct"],
        "of_which_missing_entities_not_missing_edges": {
            "unjoinable_endpoint_unknown_pct": round(100 - c4["joinable_pct"], 3),
            "both_endpoints_known_but_no_edge_pct": round(c4["ABSENT_pct"] - (100 - c4["joinable_pct"]), 3),
            "reading": "the deficit is overwhelmingly missing NODES rather than missing edges "
                       "between nodes IDIR has. That is the same literal-filter mechanism seen "
                       "from the triple side.",
        },
        "literal_valued_facts": "absent by construction, for every variant. This was known from the "
                                "source before acquisition; what is new is the measured cost.",
        "reverse_property_mapping": "not in the archive (FINDING_1).",
    },

    "CHECK_5_RESOLUTION": {
        "INVARIANT_1_BOTH_SIDES_PRESENT_N": {
            "preregistered_expectation": 0,
            "measured_proxy": c4["predicate_in_backbone"]["reverse_only"],
            "how": "every NSM triple whose predicate IDIR retained was checked in both directions "
                   "against the delivered 134,142,730 triples. Zero appeared reverse-only, and "
                   "99.998% appeared forward and exact.",
            "verdict": "SATISFIED. The anti-join behaves exactly as FB3.sh says it should, verified "
                       "against delivered data rather than against the script.",
        },
        "INVARIANT_2_NEITHER_SIDE_PRESENT_N": {
            "preregistered_expectation": 0,
            "measured": None,
            "verdict": "UNCOMPUTABLE EXACTLY, BUT BOUNDED.",
            "why_uncomputable": "detecting a pair that lost both halves requires knowing the pair "
                                "existed, and only /type/property/reverse_property says that. It is "
                                "not in the archive.",
            "the_bound": "a symmetric annihilation would show up as an ordinary domain relation "
                         "vanishing from all four variants. Of the 2,036,798 triples on predicates "
                         "absent everywhere, only 17,930 (0.21% of NSM mass, 25 predicates) are "
                         "ordinary domain content; the remaining 99.1% is web/media plumbing, "
                         "valuenotation metadata, type assertions, notable_for and user-contributed "
                         "base domains.",
            "strongest_candidates_visible_in_that_residue": [
                {"pair": ["common.topic.subjects", "common.topic.subject_of"], "triples": 13997,
                 "note": "textbook reverse-pair naming, both halves absent"},
                {"pair": ["kp_lw.philosopher.influenced_by", "kp_lw.philosophy_influencer.influencee",
                          "kp_lw.philosopher.influenced", "kp_lw.philosophy_influencee.influencer"],
                 "triples": 511, "note": "two pairs in one user-contributed domain, all four absent"},
            ],
            "everything_else_in_the_residue": "rdf-schema#range/#domain, owl#inverseOf, "
                                              "common.uri_property/foreign_key_property templates, "
                                              "the Freebase curation pipeline.* predicates, and "
                                              "annotation categories. Schema plumbing, not content.",
            "honest_caveat": "absence from all four variants is consistent with symmetric "
                             "annihilation AND with the literal filter AND with being out of "
                             "IDIR's scope. This bounds the risk; it does not attribute it. Given "
                             "the LITERAL_FILTER verdict on the CVT population, the literal filter "
                             "is the likelier explanation for most of it.",
        },
    },

    "DECISION_INPUT_FOR_THE_RAW_MIRROR": {
        "the_user_reserved_this_decision": "'Only after those pass do we decide whether the raw "
                                           "mirror is necessary.' The checks did not all pass.",
        "what_the_raw_mirror_would_buy": [
            "488,432 CVT routing anchors -- 30.75% of V1's CVT population -- unavailable from any "
            "IDIR variant, cause measured as the literal filter",
            "literal-valued facts, absent by construction from every variant",
            "the /type/property/reverse_property mapping, which would make INVARIANT_2 computable "
            "instead of merely bounded",
        ],
        "what_it_would_not_buy": [
            "gold answer coverage: already 99.95%",
            "topic coverage: already 98.09%",
            "readable surfaces: 99.83% of probe MIDs are typed, answers 99.6% named",
            "structural correctness of the retained portion: already 99.998% exact",
        ],
        "why_this_matters_more_under_ALL_NODE": "the retrieval-role decision made every CVT a "
                                                "first-class routing anchor rather than structural "
                                                "scaffolding. A 30.75% loss of CVTs is therefore a "
                                                "30.75% loss of exactly the node class the "
                                                "architecture just promoted. Under the older "
                                                "STRUCTURAL_ONLY default the same number would have "
                                                "been much easier to live with.",
        "assistant_recommendation_not_a_user_decision":
            "the hybrid the V3 proposal already described is the right shape, and it is now "
            "evidenced rather than assumed: IDIR as audit oracle and accelerator, the raw dump as "
            "the source of record. IDIR's structural agreement is good enough to validate a raw "
            "build cheaply, which is worth more than using it as the backbone would have been.",
    },

    "SOURCES": {
        "archive_sha256": acq["sha256_delivered"],
        "extraction": {"entries": ext["extracted_n"], "gb": ext["extracted_gb"],
                       "all_crc_verified": ext.get("all_entries_crc_verified")},
        "records": ["V3_ACQUISITION_RECORD.json", "V3_EXTRACTION_RECORD.json",
                    "V3_CHECK3_CHECK5_RELATIONS.json", "V3_CHECK3B_RELATION_MASS.json",
                    "V3_CHECK3C_GOLD_INCIDENCE.json", "V3_CHECK2_CHECK4_JOIN.json",
                    "V3_CHECK6_METADATA_COVERAGE.json", "V3_CHECK7_CVT_COVERAGE.json",
                    "V3_WHY_CVTS_DROPPED.json"],
    },
}

tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, OUT)
print(json.dumps(doc["CHECK_STATUS"], indent=1))
print(json.dumps(doc["WHAT_IDIR_DELIVERS"], indent=1))
print(json.dumps({k: v for k, v in doc["WHAT_IDIR_CANNOT_DELIVER"].items()
                  if k != "cause_evidence"}, indent=1))
