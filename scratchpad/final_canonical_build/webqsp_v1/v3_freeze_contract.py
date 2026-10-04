"""Freeze the CRAG_FREEBASE_CANONICAL contract before any raw byte is acquired.

    python scratchpad/final_canonical_build/webqsp_v1/v3_freeze_contract.py

Written BEFORE the download deliberately, so the contract cannot be shaped by what the dump turns
out to contain. Same discipline as the V1 pre-registrations: the document that says what we intended
has to predate the evidence about whether it was easy.

Everything under DECIDED_BY_USER is the user's wording, transcribed. Everything under
EVIDENCE_* is a measured number carried over from checks 1-7, with its source record named.
"""
import hashlib, json, os, time

D = "data/final_canonical/freebase_v3"
OUT = f"{D}/V3_CANONICAL_CONTRACT.json"
c6 = json.load(open(f"{D}/V3_CHECK6_METADATA_COVERAGE.json", encoding="utf-8"))
c7 = json.load(open(f"{D}/V3_CHECK7_CVT_COVERAGE.json", encoding="utf-8"))
c24 = json.load(open(f"{D}/V3_CHECK2_CHECK4_JOIN.json", encoding="utf-8"))
why = json.load(open(f"{D}/V3_WHY_CVTS_DROPPED.json", encoding="utf-8"))
lock = json.load(open(f"{D}/V3_DESIGN_LOCK.json", encoding="utf-8"))

cov = c6["CHECK_6_COVERAGE"]
kinds = c6["CHECK_6_COVERAGE"]["by_v1_node_kind"]
all_named = cov["overall"]["named"]
all_n = cov["overall"]["n"]
cvt_n = kinds["CVT_MEDIATOR"]["n"]
v1_total = sum(k["n"] for k in kinds.values())
v1_named = sum(k["named"] for k in kinds.values())

body = {
    "schema": "V3_CANONICAL_CONTRACT/v1",
    "name": "CRAG_FREEBASE_CANONICAL",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "FROZEN BEFORE ACQUISITION",
    "written_before": "any byte of the 31.3 GB raw mirror was downloaded, so the contract cannot "
                      "be shaped by what the dump turns out to make convenient.",

    "LANE_VERDICTS": {
        "IDIR_BACKBONE_AS_V3": {
            "verdict": "REFUTED",
            "not_because": "IDIR is bad. Its retained graph is remarkably clean -- 99.998% exact "
                           "structural agreement with zero reverse-only cases on every predicate "
                           "it keeps.",
            "but_because": "its preprocessing intentionally removes information CRAG treats as "
                           "first-class.",
            "decisive_evidence": {
                "v1_cvt_nodes": why["v1_cvt_nodes"],
                "absent_from_FB+CVT-REV": why["cvt_missing_from_FB+CVT-REV"],
                "absent_pct": round(100 * why["cvt_missing_from_FB+CVT-REV"] / why["v1_cvt_nodes"], 3),
                "rescued_by_dropping_the_anti_join": why["of_those_present_in_FB+CVT+REV"],
                "cause": why["VERDICT"],
                "reading": "FB+CVT+REV differs from the backbone in exactly one step -- the reverse "
                           "anti-join -- and rescues 97 of 488,432 nodes. The cause is isolated to "
                           "the MID-only endpoint filter, not reverse-edge removal.",
                "source": "V3_WHY_CVTS_DROPPED.json",
            },
            "why_it_is_decisive_for_CRAG_specifically": "the retrieval-role decision made every CVT "
                                                        "retrieval-eligible and a first-class "
                                                        "partition-routing anchor. Adopting IDIR as "
                                                        "the backbone would knowingly delete "
                                                        "30.76% of the node class the architecture "
                                                        "had just promoted.",
        },
        "RAW_FREEBASE_REQUIRED_FOR_V3": {
            "verdict": "SUPPORTED",
            "because": "the missing CVTs and literals are now directly measured rather than "
                       "hypothesised.",
            "measured": {
                "cvt_nodes_unavailable_from_any_idir_variant": why["of_those_absent_from_both"],
                "nsm_triple_mass_absent_pct": c24["CHECK_4_STRUCTURAL_AGREEMENT"]["ABSENT_pct"],
                "of_which_missing_nodes_pp": round(
                    100 - c24["CHECK_4_STRUCTURAL_AGREEMENT"]["joinable_pct"], 3),
                "literal_valued_facts": "absent by construction from every IDIR variant",
                "reverse_property_mapping": "absent from the archive (FINDING_1), leaving "
                                            "INVARIANT_2 bounded but uncomputable",
            },
        },
    },

    "ROLES": {
        "IDIR": ["audit oracle", "metadata accelerator", "structural validation reference",
                 "NOT the canonical V3 graph"],
        "RAW_FREEBASE": ["source of record for CRAG_FREEBASE_CANONICAL"],
        "how_they_combine": "raw Freebase is authority; IDIR metadata is fast lookup and "
                            "validation, with raw filling whatever IDIR lacks.",
        "the_oracle_is_load_bearing": "IDIR proved its retained graph is trustworthy where it "
                                      "exists. That means the raw parser does not have to be built "
                                      "blind: every stage can be validated against 134,142,730 "
                                      "delivered triples with a known 99.998% agreement target.",
        "and_it_doubles_as_a_provenance_check": "the archive.org mirror is a third-party custody "
                                                "copy; Google's own endpoint is dead (403), so "
                                                "there is no upstream digest to check against. If "
                                                "a raw build reproduces IDIR's retained triples at "
                                                "the expected rate, that is independent evidence "
                                                "the mirror is the dataset it claims to be. Recorded "
                                                "as a real limitation with a real mitigation, not "
                                                "as a solved problem.",
    },

    "DECIDED_BY_USER": {
        "IDENTITY": {
            "m_star_mid": "preserve exactly",
            "g_star_mid": "preserve exactly",
            "cvt_mid": "preserve exactly",
            "literal": "canonical RDF-term identity",
        },
        "KEEP": ["entity <-> entity facts", "entity <-> CVT facts", "CVT <-> entity facts",
                 "CVT <-> literal facts", "entity <-> literal facts", "types", "names",
                 "aliases / useful metadata", "reverse_property schema", "relation identities"],
        "DO_NOT_DO": ["name-as-node-identity", "MID-only object filtering",
                      "query-conditioned extraction", "silent literal coercion",
                      "blind duplicate reverse edges"],
        "LITERAL_UID": {
            "rule": "literal_uid = lexical form + datatype + language tag / literal class",
            "not": 'the visible string alone, e.g. "1992"',
            "consequence": 'plain "1992", a typed gYear 1992 and another typed numeric 1992 cannot '
                           "silently become one node.",
            "provenance": "the original RDF lexical representation is preserved separately.",
            "first_locked": "V3_DESIGN_LOCK.json RULE_1, restated here unchanged.",
            "why_restated": "it is the same identity-collapse bug class that name-keying caused in "
                            "the RoG lineage, in a different form. Restating it in the contract "
                            "keeps the two locks from drifting apart.",
        },
        "SOURCE_EXPANSION": "the source must NEVER be expanded to its documented ~250 GB "
                            "uncompressed form. Stream from gzip throughout.",
    },

    "PIPELINE": {
        "shape": "streaming staged, not one pass to a perfect 60M+ node Parquet graph",
        "reason": "space is the real constraint, not compute.",
        "PASS_A": {"input": "31.3 GB gzip mirror, streamed",
                   "does": "schema + metadata extraction",
                   "emits": ["reverse-property map", "names", "types", "aliases"]},
        "PASS_B": {"does": "subject-matter triples -> canonical node IDs -> external-sort edge shards"},
        "PASS_C": {"does": "literal terms -> typed literal nodes, under LITERAL_UID"},
        "PASS_D": {"does": "reverse canonicalisation -> final edge stream"},
        "PASS_E": {"does": "node table + relation table"},
        "note_on_pass_D": "RULE_2 aligns to IDIR's orientation, which check 4 verified against "
                          "delivered data. But CRAG stores ONE canonical edge with a "
                          "reverse_relation attribute rather than deleting the reverse predicate, "
                          "so PASS_D canonicalises rather than filters. IDIR's anti-join and CRAG's "
                          "canonicalisation agree on direction and disagree on what happens to the "
                          "other name.",
    },

    "ADMIN_MEDIATORS_ABLATION_PREREGISTERED": {
        "families": ["/common/notable_for", "/common/webpage", "/common/document",
                     "/location/geocode"],
        "approx_nodes_in_v1": 464663,
        "decision": "DO NOT REMOVE THEM FROM THE CORPUS.",
        "why": "CRAG begins from the complete graph and lets evidence say whether a family is "
               "noise. Deciding during corpus construction that they are not knowledge is exactly "
               "the move the ALL_NODE decision rejected.",
        "becomes": "ALL_NODE vs ALL_NODE - ADMIN_MEDIATORS, an ablation arm alongside the existing "
                   "ENTITY_ONLY / ALL_NODE / ALL_NODE - LITERAL / ALL_NODE - CVT set.",
        "registered_before": "the raw build exists, so the arm cannot be invented after seeing "
                             "which way the numbers went.",
    },

    "EVIDENCE_CORRECTION_TO_CHECK_6": {
        "what_was_reported": f"{cov['overall']['named_pct']}% of NSM MIDs are named by IDIR.",
        "why_that_framing_understates_it": "the denominator is dominated by CVTs, which genuinely "
                                           "have no names in Freebase. 0 of "
                                           f"{cvt_n:,} V1 CVT nodes are named, correctly.",
        "named_excluding_v1_classified_cvts": round(100 * all_named / (all_n - cvt_n), 3),
        "named_among_nsm_mids_outside_v1s_mid_labelled_set": round(
            100 * (all_named - v1_named) / (all_n - v1_total), 3),
        "reading": "IDIR names essentially every ordinary entity it holds. The overall figure is a "
                   "composition artifact, not a coverage gap, and the earlier presentation of it "
                   "was misleading.",
        "unchanged": "the gold-answer and topic figures were already reported on their own "
                     "denominators and stand: answers 99.606% named / 99.966% typed, topics "
                     "99.992% both.",
    },

    "WHAT_THIS_CONTRACT_DOES_NOT_SETTLE": [
        "whether the raw dump's CVT population matches V1's 1,588,085 classification -- V1's node "
        "kinds came from a lossy RoG surface and the raw build gets to reclassify from source",
        "the CVT-vs-entity classifier itself: types alone were measured NOT separable (68.1% of CVT "
        "type mass sits on types entities also carry, against a 5% pre-registered threshold). A "
        "ratio-threshold rule looks promising but needs its own pre-registration",
        "the encoder bill for a 60M+ node corpus, which remains a delayed-representation question "
        "and not a corpus-size question",
    ],
}

payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
body["CONTRACT_HASH"] = hashlib.sha256(payload).hexdigest()
body["CONTRACT_HASH_NOTE"] = "sha256 over the contract with this field absent. Any later change to "\
                             "the contract must appear as a new record with its own hash, never as "\
                             "an edit that leaves this one looking prescient."

tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(body, fh, indent=2)
os.replace(tmp, OUT)
print(json.dumps(body["LANE_VERDICTS"]["IDIR_BACKBONE_AS_V3"]["decisive_evidence"], indent=1))
print(json.dumps(body["EVIDENCE_CORRECTION_TO_CHECK_6"], indent=1))
print("CONTRACT_HASH", body["CONTRACT_HASH"])
