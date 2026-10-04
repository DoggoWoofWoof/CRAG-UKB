"""Record the V3 source assessment: what was verified, what is broken, what is unmeasurable."""
import json, os, time

OUT = "data/final_canonical/webqsp/V3_SOURCE_ASSESSMENT.json"
DEF = "data/final_canonical/webqsp/V3_LITERAL_DEFICIT.json"

d = json.load(open(DEF, encoding="utf-8"))

doc = {
    "schema": "V3_SOURCE_ASSESSMENT/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "proposal": "CRAG_FREEBASE_CANONICAL (V3): source-derived, query-independent, MID-preserving, "
                "CVT-preserving, literal-preserving full-scale Freebase graph, built from the "
                "official RDF dump with IDIR FB+CVT-REV as an audit oracle.",
    "directive_note": "directive section 5 forbade downloading full Freebase. That section governs "
                      "the V1 ROG build, which is closed and unchanged. The user has revised lane "
                      "scope to add V3; this is a scope change by its owner, not an override.",

    "CLAIMS_VERIFIED": {
        "idir_filter": {
            "claim": "IDIR keeps only triples whose subject AND object match ^/m/|^/g/",
            "status": "CONFIRMED VERBATIM",
            "source": "https://github.com/idirlab/freebases/blob/main/DataPreparationScripts/FB1.sh",
            "quoted": "WHERE subject REGEXP '^/m/|^/g/' AND object REGEXP '^/m/|^/g/' AND "
                      "predicate NOT REGEXP ...",
            "consequence": "literal-valued facts are removed before any variant is constructed.",
        },
        "idir_variant_statistics": {
            "status": "CONFIRMED, all figures match the user's citation",
            "FB+CVT-REV": {"entities": 59894890, "relations": 2641, "triples": 134213735},
            "FB-CVT-REV": {"entities": 46069321, "relations": 3055, "triples": 125124274},
            "FB-CVT+REV": {"entities": 46077533, "relations": 5028, "triples": 238981274},
            "FB+CVT+REV": {"entities": 59896902, "relations": 4425, "triples": 244112599},
            "derived_facts_worth_keeping": {
                "cvt_nodes_add_entities": 59894890 - 46069321,
                "reverse_removal_drops_triples": 244112599 - 134213735,
                "reverse_removal_drops_pct": round(
                    100.0 * (244112599 - 134213735) / 244112599, 1),
                "note": "reverse removal cuts 45% of triples. That is the scale of the "
                        "double-counted-support problem the user flagged, quantified.",
            },
        },
        "freebase_dump_description": {
            "status": "CONFIRMED on developers.google.com/freebase",
            "triples": "1.9 billion", "gzip": "22 GB", "uncompressed": "250 GB",
            "license": "CC-BY", "format": "N-Triples, UTF-8, gzip",
        },
    },

    "CLAIMS_THAT_DID_NOT_HOLD": {
        "official_dump_is_downloadable": {
            "status": "BROKEN",
            "probe": "HTTP HEAD https://storage.googleapis.com/freebase-public/rdf/"
                     "freebase-rdf-latest.gz",
            "result": "403 Forbidden -- 'Anonymous caller does not have storage.objects.get "
                      "access'. Bucket listing is likewise denied.",
            "reading": "the documentation page survives and describes the dump, but the Google "
                       "Cloud Storage object is no longer anonymously readable. The plan cannot "
                       "source from Google directly.",
            "working_alternative": {
                "url": "https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz",
                "probe_result": "HTTP 200 after redirect",
                "content_length_bytes": 31305093084,
                "content_length_gb": 31.3,
                "size_discrepancy": "31.3 GB against the documented 22 GB. Different compression "
                                    "or a different snapshot; the mirror must be checksummed and "
                                    "its provenance recorded before it is trusted as 'the official "
                                    "dump'.",
            },
        },
    },

    "LOCAL_CONSTRAINTS": {
        "disk_free_gb": 82,
        "dump_gb": 31.3,
        "idir_package_gb": 14.1,
        "both_sources_gb": 45.4,
        "headroom_after_sources_gb": 36.6,
        "uncompressed_250gb_is_impossible": True,
        "consequence": "the builder MUST stream from gzip and never materialise the decompressed "
                       "dump. That is normal for N-Triples but it constrains the design: one "
                       "streaming pass, external sort or on-disk maps for joins, no random access "
                       "to the source.",
    },

    "THE_LITERAL_DEFICIT_MEASURED": {
        "headline": "the empirical case for literal preservation is much weaker than the "
                    "architectural case, and this is stated against the proposal it was gathered "
                    "to support.",
        "measured_on": "RoG V1, the only local substrate that retained any literals",
        "edges_literal_incident": d["EDGES"]["literal_incident"],
        "edges_total": d["EDGES"]["total"],
        "edges_literal_incident_pct": d["EDGES"]["literal_incident_pct"],
        "gold_answers_that_are_literals_distinct": d["GOLD_ANSWERS_THAT_ARE_LITERALS"]["distinct"],
        "gold_distinct_pct": d["GOLD_ANSWERS_THAT_ARE_LITERALS"]["distinct_pct_of_gold"],
        "gold_occurrence_pct": d["GOLD_ANSWERS_THAT_ARE_LITERALS"]["occurrence_pct"],
        "cvt_emptied_by_the_filter": d["CVT_DAMAGE"]["cvt_losing_every_argument_under_the_filter"],
        "gold_literal_character": "predominantly postal codes (08540, 08544 ...), i.e. "
                                  "location.postal_code-shaped facts, plus dates and years.",

        "WHY_THE_NUMBER_CANNOT_BE_TRUSTED_AS_AN_UPPER_BOUND": {
            "reason_1": "RoG collapsed names onto MIDs, so a literal STRING is indistinguishable "
                        "from a collapsed entity name. Only date- and numeric-shaped literals are "
                        "identifiable. Every count is a floor.",
            "reason_2": "RoG's own extraction already discarded most literals before we saw it: "
                        "14,515 VALUE_LITERAL nodes in 2,592,894 is 0.56%, which is nothing like "
                        "the literal density of real Freebase, where names, dates, measurements "
                        "and descriptions are all literal-valued.",
            "reason_3": "the NSM sibling, the obvious MID-preserving cross-check, ALSO strips "
                        "literals -- 149 non-MID entries of 1,441,420 (webqsp) and 226 of "
                        "2,429,346 (CWQ).",
            "conclusion": "NO existing WebQSP-lineage graph preserves literal-valued facts. The "
                          "true deficit is therefore NOT MEASURABLE from any local substrate, and "
                          "measuring it is itself a reason to acquire the raw dump -- but that is "
                          "an argument from ignorance, not from evidence. It should not be "
                          "presented as if the 0.63% figure supported the proposal.",
        },
    },

    "ENCODER_FEASIBILITY_ORDER_OF_MAGNITUDE": {
        "basis": "V1 measured 57.3M est tokens over 2,592,894 nodes = ~22 tokens/node under "
                 "ALL_NODE + entity NAME_ONLY.",
        "v3_nodes_if_fb_cvt_rev_plus_literals": "59.9M MID/CVT nodes plus a literal population of "
                                                "unknown size",
        "naive_token_extrapolation": "~1.3B tokens at the same tokens/node rate",
        "cost_note": "against the measured Modal per-token rate recorded in memory "
                     "(0.086-0.129 USD/Mtok), 1.3B tokens is roughly 115-170 USD for one "
                     "representation pass, so order 200-350 USD for dense plus SPLADE.",
        "confidence": "LOW. The tokens/node rate is taken from a graph whose text is 88% CVT "
                      "records; V3's mix is unknown, literals are unaccounted, and the rate was "
                      "measured for a different corpus. Treat as a feasibility signal only -- it "
                      "says the bill is plausibly hundreds of dollars rather than tens of "
                      "thousands, nothing more precise.",
        "local_encoding": "OUT OF REACH. Recorded ceiling is ~2.57M bare-text nodes locally; V3 is "
                          "over 20x that.",
    },

    "DESIGN_POINTS_RAISED": {
        "reverse_property_canonicalisation": "keeping one stored edge plus reverse_relation "
                                             "metadata is right, but WHICH direction is canonical "
                                             "needs a deterministic rule fixed in advance. If our "
                                             "choice differs from IDIR's, the audit-oracle "
                                             "comparison breaks on direction rather than on "
                                             "content. Fix the rule, then compare.",
        "metadata_vs_structure": "splitting type.object.name / type / alias / key out of the "
                                 "structural graph is right and matches section 2: those become "
                                 "node metadata feeding rendering and classification, never "
                                 "structural topology.",
        "identity_rule_carries_over": "node_uid stays the MID; the readable surface is a derived "
                                      "view. The V1 lesson (name-keying destroyed 97,557 "
                                      "identities and 133,602 triples at only 19.87% coverage) is "
                                      "exactly what V3 must not repeat.",
        "literal_node_identity_is_an_open_problem": "a literal has no MID. Its canonical identity "
                                                    "must be defined deterministically -- most "
                                                    "likely (value, datatype) rather than the bare "
                                                    "string, or '1992' the year merges with '1992' "
                                                    "the string and we recreate the name-collision "
                                                    "failure in a new place.",
    },

    "RECOMMENDED_SEQUENCE": [
        "1. Do NOT download the 31.3 GB dump yet. Download the 14.1 GB IDIR package first and run "
        "the user's three checks -- WebQSP topic/gold MID coverage, relation coverage, and the "
        "structural-edge agreement against V1. Those need no raw dump.",
        "2. The literal-deficit check the user placed third CANNOT be answered by that package or "
        "by any local substrate, for the reasons recorded above. It moves to the raw-dump stage.",
        "3. Only if IDIR coverage is good, acquire the mirror, checksum it, record provenance, and "
        "stream one pass to extract literal-valued facts and metadata.",
        "4. Decide literal node identity BEFORE that pass, not during it.",
    ],
    "no_downloads_performed": True,
}
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, OUT)
print("written", OUT)
