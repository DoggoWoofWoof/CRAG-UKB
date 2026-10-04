"""Fix the two V3 design rules before the IDIR package is inspected.

The rules themselves were fixed by the user before any byte landed; acquisition was started
concurrently with writing this file. Neither rule can be informed by the data, which is the point.
"""
import json, os, time

OUT = "data/final_canonical/freebase_v3/V3_DESIGN_LOCK.json"

doc = {
    "schema": "V3_DESIGN_LOCK/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "lane": "CRAG_FREEBASE_CANONICAL (V3)",
    "fixed_by": "user, 2026-09-06, before acquisition",
    "why_this_file_exists": "both rules decide graph IDENTITY. Fixing them after looking at the "
                            "package would let the data choose the rule, which is the failure V1 "
                            "spent section 2 avoiding.",

    "RULE_1_LITERAL_IDENTITY": {
        "statement": "never use the visible string alone. Canonical literal identity is the tuple "
                     "(lexical_form, datatype, language_tag / source literal class).",
        "provenance": "the original RDF lexical representation is preserved separately as "
                      "provenance, never as the key.",
        "no_implicit_coercion": True,
        "worked_consequence": "plain \"1992\", a typed gYear 1992, and a typed numeric 1992 are "
                              "THREE distinct canonical nodes. They may render identically; they "
                              "may never merge.",
        "rationale": "this is the V1 name-collision failure moved one level down. Keying a literal "
                     "on its visible string is exactly the mistake that cost the released RoG "
                     "graph 97,557 identities and 133,602 triples at only 19.87% naming coverage. "
                     "A datatype-blind literal key would recreate it in a new place.",
        "display_is_not_identity": "as everywhere else in this build, the readable surface is a "
                                   "derived view over the tuple, never the key.",
    },

    "RULE_2_REVERSE_ORIENTATION": {
        "statement": "align to IDIR's orientation by default rather than inventing a CRAG "
                     "orientation, so the structural comparison against FB+CVT-REV is clean.",
        "status": "DEFAULT, NOT FROZEN. Held provisionally until check 5 passes.",
        "storage_design": "one canonical stored edge plus reverse_relation metadata. Traversal "
                          "remains bidirectional; the graph stops asserting two independent facts "
                          "where Freebase recorded one.",
        "why_it_matters_quantitatively": "IDIR goes from 244,112,599 triples with reverse edges to "
                                         "134,213,735 without: 45% of the graph is redundant "
                                         "reverse representation. Matching the convention before "
                                         "H4 avoids inflating degree and hypergraph work for zero "
                                         "new fact content.",

        "VERIFICATION_STATUS": {
            "reverse_property_derivation": {
                "claim": "IDIR derives reverse_properties from Freebase's "
                         "/type/property/reverse_property assertions",
                "status": "CONFIRMED VERBATIM",
                "quoted": "CREATE TABLE reverse_properties SELECT * FROM freebase WHERE predicate "
                          "= '/type/property/reverse_property'",
                "source": "DataPreparationScripts/FBDataDump.sh",
            },
            "retained_orientation": {
                "claim": "the -REV variant removes triples whose predicate occurs on the OBJECT "
                         "side of that mapping, thereby retaining the subject-side relation",
                "status": "NOT CONFIRMED. FBDataDump.sh creates the reverse_properties table but "
                          "contains no subsequent logic showing which side is dropped; the "
                          "removal step lives elsewhere in the pipeline and was not readable from "
                          "the sources checked.",
                "consequence": "check 5 is LOAD-BEARING, not confirmatory. The retained "
                               "orientation must be established empirically from the delivered "
                               "data before rule 2 is frozen.",
            },
        },

        "CHECK_5_MUST_ENUMERATE": [
            "the actual retained orientation, measured on delivered triples rather than assumed",
            "pathological pairs where BOTH predicates would be considered removable",
            "reverse assertions with no inverse present (missing mappings)",
            "self-inverse predicates, where subject and object side coincide",
            "asymmetric pairs where one direction carries triples the other does not",
        ],
        "if_ambiguity_is_found": "do NOT silently pick a side. Record the ambiguous set, choose a "
                                 "deterministic tie-break stated in advance of applying it, and "
                                 "report how many pairs the tie-break decided.",
    },

    "ACQUISITION": {
        "record": "https://zenodo.org/records/7909511",
        "file": "idirlab-freebases.zip",
        "expected_bytes": 14148416296,
        "expected_md5": "170689b7aad9f029566a4deb36605b01",
        "url": "https://zenodo.org/api/records/7909511/files/idirlab-freebases.zip/content",
        "raw_mirror_NOT_downloaded": "the 31.3 GB archive.org Freebase dump is deliberately not "
                                     "acquired at this stage.",
    },

    "DISK_RISK_FLAGGED_BEFORE_EXTRACTION": {
        "free_gb_at_start": 82,
        "zip_gb": 14.1,
        "free_after_download_gb": 67.9,
        "risk": "the record is a single ZIP holding all FOUR variants plus mapping files. "
                "FB+CVT+REV alone is 244,112,599 triples; extracting everything could plausibly "
                "exceed the remaining space.",
        "mitigation": "extract SELECTIVELY -- FB+CVT-REV plus the mapping/support files only -- "
                      "and list the archive before extracting anything.",
    },

    "CHECK_ORDER_PREREGISTERED": [
        "1. byte provenance: filename, exact bytes, SHA256, source URL",
        "2. WebQSP MID coverage: topic entities, answer MIDs where available, known MID populations",
        "3. relation coverage: how many of V1's 7,058 relation keys map into IDIR",
        "4. structural agreement: V1 MID-resolvable edges against FB+CVT-REV",
        "5. reverse-pair audit: retained orientation, ambiguous and missing mappings",
        "6. metadata coverage: names and types for m. and especially the g. population",
        "7. CVT coverage: V1 classifier behaviour against IDIR's mediator population where "
        "mappings overlap",
    ],
    "gate_on_the_raw_mirror": "only after these pass do we decide whether the raw dump is needed "
                              "for literal-valued facts, metadata absent from IDIR, and whatever "
                              "IDIR's subject-matter filter removed.",

    "STANDING_UNKNOWN_CARRIED_FORWARD": "the g. population is the sharpest open question. In V1, "
                                        "0 of 405,358 g. MIDs have any outgoing edge, so every g. "
                                        "classification rests on incoming evidence alone. Check 6 "
                                        "is the first opportunity to learn whether that is a "
                                        "property of Freebase g. objects or an artifact of RoG's "
                                        "extraction -- a question V1 could not settle.",
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, OUT)
print("written", OUT)
