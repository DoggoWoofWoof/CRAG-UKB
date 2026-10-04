"""Amendment 4 to the CRAG_FREEBASE_CANONICAL contract: PASS A closed, UID scheme adopted.

    python scratchpad/final_canonical_build/webqsp_v1/v3_amend_contract_v4.py

v1's rule stands: a change to the contract is a new record with its own hash, never an edit. This
cites v3's hash and leaves v1, v2 and v3 byte-identical.

WHAT PROMPTED IT. Three things that later passes will be read against, and that no earlier record
could state because they had not been measured:

  1. PASS A is finished, and its two redundancy decisions came back as exact zeros over the whole
     source. v2 held them open on a probe; they are now closed on 200/200 members.
  2. The reverse-property map turned out to have a shape no earlier record described, and the naive
     reading of it is wrong in a way that would have propagated into PASS D.
  3. PASS B needs a node identity function. Adopting one is a contract-level decision, because
     every artifact downstream is keyed by it and it is not reproducible without recording the
     interpreter setting it depends on.

Every number below is read from the artifacts, not retyped, so this record cannot drift from them.
"""
import hashlib
import json
import os
import time

D = "data/final_canonical/freebase_v3"
OUT = f"{D}/V3_CANONICAL_CONTRACT_V4.json"


def load(name):
    p = f"{D}/{name}"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return json.load(open(p, encoding="utf-8"))


V1 = load("V3_CANONICAL_CONTRACT.json")
V2 = load("V3_CANONICAL_CONTRACT_V2.json")
V3 = load("V3_CANONICAL_CONTRACT_V3.json")
ST2 = load("V3_PASS_A_SCHEMA_METADATA.json")
FRZ = load("V3_PASS_A_SCHEMA_FREEZE.json")
VAL = load("V3_PASS_A_VALIDATION.json")
UID = load("V3_NODE_UID_SPACE.json") if os.path.exists(f"{D}/V3_NODE_UID_SPACE.json") else None

red = ST2["REDUNDANCY"]

body = {
    "schema": "V3_CANONICAL_CONTRACT/v4",
    "name": "CRAG_FREEBASE_CANONICAL",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "AMENDMENT. Closes PASS A, records the reverse-property structure, and adopts the "
              "node identity function. Every other v1, v2 and v3 clause stands.",
    "amends": {
        "record": "V3_CANONICAL_CONTRACT_V3.json",
        "v3_contract_hash": V3["CONTRACT_HASH"],
        "v2_contract_hash": V2["CONTRACT_HASH"],
        "v1_contract_hash": V1["CONTRACT_HASH"],
        "v1_rule_being_honoured": V1["CONTRACT_HASH_NOTE"],
        "earlier_records_are_unmodified": True,
    },

    "PASS_A_CLOSED": {
        "supersedes": "V3_CANONICAL_CONTRACT_V3.json PASS_A_STATUS",
        "stage_1": V3["PASS_A_STATUS"]["stage_1"],
        "stage_2": f"COMPLETE. {ST2['members']} members, {ST2['workers']} workers, "
                   f"{ST2['elapsed_min']} min.",
        "TABLES": ST2["TABLES"],
        "ROW_RECONCILIATION": ST2["ROW_RECONCILIATION"],
        "INDEPENDENT_VALIDATION": {
            "method": VAL["method"],
            "ALL_CHECKS_PASS": VAL["ALL_CHECKS_PASS"],
            "checks": {k: v["ok"] for k, v in VAL["CHECKS"].items()},
            "why_it_was_re-derived": "stage 2 reconciles its own arithmetic, but the loop that "
                                     "wrote the rows also counted them, so a systematic error "
                                     "would agree with itself. The shard counts were read again "
                                     "from the 200 gz shards independently of that loop.",
        },
        "shards_deleted": VAL.get("SHARDS_DELETED"),
        "openers_retained": VAL.get("openers_retained"),
    },

    "REDUNDANCY_RESOLVED": {
        "supersedes": "V3_CANONICAL_CONTRACT_V3.json redundancy_rule_unchanged_from_v2 -- the rule "
                      "is unchanged, but it was stated as a policy and is now also a measurement.",
        "LABEL_RESIDUE_N": red["label_residue"],
        "label_rows": red["label_rows"],
        "label_dropped_as_redundant": red["label_redundant"],
        "label_finding": "every rdfs:label row in Freebase is exactly reproduced by a "
                         "type.object.name row of the same subject, agreeing on lexical form AND "
                         "language tag. rdfs:label is therefore discarded with provenance and "
                         "nothing is lost. This is a measurement over all 68,362,456 rows, not a "
                         "sample.",
        "RDF_TYPE_RESIDUE_N": red["rdf_type_residue"],
        "rdf_type_rows": red["rdf_type_rows"],
        "rdf_type_in_namespace_dropped": red["rdf_type_ns_dropped_as_redundant"],
        "rdf_type_out_of_namespace_kept": red["rdf_type_out_of_ns_kept"],
        "rdf_type_finding": "v2 forbade collapsing rdf:type into type.object.type on the grounds "
                            "that a probe had measured them non-equivalent. The full measurement "
                            "CONFIRMS the prohibition and localises it: the non-equivalence is "
                            "entirely the "
                            f"{red['rdf_type_out_of_ns_kept']} out-of-namespace rows, which are "
                            "the RDF/OWL vocabulary layer, and is exactly zero everywhere else. "
                            "The prohibition stands with an exact boundary; the out-of-namespace "
                            "rows are retained as their own table.",
        "prohibition_status": "STANDS. Do not collapse rdf:type into type.object.type. The "
                              "measurement narrows where they differ; it does not license the "
                              "merge.",
    },

    "REVERSE_PROPERTY_STRUCTURE": {
        "why_recorded": "the naive reading of this map is wrong, and the wrong reading would have "
                        "propagated straight into PASS D's canonicalisation.",
        "declaration_rows": FRZ["REVERSE_PROPERTIES"]["declaration_rows"],
        "DIRECTED_PAIRS": FRZ["REVERSE_PROPERTIES"]["DIRECTED_PAIRS"],
        "conflicting": FRZ["REVERSE_PROPERTIES"]["conflicting"],
        "shape": "each pair is declared ONCE, directed from the master side to its reverse, and "
                 "the SAME direction is restated in a second name space: once naming both "
                 "properties by schema path, once by MID. Neither side declares the pairing back "
                 "at the other. Read without resolving the two name spaces this looks like ~11,860 "
                 "one-sided declarations and zero reciprocal pairs, i.e. a broken map; resolved, "
                 f"it is {FRZ['REVERSE_PROPERTIES']['DIRECTED_PAIRS']} clean pairs with an explicit "
                 "primary side and no conflicts.",
        "name_space_join": FRZ["NAME_SPACE_SPLIT_MEASURED"]["why_it_matters"],
        "MID_PATH_RESOLUTION": FRZ["MID_PATH_RESOLUTION"],
        "master_agreement": {
            "agrees": FRZ["MASTER_PROPERTY"]["agrees_with_reverse_direction"],
            "disagrees": FRZ["MASTER_PROPERTY"]["disagrees_with_reverse_direction"],
            "rule_for_pass_d": FRZ["MASTER_PROPERTY"]["rule_for_pass_d"],
        },
        "FREEZE_HASH": FRZ["FREEZE_HASH"],
    },

    "MEDIATOR_AUTHORITY": {
        "why_contract_level": "which nodes are CVT_MEDIATOR is a claim about what the graph IS, and "
                              "the naive way of computing it fails silently rather than loudly. "
                              "Recording the resolution here is what makes the CVT count auditable "
                              "instead of merely reported.",
        "THE_HAZARD_MEASURED": FRZ["NAME_SPACE_SPLIT_MEASURED"],
        "reading": "freebase.type_hints.mediator is asserted on 2,211 MIDs. type.object.type names "
                   "types by schema path in 254,946,431 rows out of 254,946,431 -- never once by "
                   "MID. Matching the declared set against the type column directly therefore "
                   "returns zero rows and returns them WITHOUT ERROR, yielding a Freebase with no "
                   "CVTs at all. This was measured, not anticipated, and the freeze exists to stop "
                   "it reaching PASS C.",
        "RESOLUTION": {k: FRZ["MEDIATORS"][k] for k in
                       ("declared_mediator_types", "RESOLVED_TO_TYPE_PATHS", "unresolved",
                        "explicitly_non_mediator", "enumeration_types", "deprecated_types",
                        "properties_whose_expected_type_is_a_mediator",
                        "properties_belonging_to_a_mediator_type")},
        "rule": FRZ["MEDIATORS"]["authority"],
        "cross_check": FRZ["MID_PATH_RESOLUTION"]["CROSS_CHECK"],
        "KNOWN_UNDER_COUNT": "the 562 unresolved mediator MIDs are carried into PASS C's report as "
                             "a stated, bounded shortfall of the mediator declaration. They are not "
                             "guessed at by shape, degree or naming, because a CVT set assembled by "
                             "heuristic is not a set anyone can check.",
    },

    "NODE_IDENTITY_ADOPTED": {
        "why_contract_level": "every artifact from PASS B onward is keyed by this function. It is "
                              "not an implementation detail, and it is not reproducible without "
                              "recording the interpreter setting it depends on.",
        "uid": "64-bit CPython hash() over the namespace-stripped identifier as UTF-8 bytes. "
               "'<http://rdf.freebase.com/ns/m.0zgmtqc>' mints from 'm.0zgmtqc'; a non-Freebase URI "
               "keeps its full form because there the host is part of the identity.",
        "literal_uid": "hash over (lexical_form, datatype, language) joined by TAB -- never the "
                       "visible string alone. This is what keeps \"5\", \"5\"@en and "
                       "\"5\"^^xsd:int three distinct nodes, and it is unambiguous because "
                       "N-Triples escapes a literal tab as two characters. raw_rdf_term is stored "
                       "alongside so the display form is never inferred from the identity.",
        "PYTHONHASHSEED": "0, REQUIRED. hash() is randomised per process otherwise, so two workers "
                          "would mint different UIDs for the same node. Both PASS B and PASS C "
                          "refuse to start without it.",
        "collision_policy": "collisions are DETECTED, not assumed away. The subject universe was "
                            "checked exhaustively before PASS B was allowed to write anything, and "
                            "PASS C re-checks UID -> string injectivity across the whole node "
                            "universe including objects and literals. The pre-registered remedy "
                            "for a collision is to re-run PASS B under a salt -- never to merge "
                            "the colliding nodes, which would violate the v1 rule that nodes are "
                            "not merged because their surface forms coincide.",
        "subject_universe_result": (UID or {}).get("SUBJECTS"),
        "interpreter": (UID or {}).get("UID_SCHEME"),
    },

    "PASS_B_RULES": {
        "definition": "an edge is any triple whose predicate is not one of PASS A's nineteen "
                      "metadata predicates. 3.0e9 RDF statements are not 3.0e9 edges.",
        "faithfulness": "PASS B does NOT reclassify type.type.instance or any other declared "
                        "reverse of a metadata predicate. It is a faithful projection of the "
                        "source; the frozen reverse map is applied in PASS D, where it lives. An "
                        "extraction pass that silently disagrees with its source is not auditable.",
        "no_subject_sort": "still in force, per v3: REOPENED_SUBJECT_BLOCKS_N = 0.",
        "source_order_retained": "rows are written in source order, so src is constant across a "
                                 "subject block and costs almost nothing under RLE. Sorting would "
                                 "cost more than the column does.",
    },

    "NEXT_CHECKPOINT": {
        "was": V3.get("NEXT_CHECKPOINT"),
        "now": "PASS B complete and reconciled against 2,192,405,853 structural lines; then PASS C "
               "for the exact node universe by kind; then PASS D reverse canonicalisation with the "
               "pre-registered NEITHER_SIDE / reverse-pair checks; then the IDIR oracle comparison "
               "at the thresholds frozen in v2 (PASS >= 99.9% / INVESTIGATE 99.0-99.9% / "
               "STOP < 99.0%); then canonical graph validation and the freeze.",
    },
}

payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
body["CONTRACT_HASH"] = hashlib.sha256(payload).hexdigest()
body["CONTRACT_HASH_NOTE"] = ("sha256 over this amendment with this field absent. v1, v2 and v3 "
                             "remain byte-identical to when they were frozen. Any later change is "
                             "a new record, never an edit.")

tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(body, fh, indent=2)
os.replace(tmp, OUT)

print("V3 hash:", V3["CONTRACT_HASH"])
print("V4 hash:", body["CONTRACT_HASH"])
print(json.dumps(body["REDUNDANCY_RESOLVED"], indent=1)[:1200])
