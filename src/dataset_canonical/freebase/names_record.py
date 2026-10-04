# -*- coding: utf-8 -*-
"""The V4 name-hierarchy record: scratchpad/fb4/FREEBASE_NAMES_V4.json (self-hashed, BUILDS_ON the
frozen V1 contract and the V3 closure; those records are not edited).

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/names_record.py

Every number is copied from a measured artifact (NAMES_V4_BASE / NAMES_V4_FINAL / FLOOR_NAMES /
V4_FLOOR_CENSUS) and cross-checked for arithmetic consistency here; nothing is recomputed.
"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical"))
from freeze_canonical import finfo, rj, utc, write_record  # noqa: E402

V3 = "data/final_canonical/freebase_v3"
SCR = "scratchpad/fb4"
N = 301977131

base = rj(SCR + "/NAMES_V4_BASE.json")
final = rj(SCR + "/NAMES_V4_FINAL.json")
floor = rj(SCR + "/FLOOR_NAMES.json")
census = rj(SCR + "/V4_FLOOR_CENSUS.json")
v1 = rj(V3 + "/NAME_HIERARCHY_CONTRACT_V1.json")
closure = rj(V3 + "/FREEBASE_V3_CLOSURE_STATUS.json")
actual = rj(V3 + "/V3_ACTUAL_NAME_LAYER_FROZEN.json")

bk = final["by_name_kind"]
assert sum(bk.values()) == N, bk
assert bk["GENERATED_FLOOR"] == final["floor_rows_filled"] == floor["floor_nodes"] == census["FLOOR"]["n"] == 7221354
assert bk["ORIGINAL"] == census["by_tier"]["ORIGINAL"] == v1["PRECEDENCE"][0]["nodes"] == 157420227
assert bk["INFERRED"] == census["by_tier"]["INFERRED_ADMISSIBLE"] == v1["PRECEDENCE"][1]["nodes"] == 62022081
assert bk["RECOVERED_ORIGINAL"] + bk["RECOVERED_EXTERNAL"] == census["by_tier"]["RECOVERED"] == 534532
assert bk["URI_SELF"] == census["by_tier"]["URI_SELF"] == 74454117 and bk["KEY_SEMANTIC"] == census["by_tier"]["KEY_SEMANTIC"] == 324820
assert final["CHECKS"]["empty_names"] == 0 and final["CHECKS"]["bare_mid_names_in_derived_kinds_3_to_6"] == 0, final["CHECKS"]
assert set(final["bare_mid_by_name_kind"]) <= {"ORIGINAL"}, final["bare_mid_by_name_kind"]
# a derived name may BEGIN with the word Unknown/Unnamed only because a Freebase name it is built from does
# ("Unknown Pleasures -- notable for Musical Release", "Unknown Soldier (Italian Wikipedia)"); the generated floor and
# the key renderings never do
assert final["placeholder_by_name_kind"].get("GENERATED_FLOOR", 0) == 0 and final["placeholder_by_name_kind"].get("KEY_SEMANTIC", 0) == 0, \
    final["placeholder_by_name_kind"]
assert base["by_name_kind"]["GENERATED_FLOOR"] == 7221354 and base["CHECK_floor_set_equals_census"]
v1_floor = v1["PRECEDENCE"][2]["nodes"]
assert v1_floor == bk["URI_SELF"] + bk["KEY_SEMANTIC"] + bk["RECOVERED_ORIGINAL"] + bk["RECOVERED_EXTERNAL"] + bk["GENERATED_FLOOR"], \
    (v1_floor, bk)

rec = {
    "RECORD": "CRAG_FREEBASE_NAME_HIERARCHY_V4",
    "created_utc": utc(),
    "STATUS": "FROZEN",
    "BUILDS_ON": {
        "NAME_HIERARCHY_CONTRACT_V1": {"file": V3 + "/NAME_HIERARCHY_CONTRACT_V1.json", "RECORD_SHA256": v1["RECORD_SHA256"], "not_edited": True},
        "FREEBASE_V3_CLOSURE_STATUS": {"file": V3 + "/FREEBASE_V3_CLOSURE_STATUS.json", "RECORD_SHA256": closure["RECORD_SHA256"], "not_edited": True,
                                       "item_advanced": "the floor: V1 tier 3 (82,534,823 nodes resolving to the frozen display name, 7,221,354 of them "
                                                        "'Unnamed ...') is replaced by four disjoint, provenance-graded kinds; nothing frozen is modified"},
        "V3_ACTUAL_NAME_LAYER_FROZEN": {"file": V3 + "/V3_ACTUAL_NAME_LAYER_FROZEN.json", "RECORD_SHA256": actual.get("RECORD_SHA256"),
                                        "file_sha256": finfo(V3 + "/V3_ACTUAL_NAME_LAYER_FROZEN.json")["sha256"], "not_edited": True,
                                        "ruling_used": "IDIR /type/object/name rows are Freebase name assertions (audit-oracle rule is about STRUCTURE); "
                                                       "the cascade's IDIR_FREEBASE_NAME rows are therefore served as RECOVERED_ORIGINAL"},
    },
    "USER_DIRECTIVE": "complete freebase all nodes should have names etc",
    "SCOPE": "one display name for every node of the canonical Freebase universe (301,977,131), with a name_kind that says what the string is. "
             "States how the frozen layers (overlay_v1, cascade_names, inference_overlay_v1 + admissibility v1/v2, semantic_kind_v2.1) compose, "
             "and adds one new generated layer for the 7,221,354 nodes that no frozen layer named.",
    "PRECEDENCE": [
        {"kind": 0, "name": "ORIGINAL", "nodes": bk["ORIGINAL"], "source": "overlay_v1 is_original_name == true",
         "what_it_is": "the name the source carries: /type/object/name of the current dump, the historical (deleted) dump name, a literal's lexical form, a schema label",
         "authority": "ACTUAL (SOURCE_DERIVED)"},
        {"kind": 1, "name": "RECOVERED_ORIGINAL", "nodes": bk["RECOVERED_ORIGINAL"], "source": "_acquisition/cascade_names.parquet is_original_name == true "
                                                                                         "(FREEBASE_HISTORICAL_ASSERTION, FREEBASE_ALIAS, IDIR_FREEBASE_NAME)",
         "what_it_is": "a Freebase name assertion for this MID found in an earlier published Freebase dump or in IDIR's name rows; exact-identifier match only",
         "authority": "ACTUAL (SOURCE_DERIVED, historical)"},
        {"kind": 2, "name": "RECOVERED_EXTERNAL", "nodes": bk["RECOVERED_EXTERNAL"], "source": "_acquisition/cascade_names.parquet is_original_name == false "
                                                                                         "(WORDNET_SENSE_LEMMA, EXTERNAL_AUTHORITY_EXACT, CURRENT_WIKIDATA_EXACT, "
                                                                                         "EXTERNAL_URL_ARCHIVE_EXACT, SAMSUNG_WIKIDATA_EXACT, EXTERNAL_URL_LIVE_EXACT)",
         "what_it_is": "the label an external authority gives the exact identifier this node carries (a WordNet sense key, a Wikidata item with the Freebase id, an authority id, an archived URL title)",
         "authority": "ACTUAL (EXTERNAL, exact identifier lookup -- no fuzzy matching)"},
        {"kind": 3, "name": "URI_SELF", "nodes": bk["URI_SELF"], "source": "overlay_v1 URI_DERIVED",
         "what_it_is": "the node IS a URI; the name is that URI rendered", "authority": "IDENTIFIER"},
        {"kind": 4, "name": "KEY_SEMANTIC", "nodes": bk["KEY_SEMANTIC"], "source": "overlay_v1 FREEBASE_KEY_EXACT",
         "what_it_is": "the node's own readable Freebase key (e.g. a Wikipedia title key) rendered", "authority": "IDENTIFIER"},
        {"kind": 5, "name": "INFERRED", "nodes": bk["INFERRED"], "source": "inference_overlay_v1 with effective reject_reason 0 (v1 gate, then the v2 amendment)",
         "what_it_is": "a deterministic rendering of the node's own relations (V1 tier 2, unchanged)", "authority": "GENERATED -- never an authentic label"},
        {"kind": 6, "name": "GENERATED_FLOOR", "nodes": bk["GENERATED_FLOOR"], "source": "scratchpad/fb4/floor_names.parquet (floor_names.py, rules below)",
         "what_it_is": "an evidence-based description of what the node is in the graph, always carrying the node's MID so it stays unique and is "
                       "recognisable as generated: the type it declares, the relation and neighbour that remain, the key it carries, or the "
                       "plain fact that no other fact remains", "authority": "GENERATED -- never an authentic label"},
    ],
    "RESOLUTION_RULE": "name(node) = the FIRST of: original overlay name (is_original_name) | cascade name (is_original_name -> RECOVERED_ORIGINAL else "
                       "RECOVERED_EXTERNAL) | overlay URI_DERIVED | overlay FREEBASE_KEY_EXACT | admissible inferred name | generated floor name. "
                       "Each row takes exactly one kind, so the kinds are disjoint by construction.",
    "EFFECTIVE_REJECT_REASON": v1["EFFECTIVE_REJECT_REASON"],
    "FLOOR": {
        "definition": "the 7,221,354 nodes with no original name, no cascade name, no URI/key rendering and no admissible inferred name: "
                      "INFERRED_REJECTED 2,015,943 (reason 1 bookkeeping relation 2,015,907; reason 2 unreadable identity 36) + NO_INFERENCE_ROW 5,205,411",
        "by_kind": census["FLOOR"]["by_kind"],
        "evidence_used": floor.get("evidence"),
        "rules": floor.get("rules"),
        "by_rule": floor.get("by_rule"),
        "by_family": floor.get("by_family"),
        "samples": floor.get("samples"),
        "uniqueness": floor.get("uniqueness"),
        "what_the_names_are_NOT": "not labels, not recovered, not inferred entity names. They describe the node's remaining evidence in plain words and "
                                  "embed the MID. A consumer wanting only authentic labels filters name_kind <= 2.",
    },
    "CHECKS": {
        "rows": final["rows"], "rows_equals_universe": final["rows"] == N,
        "kinds_sum_to_universe": sum(bk.values()) == N,
        "floor_set_equals_census": base["CHECK_floor_set_equals_census"],
        "empty_names": final["CHECKS"]["empty_names"],
        "bare_mid_names_all_kinds": final["CHECKS"]["bare_mid_names"], "bare_mid_by_name_kind": final["bare_mid_by_name_kind"],
        "bare_mid_names_in_derived_kinds_3_to_6": final["CHECKS"]["bare_mid_names_in_derived_kinds_3_to_6"],
        "placeholder_names_all_kinds": final["CHECKS"]["placeholder_names"], "placeholder_by_name_kind": final["placeholder_by_name_kind"],
        "placeholder_names_in_derived_kinds_3_to_6": final["CHECKS"]["placeholder_names_in_derived_kinds_3_to_6"],
        "placeholder_note": "the placeholder regex flags any name that BEGINS with the word Unknown/Unnamed (or is blank), so it is an upper bound: an "
                            "ORIGINAL name may be 'Unknown (deceased)' or look like a MID ('m.jpg') because the source asserted it, and a URI_SELF / "
                            "INFERRED name begins that way only when the Freebase name it is built from does ('Unknown Pleasures -- notable for "
                            "Musical Release', 'Unknown Soldier (Italian Wikipedia)', 'Unknown Hinson in The Boxmasters'; measured 2026-09-12 over "
                            "all 2,941 derived hits: 2,416 common.notable_for#0, 255 URI_DERIVED, 104 MERGE_SUCCESSOR_NAME, 166 other mediator roles, "
                            "every one carrying a neighbour's or the URI's own name). No derived name is a bare MID, and no GENERATED_FLOOR or "
                            "KEY_SEMANTIC name is a placeholder (checked per row in compose_final and, for the floor, verify_freebase).",
        "display_text_rows_checked": base["display_text_rows_checked"],
        "display_text_disagreeing_with_original_name": base["display_text_disagreeing_with_original_name"],
        "v1_tier3_floor_partition": {"v1_tier3_nodes": v1_floor, "now": {k: bk[k] for k in ("URI_SELF", "KEY_SEMANTIC", "RECOVERED_ORIGINAL", "RECOVERED_EXTERNAL", "GENERATED_FLOOR")},
                                     "sums": True},
    },
    "COVERAGE": {"nodes_total": N,
                 "actual_names(kind 0-2)": bk["ORIGINAL"] + bk["RECOVERED_ORIGINAL"] + bk["RECOVERED_EXTERNAL"],
                 "identifier_names(kind 3-4)": bk["URI_SELF"] + bk["KEY_SEMANTIC"],
                 "generated_names(kind 5-6)": bk["INFERRED"] + bk["GENERATED_FLOOR"],
                 "pct_actual": round(100.0 * (bk["ORIGINAL"] + bk["RECOVERED_ORIGINAL"] + bk["RECOVERED_EXTERNAL"]) / N, 4),
                 "pct_named_any_kind": 100.0, "nodes_without_a_name": 0},
    "INVARIANTS": [
        "no string of kind 5 or 6 is ever relabelled into kinds 0-2; a consumer can always ask which kind a name came from",
        "the V1 floor is never lowered: every node keeps its frozen overlay name in overlay_v1 (unmodified); V4 only adds a served name on top",
        "the three nameless grades of semantic_kind_v2.1 are kept per node as an audit column (nameless_grade), never collapsed",
        "PYTHONHASHSEED=0 is part of dataset identity (node_uid rule)",
    ],
    "WHAT_THIS_IS_NOT": v1["WHAT_THIS_IS_NOT"] + [
        "NOT a claim that the 7,221,354 floor nodes were named by Freebase or by any external source: their names are generated descriptions.",
    ],
    "PROVENANCE": {"base": finfo(SCR + "/NAMES_V4_BASE.json"), "final": finfo(SCR + "/NAMES_V4_FINAL.json"),
                   "floor": finfo(SCR + "/FLOOR_NAMES.json"), "census": finfo(SCR + "/V4_FLOOR_CENSUS.json"),
                   "served_as": "data/final_canonical/freebase/nodes/shard_*.parquet columns name, name_kind, name_source, name_rule, "
                                "is_original_name, nameless_grade, recovery_class, floor_reason; nodes/name_kind.npy"},
}
write_record(SCR + "/FREEBASE_NAMES_V4.json", rec)
print("wrote", SCR + "/FREEBASE_NAMES_V4.json", rec["RECORD_SHA256"])
print(json.dumps(rec["COVERAGE"]))
