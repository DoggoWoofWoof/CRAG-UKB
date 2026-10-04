import json, os

FIX = {
    "correction_id": "BAND_PREFIX_COMPOSITION_2026_09_06",
    "what_was_claimed": "That the NAME_RESOLVABLE band is 'overwhelmingly g. machine ids'. This "
                        "assistant repeated that gloss from the 2026-09-05 pipeline audit into the "
                        "section 3 freeze document and the classification audit.",
    "what_is_actually_true": {
        "whole_band": {"n": 62375, "m.": 47739, "g.": 14636, "g_pct": 23.5},
        "leaf_subset_decided_by_pass_2": {"n": 15035, "m.": 399, "g.": 14636, "g_pct": 97.3},
        "subset_decided_by_pass_1": {"n": 47340, "m.": 47340, "g.": 0, "g_pct": 0.0},
    },
    "diagnosis": "The prior audit's sentence was about the LEAVES it was discussing, where it is "
                 "correct at 97.3%. Generalising it to the whole band was this assistant's error. "
                 "The band is 76.5% m., and it splits perfectly by classification pass.",
    "why_the_split_is_perfect": "Not one of the 405,358 g. MIDs in the released RoG graph has an "
                                "outgoing edge -- 0.00%, against 81.50% of m. MIDs. So pass 1, which "
                                "reads outgoing types, can never see a g. node, and every g. node is "
                                "necessarily decided by pass 2.",
    "STRUCTURAL_FACT_g_MIDS_HAVE_NO_OUT_EDGES": {
        "m_total": 1247264, "m_with_out_edges": 1016520, "m_pct": 81.50,
        "g_total": 405358, "g_with_out_edges": 0, "g_pct": 0.0,
        "status": "MEASURED on the released RoG v1 tables",
        "interpretation": "UNKNOWN which of two causes this is, and the released graph cannot "
                          "separate them: (a) a genuine property of Freebase g. objects, which are "
                          "late-added and may only ever appear as targets; or (b) an artefact of "
                          "RoG's extraction, which may only have expanded outward from m. seeds. "
                          "A boundary/cap effect is a poor fit because it would truncate BOTH "
                          "prefixes, and m. sits at 81.5% while g. is exactly 0.",
        "consequence": "All g. classification rests entirely on incoming-relation range evidence. "
                       "The pass-1/pass-2 cross-validation agreement of 95.75% therefore says "
                       "nothing about g. nodes -- there is no second signal for them.",
    },
    "why_it_matters_for_section_13": "It splits ORDINARY_ENTITY_UNRESOLVED_N = 62,375 into two "
                                     "populations with DIFFERENT causes, which the single 'no g. "
                                     "entries' explanation wrongly merged: 14,636 g. nodes that "
                                     "entities_names.json structurally cannot name (it holds no g. "
                                     "keys at all), and 47,739 m. nodes it could hold and simply "
                                     "does not. The second group is the file's incompleteness, not "
                                     "a structural barrier -- so a more complete m. naming source "
                                     "would reduce, though not close, the section 13 gap.",
    "classifier_unaffected": "This corrects RATIONALE TEXT only. No threshold, rule, mapping or "
                             "label changed; NODE_KIND_HASH is unchanged.",
}

for P, key in (("data/final_canonical/webqsp/V1_SECTION3_CLASSIFIER_FREEZE.json",
                "CORRECTIONS_TO_RATIONALE_TEXT"),
               ("data/final_canonical/webqsp/V1_CLASSIFICATION_AUDIT.json",
                "CORRECTIONS_TO_RATIONALE_TEXT")):
    d = json.load(open(P, encoding="utf-8"))
    d[key] = [FIX]
    d["_correction_policy"] = ("Corrections are APPENDED, never applied by editing the original "
                               "text. The superseded wording stays visible so the record shows what "
                               "was believed when the rule was frozen.")
    json.dump(d, open(P + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
    os.replace(P + ".tmp", P)

# and the two earlier files that carry the same superseded expectation
P = "data/final_canonical/webqsp/NAME_MAPPING_COVERAGE.json"
d = json.load(open(P, encoding="utf-8"))
d["NAME_MAPPING_COVERAGE_OF_NAME_RESOLVABLE_BAND"]["status"] = "SUPERSEDED_NOW_MEASURED"
d["NAME_MAPPING_COVERAGE_OF_NAME_RESOLVABLE_BAND"]["measured_in"] = (
    "data/final_canonical/webqsp/V1_CLASSIFICATION_AUDIT.json")
d["NAME_MAPPING_COVERAGE_OF_NAME_RESOLVABLE_BAND"]["measured_result"] = {
    "at_lower_end": {"denominator": 19566, "named": 0, "pct": 0.0},
    "at_upper_end": {"denominator": 62375, "named": 0, "pct": 0.0},
    "note": "The structural expectation recorded here was directionally right but rested on a "
            "wrong premise -- see CORRECTIONS_TO_RATIONALE_TEXT in V1_CLASSIFICATION_AUDIT.json. "
            "The band is 76.5% m., not overwhelmingly g.; coverage is 0 anyway, and for the m. "
            "majority the cause is the file's incompleteness rather than a structural barrier.",
}
json.dump(d, open(P + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
os.replace(P + ".tmp", P)
print("corrections appended to 3 documents")
