"""Append the source-verified reverse orientation and the check-5 metric set to the design lock.

Appends. The earlier NOT CONFIRMED record stays: it is the evidence that the rule was held
provisional until a source line was actually read, rather than assumed from the outset.
"""
import json, os, time

P = "data/final_canonical/freebase_v3/V3_DESIGN_LOCK.json"
d = json.load(open(P, encoding="utf-8"))

d["RULE_2_ORIENTATION_SOURCE_VERIFIED_2026_09_06"] = {
    "resolves": "RULE_2_REVERSE_ORIENTATION.VERIFICATION_STATUS.retained_orientation, which was "
                "recorded as NOT CONFIRMED because FBDataDump.sh creates the reverse_properties "
                "table and stops.",
    "found_in": "DataPreparationScripts/FB3.sh, not FBDataDump.sh",
    "quoted_verbatim": "CREATE TABLE freebase_clean_no_reverse SELECT f.subject, f.predicate, "
                       "f.object from freebase_clean f LEFT OUTER JOIN reverse_properties r ON "
                       "f.predicate=r.object WHERE r.object IS NULL",
    "semantics": "an anti-join. A triple survives only when its predicate does NOT appear on the "
                 "OBJECT side of any /type/property/reverse_property assertion. Object-side "
                 "predicates are removed; subject-side predicates are retained.",
    "status": "the provisional rule is now source-verified rather than inferred.",
    "still_audited_against_delivered_data": "package and script can drift; the archive is the real "
                                            "audit target. Check 5 stands.",

    "MATERIAL_RISK_THIS_QUOTE_EXPOSES": {
        "finding": "reverse_properties is used in that join with NO deduplication and NO filtering "
                   "applied beforehand.",
        "consequence": "if Freebase asserts a pair symmetrically -- both (P, reverse_property, Q) "
                       "and (Q, reverse_property, P) -- then BOTH P and Q appear on the object "
                       "side, and the anti-join removes BOTH predicates. Every fact expressed by "
                       "that pair disappears from the variant entirely, in either direction.",
        "this_is_not_hypothetical": "it follows directly from an un-deduplicated anti-join. "
                                    "Whether Freebase actually asserts pairs symmetrically is an "
                                    "empirical question the delivered data answers.",
        "bounding_evidence_already_in_hand": "FB+CVT-REV retains 134,213,735 of 244,112,599 "
                                             "triples (55.0%), so the effect is certainly not "
                                             "universal. It could still be locally severe on "
                                             "specific relation families.",
    },
}

d["CHECK_5_METRICS_PREREGISTERED"] = {
    "fixed_by": "user, 2026-09-06, before the archive was inspected",
    "emit": [
        "REVERSE_PAIR_ASSERTIONS_N",
        "OBJECT_SIDE_PREDICATES_N",
        "SUBJECT_SIDE_PREDICATES_N",
        "BOTH_SIDES_PRESENT_IN_FB_CVT_REV_N",
        "NEITHER_SIDE_PRESENT_N",
        "AMBIGUOUS_CYCLES_N",
    ],
    "INVARIANT_1": {
        "statement": "BOTH_SIDES_PRESENT_IN_FB_CVT_REV_N = 0 for true reverse pairs, modulo "
                     "schema/pathological cases explicitly classified.",
        "detects": "failure of the anti-join to remove redundancy -- both directions of one fact "
                   "still stored, degrees doubled.",
        "severity_if_violated": "cost and double-counted support; no information is lost.",
    },
    "INVARIANT_2": {
        "statement": "NEITHER_SIDE_PRESENT_N = 0.",
        "detects": "symmetric reverse_property assertions annihilating both predicates through the "
                   "un-deduplicated anti-join.",
        "severity_if_violated": "FACT LOSS. The pair's content is absent from the variant in both "
                                "directions and cannot be recovered from it at all.",
        "note": "this is the dangerous invariant of the two. A violation of INVARIANT_1 costs "
                "tokens and inflates degree; a violation of INVARIANT_2 means CRAG would be "
                "building on a graph that silently lost whole relation families. Report it first "
                "and enumerate the affected predicates by name, not just by count.",
    },
    "AMBIGUOUS_CYCLES_N_definition": "reverse_property assertions forming cycles longer than the "
                                     "expected 2-cycle, or self-referential assertions where a "
                                     "predicate is its own reverse. Both make 'subject side' "
                                     "undefined.",
    "if_invariant_2_is_violated": "do not patch around it and do not fall back to FB+CVT+REV "
                                  "wholesale. Record the affected predicate set, measure how many "
                                  "of V1's 7,058 relations and of the 8,558,342 NSM MID-to-MID "
                                  "triples it touches, and let that number decide whether the raw "
                                  "dump is required to restore them.",
}
tmp = P + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(d, fh, indent=2)
os.replace(tmp, P)
print("design lock updated")
