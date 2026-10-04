import json, os

P = "data/final_canonical/webqsp/COLLISION_DEGREE_TEST.json"
d = json.load(open(P, encoding="utf-8"))
s = d["by_collision_group_size"]
labs = ["g=2", "g=3-4", "g=5-9", "g=10-29", "g>=30"]
meds = [s[k]["median"] for k in labs]
d["MEDIAN_DEGREE_BY_GROUP_SIZE"] = dict(zip(labs, meds))
d["MEDIAN_DEGREE_MONOTONE_IN_GROUP_SIZE"] = all(meds[i] < meds[i+1] for i in range(len(meds)-1))
d["MEDIAN_DEGREE_RATIO_collision_over_control"] = round(
    d["groups"]["COLLISION_surfaces_g_ge_2"]["median"]
    / d["groups"]["CONTROL_singleton_named_surfaces"]["median"], 3)
d["STATISTIC_DISCIPLINE"] = {
    "prestated_rule_used": "MEAN",
    "prestated_rule_outcome": d["MEAN_DEGREE_MONOTONE_IN_GROUP_SIZE"],
    "where_it_failed": "only at g>=30 (n=200, mean 224.35 vs 229.08 at g=10-29)",
    "post_hoc_observation": "The MEDIAN is strictly monotone across all five strata "
                            "(13, 21, 35, 54, 85.5). Median is the more robust statistic here -- the "
                            "control max is 137,000 and one g=2 node has degree 122,343, so a "
                            "200-node top stratum's mean is tail-dominated.",
    "honesty_note": "The mean rule was fixed BEFORE the numbers were read and it returned false. "
                    "The median result is reported as a POST-HOC observation and is not substituted "
                    "for the pre-stated rule. It is corroboration, not the registered test."}
d["reading"] = ("Collision surfaces carry 5.74x the mean and 4.25x the median degree of "
                "singleton-named control surfaces, and degree rises with collision-group size. "
                "That is the signature of edge UNION under merging, not of the extra MIDs being "
                "absent from RoG's extraction -- absent MIDs contribute no edges and would leave "
                "degree flat in g. The popularity confound in the caveat still applies.")
json.dump(d, open(P + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
os.replace(P + ".tmp", P)

Q = "data/final_canonical/webqsp/ROG_ATTRIBUTABLE_COLLAPSE.json"
q = json.load(open(Q, encoding="utf-8"))
q["BOUND_DEGENERACY_CORRECTION"] = {
    "observed": "MERGES_UPPER, MERGES_NSM_presence_weighted and MERGES_LOWER all returned 97,538.",
    "cause": "entities_names.json ships with the NSM data, so every MID it names is in the NSM "
             "vocabulary BY CONSTRUCTION. The NSM-presence conditioning is therefore vacuous.",
    "consequence": "Three identical numbers are ONE measurement, not three independent "
                   "confirmations. The NSM check provides no evidence about presence in RoG.",
    "corrected_status_of_97538": "An UPPER bound on the identities RoG merged across the 36,927 "
                                 "surfaces this file can see -- it assumes every group member is in "
                                 "RoG's extraction. It is NOT a bound on RoG's total collapse, "
                                 "because the ~80% of the MID population the file cannot see is "
                                 "unmeasured in either direction.",
    "independent_evidence_that_merging_really_occurred":
        "data/final_canonical/webqsp/COLLISION_DEGREE_TEST.json -- collision surfaces carry 4.25x "
        "the median degree of singleton-named controls and degree rises with group size, which is "
        "what edge union under merging predicts and what non-extraction does not."}
q["interpretation"] = ("Bounds the identity loss inside the ~19.9% of the MID population this "
                       "third-party file can see. See BOUND_DEGENERACY_CORRECTION for what the "
                       "number does and does not bound.")
q["superseded_claim"] = ("An earlier draft of this file said 'every number here is a floor' for "
                         "RoG's full collapse. That was wrong and is retracted: 97,538 is an upper "
                         "bound WITHIN the visible subpopulation, and the unseen remainder is not "
                         "bounded by this measurement at all.")
json.dump(q, open(Q + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
os.replace(Q + ".tmp", Q)
print("patched both")
