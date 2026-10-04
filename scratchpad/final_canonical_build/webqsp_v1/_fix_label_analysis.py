import json, os
P = "data/final_canonical/webqsp/V1_RELATION_LABEL_ANALYSIS.json"
d = json.load(open(P, encoding="utf-8"))
d["COUNTING_CORRECTION_2026_09_06"] = {
    "what_was_wrong": "The 'colliding_relations' figures in measurement.schemes were LABELS LOST "
                      "(relations minus distinct labels), not the number of relations involved in a "
                      "collision. The two differ whenever a group has more than 2 members.",
    "corrected": {
        "LAST_SEGMENT": {"distinct_labels": 4657, "collision_groups": 893,
                         "relations_involved": 3294, "labels_lost": 2401,
                         "relations_involved_pct": 46.67},
        "TYPE_PLUS_PROPERTY": {"distinct_labels": 7009, "collision_groups": 47,
                               "relations_involved": 96, "labels_lost": 49,
                               "relations_involved_pct": 1.36},
    },
    "impact_on_the_decision": "None. The ladder escalates whatever actually collides, so the design "
                              "is unchanged; only the size of the escalated set moves, 49 -> 96.",
}
d["RESOLVED_BY"] = {
    "decision": "user, 2026-09-06 -- minimum qualification necessary for uniqueness",
    "implemented": "scratchpad/final_canonical_build/webqsp_v1/build_relation_labels.py",
    "result": "data/final_canonical/webqsp/V1_RELATION_LABELS.json",
    "outcome": "6,959 TYPE_PROPERTY + 93 DOMAIN_TYPE_PROPERTY + 6 FULL_KEY, 0 collisions, "
               "relation_uid -> relation_key identity preserved.",
    "why_a_third_rung_was_needed": "Escalating only the colliding relations can create a NEW "
                                   "collision with a different relation's un-escalated label. "
                                   "music.artist.album -> 'music artist album' collides with "
                                   "base.svocab.music_artist.album, whose type segment is already "
                                   "'music_artist'. Once underscores become spaces the two are "
                                   "identical, so the dots in the naturalized full key are "
                                   "load-bearing and the terminal rung must be unique BY "
                                   "CONSTRUCTION rather than assumed.",
}
d["recommendation"]["not_done_yet"] = False
d["recommendation"]["open_decision"] = "RESOLVED: SPLADE/CVT rendering consumes " \
                                       "relation_label_qualified; GRAPH_SAMPLES/UI consume " \
                                       "relation_label_short."
json.dump(d, open(P + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
os.replace(P + ".tmp", P)
print("corrected")
