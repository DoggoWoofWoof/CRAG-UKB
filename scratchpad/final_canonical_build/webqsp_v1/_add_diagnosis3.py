"""Record the structural limitation of the gate metric, discovered while diagnosing its result."""
import json, os

import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
P = "data/final_canonical/webqsp/V1_CVT_GATE_DIAGNOSTICS.json"

# Verify the claim rather than assert it: every CVT argument VALUE is some node's display name, and
# every non-CVT node's display name is exactly its NAME_ONLY entity text.
nodes = pq.read_table(f"{D}/nodes.parquet", columns=["node_kind", "display_name"])
kinds = nodes.column("node_kind").to_pylist()
disp = nodes.column("display_name").to_pylist()
noncvt_disp = {d for k, d in zip(kinds, disp) if k != "CVT_MEDIATOR"}
ent = pq.read_table(f"{D}/entity_text.parquet", columns=["text_name_only"])
name_only = set(ent.column("text_name_only").to_pylist())
identical = noncvt_disp == name_only

d = json.load(open(P, encoding="utf-8"))
d["DIAGNOSIS_3_STRUCTURAL_LIMIT_OF_THE_METRIC"] = {
    "claim": "GOLD_ONLY_IN_CVT_TEXT_N is near-zero BY CONSTRUCTION in this graph, so the gate could "
             "not have returned a large number whatever the value of CVT records.",
    "why": "A CVT record's argument values are the display surfaces of its neighbours. Every "
           "non-CVT neighbour's display surface IS its NAME_ONLY entity text. So any gold string "
           "appearing as a CVT argument necessarily also appears in some entity text. The only "
           "strings a CVT record can hold exclusively are its schema-derived header words, the "
           "structural labels of CVT-to-CVT neighbours, and punctuation-span artifacts.",
    "verified_not_assumed": {
        "noncvt_display_surfaces_equal_name_only_texts": bool(identical),
        "distinct_noncvt_display_surfaces": len(noncvt_disp),
        "distinct_name_only_texts": len(name_only),
    },
    "and_the_measured_20_are_exactly_that": "all recorded CVT-only examples end in '.', matching "
                                            "the terminal period of the '<role>: <value>.' grammar",

    "what_this_means_for_the_gate": "The pre-registered question -- does CVT text expose gold "
                                    "evidence unavailable from entity identity text -- is answered "
                                    "honestly and in the negative. But the answer is structural, "
                                    "not empirical: in the released RoG graph every value is itself "
                                    "a node with its own text, including literals like a year. The "
                                    "motivating scenario, where a marriage year lives only inside a "
                                    "mediator and nowhere else, does not occur here.",

    "what_the_metric_therefore_does_NOT_measure": "Whether CVT records are useful RETRIEVAL UNITS. "
                                                  "A CVT's value is COMPOSITION -- binding nominee, "
                                                  "award, ceremony and year into one retrievable "
                                                  "object so a multi-constraint query can match all "
                                                  "of them at once. A string-containment metric "
                                                  "cannot see composition, because it asks only "
                                                  "whether a string appears somewhere, never "
                                                  "whether the co-occurrence is retrievable in one "
                                                  "unit.",

    "recommendation": "Do NOT flip retrieval_role off UNDECIDED on the strength of this gate alone. "
                      "The gate was pre-registered to decide CVT encoding, and it has returned "
                      "cleanly in the low/low cell; but the structural blind spot above was not "
                      "known when it was registered, and it is the assistant's obligation to "
                      "surface that rather than let the number decide by default. The composition "
                      "question needs a different instrument -- a multi-constraint retrieval test "
                      "comparing short entity text + CVT records against enriched entity text with "
                      "no CVT index, scored on coverage, token cost, pool composition and P50 "
                      "exposure, which is the comparison the interpretation matrix already "
                      "anticipates for the high/low cell.",
    "gate_result_not_altered": True,
}
json.dump(d, open(P + ".tmp", "w", encoding="utf-8", newline="\n"), indent=2)
os.replace(P + ".tmp", P)
print("diagnosis 3 recorded; noncvt_display == name_only:", identical)
