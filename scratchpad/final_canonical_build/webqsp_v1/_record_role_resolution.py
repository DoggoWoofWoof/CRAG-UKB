"""Append the retrieval_role resolution to every document that recorded it as undecided.

Appends. Does not rewrite prior text: the record that these documents once said UNDECIDED is
itself the evidence that the decision came after the measurement, not before it.
"""
import json, os, time

STAMP = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
DEC = "data/final_canonical/webqsp/V1_RETRIEVAL_ROLE_DECISION.json"
BASE = "data/final_canonical/webqsp"

BLOCK = {
    "resolved_utc": STAMP,
    "resolution": "ALL_NODE. Every canonical node is RETRIEVAL_ELIGIBLE.",
    "decided_by": "user, 2026-09-06",
    "record": DEC,
    "supersedes_in_this_document": None,   # filled per file
}

PATCH = {
    "V1_NODE_KIND_FROZEN.json":
        "the field retrieval_role_still_undecided, which was true when this document was written "
        "and is now false. node_kind itself is unchanged and its hash still verifies.",
    "V1_SECTION3_CLASSIFIER_FREEZE.json":
        "retrieval_role_POLICY, which routed CVT_MEDIATOR to UNDECIDED_PENDING_CVT_GOLD_GATE and "
        "every other kind to UNDECIDED_PENDING_RETRIEVAL_POLICY. Both are now RETRIEVAL_ELIGIBLE. "
        "The separation this document insisted on -- node_kind is what the node IS, retrieval_role "
        "is how L1 treats it -- is unchanged and is what made the reversal cheap: no graph "
        "semantics moved.",
    "V1_CVT_GATE_PREREGISTRATION.json":
        "the standing of the gate itself. This document registered NAME_ONLY x NORMALIZED as the "
        "cell that would DECIDE CVT encoding. That framing is superseded: the gate is retained as "
        "analysis and no longer functions as an on/off switch. The pre-registration did its job -- "
        "it prevented cell-shopping and the recorded value stands exactly as measured -- but the "
        "question it was built to answer (does CVT text expose unique answer vocabulary) turned "
        "out not to be the question that governs CVT retrieval in CRAG (does a CVT provide routing "
        "evidence for the partition containing it).",
    "V1_CVT_GATE_RESULT.json":
        "the interpretation of the low/low cell as an architectural verdict. The measurement is "
        "unchanged and is not re-run. Its scope limit is recorded in V1_CVT_GATE_DIAGNOSTICS "
        "DIAGNOSIS_3: a string-containment metric cannot observe composition, and CVT arguments are "
        "by construction the display surfaces of nodes that already carry them.",
}

for fn, sup in PATCH.items():
    p = os.path.join(BASE, fn)
    d = json.load(open(p, encoding="utf-8"))
    blk = dict(BLOCK)
    blk["supersedes_in_this_document"] = sup
    d["RETRIEVAL_ROLE_RESOLVED_2026_09_06"] = blk
    if "retrieval_role_still_undecided" in d:
        blk["prior_field_value"] = d["retrieval_role_still_undecided"]
        d["retrieval_role_still_undecided"] = False
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=2)
    os.replace(tmp, p)
    print("appended:", fn)
