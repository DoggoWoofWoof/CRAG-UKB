"""V1 section 3 step 7 -- resolve retrieval_role under the ALL_NODE decision.

    python scratchpad/final_canonical_build/webqsp_v1/decide_retrieval_role.py

The user's decision, locked 2026-09-06:

    CRAG's canonical graph contains every node. Every node participates in partition construction.
    Every semantically renderable node, including CVTs and literals, is eligible to provide L1
    partition-routing evidence. Node-type exclusion is an ablation, not the default architecture.

node_kind is frozen and READ ONLY here (hash guard below). This writes retrieval_role only.
"""
import hashlib, json, os, time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
FROZEN = "data/final_canonical/webqsp/V1_NODE_KIND_FROZEN.json"
COST = "data/final_canonical/webqsp/V1_ALL_NODE_COST_TABLE.json"
OUT = "data/final_canonical/webqsp/V1_RETRIEVAL_ROLE_DECISION.json"

ELIGIBLE = "RETRIEVAL_ELIGIBLE"


def main():
    t0 = time.time()
    frz = json.load(open(FROZEN, encoding="utf-8"))
    cost = json.load(open(COST, encoding="utf-8"))

    nodes = pq.read_table(f"{D}/nodes.parquet")
    kind = nodes.column("node_kind").to_pylist()
    N = len(kind)
    h = hashlib.sha256()
    for k in kind:
        h.update(k.encode())
        h.update(b"\n")
    assert h.hexdigest() == frz["NODE_KIND_HASH"], "node_kind changed since it was frozen"

    prior = Counter(nodes.column("retrieval_role").to_pylist())
    role = [ELIGIBLE] * N
    tbl = nodes.set_column(nodes.schema.get_field_index("retrieval_role"), "retrieval_role",
                           pa.array(role, pa.string()))
    pq.write_table(tbl, f"{D}/nodes.parquet.tmp", compression="zstd")
    os.replace(f"{D}/nodes.parquet.tmp", f"{D}/nodes.parquet")

    rh = hashlib.sha256()
    for r in role:
        rh.update(r.encode())
        rh.update(b"\n")

    cvt_share = round(100.0 * (cost["ABLATION_ARMS_PREREGISTERED"][0]["chars"]
                               - cost["ABLATION_ARMS_PREREGISTERED"][3]["chars"])
                      / cost["ABLATION_ARMS_PREREGISTERED"][0]["chars"], 1)

    doc = {
        "schema": "V1_RETRIEVAL_ROLE_DECISION/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decided_by": "user, 2026-09-06",
        "DECISION": "CRAG's canonical graph contains every node. Every node participates in "
                    "partition construction. Every semantically renderable node, including CVTs "
                    "and literals, is eligible to provide L1 partition-routing evidence. "
                    "Node-type exclusion is an ablation, not the default architecture.",

        "WHAT_CHANGED_AND_WHY": {
            "superseded_direction": "the build was drifting toward CVT_MEDIATOR -> STRUCTURAL_ONLY, "
                                    "with CVTs required to earn retrieval permission by passing "
                                    "the gold-text gate.",
            "why_that_was_wrong": "L1 in CRAG does not return standalone documents; it selects "
                                  "graph REGIONS for L3 to traverse. A node's retrieval value is "
                                  "therefore its ability to route to the partition containing it, "
                                  "not its ability to be the answer. A CVT is the node whose text "
                                  "best matches the COMPOSITION of a multi-constraint query "
                                  "(which award was X nominated for in 2008), even though the "
                                  "answer string itself lives on an adjacent entity.",
            "status_of_the_gold_gate": "retained as analysis, demoted from switch. It established a "
                                       "true and narrow fact -- CVT text introduces no unique "
                                       "answer vocabulary (GOLD_ONLY_IN_CVT_TEXT_N = 20, all "
                                       "punctuation artifacts). It never addressed routing, and "
                                       "could not have: see V1_CVT_GATE_DIAGNOSTICS DIAGNOSIS_3.",
        },

        "NODE_CONTRACT": {
            "statement": "every canonical node satisfies the same interface; node_kind selects the "
                         "SERIALIZER, never the permission.",
            "interface": ["stable identity", "graph edges", "deterministic semantic view",
                          "retrieval representation", "partition membership"],
            "node_kind_is_not_a_permission": True,
        },

        "retrieval_role_ASSIGNED": {
            "value_for_every_node": ELIGIBLE,
            "n_nodes": N,
            "by_node_kind": dict(Counter(kind).most_common()),
            "prior_values_superseded": dict(prior),
            "RETRIEVAL_ROLE_HASH": rh.hexdigest(),
        },

        "THE_CONDITIONAL_IN_THE_POLICY_IS_VACUOUS": {
            "policy_text": "UNRESOLVED_OTHER eligible 'if deterministic semantic/structural text "
                           "exists'",
            "measured_nodes_failing_that_test": cost["DETERMINISTIC_TEXT_CONDITION"]
                                                    ["nodes_with_contentless_text_total"],
            "consequence": "no node renders to a contentless fallback, so the conditional excludes "
                           "nobody and the policy is unconditional in practice. All canonical "
                           "nodes are eligible.",
            "weakest_text_retained_deliberately": {
                "cvt_records_with_a_type_but_no_arguments":
                    cost["DETERMINISTIC_TEXT_CONDITION"]["cvt_header_only_no_arguments"],
                "entities_whose_text_is_only_their_name":
                    cost["DETERMINISTIC_TEXT_CONDITION"]["entity_with_no_facts_beyond_its_name"],
                "note": "weak text is not absent text; a Freebase type is deterministic signal.",
            },
        },

        "TYPE_AWARE_SERIALIZER": {
            "principle": "equal participation in the retrieval mechanism, NOT identical "
                         "representation.",
            "READABLE_ENTITY": "entity_text.parquet :: RoG surface",
            "MID_NAMED_ENTITY": "entity_text.parquet :: structural descriptor from incident schema",
            "UNRESOLVED_OTHER": "entity_text.parquet :: structural descriptor",
            "VALUE_LITERAL": "entity_text.parquet :: literal verbatim",
            "CVT_MEDIATOR": "cvt_text.parquet :: record type + role:value arguments, roles taken "
                            "from relation_label_qualified",
        },

        "OPEN_AXIS_NOT_DECIDED_HERE": {
            "axis": "entity text = NAME_ONLY vs NAME_PLUS_FACTS",
            "status": "NOT settled by this decision and NOT written into any artifact.",
            "measured_consequence": "under ALL_NODE, NAME_PLUS_FACTS serialises every CVT argument "
                                    "twice -- once in the CVT record, again inside each adjacent "
                                    "entity -- costing 2.554x the default arm, for information "
                                    "that is already retrievable as its own unit.",
            "assistant_recommendation_not_a_user_decision":
                "pair ALL_NODE with entity NAME_ONLY. The substitution rationale for "
                "NAME_PLUS_FACTS was to compensate for excluding CVTs; once CVTs are eligible the "
                "compensation is redundant. NAME_PLUS_FACTS remains the correct rendering for the "
                "NO_CVT_RETRIEVAL ablation arm, which is exactly where the substitution hypothesis "
                "is under test.",
        },

        "H4_SK_SUBSTRATE_CONSEQUENCE": {
            "user_observation": "if CVTs get no embeddings then KNN is built over only part of the "
                                "graph, giving ENTITY = STRUCT+KNN while CVT and LITERAL = STRUCT "
                                "only -- a hidden node-type asymmetry in the substrate.",
            "verified_current_state": "the asymmetry does NOT exist in any shipped artifact today, "
                                      "but not because it was avoided on purpose.",
            "evidence": {
                "existing_partition_manifest": "data/canonical/webqsp/partition_manifest_C.json",
                "existing_partition_n_nodes": 1316466,
                "existing_knn_manifest": "data/canonical/webqsp/knn_manifest.json",
                "existing_knn_n_nodes": 1316466,
                "knn_covers_every_node_in_that_build": True,
                "why": "that build is a DIFFERENT and superseded lineage which never classified "
                       "node kinds at all, so it could not have discriminated by kind even in "
                       "principle. It embedded whatever text it held for every node.",
                "v1_has_no_partition_or_knn_yet": True,
                "v1_n_nodes": N,
                "node_count_ratio_v1_over_superseded": round(N / 1316466, 3),
            },
            "reading": "section 3 classification is what first made node-type discrimination "
                       "POSSIBLE on this substrate. This decision declines to spend it on "
                       "exclusion. H4_SK for V1 is entirely forward-looking, so KNN over all "
                       "semantically renderable nodes -- including CVT-to-CVT neighbourhoods -- is "
                       "available as a design option rather than a retrofit.",
            "not_asserted": "whether CVT-to-CVT semantic KNN improves H4 organisation. Untested.",
        },

        "ABLATION_LADDER": {
            "default_arm": "ALL_NODE",
            "framing": "the full-graph version IS the system; filtered versions measure which parts "
                       "carry value. No node kind has to earn its place.",
            "arms": cost["ABLATION_ARMS_PREREGISTERED"],
            "cost_facts_worth_stating": {
                "cvt_records_share_of_default_arm_text_pct": cvt_share,
                "excluding_literals_saves_pct": 0.1,
                "note": "literal exclusion is not a cost lever. If that arm is ever run it tests "
                        "whether literals add retrieval NOISE, not whether they are affordable.",
            },
        },

        "WHAT_THIS_DOES_NOT_TOUCH": [
            "node_kind (frozen; hash verified unchanged before writing)",
            "graph topology, edges, or partition membership -- every node stays in the graph and "
            "in its partition regardless of any retrieval ablation",
            "relation_key identity",
            "the gate result, which stands as measured",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("retrieval_role_ASSIGNED",
                                          "THE_CONDITIONAL_IN_THE_POLICY_IS_VACUOUS")}, indent=1))


if __name__ == "__main__":
    main()
