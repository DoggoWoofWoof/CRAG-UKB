"""Measurements the ALL_NODE decision needs, BEFORE retrieval_role is written.

    python scratchpad/final_canonical_build/webqsp_v1/measure_all_node.py

Two questions only:
  (1) The policy says UNRESOLVED_OTHER (and by extension any node) is eligible "if deterministic
      semantic/structural text exists".  Which nodes render to a CONTENTLESS fallback, i.e. text
      that carries no discriminative signal at all?  Those are the only ones the conditional can
      exclude, and the count decides whether the conditional matters.
  (2) What does ALL_NODE actually cost against each ablation arm, in nodes and characters?
      The arms are pre-registered here so the cost table exists before any arm is preferred.
"""
import json, os, time
from collections import Counter

import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
OUT = "data/final_canonical/webqsp/V1_ALL_NODE_COST_TABLE.json"

# a fallback emitted by struct_label() when a node asserts no type at all: carries zero signal
CONTENTLESS = {"unnamed entity", "unlabelled record"}


def main():
    t0 = time.time()
    nodes = pq.read_table(f"{D}/nodes.parquet", columns=["node_uid", "node_kind", "display_name"])
    uid = nodes.column("node_uid").to_pylist()
    kind = nodes.column("node_kind").to_pylist()
    disp = nodes.column("display_name").to_pylist()
    kind_of = dict(zip(uid, kind))
    N = len(uid)

    # ---- (1) contentless text ----
    contentless = Counter()
    for k, d in zip(kind, disp):
        if d.strip().rstrip(".").strip() in CONTENTLESS:
            contentless[k] += 1

    ent = pq.read_table(f"{D}/entity_text.parquet",
                        columns=["node_uid", "node_kind", "text_name_only", "text_name_plus_facts"])
    e_uid = ent.column("node_uid").to_pylist()
    e_kind = ent.column("node_kind").to_pylist()
    e_only = ent.column("text_name_only").to_pylist()
    e_facts = ent.column("text_name_plus_facts").to_pylist()
    del ent
    cvt = pq.read_table(f"{D}/cvt_text.parquet", columns=["node_uid", "text_cvt_record"])
    c_uid = cvt.column("node_uid").to_pylist()
    c_txt = cvt.column("text_cvt_record").to_pylist()
    del cvt

    # a CVT that renders to its header and nothing else has a type but no arguments: weak, but the
    # type IS deterministic signal.  Counted separately from contentless.
    header_only = sum(1 for t in c_txt if t.count(":") == 0)
    entity_no_facts = sum(1 for a, b in zip(e_only, e_facts) if a == b)

    # ---- (2) cost per arm ----
    def chars(texts):
        return sum(len(t) for t in texts)

    ent_only_by_kind = Counter()
    for k in e_kind:
        ent_only_by_kind[k] += 1

    ENTITYLIKE = ("READABLE_ENTITY", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")
    sel_entitylike = [i for i, k in enumerate(e_kind) if k in ENTITYLIKE]
    sel_literal = [i for i, k in enumerate(e_kind) if k == "VALUE_LITERAL"]

    def arm(name, n, ch, note):
        return {"arm": name, "nodes": n, "chars": ch, "est_tokens_chars_over_4": round(ch / 4),
                "note": note}

    ch_only_all = chars(e_only)
    ch_facts_all = chars(e_facts)
    ch_cvt = chars(c_txt)
    ch_only_el = sum(len(e_only[i]) for i in sel_entitylike)
    ch_facts_el = sum(len(e_facts[i]) for i in sel_entitylike)
    ch_only_lit = sum(len(e_only[i]) for i in sel_literal)

    arms = [
        arm("ALL_NODE__entity_NAME_ONLY", N, ch_only_all + ch_cvt,
            "DEFAULT under the ALL_NODE decision. CVT records carry composition, so entity text "
            "does not also need to carry the same arguments."),
        arm("ALL_NODE__entity_NAME_PLUS_FACTS", N, ch_facts_all + ch_cvt,
            "every CVT argument is serialised twice, once in the CVT record and again inside each "
            "adjacent entity. Redundant by construction under ALL_NODE."),
        arm("NO_CVT_RETRIEVAL__entity_NAME_PLUS_FACTS", len(e_uid), ch_facts_all,
            "the substitution arm: CVT information is flattened into entity text instead of being "
            "retrievable as its own unit."),
        arm("NO_CVT_RETRIEVAL__entity_NAME_ONLY", len(e_uid), ch_only_all,
            "strict exclusion with no compensation. Lower bound on corpus cost."),
        arm("ENTITY_ONLY", len(sel_entitylike), ch_only_el,
            "no CVT, no literal."),
        arm("ENTITY_ONLY__NAME_PLUS_FACTS", len(sel_entitylike), ch_facts_el,
            "no CVT, no literal, flattened facts."),
        arm("ALL_NODE_MINUS_LITERAL", N - len(sel_literal), ch_only_all - ch_only_lit + ch_cvt,
            "literals excluded from retrieval but retained in the graph and its partitions."),
    ]

    base = arms[0]["chars"]
    for a in arms:
        a["ratio_vs_ALL_NODE_entity_NAME_ONLY"] = round(a["chars"] / base, 3)

    doc = {
        "schema": "V1_ALL_NODE_COST_TABLE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Inputs to the ALL_NODE retrieval_role decision, measured before the decision is "
                   "written into nodes.parquet. This file ranks nothing and prefers no arm.",
        "TOTAL_CANONICAL_NODES": N,
        "DETERMINISTIC_TEXT_CONDITION": {
            "question": "the policy admits UNRESOLVED_OTHER 'if deterministic semantic/structural "
                        "text exists'. Which nodes fail that test?",
            "contentless_fallback_texts": sorted(CONTENTLESS),
            "nodes_with_contentless_text_by_kind": dict(contentless),
            "nodes_with_contentless_text_total": sum(contentless.values()),
            "cvt_header_only_no_arguments": header_only,
            "entity_with_no_facts_beyond_its_name": entity_no_facts,
            "reading": "A header-only CVT still asserts its Freebase type, which is deterministic "
                       "signal; it is weak text, not absent text. Only the contentless fallbacks "
                       "carry nothing at all.",
        },
        "node_kind_counts": dict(Counter(kind).most_common()),
        "ABLATION_ARMS_PREREGISTERED": arms,
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
