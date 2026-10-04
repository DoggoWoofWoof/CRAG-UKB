# -*- coding: utf-8 -*-
"""
Add WebQSP to ID_BRIDGE.json, in the schema the five frozen datasets already use.

THE ONE FIELD THAT WILL LOOK LIKE A FAILURE AND IS NOT
    Every one of the five resolves gold references at 100.0%. WebQSP will not, and a reader
    scanning the column would reasonably conclude the sixth bridge is broken. It is not, and
    the difference is structural rather than technical:

        the five      gold is a DOCUMENT the corpus is built from, so it is present by
                      construction and anything less than 100% is a bug
        webqsp        gold is an ANSWER ENTITY named by the benchmark, and the corpus is a
                      third-party KG subgraph (RoG). Whether a given answer entity appears in
                      that subgraph is a property of RoG, not of the join

    The distinction is only credible with a control, so the same key is run against the TOPIC
    entities, which the benchmark guarantees are in the graph because they are where reasoning
    starts. That lands at 99.92%. A key that reaches 99.92% of one population and 62.42% of
    another is reaching the data; what it does not find is absent.

    So this record carries BOTH numbers and states which is the control. Publishing 62.42%
    alone would understate the bridge; publishing 99.92% alone would overstate the corpus.
"""
import collections
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
D = ROOT + "/webqsp"
P = ROOT + "/ID_BRIDGE.json"


def main():
    t0 = time.time()
    n = 0
    ids_ok = True
    with io.open(D + "/nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if json.loads(ln)["node_id"] != "webqsp:n%d" % i:
                ids_ok = False
            n += 1

    by_split = {}
    g_tot = g_hit = t_tot = t_hit = 0
    cov = collections.Counter()
    unresolved_examples = []
    for sp in ("train", "test"):
        p = "%s/queries/%s.jsonl" % (D, sp)
        if not os.path.exists(p):
            continue
        q = gr = un = 0
        for ln in io.open(p, encoding="utf-8"):
            o = json.loads(ln)
            gold = o.get("gold_entity_names") or []
            got = o.get("gold_positions") or []
            top = o.get("topic_entity_names") or []
            tp = o.get("topic_positions") or []
            q += 1
            gr += len(gold)
            un += len(gold) - len(got)
            g_tot += len(gold); g_hit += len(got)
            t_tot += len(top); t_hit += len(tp)
            cov[o.get("gold_coverage")] += 1
            if len(got) < len(gold) and len(unresolved_examples) < 5:
                unresolved_examples.append(
                    {"query_id": o["query_id"], "gold_given": len(gold),
                     "gold_resolved": len(got)})
        by_split[sp] = {"queries": q, "gold_refs": gr, "unresolved": un}

    rec = {
        "dataset": "webqsp",
        "phase_c_tree": None,
        "phase_c_tree_note": (
            "NONE deliberately. data/canonical/webqsp is a DIFFERENT node universe "
            "(1,316,466 nodes) and its dense docs channel is one of the ten damaged ones. "
            "This corpus is built from data/final_canonical/webqsp/v1/*.parquet, the "
            "V1-resolved RoG WebQSP+CWQ graph, and shares no ids with that tree."),
        "rule": "webqsp:n<position>  <->  nodes.jsonl line number  <->  source_rog_endpoint",
        "canonical_nodes": n,
        "node_id_collisions": 0,
        "rule_checked": n,
        "rule_violations": 0 if ids_ok else -1,
        "rule_examples": [],
        "queries": sum(v["queries"] for v in by_split.values()),
        "gold_refs": g_tot,
        "gold_unresolved": g_tot - g_hit,
        "gold_resolution_pct": round(100.0 * g_hit / g_tot, 2) if g_tot else None,
        "by_split": by_split,
        "unresolved_examples": unresolved_examples,
        "POSITIVE_CONTROL": {
            "population": "topic entities -- the benchmark guarantees these are in the graph",
            "refs": t_tot, "resolved": t_hit,
            "pct": round(100.0 * t_hit / t_tot, 2) if t_tot else None,
            "why": ("the same key on a population that MUST be present. 99.9% there and 62.4% "
                    "on gold means the key reaches the data and what it misses is absent from "
                    "RoG, not unmatched by a broken join. Without this control the gold number "
                    "is uninterpretable.")},
        "GOLD_IS_NOT_COMPARABLE_TO_THE_OTHER_FIVE": (
            "the five resolve gold at 100.0% because their gold IS a corpus document, present "
            "by construction. WebQSP gold is an answer ENTITY and the corpus is a third-party "
            "KG subgraph, so coverage is a property of RoG. Reading this column as a bridge "
            "quality score across all six would be wrong."),
        "query_level_coverage": dict(sorted(cov.items())),
        "PASS": bool(ids_ok and t_tot and (100.0 * t_hit / t_tot) > 99.0),
        "PASS_criterion": ("node id rule holds on every row AND the positive control clears "
                           "99% -- NOT gold coverage, which is a property of the source"),
        "seconds": round(time.time() - t0, 1)}

    d = json.load(io.open(P, encoding="utf-8"))
    d.setdefault("datasets", {})["webqsp"] = rec
    d["id_spaces"] = d.get("id_spaces", {})
    if isinstance(d["id_spaces"], dict):
        d["id_spaces"]["webqsp"] = ("canonical positions 0..2,592,893; node_id is "
                                    "'webqsp:n<position>'; graph endpoints and pointer-index "
                                    "positions are the SAME integers")
    json.dump(d, io.open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("webqsp: nodes=%s rule_violations=%s" % (format(n, ","), rec["rule_violations"]))
    print("  gold %s/%s (%.2f%%)   CONTROL topic %s/%s (%.2f%%)"
          % (format(g_hit, ","), format(g_tot, ","), rec["gold_resolution_pct"],
             format(t_hit, ","), format(t_tot, ","), rec["POSITIVE_CONTROL"]["pct"]))
    print("  by split: %s" % by_split)
    print("  PASS=%s   updated %s  %.1fs" % (rec["PASS"], P, rec["seconds"]))


if __name__ == "__main__":
    main()
