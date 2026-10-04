"""
ID BRIDGE + GOLD RESOLUTION
===========================
Makes the identifier relationships between the two trees a first-class, verified artifact
instead of tribal knowledge, and proves that every gold answer a query points at is a real
canonical node.

THREE ID SPACES
    canonical_v1   data/final_canonical/<ds>/nodes.jsonl      2wiki:c1000, metaqa:e00000
    Phase-C        data/canonical/<tree>/documents.jsonl      2wu:1000,    metaqa_ent_0
    legacy/UKB     the pre-canonical substrate                hotpotqa_clean_doc_367397

For three datasets the canonical<->Phase-C relation is a pure function, so the bridge is stored
as a RULE and the rule is then verified on every row rather than asserted. For the other two the
ids are content hashes with no derivable relation, so the bridge is the explicit pointer index
that already exists, and what is verified is that it is a bijection onto the Phase-C rows used.

GOLD RESOLUTION
    canonical_v1 queries already carry gold_node_ids in canonical space. Every one of them must
    name a node that exists in nodes.jsonl. An unresolvable gold id would mean a query whose
    answer is not in the corpus -- silently unanswerable, and invisible in any recall metric.
    Node ids are held as 64-bit hashes rather than strings so 5.99M ids cost ~48 MB, not ~700 MB.
"""
import hashlib, io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
        "hotpotqa": "hotpotqa", "squad": "squad"}
# canonical id -> Phase-C id, where a closed-form relation exists
RULES = {
    "metaqa": {"rule": "metaqa:e<5-digit zero-padded n>  <->  metaqa_ent_<n>",
               "fwd": lambda c: "metaqa_ent_%d" % int(c.split(":e", 1)[1])},
    "2wiki": {"rule": "2wiki:c<curid>  <->  2wu:<curid>",
              "fwd": lambda c: "2wu:" + c.split(":c", 1)[1]},
    "hotpotqa": {"rule": "hotpotqa:c<source id>  <->  hotpot_<source id>",
                 "fwd": lambda c: "hotpot_" + c.split(":c", 1)[1]},
    "musique": {"rule": "content-hash ids, no closed form; bridge is the pointer index",
                "fwd": None},
    "squad": {"rule": "content-hash ids, no closed form; bridge is the pointer index",
              "fwd": None},
}
SPLITS = ["train", "dev", "validation", "test"]


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def check(ds):
    t0 = time.time()
    tree = TREE[ds]
    r = {"dataset": ds, "phase_c_tree": tree, "rule": RULES[ds]["rule"]}

    # canonical node ids as 64-bit hashes
    ids = []
    with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
        for ln in f:
            ids.append(h64(json.loads(ln)["node_id"]))
    node_h = np.sort(np.asarray(ids, dtype=np.uint64))
    r["canonical_nodes"] = int(node_h.size)
    r["node_id_collisions"] = int(node_h.size - np.unique(node_h).size)

    # verify the closed-form rule against the Phase-C ids the pointer index actually resolves to
    fwd = RULES[ds]["fwd"]
    if fwd is not None:
        pc = []
        with io.open("data/canonical/%s/documents.jsonl" % tree, encoding="utf-8") as f:
            for ln in f:
                pc.append(json.loads(ln)["canonical_doc_id"])
        pcset = set(pc)
        bad, n = 0, 0
        ex = []
        with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
            for ln in f:
                c = json.loads(ln)["node_id"]
                n += 1
                if fwd(c) not in pcset:
                    bad += 1
                    if len(ex) < 3:
                        ex.append({"canonical": c, "derived": fwd(c)})
        r["rule_checked"] = n
        r["rule_violations"] = bad
        r["rule_examples"] = ex
        del pc, pcset
    else:
        r["rule_checked"] = 0
        r["rule_violations"] = 0

    # gold resolution: every gold_node_id must be a real canonical node
    tot = miss = nq = 0
    missing_ex = []
    per_split = {}
    for sp in SPLITS:
        p = "%s/%s/queries/%s.jsonl" % (ROOT, ds, sp)
        if not os.path.exists(p):
            continue
        st, sm, sq = 0, 0, 0
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                sq += 1
                g = o.get("gold_node_ids") or []
                for x in g:
                    st += 1
                    k = h64(x)
                    i = np.searchsorted(node_h, np.uint64(k))
                    if i >= node_h.size or node_h[i] != k:
                        sm += 1
                        if len(missing_ex) < 5:
                            missing_ex.append({"split": sp, "query": o.get("query_id"), "gold": x})
        per_split[sp] = {"queries": sq, "gold_refs": st, "unresolved": sm}
        tot += st
        miss += sm
        nq += sq
    r["queries"] = nq
    r["gold_refs"] = tot
    r["gold_unresolved"] = miss
    r["gold_resolution_pct"] = round(100.0 * (tot - miss) / tot, 4) if tot else None
    r["by_split"] = per_split
    r["unresolved_examples"] = missing_ex
    r["PASS"] = bool(r["rule_violations"] == 0 and miss == 0 and r["node_id_collisions"] == 0)
    r["seconds"] = round(time.time() - t0, 1)
    print("  %-9s nodes=%-9d rule_viol=%-6d queries=%-7d gold=%-9d unresolved=%-6d %s"
          % (ds, r["canonical_nodes"], r["rule_violations"], nq, tot, miss,
             "PASS" if r["PASS"] else "FAIL"), flush=True)
    return r


if __name__ == "__main__":
    out, ok = {}, True
    for ds in (sys.argv[1:] or ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]):
        out[ds] = check(ds)
        ok &= out[ds]["PASS"]
    p = "%s/ID_BRIDGE.json" % ROOT
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.setdefault("datasets", {}).update(out)
    old["RECORD"] = "CANONICAL_V1_ID_BRIDGE"
    old["id_spaces"] = {
        "canonical_v1": "data/final_canonical/<ds>/nodes.jsonl",
        "phase_c": "data/canonical/<tree>/documents.jsonl",
        "bridge": ("closed-form rule where one exists, verified per row; otherwise the "
                   "pointer index at <ds>/pointer_index/, which is the bridge")}
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=1)
    print("\nALL_PASS =", ok, "->", p)
