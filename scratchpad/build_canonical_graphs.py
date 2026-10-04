"""
CANONICAL-SPACE GRAPHS
======================
canonical_v1 shipped nodes, queries, embeddings and a pointer index, but no graph. The edges
existed only in the Phase-C trees under data/canonical/, addressed by Phase-C document ids.
A consumer holding only data/final_canonical/ could not use them, and "endpoint alignment
verified" could not be stated because nothing had ever resolved a Phase-C endpoint into a
canonical node.

This materialises every edge family into the canonical POSITIONAL space -- the same space as
nodes.jsonl line numbers and the pointer index -- so an edge endpoint indexes an embedding
directly, with no join:

    src[k], dst[k]  are positions i, j
    nodes.jsonl line i is the source node
    CanonicalEmbeddings(ds, model).get(i) is its vector

ENDPOINT ALIGNMENT
    Every endpoint token in every TSV is resolved against the canonical node table, and the
    resolution is exact -- integer identity where the id space is numeric, 128-bit digest
    equality where it is a content hash. Nothing is fuzzy-matched. An endpoint that does not
    resolve is COUNTED and SAMPLED, never silently dropped, and the run aborts if the
    unresolved fraction exceeds a hair, because a systematically unresolvable endpoint means
    the id spaces were misidentified rather than that a few edges are stray.

WHAT IS DELIBERATELY NOT DONE
    The five datasets do not have the same edge families and are not made to. metaqa has no
    NER graph; hotpotqa and 2wiki_universe have no kNN graph. That is a property of what the
    sources support, so it is recorded in the manifest and in EDGE_FAMILY_MATRIX.json. No
    edge is invented to make the five look uniform.

    Direction is also preserved as found, not symmetrised. Every source manifest declares
    directed: true, and both directions of an undirected reading can be produced by a
    consumer in one line; fabricating the reverse edges here would make an asymmetric source
    look symmetric.

ONE ENDPOINT SPACE IS NOT WHAT IT LOOKS LIKE
    hotpotqa's structural TSV carries BARE curids (7533751), while its NER TSV carries
    hotpot_-prefixed ids. Both mean the same node. The normaliser makes that explicit rather
    than letting it fail as 15.4M unresolved endpoints.
"""
import glob
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"
TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
        "hotpotqa": "hotpotqa", "squad": "squad"}
FAMILIES = ["structural", "ner", "knn"]
MAX_UNRESOLVED_FRAC = 1e-4


def canon_keys(ds):
    """
    (kind, keys) for the canonical node table in nodes.jsonl order.
      kind "int"    keys are int64 -- the dataset's id space is numeric
      kind "hash"   keys are 128-bit digests of the Phase-C document id
    The numeric path is not an optimisation detail: it makes endpoint resolution exact
    integer identity for the three large datasets instead of a string join.
    """
    p = "%s/%s/nodes.jsonl" % (ROOT, ds)
    if ds in ("metaqa", "hotpotqa", "2wiki"):
        sep = ":e" if ds == "metaqa" else ":c"
        out = []
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                out.append(int(json.loads(ln)["node_id"].split(sep, 1)[1]))
        a = np.asarray(out, dtype=np.int64)
        return "int", a, np.ones(a.size, dtype=bool), []

    # squad / musique: content-hash ids with no closed form. The node's OWN Phase-C row is
    # phase_c_row_index in the reuse map; turn that into the Phase-C document id.
    ix = json.load(io.open("%s/%s/reuse_map/index.json" % (ROOT, ds), encoding="utf-8"))
    own = []
    for sh in ix["shards"]:
        with io.open("%s/%s/%s" % (ROOT, ds, sh["file"]), encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                if o.get("canonical_node_id") is None:
                    continue
                own.append(int(o["phase_c_row_index"]))
    own = np.asarray(own, dtype=np.int64)
    docids, pctitle = [], []
    with io.open("%s/%s/documents.jsonl" % (CANON, TREE[ds]), encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            docids.append(o["canonical_doc_id"])
            pctitle.append(o.get("title") or "")
    buf = bytearray()
    for r in own:
        buf += hashlib.blake2b(docids[r].encode("utf-8"), digest_size=16).digest()
    keys = np.frombuffer(buf, dtype=np.uint64).reshape(own.size, 2)

    # Phase-C may hold FEWER documents than canonical holds nodes, because Phase-C
    # deduplicated by text while canonical_v1 keeps nodes whose ids differ. musique is the
    # real case: 117,534 canonical nodes over 117,533 Phase-C documents, because "Etz Efraim"
    # and "Adei Ad" carry byte-identical settlement boilerplate. One Phase-C endpoint then
    # names two canonical nodes and edge resolution is ambiguous.
    #
    # The faithful reading is that the Phase-C graph was DERIVED OVER PHASE-C DOCUMENTS, so
    # the endpoint means the document Phase-C actually had -- the one whose title it carries.
    # That node keeps the edges; the collapsed twin is recorded as unrepresented in the
    # graph rather than being handed edges that were never computed for it. Copying the
    # edges across would assert title_mention relations for a title the derivation never
    # searched for.
    v = keys.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
    o = np.argsort(v, kind="stable")
    sv = v[o]
    dup_at = np.nonzero(sv[1:] == sv[:-1])[0]
    usable = np.ones(own.size, dtype=bool)
    collapsed = []
    if dup_at.size:
        titles = []
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                titles.append(json.loads(ln).get("title") or "")
        # expand each run of equal keys into its full member list, not just adjacent pairs
        runs, start = [], int(dup_at[0])
        for a, b in zip(dup_at[:-1], dup_at[1:]):
            if int(b) != int(a) + 1:
                runs.append((start, int(a) + 1))
                start = int(b)
        runs.append((start, int(dup_at[-1]) + 1))
        for lo, hi in runs:
            members = [int(o[j]) for j in range(lo, hi + 1)]
            want = pctitle[own[members[0]]]
            keep = [m for m in members if titles[m] == want]
            if len(keep) != 1:
                raise SystemExit("%s: %d canonical nodes share one Phase-C endpoint and the "
                                 "title does not disambiguate them: %s"
                                 % (ds, len(members), members))
            for m in members:
                if m != keep[0]:
                    usable[m] = False
                    collapsed.append({"canonical_position": m, "canonical_title": titles[m],
                                      "phase_c_doc_id": docids[own[m]],
                                      "phase_c_title": want,
                                      "kept_position": keep[0]})
    return "hash", keys, usable, collapsed


PREFIX = {"metaqa": "metaqa_ent_", "2wiki": "2wu:", "hotpotqa": "hotpot_"}


def endpoint_ints(ds, family, col):
    """
    int64 endpoint keys from a column of endpoint tokens.

    The prefix is detected from the first token rather than assumed, because hotpotqa writes
    BARE curids in its structural TSV and hotpot_-prefixed ids in its NER TSV. Conformance of
    the rest of the column is then an assertion, not a hope: to_numeric raises on any row
    that does not parse, so a column that mixes the two forms fails loudly instead of
    resolving half its endpoints to nothing.
    """
    import pandas as pd
    pre = PREFIX[ds]
    k = len(pre) if str(col.iat[0]).startswith(pre) else 0
    return pd.to_numeric(col.str.slice(k), errors="raise").to_numpy(dtype=np.int64)


def resolve(kind, keys, order, sortedk, col, ds, family):
    """endpoint tokens -> canonical positions; -1 where unresolved."""
    if kind == "int":
        q = endpoint_ints(ds, family, col)
        p = np.searchsorted(sortedk, q)
        ok = (p < sortedk.size) & (sortedk[np.minimum(p, sortedk.size - 1)] == q)
    else:
        buf = bytearray()
        for t in col.to_numpy():
            buf += hashlib.blake2b(str(t).encode("utf-8"), digest_size=16).digest()
        q = np.frombuffer(buf, dtype=np.uint64).reshape(len(col), 2)
        # lexicographic search over the 2-column digest, via a structured view
        v = q.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
        s = sortedk.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
        p = np.searchsorted(s, v)
        ok = (p < s.size) & (s[np.minimum(p, s.size - 1)] == v)
    out = np.full(len(col), -1, dtype=np.int64)
    out[ok] = order[p[ok]]
    return out


def read_tsv(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", header=None, names=["s", "d", "r"], dtype=str,
                       quoting=3, keep_default_na=False, na_filter=False,
                       engine="c", on_bad_lines="error")


def build(ds):
    t0 = time.time()
    tree = TREE[ds]
    kind, keys, usable, collapsed = canon_keys(ds)
    n_nodes = keys.shape[0]
    pos = np.nonzero(usable)[0]
    k = keys[pos]
    if kind == "int":
        o = np.argsort(k, kind="stable")
        sortedk = k[o]
        dup = int((np.diff(sortedk) == 0).sum())
    else:
        v = k.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
        o = np.argsort(v, kind="stable")
        sortedk = k[o]
        sv = sortedk.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
        dup = int((sv[1:] == sv[:-1]).sum())
    if dup:
        raise SystemExit("%s: %d duplicate canonical endpoint keys remain after "
                         "disambiguation -- endpoint resolution would be ambiguous"
                         % (ds, dup))
    order = pos[o].astype(np.int32)

    d = "%s/%s/graph" % (ROOT, ds)
    os.makedirs(d, exist_ok=True)
    fam_out = {}
    for family in FAMILIES:
        src_tsv = "%s/%s/graph_%s.tsv" % (CANON, tree, family)
        if not os.path.exists(src_tsv):
            fam_out[family] = {"present": False,
                               "reason": "no graph_%s.tsv in the %s tree" % (family, tree)}
            print("  %-9s %-10s ABSENT" % (ds, family), flush=True)
            continue
        df = read_tsv(src_tsv)
        s = resolve(kind, keys, order, sortedk, df["s"], ds, family)
        t = resolve(kind, keys, order, sortedk, df["d"], ds, family)
        good = (s >= 0) & (t >= 0)
        n_all = int(s.size)
        n_bad = int((~good).sum())
        frac = n_bad / float(n_all) if n_all else 0.0
        ex = []
        if n_bad:
            bi = np.nonzero(~good)[0][:5]
            ex = [{"line": int(k), "src": str(df["s"].to_numpy()[k]),
                   "dst": str(df["d"].to_numpy()[k]),
                   "src_resolved": bool(s[k] >= 0), "dst_resolved": bool(t[k] >= 0)}
                  for k in bi]
        if frac > MAX_UNRESOLVED_FRAC:
            raise SystemExit("%s/%s unresolved endpoints %d/%d (%.4f%%) exceeds tolerance -- "
                             "this is an id-space mismatch, not stray edges: %s"
                             % (ds, family, n_bad, n_all, 100 * frac, json.dumps(ex)))

        si = s[good].astype(np.int32)
        ti = t[good].astype(np.int32)
        third = df["r"].to_numpy()[good]
        payload = {"src": si, "dst": ti}
        rel_vocab = None
        try:
            w = third.astype(np.float32)
            payload["weight"] = w
            attr = "weight(float32)"
        except ValueError:
            rel_vocab = sorted(set(third.tolist()))
            code = {v: i for i, v in enumerate(rel_vocab)}
            payload["rel"] = np.asarray([code[v] for v in third], dtype=np.int16)
            attr = "rel(int16) + rel_vocab"
        np.savez(os.path.join(d, "%s.npz" % family), **payload)

        gm = {}
        p = "%s/%s/graph_manifest.json" % (CANON, tree)
        if os.path.exists(p):
            j = json.load(io.open(p, encoding="utf-8"))
            gm = j if j.get("edge_family", "").startswith(family[:4]) or family == "structural" \
                else {}
        fam_out[family] = {
            "present": True, "file": "graph/%s.npz" % family,
            "n_edges": int(si.size), "n_edges_in_source_tsv": n_all,
            "unresolved_endpoint_edges": n_bad,
            "unresolved_examples": ex,
            "attributes": attr,
            "relation_vocabulary": rel_vocab,
            "self_loops": int((si == ti).sum()),
            "source_tsv": src_tsv,
            "source_tsv_bytes": os.path.getsize(src_tsv),
            "phase_c_manifest_sha256": gm.get("graph_tsv_sha256"),
            "directed": gm.get("directed", True),
            "provenance": gm.get("provenance"),
            "edge_subtype": gm.get("edge_subtype")}
        print("  %-9s %-10s edges=%-12s unresolved=%-6d self_loops=%-7d %s"
              % (ds, family, format(int(si.size), ","), n_bad,
                 fam_out[family]["self_loops"], attr), flush=True)
        del df, s, t, si, ti, third, payload

    man = {"RECORD": "CANONICAL_V1_GRAPH", "dataset": ds, "phase_c_tree": tree,
           "n_nodes": int(n_nodes),
           "endpoint_space": ("edge endpoints are POSITIONS: the line number in "
                              "%s/%s/nodes.jsonl, which is also the pointer-index position, "
                              "so an endpoint indexes an embedding with no join" % (ROOT, ds)),
           "endpoint_resolution": ("exact integer identity" if kind == "int"
                                   else "exact 128-bit digest identity on the Phase-C "
                                        "document id"),
           "direction_policy": ("preserved as found; sources declare directed=true and are "
                                "NOT symmetrised here"),
           "family_policy": ("families are recorded, never invented: a dataset whose source "
                             "supports no NER or kNN graph simply has none"),
           "nodes_unrepresented_in_graph": collapsed,
           "nodes_unrepresented_reason": (
               "Phase-C deduplicated these nodes into another document by identical text, so "
               "the graph derivation never ran for their own id. They keep every other "
               "canonical property; they simply have no edges from a Phase-C-derived family. "
               "Edges were NOT copied from the twin, because that would assert relations the "
               "derivation never computed." if collapsed else None),
           "families": fam_out,
           "seconds": round(time.time() - t0, 1)}
    json.dump(man, io.open(os.path.join(d, "GRAPH_MANIFEST.json"), "w", encoding="utf-8"),
              indent=1)
    return man


if __name__ == "__main__":
    which = sys.argv[1:] or ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
    allman = {}
    for ds in which:
        allman[ds] = build(ds)
    p = "%s/EDGE_FAMILY_MATRIX.json" % ROOT
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old["RECORD"] = "CANONICAL_V1_EDGE_FAMILY_MATRIX"
    old["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    old["POLICY"] = ("The five datasets do not carry the same edge families and are not made "
                     "to. This matrix is the record of which families exist, so a consumer "
                     "can condition on it instead of assuming uniformity. Absence is a "
                     "property of the source, never a gap to be filled with invented edges.")
    old.setdefault("matrix", {}).update(
        {ds: {f: (m["families"][f]["n_edges"] if m["families"][f]["present"] else None)
              for f in FAMILIES} for ds, m in allman.items()})
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=1)
    print("\nEDGE FAMILY MATRIX")
    print("  %-10s %14s %14s %14s" % ("dataset", "structural", "ner", "knn"))
    for ds, row in sorted(old["matrix"].items()):
        print("  %-10s %14s %14s %14s"
              % (ds, *[("--" if row[f] is None else format(row[f], ",")) for f in FAMILIES]))
    print("\nwrote", p)
