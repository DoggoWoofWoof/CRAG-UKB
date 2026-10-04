# -*- coding: utf-8 -*-
"""
CONSOLIDATION V3, part 2 -- one tree per dataset, one query store, one graph tree.

Runs after consolidate_v3.py heal/fold/pointers. Same log (CONSOLIDATION_V3_LOG.json), same
rule: nothing is encoded or invented, every byte written already existed in a served store,
and every move is proved by bytes before the record is rewritten.

  queries   metaqa's three hop trees (metaqa_1hop/2hop/3hop, plus the three DENSE_REPAIR
            query patches the pointers still reference) become ONE position-ordered store
            data/canonical/metaqa/encodings/<model>/queries: row i == query_ids.json[i].
            Proof per position: the hop tree's own id at (tree,row) == that tree's
            queries.jsonl id, question text == the canonical question, hop == canonical hop;
            then every row of the merged store == what the OLD resolver gathers for that
            position. The query pointer index becomes the identity.
  trees     the queries-only Phase-C tree data/canonical/2wiki is merged into
            2wiki_universe, which is then renamed 2wiki; webqsp_rog_v1 is renamed webqsp; the
            hop trees and the two patch directories move to data/_retired_v3/ (a move, not a
            deletion -- deleting is the user's step). POINTER_INDEX paths follow.
  graphs    every graph2/ family moves into graph/; the one superseded frozen family
            (musique knn) is retired and replaced by its audited successor; ONE
            GRAPH_MANIFEST.json per dataset carries every family with its provenance and its
            history; graph2/ disappears.
"""
import io
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consolidate_v3 import (ROOT, FC, CAN, PATCH, REV2, rj, wj, sha_file, log_phase,   # noqa: E402
                            store_ids, store_dir)

PI = FC + "/POINTER_INDEX.json"
HIST = FC + "/_history/pre_v3"
RET = "data/_retired_v3"
SS = 40000
DIM = 1536
VOCAB = 30522
DATASETS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
HOP_TREES = ["metaqa_1hop", "metaqa_2hop", "metaqa_3hop"]
RENAMES = {"2wiki_universe": "2wiki", "webqsp_rog_v1": "webqsp"}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(p):
    out = []
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def tree_of_path(path):
    parts = path.replace(chr(92), "/").split("/")
    if parts[:2] != ["data", "canonical"]:
        sys.exit("not a Phase-C store path: %s" % path)
    return parts[2]


def resolver():
    sys.path.insert(0, FC)
    import pointer_resolver as v1
    return v1


def mv(s, d, moves):
    if not os.path.exists(s):
        sys.exit("move: missing %s" % s)
    if os.path.exists(d):
        sys.exit("move: target exists %s" % d)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    os.rename(s, d)
    moves.append({"from": s, "to": d})
    journal({"from": s, "to": d})
    print("  moved %s -> %s" % (s, d))


def journal(entry):
    """append-only record of every move, written before anything else can fail"""
    os.makedirs(HIST, exist_ok=True)
    with io.open(HIST + "/v3_moves.jsonl", "a", encoding="utf-8", newline=chr(10)) as f:
        f.write(json.dumps(dict(entry, utc=utc())) + chr(10))


# -------------------------------------------------------------------------------------- queries
def queries():
    v1 = resolver()
    pi = rj(PI)
    q = pi["queries"]["metaqa"]
    trees = list(q["trees"])
    if trees == ["metaqa"]:
        sys.exit("metaqa queries are already merged")
    if trees != HOP_TREES:
        sys.exit("unexpected metaqa trees %s" % trees)
    n = int(q["n_queries"])
    qids = rj("%s/metaqa/queries/pointer_index/query_ids.json" % FC)
    if len(qids) != n:
        sys.exit("query_ids.json has %d entries, record says %d" % (len(qids), n))
    canon = {}
    for split in ("train", "dev", "test"):
        for r in read_jsonl("%s/metaqa/queries/%s.jsonl" % (FC, split)):
            canon[r["query_id"]] = (r["question"], int(r["hop"]))
    missing = [x for x in qids if x not in canon]
    if missing:
        sys.exit("%d query ids are not in the canonical query files" % len(missing))
    tree_rows = [read_jsonl("%s/%s/queries.jsonl" % (CAN, t)) for t in trees]

    mapping = {}
    positions_from = {}
    for model in ("dense", "splade"):
        p = "%s/metaqa/queries/pointer_index/%s.npz" % (FC, model)
        z = np.load(p)
        src, row = z["src"].copy(), z["row"].copy()
        z.close()
        if len(src) != n:
            sys.exit("pointer index %s has %d positions" % (model, len(src)))
        tree_of = np.full(n, -1, dtype=np.int64)
        trow = np.full(n, -1, dtype=np.int64)
        kinds = {}
        for si, st in enumerate(q[model]["stores"]):
            m = np.nonzero(src == si)[0]
            if not len(m):
                continue
            kind = st.get("store")
            if kind == "PHASE_C":
                ti = trees.index(tree_of_path(st["path"]))
                tree_of[m] = ti
                trow[m] = row[m]
            elif kind == "DENSE_REPAIR":
                d = os.path.dirname(st["path"])
                ti = trees.index(os.path.basename(d).split("__")[0])
                rows_map = np.asarray(rj(d + "/dense_rows.json"), dtype=np.int64)
                tree_of[m] = ti
                trow[m] = rows_map[row[m]]
            else:
                sys.exit("unknown store kind %s" % kind)
            kinds[kind] = kinds.get(kind, 0) + int(len(m))
        if (tree_of < 0).any():
            sys.exit("%s: unmapped positions" % model)
        # id + text proof, every position
        ids_by_tree = [store_ids(store_dir(t, model, "queries")) for t in trees]
        bad_id = bad_text = bad_hop = 0
        for i in range(n):
            ti, r = int(tree_of[i]), int(trow[i])
            rec = tree_rows[ti][r]
            if ids_by_tree[ti].get(r) != rec["query_id"]:
                bad_id += 1
            qtext, hop = canon[qids[i]]
            if rec["question"] != qtext:
                bad_text += 1
            if rec["hop"] != "%dhop" % hop:
                bad_hop += 1
        if bad_id or bad_text or bad_hop:
            sys.exit("%s: join proof failed: ids %d text %d hop %d" % (model, bad_id, bad_text, bad_hop))
        mapping[model] = (tree_of, trow)
        positions_from[model] = kinds
        print("  metaqa queries %-6s join proved on %d positions %s" % (model, n, kinds))
    if not (np.array_equal(mapping["dense"][0], mapping["splade"][0])
            and np.array_equal(mapping["dense"][1], mapping["splade"][1])):
        sys.exit("dense and splade pointers disagree on (tree,row)")
    tree_of, trow = mapping["dense"]

    results = []
    tree_manifests = {}
    for model in ("dense", "splade"):
        out = store_dir("metaqa", model, "queries")
        if os.path.exists(out):
            sys.exit("%s exists; remove it (or the merge already ran)" % out)
        os.makedirs(out)
        mans = [rj("%s/manifest.json" % store_dir(t, model, "queries")) for t in trees]
        for key in ("encoder", "dim", "dtype", "query_instruction", "shard_size"):
            vals = {json.dumps(m.get(key)) for m in mans}
            if len(vals) != 1:
                sys.exit("%s: hop trees disagree on %s: %s" % (model, key, vals))
        tstores = [v1._Store(model, os.path.join(ROOT, store_dir(t, model, "queries")), SS) for t in trees]
        n_shards = (n + SS - 1) // SS
        shards = []
        for k in range(n_shards):
            lo, hi = k * SS, min(n, (k + 1) * SS)
            sel = np.arange(lo, hi)
            pieces, order = [], []
            for ti in range(len(trees)):
                m = np.nonzero(tree_of[sel] == ti)[0]
                if len(m):
                    pieces.append(tstores[ti].read(trow[sel][m]))
                    order.append(m)
            X = v1._reorder(pieces, order, len(sel), model)
            if model == "dense":
                X = np.ascontiguousarray(X, dtype=np.float16)
                if X.shape != (hi - lo, DIM):
                    sys.exit("shape %s" % (X.shape,))
                fn = "shard_%05d.npy" % k
                np.save(os.path.join(out, fn), X)
            else:
                X = X.tocsr()
                if X.shape != (hi - lo, VOCAB):
                    sys.exit("shape %s" % (X.shape,))
                fn = "shard_%05d.npz" % k
                np.savez(os.path.join(out, fn), indices=X.indices.astype(np.int32),
                         indptr=X.indptr.astype(np.int32), data=X.data.astype(np.float32),
                         shape=np.array([hi - lo, VOCAB], dtype=np.int64), format=np.bytes_(b"csr"))
            io.open(os.path.join(out, "ids_%05d.json" % k), "w", encoding="utf-8").write(json.dumps(qids[lo:hi]))
            shards.append({"shard": k, "rows": hi - lo, "file": fn, "ids": "ids_%05d.json" % k})
        wj(out + "/index.json", {"shard_size": SS, "n_shards": n_shards, "shards": shards})
        base = mans[0]
        man = {"dataset": "metaqa", "kind": "queries", "model": model, "n_items": n, "rows_covered": n,
               "shard_size": SS, "n_shards": n_shards, "shards_present": list(range(n_shards)), "complete": True,
               "source_sha256": None, "encoder": base.get("encoder"), "dim": base.get("dim"), "dtype": base.get("dtype")}
        if "query_instruction" in base:
            man["query_instruction"] = base["query_instruction"]
        man["consolidated_v3"] = {
            "utc": utc(),
            "row_order": "row i == query_ids.json[i] == canonical query position i (the query pointer index is the identity)",
            "merged_from": [{"tree": t, "manifest_sha256": sha_file("%s/manifest.json" % store_dir(t, model, "queries")),
                             "n_items": m.get("n_items"), "source_sha256": m.get("source_sha256")}
                            for t, m in zip(trees, mans)],
            "positions_from": positions_from[model],
            "proof": "per position: hop-tree store id == hop-tree queries.jsonl query_id, question == canonical question, "
                     "hop == canonical hop; per row: bytes == old resolver gather(position)"}
        wj(out + "/manifest.json", man)
        tree_manifests[model] = man["consolidated_v3"]["merged_from"]
        # proof against the OLD resolver (old pointer index, hop trees + query patches still on disk)
        E = v1.CanonicalEmbeddings("metaqa", model, "queries")
        new_store = v1._Store(model, os.path.join(ROOT, out), SS)
        for k in range(n_shards):
            lo, hi = k * SS, min(n, (k + 1) * SS)
            old = E.gather(np.arange(lo, hi))
            new = new_store.read(np.arange(lo, hi))
            if model == "dense":
                ok = np.array_equal(np.asarray(old), np.asarray(new))
            else:
                ok = (old != new).nnz == 0 and np.array_equal(old.indptr, new.indptr)
            if not ok:
                sys.exit("%s shard %d: merged rows differ from the old resolver" % (model, k))
        # the pointer index becomes the identity
        p = "%s/metaqa/queries/pointer_index/%s.npz" % (FC, model)
        os.makedirs(HIST, exist_ok=True)
        bak = "%s/metaqa__queries__pointer_index__%s.npz" % (HIST, model)
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        np.savez(p + ".tmp.npz", src=np.zeros(n, dtype=np.int8), row=np.arange(n, dtype=np.int32))
        os.replace(p + ".tmp.npz", p)
        q[model]["stores"] = [{"store": "PHASE_C", "tree": "metaqa", "path": out, "shard_size": SS}]
        q[model]["shard_size_per_tree"] = [SS]
        q[model]["n_items_per_tree"] = [n]
        q[model]["repaired_pointers"] = 0
        q[model]["consolidated_v3"] = {"merged_hop_trees": trees, "identity": True, "positions_from": positions_from[model]}
        shard_hashes = {s["file"]: sha_file(os.path.join(out, s["file"])) for s in shards}
        results.append({"model": model, "store": out, "rows": n, "n_shards": n_shards, "positions_from": positions_from[model],
                        "shard_sha256": shard_hashes, "proof": "bytes == old resolver gather on all %d positions" % n})
        print("  metaqa queries %-6s merged -> %s (%d rows, %d shards), identity pointers" % (model, out, n, n_shards))
    q["trees"] = ["metaqa"]
    q["join"] = ("row i of data/canonical/metaqa/encodings/<model>/queries == query_ids.json[i]; the hop-tree join "
                 "(hop, official_split, original_index) was re-proved per position by id, question text and hop "
                 "before the merge (CONSOLIDATION_V3_LOG phase queries)")
    q["merged_from_hop_trees_v3"] = trees
    wj(PI, pi)
    log_phase("queries", {"results": results, "hop_tree_manifests": tree_manifests})
    return True


# ---------------------------------------------------------------------------------------- trees
def _rename_store_entry(st):
    for k in ("tree", "path", "repairs_tree"):
        v = st.get(k)
        if isinstance(v, str):
            for old, new in RENAMES.items():
                v = v.replace("data/canonical/%s/" % old, "data/canonical/%s/" % new)
                if v == old:
                    v = new
            st[k] = v
    return st


def trees():
    pi = rj(PI)
    if pi["queries"]["metaqa"]["trees"] != ["metaqa"]:
        sys.exit("run the queries phase first (metaqa hop trees are still referenced)")
    moves = []
    markers = []
    # A. the queries-only Phase-C tree `2wiki` folds into `2wiki_universe`, which becomes `2wiki`
    old, uni = CAN + "/2wiki", CAN + "/2wiki_universe"
    if os.path.isdir(uni):
        if os.path.isdir(old):
            for rel in ("encodings/dense/queries", "encodings/splade/queries", "encodings/_src/queries",
                        "queries.jsonl", "query_manifest.json"):
                mv(os.path.join(old, rel), os.path.join(uni, rel), moves)
            left = [os.path.join(dp, f) for dp, _, fn in os.walk(old) for f in fn]
            if left:
                sys.exit("old 2wiki tree still holds files: %s" % left[:5])
            shutil.rmtree(old)          # empty directories only (checked above)
            moves.append({"removed_empty_dir": old})
            journal({"removed_empty_dir": old})
        mv(uni, old, moves)
        markers.append((old, {"RECORD": "TREE_RENAMED_V3", "utc": utc(), "from": "2wiki_universe", "to": "2wiki",
                              "absorbed": "the queries-only Phase-C tree data/canonical/2wiki (encodings/*/queries, "
                                          "encodings/_src/queries, queries.jsonl, query_manifest.json)",
                              "note": "manifest.json files inside the stores still say dataset=2wiki_universe: they are "
                                      "the Phase-C build records and are kept verbatim"}))
    # B. webqsp_rog_v1 -> webqsp
    if os.path.isdir(CAN + "/webqsp_rog_v1"):
        mv(CAN + "/webqsp_rog_v1", CAN + "/webqsp", moves)
        markers.append((CAN + "/webqsp", {"RECORD": "TREE_RENAMED_V3", "utc": utc(), "from": "webqsp_rog_v1", "to": "webqsp",
                                           "note": "WEBQSP_ROG_STANDARD provenance is unchanged: the V1-resolved RoG "
                                                   "WebQSP+CWQ graph, never 'full Freebase'"}))
    # C. the merged-away hop trees and the folded patch directories leave the served layout
    for t in HOP_TREES:
        if os.path.isdir(os.path.join(CAN, t)):
            mv(os.path.join(CAN, t), os.path.join(RET, "canonical", t), moves)
    if os.path.isdir(PATCH):
        mv(PATCH, os.path.join(RET, "final_canonical", "_dense_repair_patch"), moves)
    if os.path.isdir(REV2):
        keep = os.path.join(HIST, "_rev2_encoder_patch")
        os.makedirs(keep, exist_ok=True)
        for f in sorted(os.listdir(REV2)):
            p = os.path.join(REV2, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(keep, f))
        mv(REV2, os.path.join(RET, "final_canonical", "_rev2_encoder_patch"), moves)
    # D. the record follows
    for ds, d in pi["datasets"].items():
        for model in ("dense", "splade"):
            if isinstance(d[model].get("phase_c_tree"), str):
                d[model]["phase_c_tree"] = RENAMES.get(d[model]["phase_c_tree"], d[model]["phase_c_tree"])
            d[model]["stores"] = [_rename_store_entry(s) for s in d[model]["stores"]]
    for ds, d in pi["queries"].items():
        if isinstance(d.get("trees"), list):
            d["trees"] = [RENAMES.get(t, t) for t in d["trees"]]
        for model in ("dense", "splade"):
            d[model]["stores"] = [_rename_store_entry(s) for s in d[model]["stores"]]
    pi["tree_renames_v3"] = {"utc": utc(), "renamed": RENAMES, "merged": {"2wiki (queries-only Phase-C tree)": "into 2wiki_universe before its rename",
                                                                           "metaqa_1hop+metaqa_2hop+metaqa_3hop": "into data/canonical/metaqa/encodings/<model>/queries"},
                             "retired_to": RET, "layout": "one Phase-C tree per dataset: data/canonical/<dataset>/encodings/<model>/{docs,queries}"}
    pi["resolver"]["PHASE_C"] = ("data/canonical/<dataset>/encodings/<model>/<docs|queries>/ ; shard = row // shard_size, "
                                 "offset = row % shard_size (shard_size 40000; webqsp 12000)")
    pi["resolver"]["QUERIES"] = ("queries/pointer_index/<model>.npz ; position i == entry i of query_ids.json ; since V3 the "
                                 "query index is the identity (row i == position i) for all six datasets")
    for k in ("REV2_PATCH", "STORES"):
        pi["resolver"].pop(k, None)
    pi["resolver"]["src_codes"] = {"0": "PHASE_C"}
    pi["resolver"]["src"] = "always 0 since V3: one store per channel"
    wj(PI, pi)
    for d, rec in markers:
        wj(os.path.join(d, "TREE_RENAMED_V3.json"), rec)
    # E. every store path resolves through the unchanged v1 resolver
    v1 = resolver()
    checked = []
    for ds in DATASETS:
        for model in ("dense", "splade"):
            for kind in ("docs", "queries"):
                E = v1.CanonicalEmbeddings(ds, model, kind)
                for s in E.store_spec:
                    if not os.path.isdir(os.path.join(ROOT, s["path"])):
                        sys.exit("store path missing after rename: %s" % s["path"])
                    if s["path"] != store_dir(ds, model, kind):
                        sys.exit("store path is not the one-tree layout: %s" % s["path"])
                if int(E.src.max()) != 0:
                    sys.exit("%s/%s/%s still has src != 0" % (ds, model, kind))
                n = len(E)
                pos = np.unique(np.concatenate([[0, n - 1], np.linspace(0, n - 1, 7).astype(np.int64)]))
                g = E.gather(pos)
                nrow = int(g.shape[0])
                if nrow != len(pos):
                    sys.exit("gather shape")
                checked.append("%s/%s/%s" % (ds, model, kind))
    print("  resolved %d channels through the renamed trees" % len(checked))
    journal_entries = [json.loads(l) for l in io.open(HIST + "/v3_moves.jsonl", encoding="utf-8") if l.strip()]
    log_phase("trees", {"moves": moves, "all_moves_journal": journal_entries, "markers": [d for d, _ in markers],
                        "channels_resolved": checked})
    return True


# --------------------------------------------------------------------------------------- graphs
def _npz_stats(p, n_nodes):
    z = np.load(p)
    src, dst = z["src"], z["dst"]
    keys = list(z.files)
    out = {"n_edges": int(src.size), "keys": keys, "max_endpoint": int(max(src.max(), dst.max())) if src.size else -1,
           "self_loops": int(np.count_nonzero(src == dst)), "src_dtype": str(src.dtype), "dst_dtype": str(dst.dtype)}
    z.close()
    if out["max_endpoint"] >= n_nodes:
        sys.exit("%s: endpoint %d >= n_nodes %d" % (p, out["max_endpoint"], n_nodes))
    return out


def graphs():
    os.makedirs(HIST, exist_ok=True)
    results = {}
    moves = []
    for ds in DATASETS:
        gd = "%s/%s/graph" % (FC, ds)
        g2 = "%s/%s/graph2" % (FC, ds)
        fm = rj(gd + "/GRAPH_MANIFEST.json")
        if fm.get("RECORD") == "CANONICAL_V3_GRAPH":
            print("  %-9s already consolidated" % ds)
            continue
        am = rj(g2 + "/GRAPH_MANIFEST.json") if os.path.isdir(g2) else None
        n_nodes = int(fm["n_nodes"])
        if am and int(am["n_nodes"]) != n_nodes:
            sys.exit("%s: n_nodes differ between graph and graph2" % ds)
        fm_sha = sha_file(gd + "/GRAPH_MANIFEST.json")
        am_sha = sha_file(g2 + "/GRAPH_MANIFEST.json") if am else None
        shutil.copy2(gd + "/GRAPH_MANIFEST.json", "%s/%s__graph__GRAPH_MANIFEST.json" % (HIST, ds))
        families = {}
        history = {}
        for fam, e in fm["families"].items():
            e = dict(e)
            if e.get("present"):
                p = "%s/%s.npz" % (gd, fam)
                e["npz_sha256"] = sha_file(p)
                e["npz_bytes"] = os.path.getsize(p)
                e["origin"] = {"tree": "graph/", "record": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES" if ds != "webqsp" else "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES",
                               "moved_by_v3": False}
            families[fam] = e
        if am:
            for fam, e in am["families"].items():
                if not e.get("present"):
                    continue
                e = dict(e)
                src_npz = "%s/%s.npz" % (g2, fam)
                if sha_file(src_npz) != e.get("npz_sha256"):
                    sys.exit("%s/graph2/%s.npz does not match its manifest digest" % (ds, fam))
                if families.get(fam, {}).get("present"):
                    dec = (am.get("supersedes") or {}).get(fam)
                    if not dec:
                        sys.exit("%s: graph2 duplicates frozen family %s without a supersedes block" % (ds, fam))
                    frozen_npz = "%s/%s.npz" % (gd, fam)
                    frozen_sha = sha_file(frozen_npz)
                    if frozen_sha != dec["frozen_sha256"]:
                        sys.exit("%s: frozen %s.npz is not the pinned artifact" % (ds, fam))
                    retired = "%s/final_canonical/%s__graph__%s__superseded_frozen.npz" % (RET, ds, fam)
                    mv(frozen_npz, retired, moves)
                    history[fam] = {"event": "SUPERSEDED_FROZEN_FAMILY_REPLACED_IN_PLACE", "utc": utc(),
                                    "frozen": dict(families[fam], retired_to=retired),
                                    "declaration": dec, "replacement_pinned_by": "LOCKED_6_OF_6_FAMILY_COMPLETION_V2"}
                mv(src_npz, "%s/%s.npz" % (gd, fam), moves)
                if sha_file("%s/%s.npz" % (gd, fam)) != e["npz_sha256"]:
                    sys.exit("%s: %s.npz changed while moving" % (ds, fam))
                e["file"] = "graph/%s.npz" % fam
                e["origin"] = {"tree": "graph2/ (moved into graph/ by CONSOLIDATION_V3)",
                               "record": "LOCKED_6_OF_6_FAMILY_COMPLETION_V2" if fam in history else "LOCKED_6_OF_6_FAMILY_COMPLETION_V1",
                               "moved_by_v3": True, "graph2_manifest_sha256": am_sha}
                families[fam] = e
            mv(g2 + "/GRAPH_MANIFEST.json", "%s/%s__graph2__GRAPH_MANIFEST.json" % (HIST, ds), moves)
            left = os.listdir(g2)
            if left:
                sys.exit("%s/graph2 still holds %s" % (ds, left))
            os.rmdir(g2)
            moves.append({"removed_empty_dir": g2})
            journal({"removed_empty_dir": g2})
        # verify every present family from the bytes
        stats = {}
        for fam, e in families.items():
            if not e.get("present"):
                continue
            p = "%s/%s" % (os.path.dirname(gd), e["file"])
            st = _npz_stats(p, n_nodes)
            if st["n_edges"] != int(e["n_edges"]):
                sys.exit("%s/%s: %d edges on disk, manifest says %d" % (ds, fam, st["n_edges"], e["n_edges"]))
            if sha_file(p) != e["npz_sha256"]:
                sys.exit("%s/%s: digest" % (ds, fam))
            e["verified_v3"] = {"utc": utc(), "n_edges": st["n_edges"], "max_endpoint": st["max_endpoint"],
                                "self_loops": st["self_loops"], "npz_keys": st["keys"]}
            stats[fam] = st
        out = {"RECORD": "CANONICAL_V3_GRAPH", "dataset": ds, "n_nodes": n_nodes,
               "endpoint_space": fm.get("endpoint_space"), "endpoint_resolution": fm.get("endpoint_resolution"),
               "direction_policy": "as found in the source, NOT symmetrised; each family carries its own `directed` flag "
                                   "(Phase-C families true; the kNN/NER families added after LOCKED_5_OF_5 are undirected, "
                                   "one row per pair)",
               "family_policy": fm.get("family_policy"),
               "nodes_unrepresented_in_graph": fm.get("nodes_unrepresented_in_graph", []),
               "nodes_unrepresented_reason": fm.get("nodes_unrepresented_reason"),
               "families": families,
               "families_present": sorted(f for f, e in families.items() if e.get("present")),
               "families_absent": sorted(f for f, e in families.items() if not e.get("present")),
               "history": history,
               "consolidated_v3": {"utc": utc(), "from_frozen_manifest_sha256": fm_sha, "from_graph2_manifest_sha256": am_sha,
                                   "previous_manifests": ["%s/%s__graph__GRAPH_MANIFEST.json" % (HIST, ds)]
                                                         + (["%s/%s__graph2__GRAPH_MANIFEST.json" % (HIST, ds)] if am else []),
                                   "precedence": "none needed: one tree, one file per family"},
               "phase_c_tree": RENAMES.get(fm.get("phase_c_tree", ds), fm.get("phase_c_tree", ds)),
               "seconds_original_build": fm.get("seconds")}
        wj(gd + "/GRAPH_MANIFEST.json", out)
        results[ds] = {"families_present": out["families_present"], "history": sorted(history.keys()),
                       "edges": {f: s["n_edges"] for f, s in stats.items()}}
        print("  %-9s graph/ %s%s" % (ds, out["families_present"], (" (replaced: %s)" % sorted(history)) if history else ""))
    log_phase("graphs", {"results": results, "moves": moves})
    return True


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    fn = {"queries": queries, "trees": trees, "graphs": graphs}.get(phase)
    if fn is None:
        sys.exit("usage: consolidate_v3_layout.py queries|trees|graphs")
    r = fn()
    print("%s: %s in %.0fs" % (phase, "OK" if r else "PROBLEM", time.time() - t0))
    sys.exit(0 if r else 1)


if __name__ == "__main__":
    main()
