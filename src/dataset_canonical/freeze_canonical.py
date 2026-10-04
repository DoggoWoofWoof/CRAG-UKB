"""
FREEZE THE SIX CANONICAL DATASETS  (2026-09-12)
===============================================
Writes the ONE record the user asked for -- no version chain, no superseding amendments, one
frozen description of six datasets in one normalised form:

    data/final_canonical/CANONICAL_FREEZE.json      the record (self-hashed, pins everything)
    data/final_canonical/<ds>/DATASET.json          per-dataset index: counts, files, digests
    data/final_canonical/_history/INDEX.json        digest inventory of the audit trail

and tidies the root: every record the freeze supersedes (LOCKED_*, the cleanup/check/ledger
records, the pointer index, the old handoff documents) moves under _history/ with its original
bytes -- nothing is edited, nothing is re-hashed, and each move is journaled -- so the audit
trail stays complete while the root holds only what a consumer needs.

Run AFTER materialize_canonical.py has finished every phase (including purge):
    python src/dataset_canonical/freeze_canonical.py            # moves + DATASET.json x6 + freeze
    python src/dataset_canonical/freeze_canonical.py --dataset 2wiki   # just that DATASET.json
    python src/dataset_canonical/freeze_canonical.py --reuse-dataset-records
        # re-freeze without rewriting the six DATASET.json files (they stay byte-identical); picks up
        # data/final_canonical/freebase/DATASET.json (written by freebase/freeze_freebase.py) as the
        # FREEBASE section.  The previous CANONICAL_FREEZE.json is archived under _history/records/
        # (journaled) before the new one is written -- supersession, never an edit.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FC = "data/final_canonical"
HIST = FC + "/_history"
OUT = FC + "/CANONICAL_FREEZE.json"
DATASETS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
SPLITS = {"metaqa": ["train", "dev", "test"], "squad": ["train", "dev"], "musique": ["train", "dev", "test"],
          "hotpotqa": ["train", "validation", "test"], "2wiki": ["train", "dev", "test"], "webqsp": ["train", "test"]}
CORE_INVARIANT = ("QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN. Every corpus here is "
                  "query-independent: the node set is a function of the source corpus alone.")
USER_RULINGS = {
    "2026-09-12_end_state": "at the end of all this i dont want versions, i just want canonical frozen datasets which "
                            "has all cleaned edges, embeddings, queries, everything that we might need in a normalized "
                            "form for all, we will try to get freebase also to same level but till then atleast these 6 "
                            "need to be ready and the rest other versions i give you permission to delete and freeup space",
    "2026-09-12_heal": "then heal it, i want everything fixed",
    "2026-09-12_consolidate": "can we clean all that up modify the pointers and only keep the proper datasets with their "
                              "fixed stuff rather than bloating up all",
}
# what stays at the root besides the six trees, freebase_v3/ and _history/
KEEP_ROOT = {"CANONICAL_FREEZE.json", "VERIFICATION.json", "HANDOFF.md", "canonical.py", "pointer_resolver.py",
             "_APPROVALS.json", "_APPROVAL_CRITERIA.md", "SOURCE_CONTRACTS_FOR_REVIEW.md", "_WEBQSP_ACCEPTANCE_GATE.json"}
GOVERNANCE = ["_APPROVALS.json", "_APPROVAL_CRITERIA.md", "SOURCE_CONTRACTS_FOR_REVIEW.md", "_WEBQSP_ACCEPTANCE_GATE.json"]
SHIM = '''"""
pointer_resolver -- compatibility name. There is no pointer index any more: since the freeze of
2026-09-12 every embedding store holds its rows in position order (row i == line i of
nodes.jsonl, row j == query j). Everything lives in canonical.py; this module re-exports it so
code written against CanonicalEmbeddings / CanonicalGraph / families keeps working.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from canonical import (DATASETS, SPLITS, Dataset, Embeddings, Graph, CanonicalEmbeddings, CanonicalGraph,  # noqa: F401,E402
                       families, provenance, history, family_source)
'''


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rj(p):
    return json.load(io.open(p, encoding="utf-8"))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def record_hash(rec, crlf=True):
    """The repo convention: json.dumps(indent=1) of the record without its own digest fields."""
    d = dict(rec)
    d.pop("RECORD_SHA256", None)
    d.pop("RECORD_SHA256_LF", None)
    txt = json.dumps(d, indent=1)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def write_record(p, rec):
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))


def say(*a):
    print(*a)
    sys.stdout.flush()


def finfo(p, with_sha=True):
    d = {"file": p.replace(os.sep, "/"), "bytes": os.path.getsize(p)}
    if with_sha:
        d["sha256"] = sha_file(p)
    return d


def count_lines(p):
    n = 0
    with io.open(p, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def first_keys(p):
    with io.open(p, encoding="utf-8") as f:
        return sorted(json.loads(f.readline()).keys())


def journal(entry):
    os.makedirs(HIST + "/logs", exist_ok=True)
    entry = dict(entry)
    entry["utc"] = utc()
    with io.open(HIST + "/logs/RECORD_MOVES.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def mv(s, d):
    if not os.path.exists(s):
        raise RuntimeError("move source missing: %s" % s)
    if os.path.exists(d):
        raise RuntimeError("move target exists: %s" % d)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    sha = sha_file(s) if os.path.isfile(s) else None
    os.rename(s, d)
    journal({"from": s, "to": d, "sha256": sha})
    say("  moved %s -> %s" % (s, d))


# ------------------------------------------------------------------------------ root tidy
def tidy_root():
    """Everything the freeze supersedes goes under _history/ unchanged; the shim replaces the V3 resolver."""
    moved = []
    # the pre-freeze HANDOFF describes the pointer-indexed package; it is archived, not overwritten
    hd = FC + "/HANDOFF.md"
    if os.path.exists(hd) and "CANONICAL_FREEZE.json" not in io.open(hd, encoding="utf-8").read():
        mv(hd, HIST + "/docs/HANDOFF_pre_freeze.md")
        moved.append([hd, HIST + "/docs/HANDOFF_pre_freeze.md"])
    for f in sorted(os.listdir(FC)):
        p = os.path.join(FC, f)
        if f in KEEP_ROOT or f in DATASETS or f in ("freebase", "freebase_v3", "_history"):
            continue
        if os.path.isdir(p):
            if f == "__pycache__":
                shutil.rmtree(p)
                continue
            if not any(fs for _, _, fs in os.walk(p)):
                shutil.rmtree(p)
                journal({"removed_empty_dir": p})
                say("  removed empty %s" % p)
                continue
            raise RuntimeError("unexpected directory at the root: %s" % p)
        if f.endswith(".log") or f in ("CONSOLIDATION_V3_LOG.json",):
            d = HIST + "/logs/" + f
        elif f.endswith(".md"):
            d = HIST + "/docs/" + f
        elif f.endswith(".json") or f.startswith("_GATE"):
            d = HIST + "/records/" + f
        else:
            raise RuntimeError("do not know where %s belongs" % p)
        mv(p, d)
        moved.append([p, d])
    # the V3 resolver is archived beside v1/v2 and replaced by the shim
    pr = FC + "/pointer_resolver.py"
    if os.path.exists(pr) and "compatibility name" not in io.open(pr, encoding="utf-8").read():
        mv(pr, HIST + "/pre_v3/pointer_resolver_v3.py")
        moved.append([pr, HIST + "/pre_v3/pointer_resolver_v3.py"])
    if not os.path.exists(pr):
        with io.open(pr, "w", encoding="utf-8", newline="\n") as f:
            f.write(SHIM)
        say("  wrote shim %s" % pr)
    return moved


# ------------------------------------------------------------------------------ DATASET.json
def dataset_record(ds):
    d = FC + "/" + ds
    for must in ("nodes.jsonl", "queries/query_ids.json", "graph/GRAPH_MANIFEST.json", "embeddings"):
        if not os.path.exists(os.path.join(d, must)):
            raise RuntimeError("%s lacks %s" % (ds, must))
    for gone in ("pointer_index", "reuse_map", "queries/pointer_index", "graph2", "encoder_row_of_node.npz"):
        if os.path.exists(os.path.join(d, gone)):
            raise RuntimeError("%s still has %s; run materialize_canonical.py layout" % (ds, gone))
    t0 = time.time()
    say("  %s: nodes" % ds)
    nodes = finfo(d + "/nodes.jsonl")
    n_nodes = count_lines(d + "/nodes.jsonl")
    nodes.update({"n": n_nodes, "fields": first_keys(d + "/nodes.jsonl"),
                  "position_semantics": "line i of nodes.jsonl is position i: embeddings row i, graph endpoint i, cache id i"})

    say("  %s: queries" % ds)
    qids = rj(d + "/queries/query_ids.json")
    by_split, cat = {}, []
    for sp in SPLITS[ds]:
        p = d + "/queries/%s.jsonl" % sp
        e = finfo(p)
        ids = []
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                ids.append(json.loads(ln)["query_id"])
        e["n"] = len(ids)
        by_split[sp] = e
        cat.extend(ids)
    if cat != qids:
        raise RuntimeError("%s: query_ids.json != split concatenation" % ds)
    queries = {"n": len(qids), "splits_in_row_order": SPLITS[ds], "by_split": by_split,
               "query_ids": finfo(d + "/queries/query_ids.json"),
               "fields": first_keys(d + "/queries/%s.jsonl" % SPLITS[ds][0]),
               "gold_field": "gold_node_ids (canonical node_ids; resolve through nodes.jsonl)",
               "row_semantics": "query j == query_ids.json[j] == the j-th line of the split files concatenated in the order above; "
                                "embeddings/*/queries row j and retrieval_cache row j are that query"}
    lanes = d + "/queries/lanes"
    if os.path.isdir(lanes):
        queries["lanes"] = {f: finfo(os.path.join(lanes, f)) for f in sorted(os.listdir(lanes))}
    for f in sorted(os.listdir(d)):
        if f.startswith("eval_") and f.endswith(".jsonl"):
            e = finfo(os.path.join(d, f))
            e["n"] = count_lines(os.path.join(d, f))
            queries.setdefault("eval_subsets", {})[f] = e

    say("  %s: embeddings" % ds)
    emb = {}
    for model in ("dense", "splade"):
        for kind in ("docs", "queries"):
            sd = d + "/embeddings/%s/%s" % (model, kind)
            m = rj(sd + "/manifest.json")
            expect = n_nodes if kind == "docs" else len(qids)
            if int(m["n_rows"]) != expect:
                raise RuntimeError("%s %s %s: %d rows vs %d expected" % (ds, model, kind, m["n_rows"], expect))
            files = sorted(os.listdir(sd))
            shard_files = [f for f in files if f.startswith("shard_")]
            if len(shard_files) != m["n_shards"] or any(f not in shard_files + ["manifest.json"] for f in files):
                raise RuntimeError("%s %s %s: store directory does not match its manifest" % (ds, model, kind))
            mat = m["materialised"]
            emb.setdefault(model, {})[kind] = {
                "dir": "embeddings/%s/%s" % (model, kind), "n_rows": m["n_rows"], "shard_size": m["shard_size"],
                "n_shards": m["n_shards"], "bytes": int(sum(s["bytes"] for s in m["shards"])),
                "manifest": finfo(sd + "/manifest.json"), "format": m["format"], "encoder": m["encoder"],
                "row_digest_chain_sha256": mat["proof"]["new_store_row_digest_chain_sha256"],
                "proof": {"method": mat["proof"]["method"], "positions_checked": mat["proof"]["positions_checked"],
                          "equal": mat["proof"]["equal"]},
                "dense_integrity": mat.get("dense_integrity"),
                "duplicate_positions": mat.get("duplicate_positions"),
                "v2_fingerprint_reproduced": (mat.get("v2_fingerprint") or {}).get("match"),
            }

    say("  %s: graph" % ds)
    gm = rj(d + "/graph/GRAPH_MANIFEST.json")
    if gm.get("RECORD") != "CANONICAL_GRAPH" or int(gm["n_nodes"]) != n_nodes:
        raise RuntimeError("%s: graph manifest not normalised or n_nodes mismatch" % ds)
    fams = {}
    for name, fam in gm["families"].items():
        if not fam.get("present"):
            fams[name] = {"present": False, "reason": fam.get("reason")}
            continue
        p = d + "/graph/%s.npz" % name
        sha = sha_file(p)
        if sha != fam["npz_sha256"]:
            raise RuntimeError("%s/%s: npz sha drifted since the manifest" % (ds, name))
        fams[name] = {"present": True, "file": "graph/%s.npz" % name, "n_edges": fam["n_edges"], "directed": fam["directed"],
                      "attributes": fam.get("attributes"), "relation_vocabulary": fam.get("relation_vocabulary"),
                      "directed_as_frozen": fam.get("directed_as_frozen"), "storage_census": fam.get("storage_census"),
                      "bytes": fam["npz_bytes"], "sha256": sha, "origin": fam.get("origin"), "edge_subtype": fam.get("edge_subtype")}
    graph = {"manifest": finfo(d + "/graph/GRAPH_MANIFEST.json"), "families": fams,
             "families_present": gm["families_present"], "families_absent": gm["families_absent"],
             "n_edges_total": int(sum(v["n_edges"] for v in fams.values() if v["present"])),
             "endpoint_space": gm["endpoint_space"], "direction_policy": gm.get("direction_policy"),
             "nodes_unrepresented_in_graph": gm.get("nodes_unrepresented_in_graph"),
             "history": gm.get("history")}

    say("  %s: caches" % ds)
    caches = {}
    for model in ("dense", "splade"):
        p = d + "/retrieval_cache/%s_top1000.npz" % model
        z = np.load(p)
        ids, sc = z["ids"], z["scores"]
        shape, K = list(ids.shape), int(ids.shape[1])
        mx = int(ids.max())
        z.close()
        if shape[0] != len(qids) or mx >= n_nodes:
            raise RuntimeError("%s %s cache: shape %s, max id %d" % (ds, model, shape, mx))
        e = finfo(p)
        e.update({"shape": shape, "K": K, "ids_dtype": str(ids.dtype), "scores_dtype": str(sc.dtype),
                  "index_space": "positions (nodes.jsonl line numbers); row j is query j", "max_id_seen": mx})
        mp = d + "/retrieval_cache/%s_top1000.meta.json" % model
        if os.path.exists(mp):
            e["meta"] = finfo(mp)
            e["meta"]["content"] = rj(mp)
        caches[model] = e

    say("  %s: records" % ds)
    records, kept_dirs = {}, {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f in ("nodes.jsonl", "DATASET.json") or f.startswith("eval_"):
            continue
        if os.path.isdir(p):
            if f in ("embeddings", "graph", "queries", "retrieval_cache"):
                continue
            if not any(fs for _, _, fs in os.walk(p)):      # an emptied staging skeleton, not a kept dir
                shutil.rmtree(p)
                journal({"removed_empty_dir": p})
                say("  removed empty %s" % p)
                continue
            inv = []
            for r, _, fs in os.walk(p):
                for x in sorted(fs):
                    inv.append(finfo(os.path.join(r, x)))
            kept_dirs[f] = {"files": len(inv), "bytes": int(sum(x["bytes"] for x in inv)), "inventory": inv,
                            "why": {"_acquisition": "acquired source data for the RoG WebQSP+CWQ subgraph (source of record for this dataset; "
                                                    "nsm/extracted/webqsp/webqsp/*_simple.json is read by k_semantics_and_splits.py)",
                                    "v1": "the structured KB tables (nodes, edges, relations, entity/CVT text) the node text was rendered from"}.get(f, "kept")}
            continue
        records[f] = finfo(p)

    tree_bytes = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(d) for x in fs)
    rec = {
        "RECORD": "CANONICAL_DATASET", "dataset": ds, "frozen_utc": utc(),
        "n_nodes": n_nodes, "n_queries": len(qids), "queries_by_split": {sp: by_split[sp]["n"] for sp in SPLITS[ds]},
        "layout": {"nodes": "nodes.jsonl", "queries": "queries/<split>.jsonl + queries/query_ids.json",
                   "embeddings": "embeddings/<dense|splade>/<docs|queries>/shard_XXXXX.<npy|npz> + manifest.json",
                   "graph": "graph/<family>.npz + graph/GRAPH_MANIFEST.json", "retrieval_cache": "retrieval_cache/<model>_top1000.npz",
                   "loader": "data/final_canonical/canonical.py"},
        "nodes": nodes, "queries": queries, "embeddings": emb, "graph": graph, "retrieval_cache": caches,
        "records": records, "kept_dirs": kept_dirs,
        "tree_bytes": int(tree_bytes), "seconds": round(time.time() - t0, 1),
    }
    write_record(d + "/DATASET.json", rec)
    say("  %s DATASET.json written (%.1f GB tree, %.0fs)" % (ds, tree_bytes / 1e9, time.time() - t0))
    return rec


# ------------------------------------------------------------------------------ query lanes
LANE_SEMANTICS = {
    "LEGACY_CONTINUITY": "this repo's frozen legacy evaluation subset mapped onto canonical ids (legacy_query_id / legacy_cache_row); "
                         "built for paired legacy -> canonical analysis ONLY. It is train-derived on squad, musique and 2wiki and mostly "
                         "train-derived on hotpotqa, so it is NOT an evaluation population; on metaqa it is 1,998 dev questions whose "
                         "legacy gold sets differ from the canonical ones on 46 rows (the legacy substrate merged case variants of "
                         "entity names; CANONICALIZATION_AUDIT.eval_subset_gold_consistency).",
    "QUALITY_LOCKED": "the Track-B (2026-09-06) choice of the official held-out split with public labels: dev where the test labels are "
                      "hidden (squad, musique, 2wiki), validation for hotpotqa, and TEST for metaqa (its test labels are public). "
                      "For metaqa this lane is therefore the official test split and is NOT an evaluation population under the standing "
                      "rule that bars test results: EVAL_SPLITS (metaqa dev, 39,138) governs. The lane file stays as a query subset "
                      "(query subsetting is allowed; the corpus never depends on it).",
    "SCALE_ALL": "every official query of every split, ids only (test rows are blind on musique, hotpotqa, 2wiki).",
}


def query_lanes():
    """Census of the served lane files, straight from their rows (n, by official split, unanswerable rows)."""
    out = {}
    for ds in DATASETS:
        d = FC + "/%s/queries/lanes" % ds
        if not os.path.isdir(d):
            out[ds] = None
            continue
        lanes = {}
        for f in sorted(os.listdir(d)):
            lane = f.split(".")[0]
            n = 0
            by = {}
            imp = 0
            has_imp = False
            with io.open(os.path.join(d, f), encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    by[r.get("split")] = by.get(r.get("split"), 0) + 1
                    if "is_impossible" in r:
                        has_imp = True
                        if r["is_impossible"]:
                            imp += 1
            # unanswerable_rows is None where the lane rows carry no is_impossible field (ids-only lanes)
            lanes[lane] = {"file": "%s/queries/lanes/%s" % (ds, f), "n": n, "by_split": dict(sorted(by.items())),
                           "unanswerable_rows": imp if has_imp else None}
        out[ds] = lanes
    return out


# ------------------------------------------------------------------------------ the freeze
HISTORY_NOT_INDEXED = {
    "logs/HANDOFF_CLAIMS_CHECK.json": "written by check_handoff.py after every freeze (the gate on HANDOFF.md), so its bytes change "
                                      "after the index is made; verify_canonical.py checks it by content instead (present, VERDICT "
                                      "ALL_PASS, made on the served HANDOFF.md)",
}


def fb_source_layer():
    """freebase_v3/ after the 2026-09-13 deletion: the terminal closure record, the deletion log, and what stays"""
    v2 = FC + "/freebase_v3/FREEBASE_V3_CLOSURE_STATUS_V2.json"
    dl = HIST + "/logs/FREEBASE_V3_SUBLAYER_DELETION.json"
    r2 = HIST + "/logs/FREEBASE_R2_IDENTITY_CENSUS.json"
    if not (os.path.exists(v2) and os.path.exists(dl)):
        return None
    V = rj(v2)
    D = rj(dl)
    return {
        "status": V["STATUS"],
        "closure_status_v2": finfo(v2), "closure_status_v1": V["BUILDS_ON"]["file"],
        "sublayer_deletion_log": finfo(dl), "deleted": {"files": D["deleted"], "bytes": D["bytes_deleted"], "status": D["status"],
                                                        "layers": sorted(D["layers"])},
        "r2_identity_census": finfo(r2) if os.path.exists(r2) else None,
        "stays": V["STILL_PROTECTED"],
        "open_items": {it["item"]: it["disposition"] for it in V["OPEN_ITEMS_DISPOSITION"]},
    }


def history_index():
    inv = []
    for r, _, fs in os.walk(HIST):
        for x in sorted(fs):
            if x == "INDEX.json" and os.path.abspath(r) == os.path.abspath(HIST):
                continue
            rel = os.path.relpath(os.path.join(r, x), HIST).replace(os.sep, "/")
            if rel in HISTORY_NOT_INDEXED:
                continue
            inv.append(finfo(os.path.join(r, x)))
    idx = {"RECORD": "HISTORY_INDEX", "utc": utc(), "what": "digest inventory of the audit trail under _history/; "
           "every superseded record, the pre-consolidation pointer indexes and resolvers, the Phase-C build manifests, and the logs",
           "files": len(inv), "bytes": int(sum(x["bytes"] for x in inv)), "not_indexed": HISTORY_NOT_INDEXED, "inventory": inv}
    write_record(HIST + "/INDEX.json", idx)
    return idx


def lineage():
    """The superseded records, by name and their own recorded digests (re-verified before pinning)."""
    out = []
    for name in ("LOCKED_5_OF_5_BENCHMARK_SUBSTRATES", "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES",
                 "LOCKED_6_OF_6_FAMILY_COMPLETION_V1", "LOCKED_6_OF_6_FAMILY_COMPLETION_V2"):
        p = HIST + "/records/%s.json" % name
        r = rj(p)
        out.append({"record": name, "moved_to": p, "created_utc": r.get("created_utc"),
                    "RECORD_SHA256": r.get("RECORD_SHA256"), "self_hash_ok": record_hash(r) == r.get("RECORD_SHA256"),
                    "file_sha256": sha_file(p)})
    return out


def summarise_logs():
    v3 = rj(HIST + "/logs/CONSOLIDATION_V3_LOG.json")
    mat = rj(HIST + "/logs/MATERIALIZE_LOG.json")
    heal = [ph for ph in v3["phases"] if ph["phase"] == "heal"]
    fold = [ph for ph in v3["phases"] if ph["phase"] == "fold"]
    healed = int(sum(r["rows_written"] + r["rows_already_equal"] for ph in heal for r in ph["results"] if "rows_written" in r))
    heal_skipped = [{"patch": r["patch"], "action": r.get("action"), "why": r.get("why")} for ph in heal for r in ph["results"] if "rows_written" not in r]
    folded = {("%s/%s" % (r["dataset"], r["model"])): int(r["rows_written"] + r["rows_already_equal"]) for ph in fold for r in ph["results"]}
    docs = {"%s/%s" % (p["dataset"], p["model"]): {"n_rows": p["n_rows"], "chain_sha256": p["chain_sha256"], "proof_equal": p["proof_equal"],
                                                  "damaged_rows": p["damaged_rows"], "v2_fingerprint_match": (p.get("v2_fingerprint") or {}).get("match")}
            for p in mat["phases"] if p["phase"] == "docs"}
    deleted = {"old_stores": [{"path": p["deleted"], "bytes": p["bytes"]} for p in mat["phases"] if p["phase"] == "retire_store"],
               "layout": {p["dataset"]: p["deleted_bytes"] for p in mat["phases"] if p["phase"] == "layout"}}
    purge = [p for p in mat["phases"] if p["phase"] == "purge"]
    if purge:
        pu = purge[-1]
        deleted["phase_c_tree_bytes"] = int(sum(t["bytes"] for t in pu["phase_c_trees"]))
        deleted["retired_v3_bytes"] = (pu.get("retired_v3") or {}).get("bytes")
        deleted["cleanup_pass_3"] = {k: v for k, v in (pu.get("cleanup_pass_3") or {}).items() if k != "removed_empty_dirs"}
    deleted["total_bytes"] = int(sum(x["bytes"] for x in deleted["old_stores"]) + sum(deleted["layout"].values())
                                 + (deleted.get("phase_c_tree_bytes") or 0) + (deleted.get("retired_v3_bytes") or 0)
                                 + ((deleted.get("cleanup_pass_3") or {}).get("bytes") or 0))
    return {
        "consolidation_2026_09_11_12": {
            "log": HIST + "/logs/CONSOLIDATION_V3_LOG.json",
            "healed_dense_rows": healed, "heal_skipped": heal_skipped,
            "healed_note": "142,633 damaged Phase-C dense rows: 129,819 healed in place from the re-encoded repair vectors (bit-exact); "
                           "the 12,814 webqsp__docs patch rows targeted the Phase-C webqsp docs store, which no pointer served and whose "
                           "rows differ from the patch (writing it would have damaged healthy rows) -- SKIPPED_ORPHAN, retired with the patch",
            "rev2_rows_folded_into_own_rows": folded,
            "musique_knn": "the frozen LOCKED_5_OF_5 kNN family (266,488 edges, 4,402 weights off by up to 0.50) was replaced by the "
                           "audited rebuild (265,366 edges, every weight within 6e-7 of the vectors); the stale copy was retired and deleted",
            "graph_trees": "graph2/ (NER/kNN families added after the five were frozen) merged into graph/, one tree per dataset",
            "metaqa_queries": "the three hop trees' query stores merged into one store in query order; every position joined by id, question and hop",
        },
        "materialisation_2026_09_12": {
            "log": HIST + "/logs/MATERIALIZE_LOG.json",
            "what": "every docs store rewritten in position order from the pointer index; every query store moved (already in order); "
                    "per-row sha1 digests of old and new stores compared through the map for every position",
            "docs_channels": docs,
        },
        "deleted": deleted,
    }


def freeze(ds_records):
    t0 = time.time()
    fb_rec = rj(FC + "/freebase/DATASET.json") if os.path.exists(FC + "/freebase/DATASET.json") else None
    lin = lineage()
    for f in sorted(os.listdir(HIST + "/records")):
        if f.startswith("CANONICAL_FREEZE_") and f.endswith(".json"):
            r = rj(HIST + "/records/" + f)
            lin.append({"record": f[:-5], "moved_to": HIST + "/records/" + f, "created_utc": r.get("frozen_utc"),
                        "RECORD_SHA256": r.get("RECORD_SHA256"), "self_hash_ok": record_hash(r) == r.get("RECORD_SHA256"),
                        "file_sha256": sha_file(HIST + "/records/" + f), "what": "the CANONICAL_FREEZE this record supersedes"})
    K = rj(HIST + "/records/K_SEMANTICS_AND_EVAL_SPLITS.json")
    lanes = query_lanes()
    for ds, ln in lanes.items():
        if ln:
            assert set(ln) == {"LEGACY_CONTINUITY", "QUALITY_LOCKED", "SCALE_ALL"}, (ds, sorted(ln))
    assert lanes["metaqa"]["QUALITY_LOCKED"]["by_split"] == {"test": 39093}, lanes["metaqa"]["QUALITY_LOCKED"]
    builders = rj(HIST + "/builders/BUILDERS.json") if os.path.exists(HIST + "/builders/BUILDERS.json") else None
    GF = rj(HIST + "/records/GOLD_FIELD_CHECK.json")
    SP = rj(HIST + "/records/SPLADE_POOLING_FINDING.json")
    IB = rj(HIST + "/records/ID_BRIDGE.json")
    logs = summarise_logs()
    hist = history_index()

    datasets = {}
    for ds in DATASETS:
        r = ds_records[ds]
        datasets[ds] = {
            "DATASET.json": finfo(FC + "/%s/DATASET.json" % ds),
            "n_nodes": r["n_nodes"], "n_queries": r["n_queries"], "queries_by_split": r["queries_by_split"],
            "nodes_sha256": r["nodes"]["sha256"],
            "embeddings": {m: {k: {"n_rows": v["n_rows"], "n_shards": v["n_shards"], "bytes": v["bytes"],
                                   "row_digest_chain_sha256": v["row_digest_chain_sha256"], "manifest_sha256": v["manifest"]["sha256"]}
                               for k, v in kinds.items()} for m, kinds in r["embeddings"].items()},
            "graph": {"manifest_sha256": r["graph"]["manifest"]["sha256"], "families_present": r["graph"]["families_present"],
                      "families_absent": r["graph"]["families_absent"],
                      "n_edges": {k: v["n_edges"] for k, v in r["graph"]["families"].items() if v["present"]},
                      "npz_sha256": {k: v["sha256"] for k, v in r["graph"]["families"].items() if v["present"]}},
            "retrieval_cache": {m: {"sha256": v["sha256"], "shape": v["shape"]} for m, v in r["retrieval_cache"].items()},
            "eval_split": K["EVAL_SPLITS"][ds],
            "gold_resolution": {sp: {"gold_refs": v["gold_refs"], "resolved": v["gold_refs_resolved"], "verdict": v["verdict"]}
                                for sp, v in GF["results"][ds]["splits"].items()} if ds in GF["results"] else None,
            "id_bridge": {k: IB["datasets"][ds].get(k) for k in ("rule", "gold_resolution_pct", "PASS")} if ds in IB["datasets"] else None,
            "tree_bytes": r["tree_bytes"],
        }
    edge_matrix = {ds: {fam: (datasets[ds]["graph"]["n_edges"].get(fam)) for fam in ("structural", "ner", "knn")} for ds in DATASETS}
    totals = {
        "datasets": 6,
        "nodes": int(sum(datasets[d]["n_nodes"] for d in DATASETS)),
        "queries": int(sum(datasets[d]["n_queries"] for d in DATASETS)),
        "edges_served": int(sum(sum(datasets[d]["graph"]["n_edges"].values()) for d in DATASETS)),
        "edge_families_served": int(sum(len(datasets[d]["graph"]["n_edges"]) for d in DATASETS)),
        "embedding_channels": 24,
        "embedding_bytes": int(sum(v["bytes"] for d in DATASETS for kinds in datasets[d]["embeddings"].values() for v in kinds.values())),
        "tree_bytes": int(sum(datasets[d]["tree_bytes"] for d in DATASETS)),
    }
    rec = {
        "RECORD": "CANONICAL_FREEZE",
        "STATUS": "FROZEN",
        "frozen_utc": utc(),
        "STATEMENT": "Six benchmark datasets -- metaqa, squad, musique, hotpotqa, 2wiki, webqsp -- each a single tree under "
                     "data/final_canonical/<ds>/ in one normalised form: nodes, queries with splits, dense and SPLADE embeddings for "
                     "documents and queries stored in position order, structural/NER/kNN edge families on positions, and top-1000 "
                     "retrieval caches. This is the only served copy. There is no version chain: the records this freeze supersedes are "
                     "kept unchanged under _history/ as the audit trail and nothing else is served."
                     + (" The seventh tree, freebase/, holds the full Freebase universe in the same position form (see FREEBASE); it "
                        "has every node named, every cleaned edge as CSR, relations and metadata on positions, and no embeddings." if fb_rec else ""),
        "CORE_INVARIANT": CORE_INVARIANT,
        "POSITION_INVARIANT": "Position i is line i of nodes.jsonl, row i of embeddings/*/docs, an endpoint value in graph/*.npz and an id "
                              "in retrieval_cache/*. Query j is entry j of queries/query_ids.json (the split files concatenated in the "
                              "recorded order), row j of embeddings/*/queries and row j of retrieval_cache/*. No pointer index, no join.",
        "USER_RULINGS": USER_RULINGS,
        "LAYOUT": {
            "root": "data/final_canonical/",
            "per_dataset": ["nodes.jsonl", "queries/<split>.jsonl", "queries/query_ids.json", "queries/lanes/*", "eval_*.jsonl (query subsets)",
                            "embeddings/{dense,splade}/{docs,queries}/shard_XXXXX.{npy,npz} + manifest.json",
                            "graph/{structural,ner,knn}.npz + GRAPH_MANIFEST.json", "retrieval_cache/{dense,splade}_top1000.npz (+ .meta.json)",
                            "DATASET.json", "build/source/audit records of that dataset (listed in DATASET.json.records)"],
            "shards": "40,000 rows per shard for every channel of every dataset; the last shard is shorter",
            "dense": "float16 (rows, 1536), unit L2 normalised, Alibaba-NLP/gte-Qwen2-1.5B-instruct; queries carry the recorded instruction prefix",
            "splade": "CSR npz (indices int32, indptr int32, data float32, shape int64 [rows, 30522], format b'csr'), naver/splade-cocondenser-ensembledistil",
            "loader": "data/final_canonical/canonical.py (Dataset, Embeddings, Graph); pointer_resolver.py is a compatibility shim",
            "governance_files_at_root": GOVERNANCE,
            "history": "_history/ (records/, docs/, logs/, pre_v3/, materialize/, phase_c/, builders/) indexed by _history/INDEX.json",
            "freebase": ("freebase/ is the seventh tree: the full CRAG_FREEBASE_CANONICAL universe (302M nodes / 2.06B edges) in the "
                         "same position form -- nodes/ (uid index, kind, name_kind, 76 parquet shards with a name on every row), graph/ "
                         "(forward and reverse CSR), relations/, metadata/ (12 tables by position), bridge/ (webqsp positions, schema ids), "
                         "records/, DATASET.json; loader canonical.Freebase. No queries, no embeddings (infeasible on this machine; not "
                         "fabricated). freebase_v3/ is its source layer: the raw mirror, the IDIR zip, the recovery sources and canonical/ "
                         "(the frozen V3 graph) stay; the data of the sub-layers the tree supersedes (overlay_v1, inferred_name_v1, "
                         "inference_overlay_v1, semantic_v1, pass_a-d, probes) was deleted on 2026-09-13 under the owner's authorisation "
                         "(FREEBASE.source_layer: the deletion log and the terminal closure record FREEBASE_V3_CLOSURE_STATUS_V2)."
                         if fb_rec else
                         "freebase_v3/ is a separate layer (CRAG_FREEBASE_CANONICAL, 302M nodes / 2.06B edges, CLOSED_FOR_COMPUTATION) and is "
                         "NOT part of this freeze; it is to be brought to the same normalised form later. Its raw mirror, IDIR zip and "
                         "overlay_v1 stay untouched."),
        },
        "DATASETS": datasets,
        "TOTALS": totals,
        "FREEBASE": ({
            "DATASET.json": finfo(FC + "/freebase/DATASET.json"),
            "n_nodes": fb_rec["n_nodes"], "n_edges": fb_rec["n_edges"], "n_relations": fb_rec["n_relations"],
            "names": {k: fb_rec["names"][k] for k in ("census", "actual_names", "identifier_names", "inferred_names", "generated_floor_names")},
            "name_kinds": fb_rec["names"]["NAME_KINDS"],
            "by_kind": fb_rec["nodes"]["by_kind"], "flipped_edges": fb_rec["graph"]["flipped_edges"],
            "tree_bytes": fb_rec["tree_bytes"],
            "position_invariant": fb_rec["POSITION_INVARIANT"],
            "not_included": fb_rec["not_included"],
            "verify": "python src/dataset_canonical/freebase/verify_freebase.py --full (also run by verify_canonical.py --full)",
            "source_layer": fb_source_layer(),
        } if fb_rec else None),
        "EDGE_FAMILIES": {"policy": "families are recorded, never invented; since the consolidation all six carry structural, ner and knn, "
                                    "each with its own provenance and direction flag in GRAPH_MANIFEST.json. structural is directed as found "
                                    "in the source; every ner and knn family is one row per unordered pair, directed:false, with a storage "
                                    "census (reverse rows, self loops, distinct pairs) recorded beside the flag",
                          "directed": {ds: {fam: bool(v["directed"]) for fam, v in ds_records[ds]["graph"]["families"].items() if v["present"]} for ds in DATASETS},
                          "matrix": edge_matrix},
        "K_SEMANTICS": K["K_SEMANTICS"],
        "EVAL_SPLITS": K["EVAL_SPLITS"],
        "QUERY_LANES": {
            "what": "the three query-lane files under <ds>/queries/lanes/ are this repo's query SUBSETS (Track B, 2026-09-06; "
                    "scratchpad/final_canonical_build/query_lanes.py, archived under _history/builders/), not the source's; the census "
                    "below is read from the rows themselves. Evaluate on EVAL_SPLITS; a lane never widens or narrows the corpus.",
            "semantics": LANE_SEMANTICS,
            "ruling": "metaqa QUALITY_LOCKED == the official test split (39,093): not an evaluation population (standing rule: no test "
                      "results); metaqa evaluates on dev (EVAL_SPLITS). LEGACY_CONTINUITY is for paired legacy comparison only: "
                      "train-derived on squad/musique/2wiki (2,000 each), 1,842 train + 158 validation on hotpotqa, 1,998 dev on metaqa.",
            "census": lanes,
        },
        "webqsp_holdout_stride": K.get("webqsp_holdout_stride"),
        "GOLD_FIELD": {"field": GF["gold_field"], "resolution_method": GF["resolution_method"]},
        "SPLADE_POOLING": {"STATUS": SP["STATUS"], "FINDING": SP["FINDING"], "CONSUMER_GUIDANCE": SP.get("CONSUMER_GUIDANCE"),
                           "WEBQSP_DECISION": SP.get("WEBQSP_DECISION"), "record": HIST + "/records/SPLADE_POOLING_FINDING.json"},
        "KNOWN_LIMITATIONS": [
            "SPLADE vectors of the five text datasets were pooled over padding positions (batch-dependent); WebQSP's are attention-masked. "
            "Ranking within a dataset is consistent; raw SPLADE scores must not be compared across datasets. Published, not repaired: "
            "re-encoding 11.4M documents would invalidate every experiment already run (SPLADE_POOLING_FINDING).",
            "Three official TEST splits are blind (no gold): musique 2,459, hotpotqa 7,405, 2wiki 12,576. Evaluate on the split named in "
            "EVAL_SPLITS; the standing rule bars test results.",
            "Structural families are directed as found in the source and never symmetrised. Every NER and kNN family is stored as one "
            "row per unordered pair and is flagged directed:false -- the frozen Phase-C manifests had declared them directed:true as a "
            "blanket convention, and a reader honouring that saw only one endpoint's adjacency; the storage census in each manifest "
            "entry (0 reverse rows, 0 self loops, distinct pairs == rows) is the evidence for the flag. canonical.Graph exposes both "
            "endpoints of an undirected edge; nothing was added to the npz files.",
            "musique position 74545 ('Adei Ad') has no structural and no NER edges: Phase-C deduplicated it into position 8381 "
            "('Etz Efraim', byte-identical boilerplate) so those derivations never ran under its own id, and edges were not copied "
            "from the twin. It does have kNN edges (3): the audited kNN rebuild of 2026-09-11 searched the resolved vectors of all "
            "117,534 positions. (The first 2026-09-12 freeze said 'no edges in any family'; that was wrong for knn -- corrected here.)",
            "WebQSP is the RoG WebQSP+CWQ subgraph (2,592,894 nodes), NAME_ONLY entity text, never 'full Freebase'. 801,361 positions "
            "share their rendered text with another position and therefore hold identical vectors. Gold coverage is 62.42% of references / "
            "97.74% of answerable queries; 106 queries have gold absent from the corpus and 37 are the benchmark's own unanswerable ones.",
            "hotpotqa 94 and 2wiki 87,763 nodes had empty rev1 text; their REV2 re-encodings are the served rows (audits in the dataset dirs).",
            "The retrieval caches store float16 scores (lossy) and break score ties with the builder's key; verify a cache with that key, "
            "not with a bare argpartition (DENSE_CACHE_ORDER_DIAGNOSIS).",
            "Experiments run before 2026-09-11 used the corrupted dense rows and are not comparable to post-freeze numbers without a rerun.",
            "The query lanes are subsets with different populations from EVAL_SPLITS: LEGACY_CONTINUITY (this repo's legacy 2,000-query "
            "eval subsets) is drawn from the TRAIN split on squad, musique and 2wiki, is 1,842 train + 158 validation on hotpotqa, and on "
            "metaqa is 1,998 dev questions whose legacy gold sets differ from the canonical ones on 46 rows; metaqa's QUALITY_LOCKED lane is "
            "the official TEST split (39,093) and is not an evaluation population under the standing rule (metaqa evaluates on dev). A "
            "continuity claim against legacy numbers therefore changes the query population, not only the substrate (QUERY_LANES).",
            "The hotpotqa builder revision recorded in its build_info.json (build_kb.py a46d745a...) no longer exists anywhere: the 2wiki "
            "build of 2026-09-05 overwrote the file in place before any copy was kept. The hotpotqa tree is verified by its own records "
            "(nodes_jsonl_sha256, integrity_report, query_independence_test, the REV2 audits) but is not reproducible from an archived "
            "builder; every other builder revision named by a build_info.json matches the copy under _history/builders/ (BUILDERS).",
        ] + ([
            "freebase/ has no embeddings and no retrieval cache: encoding 301,977,131 nodes is infeasible on this machine "
            "(FREEBASE_V3_CLOSURE_STATUS: universe encodings INFEASIBLE (CPU)); nothing was fabricated in their place.",
            "freebase/ names: only name_kind 0-2 (ORIGINAL / RECOVERED_ORIGINAL / RECOVERED_EXTERNAL) are Freebase labels. URI_SELF and "
            "KEY_SEMANTIC are the node's own identifier made readable; INFERRED (structural inference, V3 contract) and GENERATED_FLOOR "
            "(evidence-based descriptions of the 7.2M residue, MID embedded) are derived strings that must never be presented as "
            "authentic labels. The three nameless grades of the V3 audit are kept per node (nameless_grade).",
            "freebase/ position(id) needs PYTHONHASHSEED=0 (the V3 uid rule); bridge/schema_index.parquet and bridge/webqsp_positions.npy "
            "resolve schema ids and webqsp nodes without it.",
        ] if fb_rec else []),
        "RESOLVED_SINCE_THE_EARLIER_FREEZES": [
            "129,819 zeroed dense rows in nine served stores: healed in place from the re-encoded repair vectors (the other 12,814 rows of the 142,633-row repair patch targeted the Phase-C webqsp docs store, which no pointer served -- SKIPPED_ORPHAN, retired with the patch); every served dense channel scans clean (0 damaged rows).",
            "REV2 re-encodings (2wiki 87,764 dense / 87,763 splade; hotpotqa 1,688 / 94) folded into the nodes' own rows; the patch stores are gone.",
            "The stale frozen musique kNN was replaced by the audited rebuild.",
            "WebQSP carries NER and kNN families and a retrieval cache like the other five; shards are 40,000 rows like the other five.",
            "The pointer index, the reuse maps, the second root (data/canonical), graph2/, the hop-tree query stores and every superseded copy are gone.",
            "The directed flag of the Phase-C NER/kNN families (metaqa knn; squad ner+knn; musique ner; hotpotqa ner; 2wiki ner) corrected "
            "from the blanket directed:true to directed:false on the evidence of a row census (MATERIALIZE_LOG phase 'direction'); the "
            "edge files are byte-identical to the frozen ones.",
        ],
        "LINEAGE": {"superseded_records": lin, "consolidation_and_materialisation": logs,
                    "v1_resolver_note": "the LOCKED_5_OF_5-pinned pointer_resolver.py (10,298 B) was overwritten before it was archived; "
                                        "_history/pre_v3/pointer_resolver_v1.py is a reconstruction whose bytecode, constants and line "
                                        "tables match the original .pyc kept beside it; the byte pin itself is not reproducible.",
                    "record_moves_journal": HIST + "/logs/RECORD_MOVES.jsonl"},
        "HISTORY": {"index": finfo(HIST + "/INDEX.json"), "files": hist["files"], "bytes": hist["bytes"], "not_indexed": HISTORY_NOT_INDEXED},
        "BUILDERS": ({"index": finfo(HIST + "/builders/BUILDERS.json"), "files": builders["files"], "bytes": builders["bytes"],
                      "build_info_revisions": builders["build_info_revisions"], "what": builders["what"]} if builders else None),
        "GOVERNANCE": {f: finfo(FC + "/" + f) for f in GOVERNANCE if os.path.exists(FC + "/" + f)},
        "LOADER": {"canonical.py": finfo(FC + "/canonical.py"), "pointer_resolver.py": finfo(FC + "/pointer_resolver.py")},
        "HASHING_CONVENTION": {
            "what_is_hashed": "the record re-serialised with json.dumps(indent=1) and its own RECORD_SHA256 / RECORD_SHA256_LF fields removed",
            "RECORD_SHA256": "CRLF newlines (the convention every earlier LOCKED_* record used)",
            "RECORD_SHA256_LF": "the same content with LF newlines; verify with this one where line endings may have been normalised",
            "verify": "python src/dataset_canonical/verify_canonical.py --full",
        },
        "seconds": round(time.time() - t0, 1),
    }
    write_record(OUT, rec)
    say("wrote %s  RECORD_SHA256 %s" % (OUT, rec["RECORD_SHA256"]))
    say("totals: %s" % json.dumps(totals))


def archive_previous_freeze():
    """Supersession, never an edit: the standing CANONICAL_FREEZE.json moves under _history/records/ with its bytes."""
    if not os.path.exists(OUT):
        return None
    prev = rj(OUT)
    tag = "%s_%s" % (prev.get("frozen_utc", "undated").replace(":", "").replace("-", ""), prev.get("RECORD_SHA256", "")[:8])
    dest = HIST + "/records/CANONICAL_FREEZE_%s.json" % tag
    mv(OUT, dest)
    return {"archived_to": dest, "RECORD_SHA256": prev.get("RECORD_SHA256"), "frozen_utc": prev.get("frozen_utc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--allow-old-tree", action="store_true")
    ap.add_argument("--reuse-dataset-records", action="store_true",
                    help="do not rewrite the six DATASET.json files; load them and re-freeze (freebase picked up if present)")
    a = ap.parse_args()
    if a.dataset:
        dataset_record(a.dataset)
        return
    if os.path.isdir("data/canonical") and not a.allow_old_tree:
        raise SystemExit("data/canonical still exists: run materialize_canonical.py purge first")
    say("tidying the root")
    tidy_root()
    recs = {}
    for ds in DATASETS:
        recs[ds] = rj(FC + "/%s/DATASET.json" % ds) if a.reuse_dataset_records else dataset_record(ds)
    arch = archive_previous_freeze()
    if arch:
        say("archived the previous freeze: %s" % json.dumps(arch))
    freeze(recs)


if __name__ == "__main__":
    main()
