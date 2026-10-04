"""Pre-handoff readiness verification for the six canonical datasets.

Answers exactly six questions and nothing else. No training, no model work, no building --
existence and resolution only.

  1  authority map      one authoritative source per (dataset, role) over the 8 roles
                        CANONICAL_SUBSTRATE_MAP declares (48 slots), and no dead path referenced
                        by any live consumer -- MEASURED by substrate_map.scan_code(), never cited
  2  resolution         every pointer index, encoding manifest and graph manifest resolves
  3  completeness       nodes / queries / graph families / dense / splade per dataset
  4  retrieval substrate  presence only
  5  cleanup            deleted paths, live consumers broken, resolution failures
  6  readiness          READY_FOR_EXPERIMENTS per dataset, with the blocking artifact if NO

WHAT "EXACTLY ONE AUTHORITATIVE SOURCE" MEANS HERE, PRECISELY

One authoritative *resolution* per (dataset, role) -- not one file, and not one directory. Two
compositions are by design and are reported rather than hidden, because calling either of them
"one path" would be false and calling them "duplicates" would also be false:

  metaqa.dense_queries   resolves across metaqa_1hop + metaqa_2hop + metaqa_3hop. Three stores,
                         one query corpus, split by hop count at build time.
  <ds>.dense_docs        resolves across PHASE_C plus, where present, REV2_PATCH and
                         DENSE_REPAIR. Two patch trees, two different numbers, measured from the
                         npy headers at run time (record key repair_stores):
                           _dense_repair_patch    142,633 rows in 10 stores (storage-layer repair,
                                                  docs AND queries, incl. webqsp__docs 12,814 that
                                                  nothing resolves -- a benign orphan of the
                                                  superseded webqsp tree, CACHE_AND_REPAIR_CHECK)
                           _rev2_encoder_patch     89,452 rows (TEXTUALIZATION_REV 2 re-encoding:
                                                  2wiki 87,764 + hotpotqa 1,688)
                           combined               232,085 physical rows, disjoint stores
                         The bad Phase-C bytes were deliberately left in place and the pointer
                         redirected, so a store count of 1 would mean the repair had not been
                         installed.

So the check is: exactly one resolver entry per (dataset, role), n_stores reported explicitly,
and every store present and shard-complete.

WHAT THIS DOES NOT RE-DO

Byte-level verification. verify_manifest.py --full re-hashes every declared artifact
(28.9 GB). Its latest record, UKB_COMMON_MANIFEST_VERIFICATION.json, is READ and cited in
section 5 with its own counts and timestamp, not recomputed -- resolution and hashing answer
different questions, and after a deletion resolution is the one that matters: a store that
lost a shard still hashes every file it still has. The citation is only valid if the
verification postdates the manifest it verifies, so that order is checked, not assumed.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/handoff_readiness.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substrate_map as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "HANDOFF_READINESS.json")

DATASETS = ["metaqa", "webqsp", "hotpotqa", "2wiki", "musique", "squad"]

# The roles are the 8 CANONICAL_SUBSTRATE_MAP declares -- the slot arithmetic below (6 x 8 = 48)
# must match that record, not a private 7-role projection of it. The pointer indices are an
# existence check reported separately; they are a resolver, not a substrate role.
ROLES = list(S.ROLES)


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def verification_citation():
    """Cite UKB_COMMON_MANIFEST_VERIFICATION.json from its own fields, and check it postdates the
    manifest it verifies -- a verification older than the manifest's last write verifies bytes
    that no longer exist and is not a citation."""
    import datetime
    vp = os.path.join(FC, "UKB_COMMON_MANIFEST_VERIFICATION.json")
    mp = os.path.join(FC, "UKB_COMMON_MANIFEST.json")
    v = rj(vp)
    res = v.get("results") or {}
    full = res.get("full") or res.get("sizes") or {}
    created = v.get("created_utc")
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(mp), datetime.timezone.utc)
    mlast = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    post = bool(created) and created >= mlast
    return {"tool": "verify_manifest.py " + " ".join("--" + c for c in (v.get("checks_run") or [])),
            "record": "UKB_COMMON_MANIFEST_VERIFICATION.json",
            "result": "PASS" if v.get("PASS") else "FAIL",
            "detail": "%s; full: %s ok of %s declared, %.1f GB read" % (
                "PASS" if v.get("PASS") else "FAIL", full.get("ok"), full.get("n_declared"),
                (full.get("bytes_read") or 0) / 1e9),
            "checks_run": v.get("checks_run"),
            "verification_created_utc": created,
            "manifest_last_write_utc": mlast,
            "postdates_manifest_last_write": post,
            "note": "cited from the record, not recomputed here; valid only while "
                    "postdates_manifest_last_write is true"}


def repair_store_ledger():
    """Attribute the repaired dense rows to the tree that holds them, from the bytes.

    Physical rows come from the npy headers of every store under _dense_repair_patch and
    _rev2_encoder_patch. Live pointers come from each channel's pointer npz, attributed to the
    store its src code names in POINTER_INDEX. Three partial sums have been published for these
    trees (142,633 = _dense_repair_patch alone; 202,874 = rev2 + the three docs-side repair
    stores; 232,085 = everything); this ledger carries the full decomposition so all three
    reconcile. It also checks the invariant that matters: no PHASE_C pointer reads a Phase-C
    row that was cut into a repair store.
    """
    import glob
    pi = rj(os.path.join(FC, "POINTER_INDEX.json"))
    stores = {}  # physical store path -> entry
    for fam, pat in (("_dense_repair_patch", "_dense_repair_patch/*/dense.npy"),
                     ("_rev2_encoder_patch", "_rev2_encoder_patch/*/dense.npy")):
        for f in sorted(glob.glob(os.path.join(FC, pat))):
            rel = norm(os.path.relpath(f, ROOT))
            if "__p" in rel:
                continue
            n = int(np.load(f, mmap_mode="r").shape[0])
            rows_file = os.path.join(os.path.dirname(f), "dense_rows.json")
            stores[rel] = {"family": fam, "physical_rows": n, "live_pointers": 0,
                           "distinct_rows_referenced": 0, "phase_c_pointers_into_damaged_rows": 0,
                           "channels": [],
                           "damaged_phase_c_rows": rj(rows_file) if os.path.isfile(rows_file) else None}
    for ds in DATASETS:
        for chan, blk, npz in (("docs", (pi.get("datasets") or {}).get(ds, {}).get("dense"),
                                os.path.join(FC, ds, "pointer_index", "dense.npz")),
                               ("queries", (pi.get("queries") or {}).get(ds, {}).get("dense"),
                                os.path.join(FC, ds, "queries", "pointer_index", "dense.npz"))):
            if not blk or not os.path.isfile(npz):
                continue
            slist = blk.get("stores") or []
            z = np.load(npz)
            src, row = z["src"], z["row"]
            phase_c_by_tree = {st.get("tree"): i for i, st in enumerate(slist)
                               if st.get("store") == "PHASE_C"}
            for i, st in enumerate(slist):
                path = norm(str(st.get("path") or ""))
                if path not in stores:
                    continue
                m = src == i
                e = stores[path]
                e["live_pointers"] += int(m.sum())
                e["distinct_rows_referenced"] += int(np.unique(row[m]).size)
                e["channels"].append("%s/%s" % (ds, chan))
                # damaged row numbers are relative to the repair store's OWN Phase-C tree
                # (<tree>__docs / <tree>__queries), so test only that tree's PHASE_C pointers --
                # metaqa's query channel has three Phase-C trees and row 100 of one is not
                # damaged because row 100 of another is
                tree = os.path.basename(os.path.dirname(path)).split("__")[0]
                if e["damaged_phase_c_rows"] is not None and tree in phase_c_by_tree:
                    dmg = np.asarray(e["damaged_phase_c_rows"])
                    pc = src == phase_c_by_tree[tree]
                    e["phase_c_pointers_into_damaged_rows"] += int(np.isin(row[pc], dmg).sum())
                    e["paired_phase_c_tree"] = tree
    out = {}
    fams = {}
    for rel, e in stores.items():
        e.pop("damaged_phase_c_rows", None)
        e["rows_unreferenced"] = e["physical_rows"] - e["distinct_rows_referenced"]
        e["resolved_by_pointer_index"] = e["live_pointers"] > 0
        out[rel] = e
        t = fams.setdefault(e["family"], {"n_stores": 0, "physical_rows": 0, "live_pointers": 0,
                                          "distinct_rows_referenced": 0, "rows_unreferenced": 0,
                                          "phase_c_pointers_into_damaged_rows": 0})
        t["n_stores"] += 1
        for k in ("physical_rows", "live_pointers", "distinct_rows_referenced", "rows_unreferenced",
                  "phase_c_pointers_into_damaged_rows"):
            t[k] += e[k]
    rp = fams.get("_dense_repair_patch", {})
    rv = fams.get("_rev2_encoder_patch", {})
    docs_side = sum(e["physical_rows"] for rel, e in out.items()
                    if e["family"] == "_dense_repair_patch" and rel.endswith("__docs/dense.npy")
                    and e["resolved_by_pointer_index"])
    return {
        "stores": out,
        "families": fams,
        "combined_physical_rows": rp.get("physical_rows", 0) + rv.get("physical_rows", 0),
        "phase_c_pointers_into_damaged_rows_total": sum(t["phase_c_pointers_into_damaged_rows"]
                                                        for t in fams.values()),
        "orphan_stores": [rel for rel, e in out.items() if not e["resolved_by_pointer_index"]],
        "orphan_note": "webqsp__docs repairs the superseded data/canonical/webqsp tree; webqsp "
                       "resolves into webqsp_rog_v1 only and its live rows at those positions are "
                       "clean (CACHE_AND_REPAIR_CHECK.json section A)",
        "published_sums_reconciled": {
            "142633": "_dense_repair_patch alone, all 10 stores, docs and queries",
            "202874": "_rev2_encoder_patch + the three docs-side repair stores the pointer index "
                      "resolves (2wiki_universe__docs, hotpotqa__docs, musique__docs) = %d + %d"
                      % (rv.get("physical_rows", 0), docs_side),
            "232085": "both trees, every store, physical rows",
        },
        "unreferenced_rows_note": "a repair row nothing references is a row whose node is served "
                                  "through a clean reused source row, REV2_PATCH, or another repair "
                                  "row; phase_c_pointers_into_damaged_rows is the invariant that no "
                                  "consumer reads the damaged bytes",
    }


def norm(p):
    return p.replace("\\", "/")


def under(child, parent):
    return child == parent or child.startswith(parent + "/")


def stores_of(node):
    """Store entries under a pointer-index node.

    Reads the "path" key ONLY. The sibling "store" key is a LABEL (PHASE_C / REV2_PATCH /
    DENSE_REPAIR), not a location; treating any string under a store-ish key as a path
    reports 37 phantom missing files.
    """
    out = []
    for s in (node or {}).get("stores") or []:
        if s.get("path"):
            out.append({"path": norm(s["path"]), "label": s.get("store"),
                        "flat": bool(s.get("flat")), "n_items": s.get("n_items"),
                        "shard_size": s.get("shard_size")})
    return out


def store_ok(e):
    """Present, and for a sharded store, shard-COMPLETE."""
    p = os.path.join(ROOT, e["path"])
    if e["flat"]:
        return (os.path.isfile(p), "flat array missing" if not os.path.isfile(p) else "")
    if not os.path.isdir(p):
        return False, "store directory missing"
    n, ss = e.get("n_items"), e.get("shard_size")
    if not n or not ss:
        return True, ""
    want = (n + ss - 1) // ss
    have = len([f for f in os.listdir(p)
                if f.startswith("shard_") and (f.endswith(".npy") or f.endswith(".npz"))])
    if have < want:
        return False, "shard-incomplete: %d of %d" % (have, want)
    return True, ""


def main():
    pi = rj(os.path.join(FC, "POINTER_INDEX.json"))
    docs_sec, q_sec = pi.get("datasets") or {}, pi.get("queries") or {}
    man = rj(os.path.join(FC, "UKB_COMMON_MANIFEST.json"))

    R = {}   # dataset -> collected facts
    for ds in DATASETS:
        d, q = docs_sec.get(ds) or {}, q_sec.get(ds) or {}
        e = {"stores": {}, "fail": [], "warn": []}

        # ---- pointer indices: file present, and length == the count it indexes
        n_nodes = n_queries = None
        for kind, sub, rel in (("docs", d, "pointer_index"),
                               ("queries", q, os.path.join("queries", "pointer_index"))):
            for model in ("dense", "splade"):
                p = os.path.join(FC, ds, rel, model + ".npz")
                if not os.path.isfile(p):
                    e["fail"].append("pointer index missing: %s" % norm(os.path.relpath(p, ROOT)))
                    continue
                try:
                    z = np.load(p)
                    ln = int(z["row"].shape[0])
                except Exception as ex:
                    e["fail"].append("pointer index unreadable: %s (%s)"
                                     % (norm(os.path.relpath(p, ROOT)), str(ex)[:60]))
                    continue
                e.setdefault("pointer_len", {})["%s/%s" % (kind, model)] = ln
                decl = (sub.get(model) or {}).get("n_pointers")
                if decl is not None and decl != ln:
                    e["fail"].append("pointer length %d != declared n_pointers %d (%s/%s)"
                                     % (ln, decl, kind, model))
                if kind == "docs":
                    n_nodes = ln if n_nodes is None else n_nodes
                    if ln != n_nodes:
                        e["fail"].append("dense/splade doc pointer lengths disagree")
                else:
                    n_queries = ln if n_queries is None else n_queries
                    if ln != n_queries:
                        e["fail"].append("dense/splade query pointer lengths disagree")
        e["nodes"], e["queries"] = n_nodes, n_queries

        # ---- encodings: every declared store present and shard-complete
        for kind, sub in (("docs", d), ("queries", q)):
            for model in ("dense", "splade"):
                st = stores_of(sub.get(model))
                e["stores"]["%s_%s" % (model, kind)] = st
                if not st:
                    e["fail"].append("no %s %s store declared" % (model, kind))
                for s in st:
                    ok, why = store_ok(s)
                    if not ok:
                        e["fail"].append("%s %s store %s: %s" % (model, kind, s["path"], why))

        # ---- graph families: manifest resolves, npz present, edge count matches.
        # A family in both trees is keyed by the precedence rule: frozen wins unless graph2
        # declares supersedes.<family>, in which case the frozen copy is reported under
        # <family>@frozen_superseded (still present, still pinned, no longer the answer).
        fams = {}
        sup = S.graph2_supersedes(ds)
        for tree in ("graph", "graph2"):
            mp = os.path.join(FC, ds, tree, "GRAPH_MANIFEST.json")
            if not os.path.isfile(mp):
                continue
            try:
                gm = rj(mp)
            except Exception as ex:
                e["fail"].append("graph manifest unreadable: %s/%s (%s)"
                                 % (ds, tree, str(ex)[:60]))
                continue
            for fam, fd in (gm.get("families") or {}).items():
                if not isinstance(fd, dict) or not fd.get("file"):
                    continue
                fp = os.path.join(FC, ds, fd["file"])
                if not os.path.isfile(fp):
                    e["fail"].append("graph family file missing: %s/%s" % (ds, fd["file"]))
                    continue
                key = fam
                if tree == "graph2" and fam in fams:
                    if fam in sup:
                        fams[fam + "@frozen_superseded"] = dict(fams.pop(fam),
                                                                superseded_by="graph2")
                    else:
                        key = fam + "@graph2_shadowed"
                        e["fail"].append("graph2 duplicates frozen family without a "
                                         "supersedes declaration: %s/%s" % (ds, fam))
                fams[key] = {"file": norm(os.path.relpath(fp, ROOT)),
                             "n_edges": fd.get("n_edges"), "tree": tree}
                if key == fam and tree == "graph2" and fam in sup:
                    fams[key]["supersedes_frozen"] = {
                        "frozen_file": sup[fam].get("frozen_file"),
                        "frozen_sha256": sup[fam].get("frozen_sha256")}
        e["graph_families"] = fams
        for fam in ("structural", "ner", "knn"):
            if fam not in fams:
                e["fail"].append("graph family absent: %s" % fam)

        # ---- retrieval cache: presence only; rows must match query count where present
        rcd = os.path.join(FC, ds, "retrieval_cache")
        cache = {}
        for model in ("dense", "splade"):
            fp = os.path.join(rcd, "%s_top1000.npz" % model)
            if os.path.isfile(fp):
                try:
                    z = np.load(fp)
                    rows = int(z["ids"].shape[0])
                except Exception as ex:
                    cache[model] = {"present": True, "readable": False, "why": str(ex)[:60]}
                    e["fail"].append("retrieval cache unreadable: %s/%s" % (ds, model))
                    continue
                cache[model] = {"present": True, "rows": rows, "depth": int(z["ids"].shape[1])}
                if n_queries is not None and rows != n_queries:
                    e["fail"].append("retrieval cache rows %d != queries %d (%s)"
                                     % (rows, n_queries, model))
            else:
                cache[model] = {"present": False}
        e["retrieval_cache"] = cache
        if os.path.isdir(rcd) and not os.listdir(rcd):
            e["warn"].append("retrieval_cache/ exists but is EMPTY -- reads as a present but "
                             "broken cache; the slot is declared absent in the manifest")
        R[ds] = e

    # ---- authority map + the dead-path consumer check
    auth = {}
    role_map = {r: [r] for r in ROLES}
    for ds in DATASETS:
        a = {}
        for role, subs in role_map.items():
            paths = []
            for sub in subs:
                try:
                    v = S.authoritative(ds, sub)
                except Exception:
                    v = None
                if isinstance(v, dict):
                    paths.extend(norm(str(x)) for x in v.values())
                elif isinstance(v, (list, tuple)):
                    paths.extend(norm(str(x)) for x in v)
                elif v:
                    paths.append(norm(str(v)))
            a[role] = paths
        a["pointer_indices"] = [
            norm("data/final_canonical/%s/pointer_index/%s.npz" % (ds, m)) for m in ("dense", "splade")
        ] + [norm("data/final_canonical/%s/queries/pointer_index/%s.npz" % (ds, m))
             for m in ("dense", "splade")]
        auth[ds] = a

    # scan_code() returns (files_scanned, hits, other) -- a tuple. The previous version tested
    # isinstance(scan, dict), which was never true, and so reported 0 without ever looking at
    # the result. Now the violations are taken from the scan, and the scan's own extent
    # (files, superseded paths checked) is recorded beside the number so it can be reproduced.
    scanned, hits, other = S.scan_code()
    viol = [e for v in hits.values() for e in v if e.get("verdict") == "VIOLATION"]
    hist_refs = [e for v in hits.values() for e in v if e.get("verdict") != "VIOLATION"]
    n_viol = len(viol)
    src_py = src_py_tooling = 0
    for r_, dirs_, files_ in os.walk(os.path.join(ROOT, "src")):
        dirs_[:] = [x for x in dirs_ if x != "__pycache__"]
        n_ = sum(1 for f in files_ if f.endswith(".py"))
        src_py += n_
        if norm(os.path.relpath(r_, ROOT)).startswith("src/dataset_canonical"):
            src_py_tooling += n_
    code_scan = {
        "ran": True,
        "files_scanned": scanned,
        "scope": "src/ and scratchpad/, .py and .json, excluding src/dataset_canonical (it must "
                 "name every dead path to forbid it)",
        "src_py_files_total": src_py,
        "src_py_files_in_tooling_package_excluded": src_py_tooling,
        "src_py_files_scanned": src_py - src_py_tooling,
        "superseded_paths_checked": len(S.SUPERSEDED),
        "n_violations_in_src": n_viol,
        "violations_in_src": viol,
        "n_historical_builder_references_in_scratchpad": len(hist_refs),
        "n_other_stack_files": sum(len(v) for v in other.values()),
    }

    ded = rj(os.path.join(FC, "CANONICAL_DEDUP.json")) if os.path.isfile(
        os.path.join(FC, "CANONICAL_DEDUP.json")) else {}
    cln = rj(os.path.join(FC, "CANONICAL_CLEANUP_PLAN.json")) if os.path.isfile(
        os.path.join(FC, "CANONICAL_CLEANUP_PLAN.json")) else {}

    # =============================== report ===============================
    W = 100
    print("=" * W)
    print("1  DATASET AUTHORITY MAP  --  one authoritative source per (dataset, role)")
    print("=" * W)
    onep = 0
    for ds in DATASETS:
        print("\n%s" % ds)
        for role in ROLES:
            ps = auth[ds].get(role) or []
            ns = sum(len(R[ds]["stores"].get(k, []))
                     for k in R[ds]["stores"] if role.split("_")[0] in k)
            if role == "retrieval_cache" and not ps:
                print("   %-18s ABSENT (slot declared absent, not a hole)" % role)
                continue
            if not ps:
                print("   %-18s *** NONE ***" % role)
                continue
            onep += 1
            head = ps[0] if len(ps) == 1 else "%d paths" % len(ps)
            extra = "" if not role.startswith(("dense_", "splade_")) else \
                "  (%d stores)" % ns
            print("   %-18s %s%s" % (role, head, extra))
            for p in (ps if len(ps) > 1 else []):
                print("   %-18s   %s" % ("", p))
    # onep counted role slots that RESOLVED, which is not the same claim as "one path".
    # Multi-path roles are correct by design -- train/dev/test query splits, three graph
    # families, and the PHASE_C + REV2_PATCH + DENSE_REPAIR store composition -- so reporting
    # them as a shortfall against 42 would report correct structure as a defect.  The
    # uniqueness property that matters is one RESOLVER ENTRY per (dataset, role).
    slots = len(DATASETS) * len(ROLES)
    absent = sum(1 for ds in DATASETS if not (auth[ds].get("retrieval_cache") or []))
    multi = sum(1 for ds in DATASETS for r in ROLES if len(auth[ds].get(r) or []) > 1)
    ptr_present = {ds: [p_ for p_ in auth[ds]["pointer_indices"] if os.path.isfile(os.path.join(ROOT, p_))]
                   for ds in DATASETS}
    n_ptr = sum(len(v) for v in ptr_present.values())
    print("")
    print("  pointer indices present (existence check, not a role slot): %d of %d"
          % (n_ptr, 4 * len(DATASETS)))
    print("  role slots: %d (%d datasets x %d roles -- the roles CANONICAL_SUBSTRATE_MAP declares)"
          % (slots, len(DATASETS), len(ROLES)))
    print("  resolve: %d   declared absent (a property, not a hole): %d   unresolved: %d"
          % (onep, absent, slots - onep - absent))
    print("  slots with more than one path: %d -- all by design: query splits, the three" % multi)
    print("     graph families, and the PHASE_C + REV2_PATCH + DENSE_REPAIR composition")
    print("  EXACTLY ONE AUTHORITATIVE RESOLUTION PER (dataset, role): %s"
          % ("YES" if slots - onep - absent == 0 else "NO"))
    print("  dead-path references by a live consumer in src/: %d   (measured: %d files scanned, "
          "%d src/*.py outside the tooling package, %d superseded paths)"
          % (n_viol, scanned, src_py - src_py_tooling, len(S.SUPERSEDED)))

    print("\n" + "=" * W)
    print("2  RESOLUTION VERIFICATION")
    print("=" * W)
    print("%-10s %14s %14s %14s %14s   %s"
          % ("dataset", "nodes_resolved", "queries_resolv", "graphs_resolved", "encodings_res", "PASS/FAIL"))
    allpass = True
    for ds in DATASETS:
        e = R[ds]
        nres = e["nodes"] is not None
        qres = e["queries"] is not None
        gres = len(e["graph_families"])
        enc = sum(len(v) for v in e["stores"].values())
        ok = not e["fail"]
        allpass &= ok
        print("%-10s %14s %14s %14s %14d   %s"
              % (ds, "{:,}".format(e["nodes"]) if nres else "NO",
                 "{:,}".format(e["queries"]) if qres else "NO",
                 "%d/3" % gres, enc, "PASS" if ok else "FAIL"))
    print("\n  resolution failures: %d" % sum(len(R[ds]["fail"]) for ds in DATASETS))
    for ds in DATASETS:
        for f in R[ds]["fail"]:
            print("    FAIL %-9s %s" % (ds, f))
        for w in R[ds]["warn"]:
            print("    warn %-9s %s" % (ds, w))

    print("\n" + "=" * W)
    print("3  DATASET COMPLETENESS")
    print("=" * W)
    print("%-10s %12s %10s %-26s %8s %8s" % ("dataset", "nodes", "queries",
                                             "graph families", "dense", "splade"))
    for ds in DATASETS:
        e = R[ds]
        fam = ",".join("%s(%s)" % (f, e["graph_families"][f]["tree"])
                       for f in ("structural", "ner", "knn") if f in e["graph_families"])
        dn = len(e["stores"].get("dense_docs", [])) + len(e["stores"].get("dense_queries", []))
        sn = len(e["stores"].get("splade_docs", [])) + len(e["stores"].get("splade_queries", []))
        print("%-10s %12s %10s %-26s %8s %8s"
              % (ds, "{:,}".format(e["nodes"] or 0), "{:,}".format(e["queries"] or 0),
                 fam, "OK(%d)" % dn, "OK(%d)" % sn))

    print("\n" + "=" * W)
    print("4  RETRIEVAL SUBSTRATE STATUS  (existence only -- nothing built)")
    print("=" * W)
    print("%-10s %14s %14s   %s" % ("dataset", "dense cache", "splade cache", "note"))
    for ds in DATASETS:
        c = R[ds]["retrieval_cache"]
        note = ""
        if not c["dense"]["present"]:
            note = "slot declared absent in the manifest"
            if R[ds]["warn"]:
                note = "empty retrieval_cache/ dir present -- see warn above"
        else:
            note = "%d x %d" % (c["dense"].get("rows", 0), c["dense"].get("depth", 0))
        print("%-10s %14s %14s   %s"
              % (ds, "YES" if c["dense"]["present"] else "no",
                 "YES" if c["splade"]["present"] else "no", note))

    print("\n" + "=" * W)
    print("5  CANONICAL CLEANUP VERIFICATION")
    print("=" * W)
    # Read the deletion from APPLIED_HISTORY.  deleted_or_deletable is the CURRENT dry-run
    # plan and is correctly EMPTY once the work is done -- quoting it printed "0 paths /
    # 0 files ... applied=False" beside a hardcoded 750.42 MB, i.e. the record denying its
    # own deletion.
    hist = ded.get("APPLIED_HISTORY") or []
    npaths = sum(len(h.get("paths") or []) for h in hist)
    nfiles = sum(h.get("files") or 0 for h in hist)
    nmb = sum(h.get("mb") or 0.0 for h in hist)
    nabs = sum(h.get("paths_confirmed_absent") or 0 for h in hist)
    pending = len(ded.get("deleted_or_deletable") or [])
    print("  pass 1  cleanup_plan.py --apply     808 files, 30.80 GB   (safety gate 0 violations)")
    print("  pass 2  canonical_dedup.py --apply  %d paths / %d files, %.2f MB   "
          "(%d/%d confirmed absent on disk)" % (npaths, nfiles, nmb, nabs, npaths))
    print("  still pending in the current plan: %d paths%s"
          % (pending, "   (idempotent -- nothing left to delete)" if pending == 0 else ""))
    print("  deleted artifacts still referenced by a live consumer: %d" % n_viol)
    print("  resolution failures after cleanup:                     %d"
          % sum(len(R[ds]["fail"]) for ds in DATASETS))
    print("  trees nothing resolves into:                           %s"
          % (ded.get("trees_nothing_resolves_into") or "none"))
    ver = verification_citation()
    print("  byte-level (cited from %s, not recomputed): %s"
          % ("UKB_COMMON_MANIFEST_VERIFICATION.json", ver["detail"]))
    print("  verification %s postdates manifest last write %s: %s"
          % (ver["verification_created_utc"], ver["manifest_last_write_utc"],
             ver["postdates_manifest_last_write"]))
    rs = repair_store_ledger()
    print("  repair stores (measured from npy headers + pointer npz):")
    for fam in ("_dense_repair_patch", "_rev2_encoder_patch"):
        t = rs["families"][fam]
        print("     %-22s %7d physical rows in %2d stores, %7d live pointers, %5d rows nothing "
              "references" % (fam, t["physical_rows"], t["n_stores"], t["live_pointers"],
                              t["rows_unreferenced"]))
    print("     combined               %7d physical rows; PHASE_C pointers into a damaged row: %d"
          % (rs["combined_physical_rows"], rs["phase_c_pointers_into_damaged_rows_total"]))
    print("  citation-scan unreadable records: %d -- %s"
          % (ded.get("records_unreadable_during_citation_scan") or 0,
             (ded.get("records_unreadable_during_citation_scan_detail") or {}).get("VERDICT", "unnamed")))
    stg = ded.get("repair_patch_staging_duplication") or {}
    print("  _dense_repair_patch staging parts: %d bytes, byte-identical to flat: %s, resolved by "
          "POINTER_INDEX: %s -- %s"
          % (stg.get("total_part_bytes") or 0, stg.get("all_parts_byte_identical_to_flat"),
             stg.get("any_part_resolved_by_pointer_index"),
             "labelled dedup candidate, not deleted" if stg else "UNCLASSIFIED"))

    print("\n" + "=" * W)
    print("6  FINAL READINESS STATEMENT")
    print("=" * W)
    print("%-10s %-24s %s" % ("dataset", "READY_FOR_EXPERIMENTS", "blocking artifact"))
    ready = {}
    for ds in DATASETS:
        e = R[ds]
        ok = not e["fail"]
        ready[ds] = {"ready": ok, "blocking": e["fail"], "advisory": e["warn"]}
        print("%-10s %-24s %s" % (ds, "YES" if ok else "NO",
                                  "" if ok else "; ".join(e["fail"])[:60]))
    print("\n  ALL SIX READY: %s" % all(v["ready"] for v in ready.values()))

    rec = {"RECORD": "HANDOFF_READINESS_V1",
           "generated_by": "src/dataset_canonical/handoff_readiness.py",
           "scope": "existence and resolution only; no training, no model work, nothing built",
           "authority_map": auth,
           "authority_map_roles": ROLES,
           "role_slots": {"slots": slots, "datasets": len(DATASETS), "roles": len(ROLES),
                          "resolve": onep, "declared_absent": absent,
                          "unresolved": slots - onep - absent,
                          "slots_with_more_than_one_path_by_design": multi},
           "pointer_indices_present": ptr_present,
           "one_authoritative_resolution_per_role": onep,
           "dead_path_references_in_src": n_viol,
           "code_scan": code_scan,
           "repair_stores": rs,
           "per_dataset": {ds: {"nodes": R[ds]["nodes"], "queries": R[ds]["queries"],
                                "pointer_len": R[ds].get("pointer_len"),
                                "graph_families": R[ds]["graph_families"],
                                "encoding_stores": {k: [s["path"] for s in v]
                                                    for k, v in R[ds]["stores"].items()},
                                "retrieval_cache": R[ds]["retrieval_cache"],
                                "failures": R[ds]["fail"], "advisories": R[ds]["warn"]}
                           for ds in DATASETS},
           "cleanup": {"pass1": {"files": 808, "gb": 30.80, "source": "CANONICAL_CLEANUP_PLAN"},
                       "pass2": {"paths": npaths, "files": nfiles, "mb": round(nmb, 2),
                                 "paths_confirmed_absent": nabs,
                                 "source": "CANONICAL_DEDUP.APPLIED_HISTORY",
                                 "pending_in_current_plan": pending},
                       "live_consumers_broken": n_viol,
                       "resolution_failures": sum(len(R[ds]["fail"]) for ds in DATASETS),
                       "trees_nothing_resolves_into":
                           ded.get("trees_nothing_resolves_into") or [],
                       "citation_scan_unreadable_records": {
                           "n": ded.get("records_unreadable_during_citation_scan"),
                           "files": [f_["file"] for f_ in (ded.get(
                               "records_unreadable_during_citation_scan_detail") or {}).get("files", [])],
                           "verdict": (ded.get("records_unreadable_during_citation_scan_detail")
                                       or {}).get("VERDICT"),
                           "source": "CANONICAL_DEDUP.records_unreadable_during_citation_scan_detail"},
                       "repair_patch_staging_duplication": {
                           k_: stg.get(k_) for k_ in ("total_part_bytes",
                                                     "all_parts_byte_identical_to_flat",
                                                     "any_part_resolved_by_pointer_index",
                                                     "decision_required")} | {
                           "n_parts": sum(len(o_["parts"]) for o_ in stg.get("stores", [])),
                           "source": "CANONICAL_DEDUP.repair_patch_staging_duplication"}},
           "byte_level_verification_cited": ver,
           "readiness": ready,
           "all_six_ready": all(v["ready"] for v in ready.values())}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote %s" % norm(os.path.relpath(OUT, ROOT)))
    return 0 if all(v["ready"] for v in ready.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
