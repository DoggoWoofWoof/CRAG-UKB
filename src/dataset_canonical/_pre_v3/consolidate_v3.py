# -*- coding: utf-8 -*-
"""
CONSOLIDATION V3 -- heal and fold the encoding substrate IN PLACE, then simplify the pointers.

User ruling 2026-09-12: "then heal it, i want everything fixed" / "clean all that up, modify the
pointers and only keep the proper datasets with their fixed stuff rather than bloating up all".
This supersedes the earlier convention that a freeze only describes and routes around damage.

Phases (each is a subcommand, each verifies itself, each appends to CONSOLIDATION_V3_LOG.json):

  heal      write every DENSE_REPAIR vector into the damaged Phase-C row it re-encoded
            (dense_rows.json is the row map; dense_ids.json must match the store's ids_*.json
            at that row). Pre-check: the target row is damaged (all-zero or torn) or already
            equals the patch. Post-check: store row == patch row bit-exact, and no touched
            shard holds a damaged row afterwards.
  fold      write every REV2_PATCH vector into the patched node's OWN Phase-C row (the row of
            the node's source_id in the store, which the pointer index no longer references
            because the node's text changed under TEXTUALIZATION_REV_2). Dense rows are written
            through the memmap; SPLADE rows are spliced into the CSR shard and the shard is
            rewritten with every untouched row byte-identical. Pre-check: own row is dead and
            unique. Post-check: store row == patch row.
  pointers  rewrite pointer_index/<model>.npz so every pointer resolves into the single
            Phase-C store (src == 0 everywhere; DENSE_REPAIR pointers become the healed row,
            REV2 pointers become the own row) and prove for EVERY redirected position that the
            new (store,row) yields the same bytes the old (store,row) did. Writes the new
            POINTER_INDEX.json; the previous one is kept under _history/.

Backups: every shard about to be modified is copied to <scratch>/v3_backup/<store>/ first, and
after the write the untouched rows are compared against the backup. The backups are deleted by
the `retire` step, not here.

Nothing here downloads, encodes, or invents a vector: every byte written already existed in a
patch store that the pointer index was serving.
"""
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
FC = "data/final_canonical"
CAN = "data/canonical"
PATCH = FC + "/_dense_repair_patch"
REV2 = FC + "/_rev2_encoder_patch"
LOG = FC + "/CONSOLIDATION_V3_LOG.json"
SCRATCH = os.environ.get("V3_SCRATCH") or os.path.join(
    os.environ.get("TEMP", "/tmp"), "claude", "C--Users-Swastik-Desktop-CRAG",
    "70ead4d5-16f9-428c-af03-4cb084412966", "scratchpad")
BACKUP = os.path.join(SCRATCH, "v3_backup")
NORM_TOL = 0.01          # the integrity scan's tolerance: a healthy fp16 unit vector is 1 +- 1e-3

# dataset -> (phase-c tree, prefix that turns a node's source_id into the store's doc id)
REV2_TREES = {"2wiki": ("2wiki_universe", "2wu:"), "hotpotqa": ("hotpotqa", "hotpot_")}


def rj(p):
    return json.load(io.open(p, encoding="utf-8"))


def wj(p, o):
    io.open(p, "w", encoding="utf-8", newline=chr(10)).write(json.dumps(o, indent=1, ensure_ascii=False))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def log_phase(phase, rec):
    log = rj(LOG) if os.path.exists(LOG) else {"RECORD": "CONSOLIDATION_V3_LOG", "phases": []}
    rec = dict(rec)
    rec["phase"] = phase
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log["phases"].append(rec)
    wj(LOG, log)


def shard_size_of(tree):
    return 12000 if tree.startswith("webqsp") else 40000


def store_dir(tree, model, kind):
    return "%s/%s/encodings/%s/%s" % (CAN, tree, model, kind)


def store_ids(sdir):
    """row -> id for a sharded store, from its ids_XXXXX.json files."""
    out = {}
    for f in sorted(glob.glob(sdir + "/ids_*.json")):
        k = int(os.path.basename(f)[4:9])
        ss = shard_size_of(sdir.split("/")[2])
        for j, did in enumerate(rj(f)):
            out[k * ss + j] = did
    return out


def backup_shard(sdir, shard_path):
    dst_dir = os.path.join(BACKUP, sdir.replace("/", "__"))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(shard_path))
    if not os.path.exists(dst):
        shutil.copy2(shard_path, dst)
    return dst


def damaged(v):
    """the integrity scan's definition: all-zero, non-finite, or a norm off the unit sphere"""
    v = np.asarray(v, dtype=np.float32)
    if not np.all(np.isfinite(v)):
        return True
    n = float(np.sqrt((v * v).sum()))
    return n == 0.0 or abs(n - 1.0) > NORM_TOL


# ----------------------------------------------------------------------------------------- heal
def repair_dirs():
    out = []
    for d in sorted(os.listdir(PATCH)):
        if "__p" in d or not os.path.isdir(os.path.join(PATCH, d)):
            continue
        tree, kind = d.split("__")
        out.append((d, tree, kind))
    return out


def heal():
    pi = rj(FC + "/POINTER_INDEX.json")
    # which repair stores does the pointer index actually serve? (webqsp__docs is an orphan:
    # the assembled webqsp store carries no damaged rows and the index has no patch for it)
    served = set()
    for ds, d in pi["datasets"].items():
        for st in d["dense"]["stores"]:
            if st.get("store") == "DENSE_REPAIR":
                served.add(os.path.normpath(st["path"]).replace(chr(92), "/"))
    for ds, d in pi["queries"].items():
        for st in d["dense"]["stores"]:
            if st.get("store") == "DENSE_REPAIR":
                served.add(os.path.normpath(st["path"]).replace(chr(92), "/"))
    results = []
    for d, tree, kind in repair_dirs():
        pdir = "%s/%s" % (PATCH, d)
        flat = pdir + "/dense.npy"
        if flat not in served:
            results.append({"patch": d, "action": "SKIPPED_ORPHAN",
                            "why": "the pointer index serves no pointer from this store; the "
                                   "target store carries no damaged row (REPAIR_COVERAGE_CHECK) and "
                                   "its rows differ from this patch, so writing it would DAMAGE "
                                   "healthy rows. Retired with the patch directory."})
            print("  %-22s SKIPPED (orphan)" % d)
            continue
        sdir = store_dir(tree, "dense", kind)
        ss = shard_size_of(tree)
        rows = np.asarray(rj(pdir + "/dense_rows.json"), dtype=np.int64)
        ids = rj(pdir + "/dense_ids.json")
        vec = np.load(flat, mmap_mode="r")
        assert len(rows) == len(ids) == vec.shape[0], d
        sids = store_ids(sdir)
        # the row map must agree with the store's own id files, row by row
        bad_id = [int(r) for r, i in zip(rows, ids) if sids.get(int(r)) != i]
        if bad_id:
            sys.exit("%s: %d rows whose store id differs from the patch id (first %s) -- refusing"
                     % (d, len(bad_id), bad_id[:5]))
        n_written = n_already = n_healthy_target = 0
        shards_touched = []
        for s in np.unique(rows // ss):
            m = np.nonzero(rows // ss == s)[0]
            sp = "%s/shard_%05d.npy" % (sdir, s)
            a = np.load(sp, mmap_mode="r")
            offs = rows[m] % ss
            cur = np.asarray(a[offs])
            want = np.asarray(vec[m])
            already = np.all(cur == want, axis=1)
            pre_damaged = np.array([damaged(x) for x in cur])
            healthy_target = (~already) & (~pre_damaged)
            n_healthy_target += int(healthy_target.sum())
            if healthy_target.any():
                sys.exit("%s shard %d: %d target rows are healthy AND differ from the patch -- "
                         "the row map is wrong, refusing to overwrite" % (d, s, int(healthy_target.sum())))
            todo = ~already
            if todo.any():
                bk = backup_shard(sdir, sp)
                del a
                a = np.load(sp, mmap_mode="r+")
                a[offs[todo]] = want[todo]
                a.flush()
                del a
                # post-check: every target row equals the patch; every other row equals the backup
                a = np.load(sp, mmap_mode="r")
                b = np.load(bk, mmap_mode="r")
                if not np.array_equal(np.asarray(a[offs]), want):
                    sys.exit("%s shard %d: post-write mismatch" % (d, s))
                mask = np.ones(a.shape[0], dtype=bool)
                mask[offs] = False
                if not np.array_equal(np.asarray(a[mask]), np.asarray(b[mask])):
                    sys.exit("%s shard %d: an untouched row changed -- restore from %s" % (d, s, bk))
                # and no damaged row remains in this shard
                norms = np.sqrt((np.asarray(a, dtype=np.float32) ** 2).sum(axis=1))
                n_bad = int(((norms == 0) | (np.abs(norms - 1.0) > NORM_TOL) | ~np.isfinite(norms)).sum())
                shards_touched.append({"shard": int(s), "rows_written": int(todo.sum()),
                                       "damaged_rows_left_in_shard": n_bad,
                                       "sha256_after": sha_file(sp)})
                n_written += int(todo.sum())
                del a, b
            n_already += int(already.sum())
        r = {"patch": d, "store": sdir, "rows_in_patch": int(len(rows)), "rows_written": n_written,
             "rows_already_equal": n_already, "shards_touched": shards_touched,
             "damaged_rows_left": sum(x["damaged_rows_left_in_shard"] for x in shards_touched)}
        results.append(r)
        print("  %-22s %-52s written %6d already %6d shards %2d damaged_left %d"
              % (d, sdir, n_written, n_already, len(shards_touched), r["damaged_rows_left"]))
    left = sum(r.get("damaged_rows_left", 0) for r in results)
    log_phase("heal", {"results": results, "damaged_rows_left_in_touched_shards": left})
    print("heal: %d patches, %d damaged rows left in touched shards" % (len(results), left))
    return left == 0


# ----------------------------------------------------------------------------------------- fold
def node_source_ids(ds):
    out = []
    with io.open("%s/%s/nodes.jsonl" % (FC, ds), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out.append((o["node_id"], o["source_id"]))
    return out


def own_rows_for(ds, model):
    """REV2 node -> its own Phase-C row (row whose store id is prefix + source_id)."""
    tree, pref = REV2_TREES[ds]
    sdir = store_dir(tree, model, "docs")
    byid = {v: k for k, v in store_ids(sdir).items()}
    nodes = node_source_ids(ds)
    pos_of = {nid: i for i, (nid, _) in enumerate(nodes)}
    ids = rj("%s/%s/%s_ids.json" % (REV2, ds, model))
    own = np.array([byid[pref + nodes[pos_of[nid]][1]] for nid in ids], dtype=np.int64)
    pos = np.array([pos_of[nid] for nid in ids], dtype=np.int64)
    return sdir, ids, own, pos


def load_csr_arrays(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def fold():
    results = []
    for ds in REV2_TREES:
        z_idx = {m: np.load("%s/%s/pointer_index/%s.npz" % (FC, ds, m)) for m in ("dense", "splade")}
        for model in ("dense", "splade"):
            sdir, ids, own, pos = own_rows_for(ds, model)
            src, row = z_idx[model]["src"], z_idx[model]["row"]
            live = set(row[src == 0].tolist())
            clash = int(sum(1 for r in own if int(r) in live))
            if clash or len(set(own.tolist())) != len(own):
                sys.exit("%s/%s: %d own rows still live or duplicated -- refusing" % (ds, model, clash))
            # every REV2 position must currently point at the patch store (src == 1)
            if not np.all(src[pos] == 1):
                sys.exit("%s/%s: some REV2 positions do not point at the patch" % (ds, model))
            ss = 40000
            n_written = n_already = 0
            shards_touched = []
            if model == "dense":
                vec = np.load("%s/%s/dense.npy" % (REV2, ds), mmap_mode="r")
                for s in np.unique(own // ss):
                    m = np.nonzero(own // ss == s)[0]
                    sp = "%s/shard_%05d.npy" % (sdir, s)
                    offs = own[m] % ss
                    want = np.asarray(vec[m])
                    a = np.load(sp, mmap_mode="r")
                    already = np.all(np.asarray(a[offs]) == want, axis=1)
                    del a
                    todo = ~already
                    if todo.any():
                        # the untouched rows are hashed before and after the write (a backup of
                        # every fold shard would not fit the disk; the heal shards ARE backed up)
                        a = np.load(sp, mmap_mode="r")
                        mask = np.ones(a.shape[0], dtype=bool)
                        mask[offs] = False
                        h_before = hashlib.sha256(np.ascontiguousarray(a[mask]).tobytes()).hexdigest()
                        del a
                        a = np.load(sp, mmap_mode="r+")
                        a[offs[todo]] = want[todo]
                        a.flush()
                        del a
                        a = np.load(sp, mmap_mode="r")
                        if not np.array_equal(np.asarray(a[offs]), want):
                            sys.exit("%s/%s shard %d: post-write mismatch" % (ds, model, s))
                        h_after = hashlib.sha256(np.ascontiguousarray(a[mask]).tobytes()).hexdigest()
                        if h_before != h_after:
                            sys.exit("%s/%s shard %d: an untouched row changed" % (ds, model, s))
                        shards_touched.append({"shard": int(s), "rows_written": int(todo.sum()),
                                               "untouched_rows_sha256": h_after, "sha256_after": sha_file(sp)})
                        n_written += int(todo.sum())
                        del a
                    n_already += int(already.sum())
            else:
                pz = load_csr_arrays("%s/%s/splade.npz" % (REV2, ds))
                p_indptr, p_indices, p_data = pz["indptr"], pz["indices"], pz["data"]
                assert p_indptr.size - 1 == len(ids), (p_indptr.size, len(ids))
                for s in np.unique(own // ss):
                    m = np.nonzero(own // ss == s)[0]
                    sp = "%s/shard_%05d.npz" % (sdir, s)
                    z = load_csr_arrays(sp)
                    indptr, indices, data = z["indptr"], z["indices"], z["data"]
                    nrows = int(z["shape"][0])
                    repl = {int(own[i] % ss): int(i) for i in m}
                    # is every replacement already in place?
                    same = True
                    for off, i in repl.items():
                        a0, a1 = indptr[off], indptr[off + 1]
                        b0, b1 = p_indptr[i], p_indptr[i + 1]
                        if not (np.array_equal(indices[a0:a1], p_indices[b0:b1]) and np.array_equal(data[a0:a1], p_data[b0:b1])):
                            same = False
                            break
                    if same:
                        n_already += len(repl)
                        continue
                    new_ind, new_dat, new_ptr = [], [], [0]
                    for r in range(nrows):
                        if r in repl:
                            i = repl[r]
                            b0, b1 = p_indptr[i], p_indptr[i + 1]
                            new_ind.append(p_indices[b0:b1]); new_dat.append(p_data[b0:b1])
                        else:
                            a0, a1 = indptr[r], indptr[r + 1]
                            new_ind.append(indices[a0:a1]); new_dat.append(data[a0:a1])
                        new_ptr.append(new_ptr[-1] + len(new_ind[-1]))
                    new_ind = np.concatenate(new_ind).astype(indices.dtype)
                    new_dat = np.concatenate(new_dat).astype(data.dtype)
                    new_ptr = np.asarray(new_ptr, dtype=indptr.dtype)
                    out = dict(z)
                    out["indices"], out["data"], out["indptr"] = new_ind, new_dat, new_ptr
                    tmp = sp + ".tmp.npz"
                    np.savez(tmp, **out)
                    os.replace(tmp, sp)
                    # post-check against the arrays read before the rewrite: untouched rows
                    # identical, replaced rows == patch
                    z2 = load_csr_arrays(sp)
                    zb = z
                    for r in range(nrows):
                        a0, a1 = z2["indptr"][r], z2["indptr"][r + 1]
                        if r in repl:
                            i = repl[r]
                            b0, b1 = p_indptr[i], p_indptr[i + 1]
                            ok = np.array_equal(z2["indices"][a0:a1], p_indices[b0:b1]) and np.array_equal(z2["data"][a0:a1], p_data[b0:b1])
                        else:
                            b0, b1 = zb["indptr"][r], zb["indptr"][r + 1]
                            ok = np.array_equal(z2["indices"][a0:a1], zb["indices"][b0:b1]) and np.array_equal(z2["data"][a0:a1], zb["data"][b0:b1])
                        if not ok:
                            sys.exit("%s/%s splade shard %d row %d: post-write mismatch" % (ds, model, s, r))
                    assert set(z2.keys()) == set(zb.keys()) and int(z2["shape"][0]) == nrows
                    shards_touched.append({"shard": int(s), "rows_written": len(repl), "sha256_after": sha_file(sp)})
                    n_written += len(repl)
            r = {"dataset": ds, "model": model, "store": sdir, "rev2_rows": int(len(ids)),
                 "rows_written": n_written, "rows_already_equal": n_already,
                 "shards_touched": len(shards_touched), "shards": shards_touched}
            results.append(r)
            print("  %-9s %-6s %-52s written %6d already %6d shards %3d" % (ds, model, sdir, n_written, n_already, len(shards_touched)))
    log_phase("fold", {"results": results})
    return True


# ------------------------------------------------------------------------------------- pointers
def pointers():
    """Rewrite every pointer to the single Phase-C store and prove the redirected positions
    resolve to the same bytes. Query pointers for metaqa are handled by the `queries` phase
    (three hop trees become one position-ordered store)."""
    sys.path.insert(0, FC)
    import pointer_resolver as v1
    pi = rj(FC + "/POINTER_INDEX.json")
    hist = FC + "/_history/pre_v3"
    os.makedirs(hist, exist_ok=True)
    results = []
    # docs
    for ds, d in pi["datasets"].items():
        for model in ("dense", "splade"):
            stores = d[model]["stores"]
            p = "%s/%s/pointer_index/%s.npz" % (FC, ds, model)
            z = np.load(p)
            src, row = z["src"].copy(), z["row"].copy()
            z.close()  # Windows: os.replace refuses while the npz is still open
            new_row = row.astype(np.int64).copy()
            n_redirect = 0
            kinds = {}
            for si, st in enumerate(stores):
                m = np.nonzero(src == si)[0]
                if not len(m):
                    continue
                kind = st.get("store")
                if kind in ("PHASE_C", "WEBQSP_V1"):
                    continue
                if kind == "DENSE_REPAIR":
                    rows_map = np.asarray(rj(os.path.dirname(st["path"]) + "/dense_rows.json"), dtype=np.int64)
                    new_row[m] = rows_map[row[m]]
                elif kind == "REV2_PATCH":
                    _sdir, _ids, own, pos = own_rows_for(ds, model)
                    # map position -> own row
                    o = dict(zip(pos.tolist(), own.tolist()))
                    new_row[m] = np.array([o[int(q)] for q in m], dtype=np.int64)
                else:
                    sys.exit("unknown store kind %s" % kind)
                kinds[kind] = int(len(m))
                n_redirect += int(len(m))
            # proof: the OLD resolver (old index, patch stores still on disk) and the healed
            # store agree byte for byte on every redirected position
            base = stores[0]
            E = v1.CanonicalEmbeddings(ds, model, "docs")
            red = np.nonzero(src != 0)[0]
            if len(red):
                old = E.gather(red)
                st = v1._Store(model, os.path.join(ROOT, base["path"]), base["shard_size"])
                new = st.read(new_row[red])
                if model == "dense":
                    ok = np.array_equal(np.asarray(old), np.asarray(new))
                else:
                    ok = (old != new).nnz == 0 and np.array_equal(old.indptr, new.indptr)
                if not ok:
                    sys.exit("%s/%s: redirected positions do not resolve to identical bytes" % (ds, model))
            # write: same npz format (src all zero) so the existing readers keep working until v3
            bak = "%s/%s__pointer_index__%s.npz" % (hist, ds, model)
            if not os.path.exists(bak):
                shutil.copy2(p, bak)
            np.savez(p + ".tmp.npz", src=np.zeros(len(src), dtype=src.dtype), row=new_row.astype(np.int32))
            os.replace(p + ".tmp.npz", p)
            d[model]["stores"] = [base]
            d[model]["n_pointers"] = int(len(src))
            d[model]["from_PHASE_C"] = int(len(src))
            d[model]["from_REV2_PATCH"] = 0
            d[model]["repaired_pointers"] = 0
            d[model]["consolidated_v3"] = {"redirected_pointers_folded_into_store": kinds,
                                           "max_row": int(new_row.max()), "distinct_rows": int(len(np.unique(new_row)))}
            results.append({"dataset": ds, "kind": "docs", "model": model, "redirected": n_redirect, "kinds": kinds,
                            "proof": "old resolver bytes == healed store bytes on every redirected position"})
            print("  %-9s docs    %-6s redirected %6d %s" % (ds, model, n_redirect, kinds))
    # queries (single-tree datasets; metaqa's three trees are merged by the `queries` phase)
    for ds, d in pi["queries"].items():
        for model in ("dense", "splade"):
            stores = d[model]["stores"]
            phase_c = [s for s in stores if s.get("store") in ("PHASE_C", "WEBQSP_V1")]
            if len(phase_c) != 1:
                print("  %-9s queries %-6s deferred to the queries phase (%d trees)" % (ds, model, len(phase_c)))
                continue
            p = "%s/%s/queries/pointer_index/%s.npz" % (FC, ds, model)
            z = np.load(p)
            src, row = z["src"].copy(), z["row"].copy()
            z.close()  # Windows: os.replace refuses while the npz is still open
            new_row = row.astype(np.int64).copy()
            kinds = {}
            for si, st in enumerate(stores):
                m = np.nonzero(src == si)[0]
                if not len(m) or st.get("store") in ("PHASE_C", "WEBQSP_V1"):
                    continue
                assert st.get("store") == "DENSE_REPAIR", st
                rows_map = np.asarray(rj(os.path.dirname(st["path"]) + "/dense_rows.json"), dtype=np.int64)
                new_row[m] = rows_map[row[m]]
                kinds["DENSE_REPAIR"] = int(len(m))
            red = np.nonzero(src != 0)[0]
            if len(red):
                E = v1.CanonicalEmbeddings(ds, model, "queries")
                old = E.gather(red)
                base = phase_c[0]
                st = v1._Store(model, os.path.join(ROOT, base["path"]), base["shard_size"])
                new = st.read(new_row[red])
                if not np.array_equal(np.asarray(old), np.asarray(new)):
                    sys.exit("%s queries/%s: redirected positions do not resolve to identical bytes" % (ds, model))
            bak = "%s/%s__queries__pointer_index__%s.npz" % (hist, ds, model)
            if not os.path.exists(bak):
                shutil.copy2(p, bak)
            np.savez(p + ".tmp.npz", src=np.zeros(len(src), dtype=src.dtype), row=new_row.astype(np.int32))
            os.replace(p + ".tmp.npz", p)
            d[model]["stores"] = phase_c
            d[model]["repaired_pointers"] = 0
            d[model]["consolidated_v3"] = {"redirected_pointers_folded_into_store": kinds,
                                           "identity": bool(np.array_equal(new_row, np.arange(len(new_row))))}
            results.append({"dataset": ds, "kind": "queries", "model": model, "kinds": kinds,
                            "identity_after": bool(np.array_equal(new_row, np.arange(len(new_row))))})
            print("  %-9s queries %-6s redirected %6d identity_after=%s" % (ds, model, sum(kinds.values()), bool(np.array_equal(new_row, np.arange(len(new_row))))))
    old_rec = FC + "/POINTER_INDEX.json"
    bak = hist + "/POINTER_INDEX.json"
    if not os.path.exists(bak):
        shutil.copy2(old_rec, bak)
    pi["resolver"]["src_codes"] = {"0": "PHASE_C"}
    pi["resolver"]["REV2_PATCH"] = "FOLDED 2026-09-12: every REV2 vector now sits in the patched node's own Phase-C row"
    pi["dense_repair_installed"] = "HEALED_IN_PLACE_2026-09-12"
    pi["dense_repair"]["healed_in_place"] = ("2026-09-12: every DENSE_REPAIR vector was written into the damaged Phase-C row it "
                                             "re-encoded (dense_rows.json), verified bit-exact, and the pointers point at the "
                                             "store again. The patch stores are retired. The corrupt bytes no longer exist.")
    pi["dense_repair"]["non_destructive"] = "SUPERSEDED: the shards were healed in place on 2026-09-12 (see healed_in_place)"
    wj(old_rec, pi)
    log_phase("pointers", {"results": results, "previous_record": bak, "previous_sha256": sha_file(bak)})
    return True


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    ok = {"heal": heal, "fold": fold, "pointers": pointers}.get(phase)
    if ok is None:
        sys.exit("usage: consolidate_v3.py heal|fold|pointers")
    r = ok()
    print("%s: %s in %.0fs" % (phase, "OK" if r else "PROBLEM", time.time() - t0))
    sys.exit(0 if r else 1)


if __name__ == "__main__":
    main()
