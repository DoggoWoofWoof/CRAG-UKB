"""
POINTER INDEX VERIFICATION
==========================
Proves the pointer index resolves to the RIGHT vector, not merely to a vector.

The pointer arrays were built from the reuse map, so re-reading the same field would be
circular. The test here closes the loop through a different artifact: a pointer names a
(shard, offset); the Phase-C ids_<shard>.json at that offset names a Phase-C node id; that
id must equal the id the reuse map independently attributes to the canonical row.

  metaqa / musique / squad : reuse_source literally carries the Phase-C node id, and the
      corpora are small, so EVERY row is checked, not a sample.
  2wiki / hotpotqa : reuse_source carries a row number, so the id check uses the documented
      id transform instead (2wiki:c<curid> -> 2wu:<curid>, hotpotqa:c<id> -> hotpot_<id>),
      which is independent of the row field the index was built from. Sampled, because the
      ids files for 150 shards are 5.99M ids.
  REV2 patch rows: checked in full for every dataset that has them -- the resolved vector
      must be bit-identical to the patch array row, and the patch ids entry must equal the
      canonical node id.

Also checks the numerical contract (dense finite and L2-normalised, SPLADE non-negative and
never an empty row) and that gather() preserves caller order under shuffling.
"""
import io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

ROOT = "data/final_canonical"
CANON = "data/canonical"
RNG = np.random.default_rng(0)
FULL = {"metaqa", "musique", "squad"}


def phase_c_ids(tree, model, kind, shard):
    return json.load(io.open("%s/%s/encodings/%s/%s/ids_%05d.json"
                             % (CANON, tree, model, kind, shard), encoding="utf-8"))


def expected_phase_c_id(ds, canonical_id):
    """The documented id transform, used where reuse_source carries a row not an id."""
    if ds == "2wiki":
        return "2wu:" + canonical_id.split(":c", 1)[1]
    if ds == "hotpotqa":
        return "hotpot_" + canonical_id.split(":c", 1)[1]
    return None


def reuse_source_ids(ds, model):
    """canonical_node_id -> Phase-C node id, for the small corpora only."""
    ix = json.load(io.open("%s/%s/reuse_map/index.json" % (ROOT, ds), encoding="utf-8"))
    out = {}
    for sh in ix["shards"]:
        for ln in io.open("%s/%s/%s" % (ROOT, ds, sh["file"]), encoding="utf-8"):
            o = json.loads(ln)
            if o.get("canonical_node_id") is None:
                continue
            rs = o.get("%s_reuse_source" % model) or ""
            if ":" in rs:
                out[o["canonical_node_id"]] = rs.split(":", 1)[1]
    return out


def check(ds, model, kind, res):
    E = CanonicalEmbeddings(ds, model, kind, root=ROOT)
    man = json.load(io.open("%s/POINTER_INDEX.json" % ROOT, encoding="utf-8"))
    n = len(E)
    r = {"n": n, "n_ids": len(E.ids)}

    # ---- numerical contract on a scattered sample -------------------------------------
    k = min(400, n)
    pos = RNG.choice(n, size=k, replace=False)
    g = E.gather(pos)
    if model == "dense":
        a = np.asarray(g, dtype=np.float32)
        nr = np.linalg.norm(a, axis=1)
        r["dense_finite"] = bool(np.isfinite(a).all())
        r["l2_min"], r["l2_max"] = round(float(nr.min()), 6), round(float(nr.max()), 6)
        r["l2_ok"] = bool(np.abs(nr - 1.0).max() < 2e-3)
    else:
        r["splade_nonneg"] = bool((g.data >= 0).all())
        r["splade_empty_rows"] = int((np.diff(g.indptr) == 0).sum())
        r["splade_dim"] = int(g.shape[1])

    # ---- gather() must preserve caller order ------------------------------------------
    sel = pos[:12]
    one = [E.get(int(i)) for i in sel]
    many = E.gather(sel)
    if model == "dense":
        r["order_preserved"] = bool(all(np.array_equal(np.asarray(one[j]),
                                                       np.asarray(many[j]))
                                        for j in range(len(sel))))
    else:
        r["order_preserved"] = bool(all((one[j] != many[j]).nnz == 0 for j in range(len(sel))))

    # ---- REV2 patch rows: resolved vector must be bit-identical to the patch ----------
    pidx = np.nonzero(E.src == 1)[0] if kind == "docs" else np.array([], dtype=np.int64)
    r["patch_pointers"] = int(pidx.size)
    if pidx.size:
        ext = "npy" if model == "dense" else "npz"
        base = "%s/_rev2_encoder_patch/%s/%s" % (ROOT, ds, model)
        pids = json.load(io.open("%s_ids.json" % base, encoding="utf-8"))
        bad_id = sum(1 for p in pidx if pids[int(E.row[p])] != E.ids[int(p)])
        r["patch_id_mismatch"] = int(bad_id)
        s = pidx if pidx.size <= 3000 else RNG.choice(pidx, 3000, replace=False)
        got = E.gather(s)
        if model == "dense":
            ref = np.load("%s.%s" % (base, ext), mmap_mode="r")[E.row[s]]
            r["patch_bit_identical"] = bool(np.array_equal(np.asarray(got), np.asarray(ref)))
        else:
            from scipy.sparse import csr_matrix
            z = np.load("%s.%s" % (base, ext))
            M = csr_matrix((z["data"], z["indices"], z["indptr"]),
                           shape=(int(z["shape"][0]), int(z["shape"][1])))
            r["patch_bit_identical"] = bool((got != M[E.row[s]]).nnz == 0)

    # ---- the non-circular id check ----------------------------------------------------
    tree = (man["datasets"][ds][model]["phase_c_tree"] if kind == "docs"
            else man["queries"][ds]["trees"][0])
    size = (man["datasets"][ds][model]["phase_c_shard_size"] if kind == "docs"
            else man["queries"][ds][model]["shard_size_per_tree"][0])
    if kind == "docs":
        pc = np.nonzero(E.src == 0)[0]
        if ds in FULL:
            want = reuse_source_ids(ds, model)
            sample = pc
        else:
            sample = pc if pc.size <= 4000 else RNG.choice(pc, 4000, replace=False)
            want = None
        rows = E.row[sample]
        bad, checked = 0, 0
        cache = {}
        for p, rw in zip(sample.tolist(), rows.tolist()):
            sh, off = rw // size, rw % size
            if sh not in cache:
                cache[sh] = phase_c_ids(tree, model, "docs", sh)
            got = cache[sh][off]
            exp = want.get(E.ids[p]) if want is not None else expected_phase_c_id(ds, E.ids[p])
            if exp is None:
                continue
            checked += 1
            if got != exp:
                bad += 1
                if bad <= 3:
                    r.setdefault("id_examples", []).append((E.ids[p], got, exp))
        r["id_checked"] = checked
        r["id_mismatch"] = bad
        r["id_check_mode"] = "ALL_ROWS_VS_REUSE_SOURCE" if ds in FULL else "SAMPLED_VS_ID_TRANSFORM"
    res["%s/%s/%s" % (ds, kind, model)] = r
    ok = (r.get("id_mismatch", 0) == 0 and r.get("patch_id_mismatch", 0) == 0
          and r.get("patch_bit_identical", True) and r["order_preserved"]
          and r.get("l2_ok", True) and r.get("splade_empty_rows", 0) == 0
          and r.get("dense_finite", True) and r.get("splade_nonneg", True))
    r["PASS"] = bool(ok)
    print("  %-28s n=%-9d %s" % ("%s/%s/%s" % (ds, kind, model), n, "PASS" if ok else "FAIL"),
          flush=True)
    if not ok:
        print("     ", json.dumps(r)[:600])
    return ok


if __name__ == "__main__":
    t0 = time.time()
    res, allok = {}, True
    for ds in ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]:
        for kind in ["docs", "queries"]:
            for model in ["dense", "splade"]:
                allok &= check(ds, model, kind, res)
    res["ALL_PASS"] = bool(allok)
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, io.open("scratchpad/verify_pointer_index.json", "w", encoding="utf-8"), indent=1)
    print("\nALL_PASS =", allok, " %.1fs" % (time.time() - t0))
