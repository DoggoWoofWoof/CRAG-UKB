"""Step 3 of the return-to-L1 plan: prove the canonical L1 adapter on every dataset.

For each dataset it (a) checks the DATASET.json record against the freeze, (b) builds the query
index (gold positions / hops / splits) and checks it against the read-only inventory audit,
(c) proves the zero-copy embedding and cache views (shape, cross-shard consistency, norms, RSS),
(d) re-derives a sample of the served dense top-k from the served vectors, (e) builds the frozen
key sets and CSRs from the served graph families and checks them against the graph manifest
census, and (f) records the L1 contract hash.

    PYTHONHASHSEED=0 python src/l1_canonical/validate_adapter.py [ds ...]

Writes results/L1_CANONICAL/ADAPTER_VALIDATION.json (merging per-dataset results). Reads only;
the derived files it creates are data/l1_canonical/<ds>/{query_index,keys}.npz.
"""
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset, DATASETS, contract_hash, sha_file  # noqa: E402
from src.dataset_canonical.freeze_canonical import record_hash  # noqa: E402

OUT = os.path.join(REPO, "results", "L1_CANONICAL", "ADAPTER_VALIDATION.json")
AUDIT = os.path.join(REPO, "results", "L1_CANONICAL", "inventory_audit_six.json")
EXPECT_EVAL = {"metaqa": 39138, "squad": 11873, "musique": 2417, "hotpotqa": 7405, "2wiki": 12576, "webqsp": 1549}
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def _avail_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


def rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


def check(rep, name, ok, detail=None):
    rep["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        log("   FAIL", name, detail)
    return bool(ok)


def validate(name, audit, exact_topk_queries=50):
    rep = {"dataset": name, "checks": [], "timings_s": {}, "rss_mb": {"start": rss_mb()}}
    t = time.time()
    d = CanonicalDataset(name)
    N, NQ = d.n_nodes, d.n_queries
    A = (audit.get("datasets") or {}).get(name, {})

    # ---- (a) records
    R = d.manifest
    check(rep, "DATASET.json self-hash", record_hash(R) == R["RECORD_SHA256"])
    fz = d.freeze["DATASETS"][name]
    check(rep, "DATASET.json file digest == CANONICAL_FREEZE pin",
          sha_file(os.path.join(d.dir, "DATASET.json")) == fz["DATASET.json"]["sha256"])
    check(rep, "n_nodes/n_queries == freeze", N == fz["n_nodes"] and NQ == fz["n_queries"], [N, NQ])
    check(rep, "eval split declared and served", d.eval_split in d.split_ranges or (name == "webqsp" and d.eval_split == "train_holdout"), d.eval_split)
    rep["timings_s"]["records"] = round(time.time() - t, 1)

    # ---- (b) query index
    t = time.time()
    qi = d.query_index(log=log)
    rep["timings_s"]["query_index"] = round(time.time() - t, 1)
    gp, gpos = qi["gold_ptr"], qi["gold_pos"]
    check(rep, "query index rows == n_queries", len(gp) - 1 == NQ)
    check(rep, "gold positions in range", gpos.size == 0 or (gpos.min() >= 0 and gpos.max() < N), [int(gpos.min()) if gpos.size else None, int(gpos.max()) if gpos.size else None])
    by = (A.get("queries") or {}).get("by_split") or {}
    for si, sp in enumerate(d.splits):
        a, b = d.split_ranges[sp]
        n_refs = int(gp[b] - gp[a])
        zero = int(((gp[a + 1:b + 1] - gp[a:b]) == 0).sum())
        exp = by.get(sp, {})
        check(rep, "gold refs %s == inventory audit" % sp, exp.get("gold_refs") is None or exp["gold_refs"] == n_refs, {"adapter": n_refs, "audit": exp.get("gold_refs")})
        check(rep, "zero-gold rows %s == inventory audit" % sp, exp.get("zero_gold_rows") is None or exp["zero_gold_rows"] == zero, {"adapter": zero, "audit": exp.get("zero_gold_rows")})
        check(rep, "split codes %s" % sp, bool((qi["split_code"][a:b] == si).all()))
    er = d.eval_rows()
    check(rep, "eval rows == expected", len(er) == EXPECT_EVAL[name], {"n": int(len(er)), "expected": EXPECT_EVAL[name], "first": int(er[0]), "last": int(er[-1])})
    ng = np.array([len(g) for g in d.gold(er)])
    rep["eval_population"] = {"split": d.eval_split, "n": int(len(er)), "with_gold": int((ng > 0).sum()), "zero_gold": int((ng == 0).sum()),
                              "mean_golds": round(float(ng.mean()), 3)}
    hp = d.hops(er)
    if (hp >= 0).any():
        u, c = np.unique(hp[hp >= 0], return_counts=True)
        rep["eval_population"]["hop_hist"] = {int(k): int(v) for k, v in zip(u, c)}
    if name == "metaqa":
        check(rep, "metaqa dev has hops 1/2/3", set(rep["eval_population"].get("hop_hist", {})) == {1, 2, 3})
    check(rep, "eval rows never in a test split", all(d.split_ranges[sp][0] > er.max() or d.split_ranges[sp][1] <= er.min() for sp in d.splits if sp == "test"))
    rep["rss_mb"]["after_query_index"] = rss_mb()

    # ---- (c) embeddings + caches, zero copy
    t = time.time()
    r0 = rss_mb()
    E, Q = d.node_embeddings, d.query_embeddings
    check(rep, "node_embeddings shape", E.shape == (N, 1536), list(E.shape))
    check(rep, "query_embeddings shape", Q.shape == (NQ, 1536), list(Q.shape))
    ss = E.shard_size
    probe = np.array([0, 1, ss - 1, ss, ss + 1, N - 1] if N > ss + 1 else [0, 1, N - 1], np.int64)
    probe = np.unique(np.clip(probe, 0, N - 1))
    ref = E.emb.read(probe)
    check(rep, "ShardedRows gather == canonical.Embeddings.read (cross-shard probe)", np.array_equal(E[probe], ref))
    check(rep, "ShardedRows slice across a shard boundary", N <= ss or np.array_equal(E[ss - 2:ss + 2], E.emb.read(np.arange(ss - 2, ss + 2))))
    rng = np.random.RandomState(0)
    smp = np.sort(rng.choice(N, size=min(2000, N), replace=False))
    nr = np.linalg.norm(E[smp].astype(np.float32), axis=1)
    check(rep, "doc vectors unit norm (|norm-1| <= 0.01, sample 2000)", float(np.abs(nr - 1).max()) <= 0.01, round(float(np.abs(nr - 1).max()), 5))
    qs = np.sort(rng.choice(NQ, size=min(500, NQ), replace=False))
    nq_ = np.linalg.norm(Q[qs].astype(np.float32), axis=1)
    check(rep, "query vectors unit norm (sample 500)", float(np.abs(nq_ - 1).max()) <= 0.01, round(float(np.abs(nq_ - 1).max()), 5))
    for mdl in ("dense", "splade"):
        z = d.cache(mdl)
        ids, sc = z["ids"], z["scores"]
        check(rep, "%s cache ids memmap [n_queries, 1000]" % mdl, ids.shape == (NQ, 1000) and z.stored("ids") and isinstance(ids, np.memmap), list(ids.shape))
        check(rep, "%s cache scores memmap" % mdl, sc.shape == (NQ, 1000) and z.stored("scores"), str(sc.dtype))
        sub = ids[qs]
        check(rep, "%s cache ids in range (sample)" % mdl, sub.min() >= 0 and sub.max() < N)
        check(rep, "%s cache 1000 distinct per row (sample)" % mdl, all(len(np.unique(r)) == 1000 for r in sub[:200]))
        ssc = sc[qs].astype(np.float32)
        check(rep, "%s cache scores non-increasing (sample)" % mdl, bool((np.diff(ssc, axis=1) <= 0).all()))
        v = d.topk(mdl, 200)
        check(rep, "%s topk(200) is a view of the cache" % mdl, v.shape == (NQ, 200) and np.shares_memory(v, ids))
    rep["rss_mb"]["after_embeddings_and_caches"] = rss_mb()
    check(rep, "zero-copy: RSS growth from opening embeddings+caches < 256 MB", r0 is None or rep["rss_mb"]["after_embeddings_and_caches"] - r0 < 256,
          None if r0 is None else round(rep["rss_mb"]["after_embeddings_and_caches"] - r0, 1))
    rep["timings_s"]["embeddings_caches"] = round(time.time() - t, 1)

    # ---- (d) exact top-k re-derivation from the served vectors (small corpora only)
    t = time.time()
    if N * 1536 * 2 <= 1.5e9 and exact_topk_queries:
        X = np.empty((N, 1536), np.float32)
        for a, b, blk in E.blocks(50000):
            X[a:b] = blk
        qrows = er[rng.choice(len(er), size=min(exact_topk_queries, len(er)), replace=False)]
        served = d.dense_topk(10, rows=qrows)
        bad = 0
        worst = 0.0
        for j, r in enumerate(qrows):
            s = X @ Q[int(r)].astype(np.float32)
            kth = np.partition(s, -10)[-10]
            gap = float(kth - s[served[j]].min())
            worst = max(worst, gap)
            if gap > 2e-3:
                bad += 1
        check(rep, "dense top-10 of %d eval queries re-derived from served vectors (fp16 tolerance 2e-3)" % len(qrows), bad == 0, {"bad": bad, "worst_gap": round(worst, 5)})
        del X
    else:
        rep["checks"].append({"check": "dense top-k re-derivation", "ok": None, "detail": "skipped: corpus too large to hold in RAM here (%d rows)" % N})
    rep["timings_s"]["exact_topk"] = round(time.time() - t, 1)

    # ---- (e) graph families -> frozen key sets + CSRs
    t = time.time()
    fams = d.families()
    check(rep, "families present == {structural, ner, knn}", set(fams) == {"structural", "ner", "knn"}, fams)
    Nk, ST, KN, NX = d.keysets(log=log)
    _, NER_RAW = d.ner_raw_keys()
    rep["keysets"] = {"N": int(Nk), "STRUCT": int(len(ST)), "KNN": int(len(KN)), "NERX": int(len(NX)), "NER_RAW": int(len(NER_RAW))}
    gm = d.graph_manifest["families"]
    cen = {f: gm[f].get("storage_census") or {} for f in fams}
    ag = (A.get("graph") or {}).get("families") or {}
    # structural: distinct unordered pairs minus the self-loop pairs
    s, dd, _, _ = d.family("structural")
    loops = int((np.asarray(s) == np.asarray(dd)).sum())
    n_loop_pairs = int(len(np.unique(np.asarray(s)[np.asarray(s) == np.asarray(dd)])))
    exp_struct = cen["structural"].get("distinct_unordered_pairs")
    check(rep, "STRUCT keys == structural distinct unordered pairs (census counts loop pairs) - loop pairs", exp_struct is None or len(ST) == exp_struct - n_loop_pairs,
          {"keys": int(len(ST)), "census_pairs": exp_struct, "loop_pairs": n_loop_pairs, "self_loop_rows": loops})
    del s, dd
    exp_ner = cen["ner"].get("distinct_unordered_pairs")
    check(rep, "NER_RAW keys == ner distinct unordered pairs", exp_ner is None or len(NER_RAW) == exp_ner, {"keys": int(len(NER_RAW)), "census": exp_ner})
    exp_knn = cen["knn"].get("distinct_unordered_pairs")
    check(rep, "KNN keys <= knn distinct pairs and KNN disjoint from STRUCT", (exp_knn is None or len(KN) <= exp_knn) and len(np.intersect1d(KN, ST, assume_unique=True)) == 0,
          {"keys": int(len(KN)), "census": exp_knn, "knn_pairs_also_structural": None if exp_knn is None else int(exp_knn - len(KN))})
    check(rep, "NERX disjoint from STRUCT", len(np.intersect1d(NX, ST, assume_unique=True)) == 0)
    for nm, K in (("STRUCT", ST), ("KNN", KN), ("NERX", NX)):
        check(rep, "%s keys sorted, unique, in range" % nm, K.size == 0 or (bool((np.diff(K) > 0).all()) and K.min() >= 0 and K.max() < np.int64(N) * N))
    xo, ao = d.struct_csr(directed=True)
    check(rep, "directed structural CSR: nnz == n_edges", int(xo[-1]) == int(gm["structural"]["n_edges"]) and ao.max() < N)
    del xo, ao
    d._csr.clear()
    d._fams.clear()
    need_b = 10 * 8 * len(ST) + 8 * N
    avail = _avail_bytes()
    if avail is None or avail > need_b + 1.5e9:
        xu, au = d.struct_csr(directed=False)
        check(rep, "undirected structural CSR: nnz == 2*|STRUCT|", int(xu[-1]) == 2 * len(ST))
        degu = np.diff(xu)
        rep["undirected_struct_degree"] = {"deg0": int((degu == 0).sum()), "max": int(degu.max()), "mean": round(float(degu.mean()), 3),
                                           "audit_directed_deg0": (ag.get("structural") or {}).get("nodes_with_degree_0")}
        del xu, au, degu
        d._csr.clear()
    else:
        rep["checks"].append({"check": "undirected structural CSR", "ok": None, "detail": "skipped: needs ~%.1f GB, %.1f GB available" % (need_b / 1e9, avail / 1e9)})
    rep["timings_s"]["graph"] = round(time.time() - t, 1)
    rep["rss_mb"]["after_graph"] = rss_mb()

    # ---- (f) pins + contract
    rep["pins"] = d.pins()
    rep["derived_files"] = {os.path.relpath(p, REPO): {"bytes": os.path.getsize(p), "sha256": sha_file(p)} for p in
                            (d._query_index_path(), d._keys_path()) if os.path.exists(p)}
    rep["n_checks"] = len([c for c in rep["checks"] if c["ok"] is not None])
    rep["n_failed"] = len([c for c in rep["checks"] if c["ok"] is False])
    rep["PASS"] = rep["n_failed"] == 0
    rep["seconds"] = round(time.time() - T0, 1)
    log("%s: %d checks, %d failed -> %s" % (name, rep["n_checks"], rep["n_failed"], "PASS" if rep["PASS"] else "FAIL"))
    return rep


def main():
    names = sys.argv[1:] or DATASETS
    audit = json.load(io.open(AUDIT, encoding="utf-8")) if os.path.exists(AUDIT) else {}
    out = json.load(io.open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {"RECORD": "L1_CANONICAL_ADAPTER_VALIDATION", "datasets": {}}
    out["contract"] = contract_hash()
    for name in names:
        log("=== %s ===" % name)
        try:
            out["datasets"][name] = validate(name, audit)
        except Exception as e:  # keep going, record the failure
            import traceback
            out["datasets"][name] = {"dataset": name, "PASS": False, "error": repr(e), "traceback": traceback.format_exc()[-2000:]}
            log("%s: ERROR %r" % (name, e))
        out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out["ALL_PASS"] = all(v.get("PASS") for v in out["datasets"].values())
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out, indent=1))
    log("wrote", OUT, "ALL_PASS" if out["ALL_PASS"] else "NOT ALL PASS", sorted(out["datasets"]))


if __name__ == "__main__":
    main()
