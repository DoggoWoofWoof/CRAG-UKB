"""G2 L1 UNIVERSAL PARTITION ROUTER SEARCH -- substrate cache builder.

ONE traversal pass per corpus.  Everything downstream (oracle ceilings, challenger construction,
boundary fusion families, successive halving) reads this cache and needs no graph work at all.

Caches, per dev query:
  base_rank[:TOPP]      canonical Dense+SPLADE partition ranking (the BASE router, unmodified)
  gold partitions       + occurrence counts (evaluation only)
  struct nodes          UNFILTERED frozen directional expansion output (RRF seeds, beam 64),
                        with provenance: residual rank j, min hop, s_dir, structural support count
  ret continuations     rrf200 / dense200 / splade200 node lists (canonical, unfiltered)

Parameter-free: nothing here is fitted, nothing uses gold except the stored evaluation labels.
No TEST split is ever read.

  python scratchpad/_l1ps_cache.py <dataset>
"""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
BEAM_MAIN = 64
TOPP = 200                # depth of the cached canonical partition ranking
SMAX = 256                # frozen M_MAX, the hard cap expand_dir can return
EVAL_CAP = int(os.environ.get("TA_CAP", "2000"))
MIN_VAL = 1000
EMB = f"data/ukb_storage/{DS}/gte_qwen/"
ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
OUT = f"{ROOT}/runs/cache_{DS}.npz"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
np.random.seed(0)


class F32View:
    def __init__(self, A):
        self.A = A

    def __getitem__(self, k):
        return np.asarray(self.A[k], np.float32)

    def __len__(self):
        return len(self.A)


def main():
    if os.path.exists(OUT):
        log(f"cache already complete: {OUT} -- nothing to do")
        return
    lock = OUT + ".lock"
    if os.path.exists(lock):
        log(f"LOCK PRESENT {lock} -- another owner is building this cache, refusing to duplicate")
        return
    open(lock, "w").write(f"{os.getpid()} {time.time()}")
    try:
        build()
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def build():
    log(f"=== L1 PARTITION-SEARCH CACHE {DS} ===")
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
    ndocs = len(hard)
    BIG = ndocs * 1536 * 4 > 1.2e9
    est_gb = (ndocs * 1536 * (2 if BIG else 4) + adji.nbytes + ndocs * 24) / 1e9
    log(f"docs={ndocs} npart={npart} predicted_peak~{est_gb:.2f}GB fp16_path={BIG}")

    j = json.load(open(EMB + "query_ids_all.json"))
    golds, hops, si = j["golds"], j.get("hops", [None] * len(j["ids"])), j["split_indices"]
    rows = list(si["val"]) if si.get("val") else list(si["all"])
    sample_rule = "val"
    if len(rows) < MIN_VAL and si.get("train"):
        rows = rows + list(si["train"]); sample_rule = "val+train (val<1000; TEST never touched)"
    val_set = set(si.get("val", []))
    if DS == "metaqa":
        buck = {1: [], 2: [], 3: []}
        for r in rows:
            if hops[r] in buck:
                buck[hops[r]].append(r)
        per = EVAL_CAP // 3; sel = []
        for h in (1, 2, 3):
            b = list(buck[h]); np.random.shuffle(b); sel += b[:per]
        rows = sorted(sel); sample_rule = "val, hop-balanced (identical to _ta_resid.py / _ta_comb.py)"
    else:
        np.random.shuffle(rows); rows = sorted(rows[:EVAL_CAP])

    dense_all = np.load(EMB + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(EMB + "splade_top200_all.npy", mmap_mode="r")
    qall = np.load(EMB + "queries_all.npy", mmap_mode="r")
    nodes = np.load(EMB + "nodes.npy", mmap_mode="r")

    keep, gold_rows = [], []
    for r in rows:
        g = [id2row[x] for x in golds[r] if x in id2row]
        if g:
            keep.append(r); gold_rows.append(g)
    rows = keep; nq = len(rows)
    log(f"dev queries={nq} ({sample_rule}) golds/q={np.mean([len(g) for g in gold_rows]):.2f}")

    d200 = np.stack([np.asarray(dense_all[i]) for i in rows]).astype(np.int64)
    s200 = np.stack([np.asarray(splade_all[i]) for i in rows]).astype(np.int64)
    dK, sK = d200[:, :TA.K_LOCK], s200[:, :TA.K_LOCK]

    log("normalizing node embeddings (" + ("fp16/fp32" if BIG else "fp32 exact") + ") ...")
    if BIG:
        X16 = np.empty((ndocs, nodes.shape[1]), np.float16)
        for a in range(0, ndocs, 50000):
            b = min(a + 50000, ndocs)
            blk = np.array(nodes[a:b], np.float32)
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
            X16[a:b] = blk.astype(np.float16); del blk
        Xn = F32View(X16)
    else:
        Xn = np.array(nodes, np.float32)
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    Qm = np.stack([np.asarray(qall[i], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)

    # ---- canonical BASE partition ranking (STEP 0)
    t = time.time()
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    t_base = time.time() - t
    base_sel = [set(int(x) for x in base_rank[qi][:TA.P_MAIN]) for qi in range(nq)]
    base_scope = np.array([int(part_sizes[sorted(s)].sum()) for s in base_sel], np.int64)
    allb = anyb = 0
    for qi in range(nq):
        c = [int(hard[g]) in base_sel[qi] for g in gold_rows[qi]]
        allb += all(c); anyb += any(c)
    BASE_ALL, BASE_ANY = allb / nq, anyb / nq
    log(f"BASE_ALL={BASE_ALL:.4f} BASE_ANY={BASE_ANY:.4f} scope={base_scope.mean():.1f} ({t_base:.1f}s)")

    parity = {"status": "NO_REFERENCE"}
    rp = f"results/GENERALIZATION/_g2_comb_{DS}.json"
    if os.path.exists(rp):
        R = json.load(open(rp))
        ok = (abs(R["BASE_ALL_P50"] - round(BASE_ALL, 4)) < 1e-9
              and abs(R["BASE_ANY_P50"] - round(BASE_ANY, 4)) < 1e-9
              and abs(R["BASE_SCOPE_NODES"] - round(float(base_scope.mean()), 1)) < 1e-6
              and R["n_dev_queries"] == nq)
        parity = {"status": "EXACT" if ok else "MISMATCH", "vs": rp, "ref_ALL": R["BASE_ALL_P50"],
                  "ref_ANY": R["BASE_ANY_P50"], "ref_scope": R["BASE_SCOPE_NODES"]}
    log(f"BASE_PARTITION_PARITY = {parity['status']}")
    assert parity["status"] != "MISMATCH", "BASE parity broken -- refusing to cache"

    # ---- retrieval continuations (canonical, unfiltered)
    fused200 = [TA.node_rrf(d200[qi], s200[qi]) for qi in range(nq)]
    ret_rrf = np.full((nq, TA.TOP200), -1, np.int32)
    for qi in range(nq):
        f = fused200[qi][:TA.TOP200]
        ret_rrf[qi, :len(f)] = f

    # ---- frozen structural expansion (RRF seeds, beam 64), UNFILTERED + provenance
    t = time.time(); esc = 0
    s_node = np.full((nq, SMAX), -1, np.int32)
    s_hop = np.zeros((nq, SMAX), np.int8)
    s_sdir = np.zeros((nq, SMAX), np.float32)
    s_cnt = np.zeros((nq, SMAX), np.int32)
    seeds_arr = np.full((nq, TA.SEED_K), -1, np.int32)
    for qi in range(nq):
        sd = fused200[qi][:TA.SEED_K]
        seeds_arr[qi, :len(sd)] = sd
        r_q = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        a, vm, e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, BEAM_MAIN)
        esc += e
        k = min(len(a), SMAX)
        for jj in range(k):
            v = int(a[jj]); m = vm.get(v, [0, 0.0, 0])
            s_node[qi, jj] = v; s_hop[qi, jj] = m[0]; s_sdir[qi, jj] = m[1]; s_cnt[qi, jj] = m[2]
        if (qi + 1) % 500 == 0:
            log(f"   struct {qi+1}/{nq} edges={esc:,}")
    t_struct = time.time() - t
    log(f"struct done: avail/q={float((s_node>=0).sum(1).mean()):.1f} edges={esc:,} {t_struct:.1f}s")

    # ---- gold partitions (evaluation labels only)
    gp, gc, gptr = [], [], [0]
    for qi in range(nq):
        cnt = {}
        for g in gold_rows[qi]:
            p = int(hard[g]); cnt[p] = cnt.get(p, 0) + 1
        for p in sorted(cnt):
            gp.append(p); gc.append(cnt[p])
        gptr.append(len(gp))

    meta = {"dataset": DS, "n_dev_queries": nq, "sample_rule": sample_rule, "n_docs": int(ndocs),
            "npart": int(npart), "fp16_path": bool(BIG), "predicted_peak_gb": round(est_gb, 2),
            "BASE_ALL_P50": round(BASE_ALL, 4), "BASE_ANY_P50": round(BASE_ANY, 4),
            "BASE_SCOPE_NODES": round(float(base_scope.mean()), 1),
            "BASE_PARTITION_PARITY": parity,
            "CONTRACT": {"K0": TA.K0, "K": TA.K_LOCK, "P_MAIN": TA.P_MAIN, "SEED_K": TA.SEED_K,
                         "BEAM": BEAM_MAIN, "MAX_HOPS": TA.MAX_HOPS, "DEG_CAP": TA.DEG_CAP,
                         "SMAX": SMAX, "TOPP": TOPP},
            "COST": {"BASE_sec": round(t_base, 1), "STRUCT_sec": round(t_struct, 1),
                     "STRUCT_edges": int(esc), "STRUCT_x_BASE": round(t_struct / max(t_base, 1e-9), 1)},
            "mean_partition_size": round(float(part_sizes.mean()), 2)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, rows=np.array(rows, np.int64),
        base_rank=base_rank[:, :TOPP].astype(np.int32),
        base_scope=base_scope,
        gold_part=np.array(gp, np.int32), gold_cnt=np.array(gc, np.int32),
        gold_ptr=np.array(gptr, np.int64),
        s_node=s_node, s_hop=s_hop, s_sdir=s_sdir, s_cnt=s_cnt, seeds=seeds_arr,
        ret_rrf=ret_rrf, ret_dense=d200[:, :TA.TOP200].astype(np.int32),
        ret_splade=s200[:, :TA.TOP200].astype(np.int32),
        hard=hard.astype(np.int32), part_sizes=part_sizes.astype(np.int64),
        hops=np.array([-1 if hops[r] is None else int(hops[r]) for r in rows], np.int8),
        in_val=np.array([1 if r in val_set else 0 for r in rows], np.int8),
        meta_json=json.dumps(meta))
    log(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")

    mp = f"{ROOT}/manifest.json"
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[DS] = {**meta, "cache": OUT,
               "cache_sha": hashlib.sha1(open(OUT, "rb").read(1 << 20)).hexdigest()[:16]}
    json.dump(man, open(mp, "w"), indent=1)
    log("manifest updated")


if __name__ == "__main__":
    main()
