"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- PHASE C: frozen L1 on a new partitioning.

Only the PARTITION-DEPENDENT artifacts are rebuilt.  Everything doc-level stays frozen and is read
straight out of the shipped cache: query embeddings, Dense top-200, SPLADE top-200, the retrieval
seeds, and the whole structural expansion (s_node / s_hop / s_sdir / s_cnt) -- the traversal runs on
doc rows and never sees a block id, so a partition swap requires no re-traversal and no re-encode.

Rebuilt for a candidate assignment `hard`:
    mem_idx        own block + the blocks of a node's DIRECTED out-neighbours (TA's exact rule)
    base_rank      TA.partition_ranking(dense[:100]) , (splade[:100]) -> TA.rrf_partitions
    gold_part      evaluation labels only, remapped through the new assignment
    part_sizes

Then the frozen selector runs unchanged: P = 50, B = 6, M_struct = 64, M_ret = 32, S4, F6.
Nothing is retuned.  PARITY: replaying the SHIPPED assignment must reproduce BASE_ALL_P50 and the
frozen F6 scoreboard exactly, which is asserted before any candidate is scored.

  python scratchpad/_l1ep_c.py <ds> [tag ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1ps_router as RT
import _l1kb_core as KB
import _l1ep_sub as EP
import _l1ep_pu as PU

OUT = EP.OUT
ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
CACHE = "scratchpad/_l1ep"
CFG = dict(KB.BASE_CFG)                    # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def dir_out(ds):
    """directed out-neighbour CSR in doc-row space, exactly the list TA iterates for mem_idx."""
    fp = f"{CACHE}/dirout_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return z["ptr"], z["idx"]
    from src.pipeline.standardizer import load_nodes
    mp = f"data/processed/master_nodes_{ds}.json"
    if not os.path.exists(mp):
        mp = "data/processed/master_nodes.json"
    nodes = load_nodes(mp)
    srcs = set(n.metadata.get("source", "") for n in nodes)
    if len(srcs) > 1:
        nodes = [n for n in nodes if n.metadata.get("source") == ds]
    docs = [n for n in nodes if n.metadata.get("type") != "question"]
    id2row = {n.node_id: i for i, n in enumerate(docs)}
    cnt = np.zeros(len(docs) + 1, np.int64)
    flat = []
    for i, nd in enumerate(docs):
        row = [id2row[x] for x in nd.neighbors if x in id2row]
        cnt[i + 1] = len(row)
        flat.extend(row)
    ptr = np.cumsum(cnt)
    idx = np.array(flat, np.int32)
    np.savez_compressed(fp, ptr=ptr, idx=idx)
    return ptr, idx


def mem_from(ds, hard):
    """mem_ptr/mem_idx for a NEW assignment, by TA's directed rule (own block + out-neighbours')."""
    ptr, idx = dir_out(ds)
    n = len(hard)
    out, mlen = [], np.empty(n, np.int64)
    for i in range(n):
        s = {int(hard[i])}
        a, b = int(ptr[i]), int(ptr[i + 1])
        if b > a:
            s.update(int(x) for x in hard[idx[a:b]])
        v = sorted(s)
        out.append(np.array(v, np.int32)); mlen[i] = len(v)
    mem_ptr = np.zeros(n + 1, np.int64); mem_ptr[1:] = np.cumsum(mlen)
    return mem_ptr, np.concatenate(out)


def rebuild(ds, hard, npart, z, meta, log=log):
    """the four partition-dependent arrays for a candidate assignment."""
    hard = np.asarray(hard, np.int64)
    mem = mem_from(ds, hard)
    dK = np.asarray(z["ret_dense"])[:, :TA.K_LOCK]
    sK = np.asarray(z["ret_splade"])[:, :TA.K_LOCK]
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    g, gptr, _, _ = PU.gold_rows(ds)
    gp, gc, gp2 = [], [], [0]
    for qi in range(len(gptr) - 1):
        cnt = {}
        for r in g[gptr[qi]:gptr[qi + 1]]:
            p = int(hard[int(r)]); cnt[p] = cnt.get(p, 0) + 1
        for p in sorted(cnt):
            gp.append(p); gc.append(cnt[p])
        gp2.append(len(gp))
    return {"hard": hard.astype(np.int32), "base_rank": base_rank[:, :np.asarray(z["base_rank"]).shape[1]],
            "gold_part": np.array(gp, np.int32), "gold_cnt": np.array(gc, np.int32),
            "gold_ptr": np.array(gp2, np.int64),
            "part_sizes": np.bincount(hard, minlength=npart).astype(np.int64)}


def replay(ds, hard, npart, tag, log=log):
    z0 = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}
    t = time.time()
    z.update(rebuild(ds, hard, npart, z, meta, log))
    C = RT.build_cache(ds, z, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    goldp = [set(int(x) for x in z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]])
             for qi in range(meta["n_dev_queries"])]
    ind_base = np.array([int(goldp[qi] <= C["base50"][qi]) for qi in range(len(goldp))], np.int8)
    ctxs = KB.contexts(z, meta, C, CFG["B"], CFG)
    sel = lambda c, qi, extra: KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"],
                                            c["cpos"], CFG["B"])[0]
    r6 = KB.run_selector(ctxs, sel, goldp, ind_base)
    rb = KB.run_selector(ctxs, KB.sel_base, goldp, ind_base)
    sizes = np.asarray(z["part_sizes"])
    scope = np.array([int(sizes[sorted(C["base50"][qi])].sum()) for qi in range(len(goldp))])
    R = {"tag": tag, "ds": ds, "npart": int(npart), "nq": len(goldp),
         "BASE_ALL_P50": round(float(ind_base.mean()), 4),
         "BASE_ANY_P50": round(float(np.mean([int(bool(goldp[qi] & C["base50"][qi]))
                                              for qi in range(len(goldp))])), 4),
         "BASE_SCOPE_NODES": round(float(scope.mean()), 1),
         "F6_ALL_P50": round(r6["ALL"], 4), "F6_ANY_P50": round(r6["ANY"], 4),
         "BND_ALL_P50": round(rb["ALL"], 4),
         "F6_gold_admitted": r6["gold_admitted"], "F6_gold_evicted": r6["gold_evicted"],
         "F6_churn": round(float(r6["churn"].mean()), 3),
         "seconds": round(time.time() - t, 1)}
    hops = np.asarray(z["hops"]) if "hops" in z else None
    if ds == "metaqa" and hops is not None:
        R["by_hop"] = {}
        for h in (1, 2, 3):
            m = hops == h
            if m.any():
                R["by_hop"][f"hop{h}"] = {"n": int(m.sum()),
                                          "BASE_ALL": round(float(ind_base[m].mean()), 4),
                                          "F6_ALL": round(float(r6["ind"][m].mean()), 4)}
    R["_ind_F6"] = r6["ind"].tolist()
    R["_ind_BASE"] = ind_base.tolist()
    return R


if __name__ == "__main__":
    ds = sys.argv[1]
    tags = sys.argv[2:] or ["PM_CURRENT_EXACT"]
    os.makedirs(f"{OUT}/INTERACTION", exist_ok=True)
    fp = f"{OUT}/INTERACTION/C_{ds}.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for tag in tags:
        if tag == "PM_CURRENT_EXACT":
            hard, npart, _ = (*PU.load_assignment(ds, "CURRENT"), None)
        else:
            p = f"scratchpad/_l1ep/parts/{ds}__{tag}.npy"
            hard = np.load(p); npart = int(hard.max()) + 1
        R = replay(ds, hard, npart, tag)
        if tag == "PM_CURRENT_EXACT":
            z0 = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
            meta = json.loads(str(z0["meta_json"]))
            ok = (abs(R["BASE_ALL_P50"] - meta["BASE_ALL_P50"]) < 1e-9
                  and abs(R["BASE_ANY_P50"] - meta["BASE_ANY_P50"]) < 1e-9
                  and abs(R["BASE_SCOPE_NODES"] - meta["BASE_SCOPE_NODES"]) < 1e-6)
            R["C1_PARITY"] = "EXACT" if ok else "MISMATCH"
            R["C1_PARITY_REF"] = {"BASE_ALL": meta["BASE_ALL_P50"], "BASE_ANY": meta["BASE_ANY_P50"],
                                  "SCOPE": meta["BASE_SCOPE_NODES"]}
            log(f"  C1_PARITY = {R['C1_PARITY']}  (ref ALL={meta['BASE_ALL_P50']} "
                f"got {R['BASE_ALL_P50']})")
        rec[tag] = R
        json.dump(rec, open(fp, "w"), indent=1)
        log(f"  {ds:16s} {tag:24s} BASE_ALL {R['BASE_ALL_P50']:.4f}  F6_ALL {R['F6_ALL_P50']:.4f}  "
            f"scope {R['BASE_SCOPE_NODES']:.0f}  ({R['seconds']}s)")
        if "by_hop" in R:
            log("      " + json.dumps(R["by_hop"]))
    print("wrote", fp)
