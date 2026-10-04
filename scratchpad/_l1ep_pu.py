"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- PHASE B: does the partitioning help?

Everything here is measured on a PARTITION ASSIGNMENT `hard` (doc row -> block id) and is
router-free: it asks whether the offline blocks co-locate evidence at all, before any selector is
allowed to try to find them.  Gold is used ONLY to score an assignment that was BUILT without it.

B0  build provenance                 read from source, see _l1ep_b0.py
B1  GOLD_COMPRESSION                 distinct gold blocks / gold docs      (1.0 = no co-location)
    PARTITION_FETCH_SAVING           1 - GOLD_COMPRESSION
B2  PAIR_COLOCATION                  P(two golds of one query share a block)
    PAIR_COLOCATION_ADJUSTED         (obs - exp) / (1 - exp), exp from the SAME size histogram
B3  ALL_GOLD_CONTAINED               P(all golds of a query in ONE block)
    GOLD_BLOCK_DENSITY               mean over queries of  max_p |golds in p| / |golds|
B4  SEED_REQUIRED_COLOCATION         fraction of required blocks that already hold a frozen seed
B5  GOLD_EDGE_CONTAINMENT            of the topology-C edges BETWEEN two golds of the same query,
                                     the fraction that is NOT cut  (MetaQA hop2/hop3 primary)
B7  MIN_PARTITIONS_REQUIRED          distribution of |gold blocks| per query
B8  ORACLE_CURVE                     full-universe oracle ALL/ANY recall at P = 1..100; this is the
                                     ceiling the ASSIGNMENT imposes, with no router in the loop
B12 graph quality                    edge cut / cut ratio / balance / modularity, per edge family
B14 exposure                         nodes exposed by the top-P blocks (matched-budget accounting)

  python scratchpad/_l1ep_pu.py <ds> [assignment.npy|CURRENT] [tag]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_sub as EP

OUT = EP.OUT
CACHE = "scratchpad/_l1ep"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ evaluation labels (gold DOCS)
def gold_rows(ds):
    """gold doc rows per dev query, rebuilt exactly as _l1ps_cache.py selected them.  Cached."""
    fp = f"{CACHE}/gold_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return z["g"], z["gptr"], z["rows"], z["hops"]
    import _l1ps_router as RT
    z, meta = RT.load(ds) if hasattr(RT, "load") else (np.load(
        f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{ds}.npz", allow_pickle=True),
        None)
    if meta is None:
        meta = json.loads(str(z["meta_json"]))
    rows = np.asarray(z["rows"], np.int64)
    import _ta_prepartition as TA
    _, _, _, _, _, id2row = TA.load_topology(ds, lambda *a: None)
    EMB = f"data/ukb_storage/{ds}/gte_qwen/"
    j = json.load(open(EMB + "query_ids_all.json"))
    golds = j["golds"]
    hops_all = j.get("hops", [0] * len(j["ids"]))
    g, gptr, hp = [], [0], []
    for r in rows:
        gr = [id2row[x] for x in golds[int(r)] if x in id2row]
        g.extend(gr); gptr.append(len(g))
        hp.append(int(hops_all[int(r)] or 0))
    g = np.array(g, np.int64); gptr = np.array(gptr, np.int64); hp = np.array(hp, np.int32)
    np.savez_compressed(fp, g=g, gptr=gptr, rows=rows, hops=hp)
    return g, gptr, rows, hp


def load_assignment(ds, spec):
    if spec in (None, "CURRENT", "P1_METIS_CURRENT"):
        import _ta_prepartition as TA
        hard, _, npart, _, _, _ = TA.load_topology(ds, lambda *a: None)
        return np.asarray(hard, np.int64), int(npart)
    h = np.load(spec)
    return np.asarray(h, np.int64), int(h.max()) + 1


# ------------------------------------------------------------------ B1-B8
def utility(ds, hard, npart, seeds=None, log=log):
    g, gptr, rows, hops = gold_rows(ds)
    nq = len(gptr) - 1
    sizes = np.bincount(hard, minlength=npart).astype(np.int64)
    N = len(hard)
    # expectation under a RANDOM assignment with EXACTLY this size histogram
    exp_pair = float((sizes * (sizes - 1)).sum()) / max(N * (N - 1), 1)
    if seeds is None:
        z = np.load(f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{ds}.npz",
                    allow_pickle=True)
        seeds = np.asarray(z["seeds"])
    S = EP.keysets(ds)
    Nn, ST, KN, NX = S
    CKEY = np.union1d(np.union1d(ST, NX), KN)

    per = {"comp": [], "pair_n": [], "pair_s": [], "allc": [], "dens": [], "minp": [],
           "seedcov": [], "seedcovn": [], "ge_tot": [], "ge_in": [], "ngold": []}
    for qi in range(nq):
        gr = g[gptr[qi]:gptr[qi + 1]]
        if not len(gr):
            continue
        gp = hard[gr]
        u, cnt = np.unique(gp, return_counts=True)
        per["ngold"].append(len(gr))
        per["comp"].append(len(u) / len(gr))
        per["minp"].append(len(u))
        per["allc"].append(int(len(u) == 1))
        per["dens"].append(float(cnt.max()) / len(gr))
        npair = len(gr) * (len(gr) - 1) // 2
        same = int((cnt * (cnt - 1) // 2).sum())
        per["pair_n"].append(npair); per["pair_s"].append(same)
        sd = [int(v) for v in seeds[qi] if v >= 0]
        gset = set(int(x) for x in gr)
        sp = set(int(hard[v]) for v in sd)
        # NONTRIVIAL: a seed that IS itself a gold document puts its own block in `sp` for free,
        # which inflates the raw number under ANY assignment (including random).  The nontrivial
        # variant counts only blocks reached by a seed that is not a gold of this query.
        spn = set(int(hard[v]) for v in sd if v not in gset)
        per["seedcov"].append(len(set(int(x) for x in u) & sp) / len(u))
        per["seedcovn"].append(len(set(int(x) for x in u) & spn) / len(u))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} B1-B8 {qi+1}/{nq}")
        if len(gr) > 1:
            a = np.repeat(gr, len(gr)); b = np.tile(gr, len(gr))
            m = a < b
            k = a[m] * np.int64(Nn) + b[m]
            ii = np.searchsorted(CKEY, k)
            hit = (ii < len(CKEY)) & (CKEY[np.minimum(ii, len(CKEY) - 1)] == k)
            per["ge_tot"].append(int(hit.sum()))
            per["ge_in"].append(int((hit & (hard[a[m]] == hard[b[m]])).sum()))
        else:
            per["ge_tot"].append(0); per["ge_in"].append(0)

    mp = np.array(per["minp"], np.int64)
    curve = {int(P): {"ALL": round(float((mp <= P).mean()), 4)} for P in
             [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]}
    R = {"ds": ds, "nq": len(mp), "npart": int(npart), "N": int(N),
         "sizes": {"min": int(sizes.min()), "max": int(sizes.max()),
                   "mean": round(float(sizes.mean()), 2), "median": float(np.median(sizes)),
                   "CV": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4),
                   "imbalance_max_over_mean": round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4),
                   "empty_blocks": int((sizes == 0).sum())},
         "B1_GOLD_COMPRESSION": round(float(np.mean(per["comp"])), 4),
         "B1_PARTITION_FETCH_SAVING": round(1.0 - float(np.mean(per["comp"])), 4),
         "B2_PAIR_COLOCATION": round(float(np.sum(per["pair_s"]) / max(np.sum(per["pair_n"]), 1)), 5),
         "B2_PAIR_EXPECTED_RANDOM": round(exp_pair, 6),
         "B2_PAIR_COLOCATION_ADJUSTED": round(
             (float(np.sum(per["pair_s"]) / max(np.sum(per["pair_n"]), 1)) - exp_pair)
             / max(1.0 - exp_pair, 1e-12), 5),
         "B3_ALL_GOLD_CONTAINED": round(float(np.mean(per["allc"])), 4),
         "B3_GOLD_BLOCK_DENSITY": round(float(np.mean(per["dens"])), 4),
         "B4_SEED_REQUIRED_COLOCATION": round(float(np.mean(per["seedcov"])), 4),
         "B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL": round(float(np.mean(per["seedcovn"])), 4),
         "B5_GOLD_EDGES_TOTAL": int(np.sum(per["ge_tot"])),
         "B5_GOLD_EDGE_CONTAINMENT": round(float(np.sum(per["ge_in"]) / max(np.sum(per["ge_tot"]), 1)), 4),
         "B7_MIN_PARTITIONS_REQUIRED": {
             "mean": round(float(mp.mean()), 4), "median": float(np.median(mp)),
             "p90": float(np.percentile(mp, 90)), "max": int(mp.max()),
             "hist": {str(int(k)): int(v) for k, v in
                      zip(*np.unique(np.minimum(mp, 20), return_counts=True))}},
         "B8_ORACLE_CURVE_ALL": curve,
         "B8_ORACLE_ALL_AT_50": curve[50]["ALL"],
         "B14_EXPOSURE_TOP50_NODES": round(float(np.sort(sizes)[::-1][:50].sum()), 1),
         "B14_MEAN_EXPOSURE_50_BLOCKS": round(float(sizes.mean() * 50), 1),
         "mean_golds_per_query": round(float(np.mean(per["ngold"])), 3)}
    if ds == "metaqa" and hops is not None and (hops > 0).any():
        R["by_hop"] = {}
        for h in (1, 2, 3):
            m = np.array([hops[qi] == h for qi in range(nq) if gptr[qi + 1] > gptr[qi]])
            if not m.any():
                continue
            mm = mp[m]
            R["by_hop"][f"hop{h}"] = {
                "n": int(m.sum()),
                "B1_GOLD_COMPRESSION": round(float(np.array(per["comp"])[m].mean()), 4),
                "B2_PAIR_COLOCATION": round(float(np.array(per["pair_s"])[m].sum()
                                                  / max(np.array(per["pair_n"])[m].sum(), 1)), 5),
                "B3_ALL_GOLD_CONTAINED": round(float(np.array(per["allc"])[m].mean()), 4),
                "B4_SEED_REQUIRED_COLOCATION": round(float(np.array(per["seedcov"])[m].mean()), 4),
                "B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL":
                    round(float(np.array(per["seedcovn"])[m].mean()), 4),
                "B5_GOLD_EDGE_CONTAINMENT": round(float(np.array(per["ge_in"])[m].sum()
                                                        / max(np.array(per["ge_tot"])[m].sum(), 1)), 4),
                "B7_MIN_PARTITIONS_REQUIRED_mean": round(float(mm.mean()), 4),
                "B8_ORACLE_ALL_AT_50": round(float((mm <= 50).mean()), 4)}
    return R


# ------------------------------------------------------------------ B12 ordinary graph quality
def quality(ds, hard, npart, fams=("E0_STRUCT", "E6_TOPOLOGY_C")):
    N, ST, KN, NX = EP.keysets(ds)
    KEY = {"E0_STRUCT": ST, "E1_NERX_ONLY": NX, "E2_KNN_ONLY": KN,
           "E4_STRUCT_KNN": np.union1d(ST, KN), "E3_STRUCT_NERX": np.union1d(ST, NX),
           "E6_TOPOLOGY_C": np.union1d(np.union1d(ST, NX), KN)}
    sizes = np.bincount(hard, minlength=npart).astype(np.int64)
    out = {}
    for f in fams:
        k = KEY[f]
        u = (k // np.int64(N)).astype(np.int64); v = (k % np.int64(N)).astype(np.int64)
        pu, pv = hard[u], hard[v]
        cut = int((pu != pv).sum()); m = len(k)
        deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.int64)
        vol = np.bincount(hard, weights=deg, minlength=npart)
        intr = np.bincount(pu[pu == pv], minlength=npart).astype(np.float64)
        tot2 = 2.0 * m
        mod = float((intr / max(m, 1) - (vol / max(tot2, 1e-9)) ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = (vol - 2 * intr) / np.maximum(np.minimum(vol, tot2 - vol), 1e-9)
        out[f] = {"edges_undirected": m, "edge_cut": cut,
                  "cut_ratio": round(cut / max(m, 1), 5), "modularity": round(mod, 5),
                  "mean_conductance": round(float(np.nanmean(cond[sizes > 0])), 5),
                  "isolated_blocks_zero_vol": int((vol == 0).sum())}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    spec = sys.argv[2] if len(sys.argv) > 2 else "CURRENT"
    tag = sys.argv[3] if len(sys.argv) > 3 else "P1_METIS_CURRENT"
    hard, npart = load_assignment(ds, spec)
    R = utility(ds, hard, npart)
    R["tag"] = tag
    R["B12_QUALITY"] = quality(ds, hard, npart)
    os.makedirs(f"{OUT}/PARTITION_UTILITY", exist_ok=True)
    fp = f"{OUT}/PARTITION_UTILITY/B_{ds}.json"
    allr = json.load(open(fp)) if os.path.exists(fp) else {}
    allr[tag] = R
    json.dump(allr, open(fp, "w"), indent=1)
    log(f"{ds} [{tag}] -> {fp}")
    for k in ["B1_GOLD_COMPRESSION", "B1_PARTITION_FETCH_SAVING", "B2_PAIR_COLOCATION",
              "B2_PAIR_EXPECTED_RANDOM", "B2_PAIR_COLOCATION_ADJUSTED", "B3_ALL_GOLD_CONTAINED",
              "B3_GOLD_BLOCK_DENSITY", "B4_SEED_REQUIRED_COLOCATION", "B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL",
              "B5_GOLD_EDGE_CONTAINMENT",
              "B8_ORACLE_ALL_AT_50"]:
        log(f"   {k:34s} {R[k]}")
    log("   B7 " + json.dumps(R["B7_MIN_PARTITIONS_REQUIRED"]))
    log("   B12 " + json.dumps(R["B12_QUALITY"]))
    if "by_hop" in R:
        for h, v in R["by_hop"].items():
            log(f"   {h} " + json.dumps(v))
