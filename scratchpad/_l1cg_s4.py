"""STEP 4 -- what ARE the MetaQA hop3 CURRENT_POOL_LIMIT missing partitions?

For every required-but-unreachable gold partition we measure the full attribute vector the directive
asks for, then group the failures FROM the measurements (no categories invented in advance).

  python scratchpad/_l1cg_s4.py [ds] [hop]
"""
import os, sys, json, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np, scipy.sparse as sp
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1cg_core as CG

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
MAXD = 6          # BFS horizon on the partition graph


def full_graph(ds):
    """the DENSE partition graph (every partition edge), used only for measurement here."""
    hard, mem, npart, adj, deg, _ = TA.load_topology(ds, log=lambda *a: None)
    ap, ai = adj; n = len(hard)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ap))
    A = sp.csr_matrix((np.ones(len(ai), np.float32), (rows, ai.astype(np.int64))), shape=(n, n))
    Mm = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n, dtype=np.int64), hard.astype(np.int64))),
                       shape=(n, npart))
    G = (Mm.T @ A @ Mm).tocsr(); G.setdiag(0); G.eliminate_zeros()
    return G, npart


def bfs(G, src, npart):
    d = np.full(npart, -1, np.int16); d[src] = 0
    fr = np.array([src]); h = 0
    while len(fr) and h < MAXD:
        h += 1
        sl = np.concatenate([np.arange(G.indptr[p], G.indptr[p + 1]) for p in fr])
        nb = np.unique(G.indices[sl]) if len(sl) else np.array([], np.int32)
        nb = nb[d[nb] < 0]
        if not len(nb):
            break
        d[nb] = h; fr = nb
    return d


def rankmap(seq, npart):
    r = np.full(npart, -1, np.int32)
    for i, p in enumerate(seq):
        p = int(p)
        if 0 <= p < npart and r[p] < 0:
            r[p] = i
    return r


def group_of(r):
    """precedence grouping, defined only after looking at the measured attributes."""
    if r["graph_dist_min"] < 0:
        return "1_UNREACHABLE_IN_PARTITION_GRAPH"
    if not r["proposed_by"]:
        return "2_GRAPH_REACHABLE_BUT_NO_FAMILY_PROPOSES_AT_256"
    best = min([x for x in (r["dense_rank"], r["splade_rank"]) if x >= 0] or [10 ** 9])
    if best <= 200:
        return "3_INSIDE_CANONICAL_TOP200_BUT_OUTSIDE_POOL"
    return "4_PROPOSED_ONLY_OUTSIDE_CANONICAL_TOP200"


def main(ds="metaqa", hop=3):
    S = CG.substrate(ds); O = CG.proposals(ds, S); T = S["T"]
    z, meta, nq = S["z"], S["meta"], S["nq"]
    hard = z["hard"]; psize = z["part_sizes"]
    ch = PP.channels(ds, z, meta)
    G, npart = full_graph(ds)
    pdeg = np.diff(G.indptr)
    log(f"{ds} npart={npart} partition-edges={G.nnz}")
    ppr = None
    f = f"{PP.PPD}/ppr/mass_{ds}.npz"
    if os.path.exists(f):
        ppr = np.load(f)["mass_global"]
    hops = S["hops"]
    sel = [qi for qi in range(nq) if T[qi]["feasible"] and T[qi]["outside"]
           and (hops is None or hop is None or int(hops[qi]) == hop)]
    log(f"target queries = {len(sel)}")

    dcache, recs = {}, []
    for qi in sel:
        seedp = sorted({int(hard[int(v)]) for v in z["seeds"][qi] if v >= 0})
        for spp in seedp:
            if spp not in dcache:
                dcache[spp] = bfs(G, spp, npart)
        D = np.stack([dcache[s] for s in seedp]) if seedp else np.full((1, npart), -1, np.int16)
        rd = rankmap(ch["PR_d"][qi], npart); rs = rankmap(ch["PR_s"][qi], npart)
        nd = rankmap(hard[z["ret_dense"][qi]], npart); ns = rankmap(hard[z["ret_splade"][qi]], npart)
        s4 = rankmap(RT.order_struct(RT.struct_aggregate_full(z, qi, hard, 256), "S4"), npart)
        pr = np.full(npart, -1, np.int32)
        if ppr is not None:
            o = np.argsort(-ppr[qi], kind="stable")
            k = int((ppr[qi] > 0).sum()); pr[o[:k]] = np.arange(k)
        prop = {fam: set(O[fam][qi][:256]) for fam in CG.FAMS}
        d1, d2, d3 = set(O["_D1"][qi]), set(O["_D2"][qi]), set(O["_D3"][qi])
        for p in sorted(T[qi]["outside"]):
            dd = D[:, p]; reach = dd[dd >= 0]
            recs.append({
                "qi": int(qi), "part": int(p),
                "dense_rank": int(rd[p]), "splade_rank": int(rs[p]),
                "node_dense_rank": int(nd[p]), "node_splade_rank": int(ns[p]),
                "s4_rank": int(s4[p]), "ppr_rank": int(pr[p]),
                "graph_dist_min": int(reach.min()) if len(reach) else -1,
                "n_seed_partitions": len(seedp),
                "n_seeds_reaching_le3": int(((dd >= 0) & (dd <= 3)).sum()),
                "precomputed_table_depth": 1 if p in d1 else (2 if p in d2 else (3 if p in d3 else -1)),
                "part_degree": int(pdeg[p]), "part_size": int(psize[p]),
                "n_missing_this_query": len(T[qi]["outside"]),
                "proposed_by": sorted(fam for fam in CG.FAMS if p in prop[fam])})
    log(f"records = {len(recs)}")

    def q(k):
        v = np.array([r[k] for r in recs], float)
        v = v[v >= 0]
        if not len(v):
            return {"n": 0, "frac_defined": 0.0}
        return {"n": int(len(v)), "frac_defined": round(len(v) / len(recs), 4),
                "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max())}

    out = {"ds": ds, "hop": hop, "n_target_queries": len(sel), "n_missing_partitions": len(recs),
           "npart": int(npart),
           "ATTRIBUTES": {k: q(k) for k in ["dense_rank", "splade_rank", "node_dense_rank",
                                            "node_splade_rank", "s4_rank", "ppr_rank",
                                            "graph_dist_min", "part_degree", "part_size"]}}
    gc = Counter(group_of(r) for r in recs)
    out["GROUPS_PARTITION_LEVEL"] = {k: {"n": v, "frac": round(v / max(1, len(recs)), 4)}
                                     for k, v in sorted(gc.items())}
    byq = {}
    for r in recs:
        byq.setdefault(r["qi"], []).append(r)
    qg = Counter(max(group_of(r) for r in v) for v in byq.values())
    out["GROUPS_QUERY_LEVEL_worst_partition"] = {k: {"n": v, "frac": round(v / max(1, len(byq)), 4)}
                                                 for k, v in sorted(qg.items())}
    out["DIST_graph_dist_min"] = {str(k): int(v) for k, v in
                                  sorted(Counter(r["graph_dist_min"] for r in recs).items())}
    out["DIST_precomputed_table_depth"] = {str(k): int(v) for k, v in
                                           sorted(Counter(r["precomputed_table_depth"] for r in recs).items())}
    out["DIST_n_missing_per_query"] = {str(k): int(v) for k, v in
                                       sorted(Counter(len(v) for v in byq.values()).items())}
    out["COVERAGE_by_family_at_256"] = {
        fam: round(float(np.mean([fam in r["proposed_by"] for r in recs])), 4) for fam in CG.FAMS}
    out["N_PARTITIONS_PROPOSED_BY_NOTHING"] = int(sum(1 for r in recs if not r["proposed_by"]))
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    tag = f"{ds}_hop{hop}" if hop else ds
    json.dump(out, open(f"{CG.CGD}/diag/step4_{tag}.json", "w"), indent=1)
    json.dump(recs[:4000], open(f"{CG.CGD}/diag/step4_{tag}_records.json", "w"))
    for k, v in out["GROUPS_PARTITION_LEVEL"].items():
        log(f"   {k:52s} {v['n']:6d}  {v['frac']:.4f}")
    log("   dist_min  " + str(out["DIST_graph_dist_min"]))
    log("   tbldepth  " + str(out["DIST_precomputed_table_depth"]))
    log("   fam@256   " + json.dumps(out["COVERAGE_by_family_at_256"]))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "metaqa", int(a[1]) if len(a) > 1 else 3)
