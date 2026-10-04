"""STEP 6 -- can candidate generation be a pure cache lookup?

Compares ONLINE structural discovery (real BFS on the partition graph at query time) against the
PRECOMPUTED top-KNBR neighbour tables, on the same target set, and prices both:

    ONLINE      edges touched = sum of partition degrees over the expanded frontier   (> 0)
    PRECOMPUTED edges touched = 0; cost = |seed partitions| x KNBR x depths integer reads

KNBR is a corpus-side knob only -- enlarging it never touches a graph edge at query time, it only
grows the table.  So the question STEP 6 really asks is: how much table do we have to store to
reproduce online discovery?

  python scratchpad/_l1cg_s6.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np, scipy.sparse as sp
import _l1cg_core as CG
import _l1cg_s4 as S4

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
KS = [32, 64, 128]


def topk_table(G, npart, K):
    out = np.full((npart, K), -1, np.int32)
    ind, ptr, dat = G.indices, G.indptr, G.data
    for p in range(npart):
        a, b = ptr[p], ptr[p + 1]
        if b <= a:
            continue
        nb, wt = ind[a:b], dat[a:b]
        o = np.lexsort((nb, -wt))[:K]
        out[p, :len(o)] = nb[o]
    return out


def run_ds(ds, OUT, hop=None):
    S = CG.substrate(ds); T = S["T"]; z = S["z"]; hard = z["hard"]
    G, npart = S4.full_graph(ds)
    deg = np.diff(G.indptr)
    hops = S["hops"]
    sel = [qi for qi in range(S["nq"]) if T[qi]["feasible"] and T[qi]["outside"]
           and (hop is None or hops is None or int(hops[qi]) == hop)]
    tabs = {K: topk_table(G, npart, K) for K in KS}
    tot = sum(len(T[qi]["outside"]) for qi in sel)
    r = {"npart": int(npart), "partition_edges": int(G.nnz), "n_target_queries": len(sel),
         "n_missing_partitions": tot, "mean_partition_degree": round(float(deg.mean()), 1),
         "ONLINE": {}, "PRECOMPUTED": {}}

    on = {1: [0, 0, 0], 2: [0, 0, 0]}       # covered, proposed-size, edges-touched
    pre = {K: {d: [0, 0] for d in (1, 2)} for K in KS}
    for qi in sel:
        seedp = sorted({int(hard[int(v)]) for v in z["seeds"][qi] if v >= 0})
        b50 = S["ctxs"][qi]["base50"]; out = T[qi]["outside"]
        f1 = set()
        e = 0
        for spp in seedp:
            e += int(deg[spp])
            f1 |= set(G.indices[G.indptr[spp]:G.indptr[spp + 1]].tolist())
        f2 = set(f1)
        e2 = e
        for p in list(f1):
            e2 += int(deg[p])
            f2 |= set(G.indices[G.indptr[p]:G.indptr[p + 1]].tolist())
        for d, fr, ee in ((1, f1, e), (2, f2, e2)):
            fr = fr - b50
            on[d][0] += len(out & fr); on[d][1] += len(fr); on[d][2] += ee
        for K in KS:
            t = tabs[K]
            d1 = {int(q) for spp in seedp for q in t[spp] if q >= 0} - b50
            d2 = ({int(q) for spp in seedp for u in t[spp] if u >= 0
                   for q in t[u] if q >= 0} | d1) - b50
            pre[K][1][0] += len(out & d1); pre[K][1][1] += len(d1)
            pre[K][2][0] += len(out & d2); pre[K][2][1] += len(d2)

    n = max(1, len(sel))
    for d in (1, 2):
        r["ONLINE"][f"depth{d}"] = {
            "missing_partition_recall": round(on[d][0] / max(1, tot), 4),
            "mean_proposals": round(on[d][1] / n, 1),
            "mean_frac_of_universe": round(on[d][1] / n / max(1, npart - CG.P), 4),
            "ONLINE_GRAPH_EDGES_TOUCHED_mean": round(on[d][2] / n, 1)}
    for K in KS:
        for d in (1, 2):
            r["PRECOMPUTED"][f"K{K}_depth{d}"] = {
                "missing_partition_recall": round(pre[K][d][0] / max(1, tot), 4),
                "mean_proposals": round(pre[K][d][1] / n, 1),
                "ONLINE_GRAPH_EDGES_TOUCHED": 0,
                "table_bytes": int(npart * K * 4 * d)}
    log(f"   {ds} npart={npart} deg={r['mean_partition_degree']} targets={len(sel)} miss={tot}")
    for k, v in r["ONLINE"].items():
        log(f"     ONLINE      {k}  recall {v['missing_partition_recall']:.4f}  "
            f"prop {v['mean_proposals']:.1f}  edges {v['ONLINE_GRAPH_EDGES_TOUCHED_mean']:.0f}")
    for k, v in r["PRECOMPUTED"].items():
        log(f"     PRECOMPUTED {k}  recall {v['missing_partition_recall']:.4f}  "
            f"prop {v['mean_proposals']:.1f}  edges 0  table {v['table_bytes']/1e6:.2f} MB")
    OUT[f"{ds}_hop{hop}" if hop else ds] = r


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/step6.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    args = sys.argv[1:] or ["metaqa", "webqsp", "2wiki_clean"]
    for ds in args:
        run_ds(ds, OUT, hop=3 if ds == "metaqa" else None)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote step6.json")


if __name__ == "__main__":
    main()
