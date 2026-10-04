"""ROUND C -- STEP 4 precomputation (exact linear diffusion basis + sparse signatures),
STEP 3 P3 partition-bounded node PPR, STEP 7 CAP_B6 vs CAP_P50 decomposition.

KEY IDENTITY (STEP 4).  The inherited iteration p_k = (1-a) S + a p_{k-1} P with p_0 = S is LINEAR
in S, so p_20 = S @ M with

    M = (1-a) * sum_{j=0}^{19} (aP)^j + (aP)^20       (M_k = (1-a) I + a M_{k-1} P, M_0 = I)

M is the diffusion basis: row p of M is exactly the PPR response to a unit personalization at
partition p.  So variant (B) "full precomputed basis" is not an approximation of (A) "online exact
PPR" -- it is the SAME function, and any disagreement is pure floating-point.  Only variant (C),
the top-L truncation of each basis row, is lossy.  M is built and consumed in row blocks so the
full dense basis is never held for the large corpora.

  python scratchpad/_l1pp_c.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1pp_b as B
import _l1kb_core as KB
import _l1kb_router as JR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P = PP.K0, PP.P
LS = (8, 16, 32, 64)
SEED_K = TA.SEED_K


# ------------------------------------------------------------------ STEP 4: diffusion basis
def basis_apply(S, Pmat, npart, Ls=LS, alpha=PP.PPR_ALPHA, iters=PP.PPR_ITERS, blk=512):
    """Stream the exact basis M in row blocks; accumulate the full reconstruction S @ M and every
    top-L truncated reconstruction in one pass.  Also returns the nnz of each sparse basis."""
    PT = sp.csr_matrix(Pmat.T)
    nq = S.shape[0]
    full = np.zeros((nq, npart), np.float64)
    trunc = {L: np.zeros((nq, npart), np.float64) for L in Ls}
    nnz = {L: 0 for L in Ls}
    Sc = np.ascontiguousarray(S)
    t_pre = time.time()
    for a in range(0, npart, blk):
        b = min(a + blk, npart)
        e = np.zeros((npart, b - a))
        e[np.arange(a, b), np.arange(b - a)] = 1.0
        p = e.copy()
        for _ in range(iters):
            p = (1.0 - alpha) * e + alpha * (PT @ p)
        Mb = np.ascontiguousarray(p.T)                 # (b-a, npart) rows a..b of M
        Sb = Sc[:, a:b]
        full += Sb @ Mb
        for L in Ls:
            if L >= npart:
                trunc[L] += Sb @ Mb; nnz[L] += Mb.size; continue
            keep = np.argpartition(-Mb, L - 1, axis=1)[:, :L]
            Mt = np.zeros_like(Mb)
            np.put_along_axis(Mt, keep, np.take_along_axis(Mb, keep, axis=1), axis=1)
            trunc[L] += Sb @ Mt
            nnz[L] += int((Mt > 0).sum())
    return full, trunc, nnz, time.time() - t_pre


def agree(m1, m2, tie, npart, goldp, psz):
    o1 = B.contrib_mass(m1, tie, npart)[1]
    o2 = B.contrib_mass(m2, tie, npart)[1]
    nq = m1.shape[0]
    j = float(np.mean([len(set(o1[q, :P].tolist()) & set(o2[q, :P].tolist())) / P
                       for q in range(nq)]))
    ex = float(np.mean([set(o1[q, :P].tolist()) == set(o2[q, :P].tolist()) for q in range(nq)]))
    return {"top50_jaccard_vs_exact": round(j, 4), "top50_exact_set_agreement": round(ex, 4),
            "max_abs_mass_diff": float(np.abs(m1 - m2).max())}


# ------------------------------------------------------------------ STEP 3: P3 local node PPR
def local_node_ppr(ds, z, hard, npart, nq, adj, cand, alpha=PP.PPR_ALPHA,
                   iters=PP.PPR_ITERS, entry=False):
    """Probability may NOT leave the partition: for candidate partition p we build the subgraph
    induced on p's own nodes and diffuse only inside it.

    Personalization: the query's seed-node masses (Dense+SPLADE node RRF, weight 1/(K0+r)) for the
    seed nodes that fall inside p.  With entry=True the mass is instead placed on p's nodes that
    are 1-hop adjacent to a seed OUTSIDE p -- the entry points through which probability would
    arrive -- which lets P3 score a partition that holds no seed itself.

    Scoring is restricted to the query's CANDIDATE partitions (`cand`), which is what P3 is
    defined over -- it is a partition SCORING signal, not a discovery mechanism, so scoring a
    partition no channel proposed would be meaningless as well as unaffordable.

    Returns the per-(query,partition) summaries as dense (nq, npart) arrays, plus two NO-GRAPH
    controls with identical support, plus the local operator footprint."""
    adj_ptr, adj_idx = adj
    order = np.argsort(hard, kind="stable")
    hs = hard[order]
    starts = np.searchsorted(hs, np.arange(npart), "left")
    ends = np.searchsorted(hs, np.arange(npart), "right")
    members = [order[starts[p]:ends[p]] for p in range(npart)]
    loc = {}                                            # node -> index inside its partition
    for p in range(npart):
        for i, nd in enumerate(members[p]):
            loc[int(nd)] = i

    # per-query seed masses at node level
    rr = z["ret_rrf"]
    qseed = []
    for qi in range(nq):
        d = {}
        for r, nd in enumerate(rr[qi][:PP.K_LOCK]):
            nd = int(nd)
            if nd < 0:
                break
            d[nd] = d.get(nd, 0.0) + 1.0 / (K0 + r)
        qseed.append(d)

    # group the work by partition so each local operator is built at most once
    touch = {}
    for qi in range(nq):
        cq = cand[qi]
        for nd, w in qseed[qi].items():
            p = int(hard[nd])
            if p < 0 or p not in cq:
                continue
            touch.setdefault(p, []).append((qi, nd, w))
        if entry:
            seen = set()
            for nd in list(qseed[qi]):
                hn = int(hard[nd])
                for v in adj_idx[adj_ptr[nd]:adj_ptr[nd + 1]]:
                    p = int(hard[v])
                    if p >= 0 and p != hn and p in cq and (p, int(v)) not in seen:
                        seen.add((p, int(v)))
                        touch.setdefault(p, []).append((qi, int(v), qseed[qi][nd]))

    MX = np.zeros((nq, npart)); TK = np.zeros((nq, npart)); CN = np.zeros((nq, npart))
    RF = np.zeros((nq, npart)); RT = np.zeros((nq, npart)); SU = np.zeros((nq, npart))
    opbytes = 0
    for p, items in touch.items():
        mem = members[p]; n = len(mem)
        if n == 0:
            continue
        idx = {int(nd): i for i, nd in enumerate(mem)}
        rows, cols = [], []
        for i, nd in enumerate(mem):
            for v in adj_idx[adj_ptr[nd]:adj_ptr[nd + 1]]:
                j = idx.get(int(v))
                if j is not None:
                    rows.append(i); cols.append(j)
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
        Ps = sp.csr_matrix((sp.diags(1.0 / d) @ A).T)
        opbytes += n * n * 4
        byq = {}
        for qi, nd, w in items:
            byq.setdefault(qi, {})
            byq[qi][idx[int(nd)]] = byq[qi].get(idx[int(nd)], 0.0) + w
        for qi, sd in byq.items():
            s = np.zeros(n)
            for i, w in sd.items():
                s[i] = w
            tot = s.sum()
            if tot <= 0:
                continue
            s /= tot
            v = s.copy()
            for _ in range(iters):
                v = (1.0 - alpha) * s + alpha * (Ps @ v)
            srt = np.sort(v)[::-1]
            MX[qi, p] = srt[0]
            TK[qi, p] = srt[:SEED_K].sum()
            CN[qi, p] = srt[:SEED_K].sum() / max(1e-12, v.sum())
            RF[qi, p] = float((v > 1e-9).sum()) / n
            RT[qi, p] = tot
            SU[qi, p] = 1.0          # CONTROL: same support, no magnitude, no graph
    return {"max_mass": MX, "topk_mass": TK, "concentration": CN, "reachable_frac": RF,
            "CTRL_seed_mass_NO_GRAPH": RT, "CTRL_support_only_NO_GRAPH": SU}, opbytes


# ------------------------------------------------------------------ STEP 7
def decompose(z, meta, C, goldp, top50, pool, B_val):
    """CAP_P50 / REACH / CAP_B6 / RANKING for the queries this method leaves uncovered."""
    nq = meta["n_dev_queries"]
    br = z["base_rank"]
    out = {"CAP_P50": [], "REACH": [], "CAP_B6": [], "RANKING": []}
    for qi in range(nq):
        fs = set(int(x) for x in top50[qi])
        if goldp[qi] <= fs:
            continue
        if len(goldp[qi]) > P:
            out["CAP_P50"].append(qi); continue
        b50 = set(int(x) for x in br[qi, :P])
        miss_pool = goldp[qi] - pool[qi]
        if miss_pool:
            out["REACH"].append(qi); continue
        if len(goldp[qi] - b50) > B_val:
            out["CAP_B6"].append(qi); continue
        out["RANKING"].append(qi)
    return out


# ------------------------------------------------------------------ driver
def run_ds(ds, OUT):
    z, meta = PP.load(ds)
    nq = meta["n_dev_queries"]; goldp = PP.goldparts(z, meta); psz = z["part_sizes"]
    hops = z["hops"]
    topo = TA.load_topology(ds, log=lambda *a: None)
    ch = PP.channels(ds, z, meta, topo)
    npart, tie, hard = ch["npart"], ch["base_rank_replay"], ch["hard"]
    G = PP.partition_graph(ds, topo, log=log)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    Cd = B.contrib_full(ch["PR_d"], npart); Cs = B.contrib_full(ch["PR_s"], npart)
    base50 = B.direct_top50(Cd + Cs, tie)
    sfull = C["sfull"][(B.M_STRUCT, "S4")]; rfull = C["rfull"][B.M_RET]
    Cstr = B.contrib_lists(sfull, npart, nq)
    S = PP.seed_personalization(z, hard, npart, nq)
    Pmat = PP.transition(G, npart)

    # ---- STEP 4 ----
    t = time.time(); online = B.global_ppr(S, Pmat); t_online = (time.time() - t) / nq
    full, trunc, nnz, t_basis = basis_apply(S, Pmat, npart)
    t = time.time(); _ = S @ np.zeros((npart, npart)) if npart <= 1 else None
    st4 = {"basis_precompute_sec": round(t_basis, 1),
           "full_basis_bytes": int(npart * npart * 4),
           "online_sec_per_query": round(t_online, 6),
           "online_graph_edges_touched_per_query": int(len(G["edge_w"]) * PP.PPR_ITERS),
           "B_full_basis": agree(full, online, tie, npart, goldp, psz),
           "C_sparse": {}}
    for L in LS:
        st4["C_sparse"][f"L{L}"] = dict(
            agree(trunc[L], online, tie, npart, goldp, psz),
            basis_bytes=int(nnz[L] * 8), basis_nnz=int(nnz[L]),
            online_graph_edges_touched_per_query=0)
    OUT.setdefault("STEP4_PRECOMPUTE", {})[ds] = st4

    # ---- STEP 3 P3 ----
    t = time.time()
    cand = [set(int(x) for x in base50[qi]) | set(int(x) for x in sfull[qi])
            | set(int(x) for x in rfull[qi]) for qi in range(nq)]
    p3, opb = local_node_ppr(ds, z, hard, npart, nq, ch["adj"], cand)
    t_p3 = (time.time() - t) / nq
    t = time.time()
    p3e, opbe = local_node_ppr(ds, z, hard, npart, nq, ch["adj"], cand, entry=True)
    t_p3e = (time.time() - t) / nq
    OUT.setdefault("STEP3_P3", {})[ds] = {
        "local_operator_bytes": int(opb), "sec_per_query_seedonly": round(t_p3, 6),
        "sec_per_query_entry": round(t_p3e, 6),
        "mean_partitions_scored_seedonly": round(float((p3["CTRL_seed_mass_NO_GRAPH"] > 0).sum() / nq), 2),
        "mean_partitions_scored_entry": round(float((p3e["CTRL_seed_mass_NO_GRAPH"] > 0).sum() / nq), 2)}

    # ---- variants ----
    Cp1, ord1 = B.contrib_mass(online, tie, npart)
    CpB, _ = B.contrib_mass(trunc[32], tie, npart)
    V = {"L0_BASE": base50,
         "L3_DIRECT_PPRGLOBAL": B.direct_top50(Cd + Cs + Cp1, tie),
         "L3p_DIRECT_PPR_PRECOMP_L32": B.direct_top50(Cd + Cs + CpB, tie)}
    for k, m in p3.items():
        V[f"L5_P3_{k}"] = B.direct_top50(Cd + Cs + B.contrib_mass(m, tie, npart)[0], tie)
    for k, m in p3e.items():
        V[f"L5e_P3ENTRY_{k}"] = B.direct_top50(Cd + Cs + B.contrib_mass(m, tie, npart)[0], tie)
    V["L1_SWAP_F6_B6"] = B.swap_top50(z, meta, C, 6, "F6")

    base = PP.score_top50(V["L0_BASE"], goldp, psz)
    f6 = PP.score_top50(V["L1_SWAP_F6_B6"], goldp, psz)
    res = {}
    for name, top in V.items():
        s = PP.score_top50(top, goldp, psz)
        e = {"ALL": round(s["ALL"], 4), "dALL": round(s["ALL"] - base["ALL"], 4),
             "dALL_vs_F6": round(s["ALL"] - f6["ALL"], 4),
             "vs_BASE": PP.mcnemar(s["ind"], base["ind"]),
             "vs_F6": PP.mcnemar(s["ind"], f6["ind"])}
        if ds == "metaqa":
            e["per_hop"] = {str(h): {"n": int((hops == h).sum()),
                                     "ALL": round(float(s["ind"][hops == h].mean()), 4),
                                     "dALL": round(float(s["ind"][hops == h].mean()
                                                         - base["ind"][hops == h].mean()), 4)}
                            for h in (1, 2, 3) if (hops == h).sum()}
        res[name] = e
        log("%-15s %-30s ALL %.4f  dBASE %+.4f (p %.4f)" % (
            ds, name, e["ALL"], e["dALL"], e["vs_BASE"]["mcnemar_p"]))
    OUT.setdefault("STEP6_ROUNDC", {})[ds] = res

    # ---- STEP 7 ----
    pool_pos = [set(int(x) for x in z["base_rank"][qi]) | set(sfull[qi]) | set(rfull[qi])
                for qi in range(nq)]
    pool_ppr = [pool_pos[qi] | set(np.nonzero(online[qi] > 0)[0].tolist()) for qi in range(nq)]
    st7 = {}
    for tag, pool in (("ROUTER_POOL", pool_pos), ("PLUS_PPR_REACHABLE", pool_ppr)):
        d = decompose(z, meta, C, goldp, V["L1_SWAP_F6_B6"], pool, 6)
        st7[tag] = {k: len(v) for k, v in d.items()}
        if ds == "metaqa":
            st7[tag]["hop3"] = {k: int(sum(1 for q in v if hops[q] == 3)) for k, v in d.items()}
            st7[tag]["hop2"] = {k: int(sum(1 for q in v if hops[q] == 2)) for k, v in d.items()}
    st7["gold_partitions_gt_50"] = int(sum(1 for g in goldp if len(g) > P))
    st7["mean_gold_partitions"] = round(float(np.mean([len(g) for g in goldp])), 2)
    st7["mean_gold_outside_base50"] = round(float(np.mean(
        [len(goldp[qi] - set(int(x) for x in z["base_rank"][qi, :P])) for qi in range(nq)])), 3)
    OUT.setdefault("STEP7_DECOMPOSITION", {})[ds] = st7
    log("%-15s STEP7 %s" % (ds, json.dumps(st7["ROUTER_POOL"])))


def main():
    dsets = sys.argv[1:] or PP.DSETS
    fp = f"{PP.PPD}/diag/round_c.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote round_c.json")


if __name__ == "__main__":
    main()
