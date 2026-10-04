"""ROUND B -- P1 global partition PPR, P2 bounded partition PPR, DIRECT exactly-P50 fusion,
and the B-sweep that removes B=6 as an axis.

Every method emits EXACTLY 50 partitions.  No learned parameter, no dataset identity, no gold at
inference.  Gold is used only to score.

  python scratchpad/_l1pp_b.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1kb_core as KB
import _l1kb_router as JR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P = PP.K0, PP.P
M_STRUCT, M_RET, DEG_CAP = 64, 32, TA.DEG_CAP


# ------------------------------------------------------------------ fusion primitives
def contrib_full(rk, npart):
    """a channel that ranks EVERY partition -> 1/(K0+pos) everywhere."""
    return 1.0 / (K0 + PP.rank_pos(rk, npart).astype(np.float64))


def contrib_lists(lists, npart, nq):
    """a channel that ranks only the partitions it has evidence for; absent -> exactly 0.0.
    This is the frozen F6 per-candidate masking semantics, lifted to a full-corpus fusion."""
    C = np.zeros((nq, npart), np.float64)
    for qi, lst in enumerate(lists):
        for r, p in enumerate(lst):
            C[qi, p] = 1.0 / (K0 + r)
    return C


def contrib_mass(mass, tie, npart, topM=None):
    """a continuous partition score -> rank -> 1/(K0+rank); zero mass contributes exactly 0."""
    fv_in = np.take_along_axis(mass, tie, axis=1)
    idx = np.argsort(-fv_in, axis=1, kind="stable")
    order = np.take_along_axis(tie, idx, axis=1)
    pos = PP.rank_pos(order, npart)
    C = 1.0 / (K0 + pos.astype(np.float64))
    C[mass <= 0] = 0.0
    if topM is not None:
        C[pos >= topM] = 0.0
    return C, order


def direct_top50(fv, tie, k=P):
    """fixed RRF -> EXACT top-50.  `tie` is the canonical base ranking, so ties are broken exactly
    the way TA.rrf_partitions breaks them and the result is deterministic."""
    fv_in = np.take_along_axis(fv, tie, axis=1)
    idx = np.argsort(-fv_in, axis=1, kind="stable")
    return np.take_along_axis(tie, idx, axis=1)[:, :k]


# ------------------------------------------------------------------ P1 / P2
def global_ppr(S, Pmat, alpha=PP.PPR_ALPHA, iters=PP.PPR_ITERS, block=1024):
    PT = sp.csr_matrix(Pmat.T)                      # csr so the sparse@dense kernel is used
    out = np.empty_like(S)
    for a in range(0, S.shape[0], block):
        s = np.ascontiguousarray(S[a:a + block].T)  # (npart, b)
        p = s.copy()
        for _ in range(iters):
            p = (1.0 - alpha) * s + alpha * (PT @ p)
        out[a:a + block] = p.T
    return out


def bounded_ppr(S, G, npart, cand, alpha=PP.PPR_ALPHA, iters=PP.PPR_ITERS, expand=False):
    """P2: a query-local partition graph induced on the query's candidate partitions.
    Returns (mass, mean_candidates, mean_edges)."""
    es, ed, ew = G["edge_src"].astype(np.int64), G["edge_dst"].astype(np.int64), \
        G["edge_w"].astype(np.float64)
    A = sp.csr_matrix((ew, (es, ed)), shape=(npart, npart))
    nq = S.shape[0]
    out = np.zeros_like(S)
    nc = ne = 0
    for qi in range(nq):
        U = np.array(sorted(cand[qi]), np.int64)
        if expand:
            sub = A[U]
            nb = np.unique(sub.indices)
            if len(nb) > DEG_CAP:
                w = np.asarray(A[U].sum(0)).ravel()
                nb = nb[np.argsort(-w[nb], kind="stable")[:DEG_CAP]]
            U = np.union1d(U, nb)
        sub = A[U][:, U]
        d = np.asarray(sub.sum(1)).ravel(); d[d == 0] = 1.0
        Ps = sp.diags(1.0 / d) @ sub
        s = S[qi, U]
        t = s.sum()
        if t <= 0:
            continue
        s = s / t
        PsT = sp.csr_matrix(Ps.T)
        p = s.copy()
        for _ in range(iters):
            p = (1.0 - alpha) * s + alpha * (PsT @ p)
        out[qi, U] = p
        nc += len(U); ne += sub.nnz
    return out, nc / nq, ne / nq


# ------------------------------------------------------------------ swap family (STEP 5A)
def es_select(bnd, chal, spos, rpos, cpos, B):
    """F6 with the DUPLICATED retrieval channel removed: canonical + structural only."""
    cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
    sc = []
    for p in cands:
        s = 1.0 / (K0 + cpos[p]) if p in cpos else 0.0
        if p in spos:
            s += 1.0 / (K0 + spos[p])
        sc.append((-s, cpos.get(p, 10 ** 6), p))
    sc.sort()
    return [p for _, _, p in sc[:B]]


def swap_top50(z, meta, C, B, mode):
    nq = meta["n_dev_queries"]
    ctxs = KB.contexts(z, meta, C, B)
    out = np.empty((nq, P), np.int32)
    for qi, c in enumerate(ctxs):
        if mode == "F6":
            X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        else:
            X = es_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        fs = c["prot"] + list(X)
        assert len(set(fs)) == P
        out[qi] = fs
    return out


# ------------------------------------------------------------------ driver
def run_ds(ds, OUT):
    z, meta = PP.load(ds)
    nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta)
    psz = z["part_sizes"]
    hops = z["hops"]
    topo = TA.load_topology(ds, log=lambda *a: None)
    ch = PP.channels(ds, z, meta, topo)
    npart = ch["npart"]
    tie = ch["base_rank_replay"]
    G = PP.partition_graph(ds, topo, log=log)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    R = {}

    Cd = contrib_full(ch["PR_d"], npart)
    Cs = contrib_full(ch["PR_s"], npart)
    base50 = direct_top50(Cd + Cs, tie)
    parity = bool(np.array_equal(base50, z["base_rank"][:, :P]))

    sfull = C["sfull"][(M_STRUCT, "S4")]
    Cstr = contrib_lists(sfull, npart, nq)

    # ---- seeds + P1 ----
    S = PP.seed_personalization(z, ch["hard"], npart, nq)
    Pmat = PP.transition(G, npart)
    t = time.time(); mass1 = global_ppr(S, Pmat); t_p1 = (time.time() - t) / nq
    Cp1, ord1 = contrib_mass(mass1, tie, npart)
    Cp1M, _ = contrib_mass(mass1, tie, npart, topM=M_STRUCT)

    # ---- P2 bounded ----
    rfull = C["rfull"][M_RET]
    dtop = ch["PR_d"][:, :P]; stop = ch["PR_s"][:, :P]
    cand = [set(int(x) for x in base50[qi]) | set(int(x) for x in dtop[qi])
            | set(int(x) for x in stop[qi]) | set(int(x) for x in sfull[qi])
            | set(int(x) for x in rfull[qi]) | set(np.nonzero(S[qi])[0].tolist())
            for qi in range(nq)]
    t = time.time(); mass2, nc2, ne2 = bounded_ppr(S, G, npart, cand); t_p2 = (time.time() - t) / nq
    Cp2, ord2 = contrib_mass(mass2, tie, npart)
    t = time.time(); mass2e, nc2e, ne2e = bounded_ppr(S, G, npart, cand, expand=True)
    t_p2e = (time.time() - t) / nq
    Cp2e, _ = contrib_mass(mass2e, tie, npart)

    # ---- independence ablation (does the PPR channel carry NEW information?) ----
    b50s = [set(int(x) for x in z["base_rank"][qi, :P]) for qi in range(nq)]
    b200 = [set(int(x) for x in z["base_rank"][qi]) for qi in range(nq)]

    def redund(lists):
        """of the partitions a channel promotes that are NOT already in the canonical top-50,
        what fraction is still inside the canonical top-200?  1.0 == the channel is telling us
        nothing the canonical Dense+SPLADE ranking does not already say."""
        tot = inc = 0
        for qi in range(nq):
            new_ = [p for p in lists[qi][:M_STRUCT] if p not in b50s[qi]]
            tot += len(new_)
            inc += sum(1 for p in new_ if p in b200[qi])
        return round(inc / max(1, tot), 4), round(tot / nq, 2)

    r_ppr, n_ppr = redund([ord1[qi, :M_STRUCT].tolist() for qi in range(nq)])
    r_s4, n_s4 = redund([list(x) for x in sfull])
    r_p2, n_p2 = redund([ord2[qi, :M_STRUCT].tolist() for qi in range(nq)])
    OUT.setdefault("STEP1_INDEPENDENCE", {})[ds] = {
        "PPRglobal_frac_promoted_already_in_canonical_top200": r_ppr,
        "PPRglobal_mean_new_partitions_promoted": n_ppr,
        "PPRbounded_frac_promoted_already_in_canonical_top200": r_p2,
        "S4struct_frac_promoted_already_in_canonical_top200": r_s4,
        "S4struct_mean_new_partitions_promoted": n_s4,
        "mean_jaccard_PPRtop50_vs_BASE50": round(float(np.mean(
            [len(set(ord1[qi, :P].tolist()) & b50s[qi]) / len(set(ord1[qi, :P].tolist()) | b50s[qi])
             for qi in range(nq)])), 4),
        "mean_jaccard_S4top50_vs_BASE50": round(float(np.mean(
            [len(set(sfull[qi][:P]) & b50s[qi]) / len(set(sfull[qi][:P]) | b50s[qi])
             for qi in range(nq)])), 4)}

    # ---- STEP 6 variants ----
    V = {
        "L0_BASE": base50,
        "L2_DIRECT_S4": direct_top50(Cd + Cs + Cstr, tie),
        "L3_DIRECT_PPRGLOBAL": direct_top50(Cd + Cs + Cp1, tie),
        "L3b_DIRECT_PPRGLOBAL_M64": direct_top50(Cd + Cs + Cp1M, tie),
        "L4_DIRECT_PPRBOUNDED": direct_top50(Cd + Cs + Cp2, tie),
        "L4b_DIRECT_PPRBOUNDED_EXP": direct_top50(Cd + Cs + Cp2e, tie),
        "L6_DIRECT_S4_PLUS_PPR": direct_top50(Cd + Cs + Cstr + Cp1, tie),
    }
    for B in (2, 4, 6, 8, 12):
        V[f"L1_SWAP_F6_B{B}"] = swap_top50(z, meta, C, B, "F6")
        V[f"SWAP_ES_B{B}"] = swap_top50(z, meta, C, B, "ES")

    base = PP.score_top50(V["L0_BASE"], goldp, psz)
    f6 = PP.score_top50(V["L1_SWAP_F6_B6"], goldp, psz)
    res = {}
    for name, top in V.items():
        s = PP.score_top50(top, goldp, psz)
        churn = float(np.mean([P - len(set(top[qi].tolist())
                                       & set(V["L0_BASE"][qi].tolist())) for qi in range(nq)]))
        e = {"ALL": round(s["ALL"], 4), "ANY": round(s["ANY"], 4),
             "dALL": round(s["ALL"] - base["ALL"], 4),
             "dALL_vs_F6": round(s["ALL"] - f6["ALL"], 4),
             "vs_BASE": PP.mcnemar(s["ind"], base["ind"]),
             "vs_F6": PP.mcnemar(s["ind"], f6["ind"]),
             "churn_mean": round(churn, 3),
             "scope_mean": round(s["scope_mean"], 1)}
        if ds == "metaqa":
            e["per_hop"] = {}
            for h in (1, 2, 3):
                m = hops == h
                if m.sum():
                    e["per_hop"][str(h)] = {
                        "n": int(m.sum()), "ALL": round(float(s["ind"][m].mean()), 4),
                        "dALL": round(float(s["ind"][m].mean() - base["ind"][m].mean()), 4),
                        "dALL_vs_F6": round(float(s["ind"][m].mean() - f6["ind"][m].mean()), 4),
                        "vs_BASE": PP.mcnemar(s["ind"][m], base["ind"][m]),
                        "vs_F6": PP.mcnemar(s["ind"][m], f6["ind"][m])}
        res[name] = e
        log("%-15s %-26s ALL %.4f  dBASE %+.4f (p %.4f)  dF6 %+.4f (p %.4f)  churn %5.2f"
            % (ds, name, e["ALL"], e["dALL"], e["vs_BASE"]["mcnemar_p"], e["dALL_vs_F6"],
               e["vs_F6"]["mcnemar_p"], e["churn_mean"]))

    OUT.setdefault("RESULTS", {})[ds] = res
    OUT.setdefault("DIRECT_PARITY", {})[ds] = "EXACT" if parity else "MISMATCH"
    OUT.setdefault("STEP11_LATENCY", {})[ds] = {
        "npart": int(npart), "nq": int(nq),
        "P1_global_ppr_sec_per_query": round(t_p1, 6),
        "P1_online_graph_edges_touched_per_query": int(len(G["edge_w"]) * PP.PPR_ITERS),
        "P2_bounded_ppr_sec_per_query": round(t_p2, 6),
        "P2_mean_candidate_partitions": round(nc2, 1),
        "P2_mean_subgraph_edges": round(ne2, 1),
        "P2_online_graph_edges_touched_per_query": int(round(ne2 * PP.PPR_ITERS)),
        "P2exp_sec_per_query": round(t_p2e, 6), "P2exp_mean_candidates": round(nc2e, 1),
        "P2exp_mean_subgraph_edges": round(ne2e, 1)}
    np.savez_compressed(f"{PP.PPD}/ppr/mass_{ds}.npz", S=S.astype(np.float32),
                        mass_global=mass1.astype(np.float32),
                        mass_bounded=mass2.astype(np.float32))
    log("%-15s DIRECT_NOEXTRA_PARITY=%s  P2 cands %.1f edges %.1f" % (
        ds, OUT["DIRECT_PARITY"][ds], nc2, ne2))


def main():
    dsets = sys.argv[1:] or PP.DSETS
    fp = f"{PP.PPD}/diag/round_b.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote round_b.json")


if __name__ == "__main__":
    main()
