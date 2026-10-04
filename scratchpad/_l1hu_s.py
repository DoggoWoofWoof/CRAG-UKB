"""L1 HIGH-CEILING RESIDUAL RANKING (Part A-I) -- can query-conditioned STRUCTURE push MetaQA/WebQSP
past a hard ceiling, or is F6's ranking already near the wall?

Part A settles the question BEFORE any new ranker is built: for the exact same H4 cores, exact
same selected-50, exact same candidate universe, compute three oracle ceilings (gold used only
diagnostically, never in a shipped ranker):

  O1_FIXED_K   same uniq_v pool, same K_q, required nodes ranked first -- ranking-alone ceiling
  O2_FULL_1HOP no K_q cutoff, every uniq_v candidate counts as fetched -- reachability ceiling
  O3_CORE_PLUS_1HOP  core members + every uniq_v candidate, budget ignored -- restated for clarity
                      in the spec; IDENTICAL to O2 in this codebase (core members are already
                      unconditionally fetched via got_core, and uniq_v IS "every FULL_C 1-hop
                      candidate") -- verified equal per query, not assumed.

If O1 < target, ranking alone cannot reach it at the current exposure; if O2 < target too, no
amount of 1-hop ranking work can, regardless of exposure -- the ceiling is the candidate universe,
not the ranker.  Both reduce to plain set arithmetic on (required nodes already in got_core,
required nodes in uniq_v, K_q) -- no lexsort/score machinery needed, so `oracle()` is a cheap,
separate, fast pass, run BEFORE the expensive G1-G6 setup below.

Three new structural signals, all reusing cached/static data (no new embeddings, no Modal):

  G1_QUERY_WEIGHTED_SUPPORT  QWS(v) = sum over raw-graph neighbours u of v with u in a selected
                              core's member set: 1/(K0+rank_q(u)) / deg(u).  Scores v by how
                              strongly ITS SUPPORTING NODES were query-activated, not by v's own
                              (possibly poor, for a 3rd-hop answer) direct lexical match.
  G2_CORE_RANK_SUPPORT       CRS(v) = sum over supporting selected cores j: norm_boundary_mass(v,j)
                              / (K0+rank(j)) -- same F2 ingredient, reweighted by how strongly the
                              CORE ITSELF was selected, not just how many cores support v.
  G3_HYPER_COHERENCE         reuses H4_SPLIT_PRESERVE's own hyperedge membership (no rebuild):
                              QSH(v) = sum over v's hyperedges e, over core members u in e:
                              1/(K0+rank_q(u)).  Rewards v for sharing an H4 GROUP with several
                              query-relevant core nodes, not just a pairwise edge to one.
  G4_STRUCT_RRF = RRF(G1,G2,G3).  G5_STRUCT_QUERY_RRF = RRF(G4, cached query-relevance rank).
  G6_STRUCT_PRECEDENCE (SP0-SP3) universal STRUCT-vs-KNN reliability tiers on top of G5, testing
                              whether WebQSP's Part-E KNN-noise finding generalises as a query-local
                              rule instead of a corpus-conditioned drop.  SP1: STRUCT-supported
                              first, tier broken by G5.  SP2: STRUCT support with cnt>=2 (own
                              overall support count, not a STRUCT-only recount -- documented
                              simplification) first.  SP3: a KNN-only candidate is demoted below
                              any STRUCT-supported tier whenever its support count does not exceed
                              the best STRUCT-supported candidate's -- "comparable-or-less support
                              cannot outrank STRUCT" operationalised without a new tunable.

Every ranker: SAME 50 selected cores, SAME FULL_C candidate universe, SAME per-query K_q as F0/F6
-- only the SCORE changes, exactly the R0-R6 discipline this phase inherits.  F6 is reproduced
in-loop (not just loaded) so Part G's rank-comparison has real per-candidate F6 ranks to diff
against, not just F6's ALL_REQUIRED indicator.

  python scratchpad/_l1hu_s.py oracle <ds>
  python scratchpad/_l1hu_s.py oracle_all
  python scratchpad/_l1hu_s.py oracle_report
  python scratchpad/_l1hu_s.py run <ds> [nsample]
  python scratchpad/_l1hu_s.py run_all
  python scratchpad/_l1hu_s.py gate
  python scratchpad/_l1hu_s.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as OVE
import _l1ep_pu as PU
import _l1kn_sub as KS
import _l1hu_hard as HH
import _l1hu_global as GL
import _l1ps_router as RTR

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
K0 = RTR.K0
FAMS = ("O1_STRUCT", "O2_NERX", "O3_KNN")
GDIR = "scratchpad/_l1hu/graphs"
METHODS = ("F6", "G1", "G2", "G3", "G4", "G5", "SP0", "SP1", "SP2", "SP3")
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def _rank_of(order):
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order))
    return r


def _ragged_gather(nodes, ptr, idx):
    """for each node in `nodes`, gather idx[ptr[node]:ptr[node+1]]; fully vectorised (no python loop
    over nodes).  Returns (src_repeated, gathered, sizes) -- sizes lets a caller np.repeat() any
    per-node value into alignment with `gathered` without a second gather."""
    nodes = np.asarray(nodes, np.int64)
    starts = ptr[nodes]
    sizes = (ptr[nodes + 1] - starts).astype(np.int64)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, idx.dtype), sizes
    cum = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    flat = np.repeat(starts, sizes) + (np.arange(total, dtype=np.int64) - np.repeat(cum, sizes))
    return np.repeat(nodes, sizes), idx[flat], sizes


def _part_csr(hard, npart, N):
    """partition id -> sorted member node ids (inverse of hard)."""
    order = np.argsort(hard, kind="stable").astype(np.int64)
    cnt = np.bincount(hard, minlength=npart).astype(np.int64)
    ptr = np.zeros(npart + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, order


def _hyperedge_csrs(ds, N):
    """H4_SPLIT_PRESERVE's own hyperedges (SK famset, the one that fed the shipped partitioner).
    eptr/eidx = hyperedge -> member nodes (eidx[eptr[e]] is the anchor).  hptr/hidx = the transpose,
    node -> containing hyperedge ids."""
    z = np.load("%s/%s__H4_SPLIT_PRESERVE__SK.npz" % (GDIR, ds))
    eptr = z["eptr"].astype(np.int64)
    eidx = z["eidx"].astype(np.int64)
    n_edges = len(eptr) - 1
    edge_of_pin = np.repeat(np.arange(n_edges, dtype=np.int64), np.diff(eptr))
    order = np.argsort(eidx, kind="stable")
    edge_sorted = edge_of_pin[order]
    cnt = np.bincount(eidx, minlength=N).astype(np.int64)
    hptr = np.zeros(N + 1, np.int64)
    hptr[1:] = np.cumsum(cnt)
    return eptr, eidx, hptr, edge_sorted, n_edges


def _setup(ds, log=log):
    hard, npart, N = GL.load_core(ds)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(nq)]

    pairs, mass = OV.boundary_mass(ds, hard, TAG, "O4_FULL_C", N, npart, log)
    bptr, bidx, bmass = GL.block_mass_csr(pairs, mass, npart, N)
    bmax = GL.block_max_mass(bptr, bmass, npart)

    f0_pairs = OV.bounded_pairs(pairs, mass, hard, npart, N, BETA_REF)
    f0_bptr_, f0_bidx_ = OV.to_block_csr(f0_pairs, npart, N)

    ret_rrf = np.asarray(z["ret_rrf"])
    return dict(hard=hard, npart=npart, N=N, f650=f650, need=need, hops=hops, nq=nq,
               bptr=bptr, bidx=bidx, bmass=bmass, bmax=bmax,
               f0_bptr_=f0_bptr_, f0_bidx_=f0_bidx_, ret_rrf=ret_rrf)


# ======================================================================= PART A -- oracle ceilings
def oracle(ds, log=log):
    E = _setup(ds, log)
    hard, npart, N = E["hard"], E["npart"], E["N"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx = E["bptr"], E["bidx"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    SEL = E["f650"]
    log("  %s: oracle setup done, entering per-query loop" % ds)

    ind = {k: np.zeros(nq, np.int8) for k in ("O1", "O2", "O3")}
    kq_arr = np.zeros(nq, np.int64)
    over_budget = np.zeros(nq, np.int64)
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 400 == 0:
            log("  %s: %d/%d, %.1fs" % (ds, qi, nq, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        S = SEL[qi]; Ss = set(S)
        selmask[:] = False; selmask[np.asarray(S, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}
        req_rem = nd_set - got_core
        if not req_rem:
            ind["O1"][qi] = ind["O2"][qi] = ind["O3"][qi] = 1
            continue

        cv = []
        for j in S:
            s, e = bptr[j], bptr[j + 1]
            if e > s:
                cv.append(bidx[s:e])
        pool_arr = np.concatenate(cv).astype(np.int64) if cv else np.zeros(0, np.int64)
        req_arr = np.fromiter(req_rem, np.int64, count=len(req_rem))
        in_pool = np.isin(req_arr, pool_arr)
        n_in_pool = int(in_pool.sum())
        n_absent = len(req_arr) - n_in_pool

        if len(f0_bidx_):
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in S]))
            K_new = u0[~selmask[hard[u0.astype(np.int64)]]]
        else:
            K_new = np.zeros(0, np.int64)
        K_q = len(K_new)
        kq_arr[qi] = K_q

        o2 = (n_absent == 0)
        o1 = o2 and (n_in_pool <= K_q)
        ind["O1"][qi] = int(o1); ind["O2"][qi] = int(o2); ind["O3"][qi] = int(o2)
        if o2 and not o1:
            over_budget[qi] = n_in_pool - K_q
    log("  %s: oracle done %d queries in %.1fs" % (ds, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    rec = {"ds": ds, "nq": nq,
          "O1_FIXED_K_ORACLE": round(float(ind["O1"][ok].mean()), 4),
          "O2_FULL_1HOP_ORACLE": round(float(ind["O2"][ok].mean()), 4),
          "O3_SELECTED_CORE_PLUS_1HOP_ORACLE": round(float(ind["O3"][ok].mean()), 4),
          "O2_EQUALS_O3": bool(np.array_equal(ind["O2"], ind["O3"])),
          "MEAN_K_q": round(float(kq_arr[ok].mean()), 1),
          "BUDGET_LIMITED_QUERIES": int((over_budget > 0).sum()),
          "MEAN_EXCESS_WHEN_BUDGET_LIMITED": (round(float(over_budget[over_budget > 0].mean()), 2)
                                              if (over_budget > 0).any() else 0),
          "_ind_O1": ind["O1"].tolist(), "_ind_O2": ind["O2"].tolist()}
    if ds == "metaqa":
        h = np.asarray(hops)[:nq]
        rec["by_hop"] = {}
        for hk in (1, 2, 3):
            mm = (h == hk) & ok
            if mm.any():
                rec["by_hop"]["hop%d" % hk] = {"n": int(mm.sum()),
                                              "O1": round(float(ind["O1"][mm].mean()), 4),
                                              "O2": round(float(ind["O2"][mm].mean()), 4)}
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    fp = "%s/ceiling/O_%s.json" % (OUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def oracle_report():
    rows = []
    for ds in HH.DS:
        fp = "%s/ceiling/O_%s.json" % (OUT, ds)
        if os.path.exists(fp):
            rows.append(json.load(open(fp)))
    for r in rows:
        print("%-16s O1_FIXED_K=%.4f O2_FULL_1HOP=%.4f O3==O2:%s meanKq=%s budget_limited_q=%d"
              % (r["ds"], r["O1_FIXED_K_ORACLE"], r["O2_FULL_1HOP_ORACLE"], r["O2_EQUALS_O3"],
                 r["MEAN_K_q"], r["BUDGET_LIMITED_QUERIES"]))
        if "by_hop" in r:
            for hk, v in r["by_hop"].items():
                print("    %s n=%-5d O1=%.4f O2=%.4f" % (hk, v["n"], v["O1"], v["O2"]))
    mq = next((r for r in rows if r["ds"] == "metaqa"), None)
    wq = next((r for r in rows if r["ds"] == "webqsp"), None)
    if mq:
        print("\nSTOP RULE: MetaQA O1_FIXED_K=%.4f %s .90" %
              (mq["O1_FIXED_K_ORACLE"], ">=" if mq["O1_FIXED_K_ORACLE"] >= 0.90 else "<"))
    if wq:
        print("STOP RULE: WebQSP O1_FIXED_K=%.4f %s .95" %
              (wq["O1_FIXED_K_ORACLE"], ">=" if wq["O1_FIXED_K_ORACLE"] >= 0.95 else "<"))
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    json.dump(rows, open("%s/ceiling/O_REPORT.json" % OUT, "w"), indent=1)
    return rows


# ================================================================= PART B-G -- structural rankers
def run(ds, nsample=None, log=log):
    E = _setup(ds, log)
    hard, npart, N = E["hard"], E["npart"], E["N"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["f650"]

    fam_bptr, fam_bidx = {}, {}
    for fam in FAMS:
        fp_, fm_ = OV.boundary_mass(ds, hard, TAG, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = _part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = _hyperedge_csrs(ds, N)
    HW = np.zeros(n_edges, np.float64)
    log("  %s: G1-G6 setup done (union CSR, part CSR, %s hyperedges), entering per-query loop"
        % (ds, format(n_edges, ",")))

    n_loop = min(nq, nsample) if nsample else nq
    rows_ = {m: {"allf": np.zeros(nq, np.int8), "expo": np.zeros(nq, np.int64)} for m in METHODS}
    miss = {"n": 0, "moved": {m: 0 for m in METHODS}, "pct_sum": {m: 0.0 for m in METHODS},
           "queries": set()}
    by_hop = {hk: {"n": 0, "moved": {m: 0 for m in METHODS}, "queries": set()} for hk in (1, 2, 3)}

    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(n_loop):
        if qi and qi % 200 == 0:
            log("  %s: %d/%d queries, %.1fs elapsed" % (ds, qi, n_loop, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        S = SEL[qi]; Ss = set(S)
        selmask[:] = False; selmask[np.asarray(S, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}

        # ---- FULL_C candidate gather, with per-pair core-RANK tracked in parallel to core id ----
        cv, cj, cm, crk = [], [], [], []
        for rk_, j in enumerate(S):
            s, e = bptr[j], bptr[j + 1]
            if e > s:
                cv.append(bidx[s:e]); cj.append(np.full(e - s, j, np.int64))
                cm.append(bmass[s:e]); crk.append(np.full(e - s, rk_, np.int64))
        if cv:
            V = np.concatenate(cv); J = np.concatenate(cj); Mm = np.concatenate(cm)
            Rk = np.concatenate(crk)
            keep = ~selmask[hard[V.astype(np.int64)]]
            V, J, Mm, Rk = V[keep], J[keep], Mm[keep], Rk[keep]
        else:
            V = np.zeros(0, np.int64)

        if len(f0_bidx_):
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in S]))
            K_new = u0[~selmask[hard[u0.astype(np.int64)]]]
        else:
            K_new = np.zeros(0, np.int64)
        K_q = len(K_new)

        if K_q == 0 or not len(V):
            for m in METHODS:
                rows_[m]["allf"][qi] = int(len(got_core) == len(nd))
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)
        k = min(K_q, nU)

        # ---- family provenance: which families support v at all (union over selected cores) ----
        fam_sets = {}
        for fam in FAMS:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in S if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        struct_supported = np.isin(uniq_v, fam_sets["O1_STRUCT"])
        knn_only = (np.isin(uniq_v, fam_sets["O3_KNN"]) & ~struct_supported
                   & ~np.isin(uniq_v, fam_sets["O2_NERX"]))

        # ---- query relevance rank: cached Dense+SPLADE RRF node continuation ----
        rrf_row = ret_rrf[qi]
        qrank_of = {int(rrf_row[r]): r for r in range(len(rrf_row)) if rrf_row[r] >= 0}
        qr = np.array([qrank_of.get(int(v), -1) for v in uniq_v], np.int64)

        # ---- F6 reproduced in-loop (this phase's reference) ----
        order3 = np.lexsort((uniq_v, -f2_score, -cnt))
        rank3 = _rank_of(order3)
        rrf6 = 1.0 / (K0 + rank3) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        rank6 = _rank_of(np.lexsort((uniq_v, -f2_score, -rrf6)))

        # ================= G2: core-rank-weighted support =================
        g2_terms = (Ms / bmax[Js]) / (K0 + Rks)
        g2_score = np.add.reduceat(g2_terms, start)
        rank_g2 = _rank_of(np.lexsort((uniq_v, -f2_score, -g2_score)))

        # ---- shared setup for G1/G3: core-member nodes of the 50 selected cores + their q-weight
        U_all = np.concatenate([part_idx[part_ptr[j]:part_ptr[j + 1]] for j in S])
        w_u_num = np.array([1.0 / (K0 + qrank_of[int(u)]) if int(u) in qrank_of else 0.0
                            for u in U_all.tolist()])

        # ================= G1: query-weighted neighbour support =================
        deg_u = np.maximum(udeg[U_all].astype(np.float64), 1.0)
        contrib_u = w_u_num / deg_u
        _, nbrs, sizes_n = _ragged_gather(U_all, uptr, uidx)
        contrib_rep = np.repeat(contrib_u, sizes_n)
        keep_n = ~selmask[hard[nbrs.astype(np.int64)]]
        nbrs_k, contrib_k = nbrs[keep_n], contrib_rep[keep_n]
        pos = np.searchsorted(uniq_v, nbrs_k)
        valid = (pos < nU) & (uniq_v[np.minimum(pos, nU - 1)] == nbrs_k)
        g1_score = np.bincount(pos[valid], weights=contrib_k[valid], minlength=nU)
        rank_g1 = _rank_of(np.lexsort((uniq_v, -f2_score, -g1_score)))

        # ================= G3: hyperedge coherence =================
        _, e_h, sizes_h = _ragged_gather(U_all, hptr, hidx)
        touched_e = np.zeros(0, np.int64)
        if len(e_h):
            w_rep_h = np.repeat(w_u_num, sizes_h)
            touched_e, inv = np.unique(e_h, return_inverse=True)
            HW[touched_e] = np.bincount(inv, weights=w_rep_h, minlength=len(touched_e))
        _, e_v, sizes_v = _ragged_gather(uniq_v, hptr, hidx)
        if len(e_v):
            cand_idx_rep = np.repeat(np.arange(nU, dtype=np.int64), sizes_v)
            g3_score = np.bincount(cand_idx_rep, weights=HW[e_v], minlength=nU)
        else:
            g3_score = np.zeros(nU)
        if len(touched_e):
            HW[touched_e] = 0.0
        rank_g3 = _rank_of(np.lexsort((uniq_v, -f2_score, -g3_score)))

        # ================= G4: RRF(G1,G2,G3)   G5: RRF(G4, cached query rank) =================
        rrf4 = 1.0 / (K0 + rank_g1) + 1.0 / (K0 + rank_g2) + 1.0 / (K0 + rank_g3)
        rank_g4 = _rank_of(np.lexsort((uniq_v, -f2_score, -rrf4)))
        rrf5 = 1.0 / (K0 + rank_g4) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        rank_g5 = _rank_of(np.lexsort((uniq_v, -f2_score, -rrf5)))

        # ================= G6 (SP0-SP3): universal STRUCT-vs-KNN precedence, tie-break = G5 =====
        rank_sp0 = rank_g5
        tier1 = (~struct_supported).astype(np.int64)
        rank_sp1 = _rank_of(np.lexsort((rank_g5, tier1)))
        multi_struct = cnt * struct_supported
        tier2 = (multi_struct < 2).astype(np.int64)
        rank_sp2 = _rank_of(np.lexsort((rank_g5, tier2)))
        max_struct_support = int(cnt[struct_supported].max()) if struct_supported.any() else 0
        demote = (knn_only & (cnt <= max_struct_support)).astype(np.int64)
        rank_sp3 = _rank_of(np.lexsort((rank_g5, demote)))

        ranks = {"F6": rank6, "G1": rank_g1, "G2": rank_g2, "G3": rank_g3, "G4": rank_g4,
                "G5": rank_g5, "SP0": rank_sp0, "SP1": rank_sp1, "SP2": rank_sp2, "SP3": rank_sp3}
        for m, rk in ranks.items():
            top = uniq_v[rk < k]
            gotX = got_core | (set(top.tolist()) & nd_set)
            rows_[m]["allf"][qi] = int(len(gotX) == len(nd))
            rows_[m]["expo"][qi] = k

        # ---- Part G: rank movement for nodes still missing under F6, hop-stratified for metaqa ----
        f6_top = uniq_v[rank6 < k]
        got_f6 = got_core | (set(f6_top.tolist()) & nd_set)
        missing_f6 = [x for x in nd if x not in got_f6]
        if missing_f6:
            idx_of = {int(v): i for i, v in enumerate(uniq_v.tolist())}
            h_this = int(np.asarray(hops)[qi]) if ds == "metaqa" and hops is not None else None
            any_m = False
            for x in missing_f6:
                if x not in idx_of:
                    continue
                any_m = True
                miss["n"] += 1
                i_ = idx_of[x]
                for m, rk in ranks.items():
                    moved = int(rk[i_] < k)
                    miss["moved"][m] += moved
                    miss["pct_sum"][m] += 1.0 - rk[i_] / max(nU - 1, 1)
                    if h_this in (1, 2, 3):
                        by_hop[h_this]["moved"][m] += moved
            if any_m:
                miss["queries"].add(qi)
                if h_this in (1, 2, 3):
                    by_hop[h_this]["n"] += sum(1 for x in missing_f6 if x in idx_of)
                    by_hop[h_this]["queries"].add(qi)

    log("  %s: done %d queries in %.1fs" % (ds, n_loop, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    if nsample:
        ok = ok & (np.arange(nq) < n_loop)
    hops_arg = hops if ds == "metaqa" else None
    cells = {}
    for m in METHODS:
        allf = rows_[m]["allf"]
        rec = {"ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4), "_ind_ALL": allf.tolist()}
        if hops_arg is not None:
            h = np.asarray(hops_arg)[:nq]
            for hk in (1, 2, 3):
                mm = (h == hk) & ok
                if mm.any():
                    rec["hop%d_ALL_REQUIRED_FETCHED" % hk] = round(float(allf[mm].mean()), 4)
        cells[m] = rec

    miss_queries = sorted(miss["queries"])
    rescue = {m: (round(float(np.mean([cells[m]["_ind_ALL"][qi] for qi in miss_queries])), 4)
                 if miss_queries else None) for m in METHODS}
    by_hop_out = {}
    if ds == "metaqa":
        for hk in (1, 2, 3):
            qset = sorted(by_hop[hk]["queries"])
            by_hop_out["hop%d" % hk] = {
                "n_missing_nodes": by_hop[hk]["n"], "n_missing_queries": len(qset),
                "frac_moved_into_Kq": {m: (round(by_hop[hk]["moved"][m] / by_hop[hk]["n"], 4)
                                           if by_hop[hk]["n"] else None) for m in METHODS},
                "query_rescue_conversion": {m: (round(float(np.mean(
                    [cells[m]["_ind_ALL"][qi] for qi in qset])), 4) if qset else None)
                                            for m in METHODS}}

    rec = {"ds": ds, "N": N, "npart": npart, "nq": nq, "nsample": nsample, "CELLS": cells,
          "MISSING_UNDER_F6": {"n_missing_nodes": miss["n"], "n_missing_queries": len(miss_queries),
                               "frac_moved_into_Kq": {m: (round(miss["moved"][m] / miss["n"], 4)
                                                          if miss["n"] else None) for m in METHODS},
                               "mean_rank_percentile": {m: (round(miss["pct_sum"][m] / miss["n"], 4)
                                                            if miss["n"] else None) for m in METHODS},
                               "query_rescue_conversion": rescue,
                               "by_hop": by_hop_out if by_hop_out else None}}
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    tag = ("S_%s_n%d.json" % (ds, nsample)) if nsample else ("S_%s.json" % ds)
    fp = "%s/ceiling/%s" % (OUT, tag)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def gate():
    recs = {}
    for ds in HH.DS:
        fp = "%s/ceiling/S_%s.json" % (OUT, ds)
        if os.path.exists(fp):
            recs[ds] = json.load(open(fp))
    if not recs:
        print("no ceiling/S_*.json results yet"); return {}

    vs_f6 = {}
    for m in ("G1", "G2", "G3", "G4", "G5", "SP0", "SP1", "SP2", "SP3"):
        rows_out = []
        for ds, rec in recs.items():
            f6 = rec["CELLS"]["F6"]; mm = rec["CELLS"][m]
            gd, ls, p = HH.mcnemar(f6["_ind_ALL"], mm["_ind_ALL"])
            floor = HH.FLOOR.get(ds, 0.0)
            d = round(mm["ALL_REQUIRED_FETCHED"] - f6["ALL_REQUIRED_FETCHED"], 4)
            rows_out.append({"ds": ds, "F6": f6["ALL_REQUIRED_FETCHED"],
                             m: mm["ALL_REQUIRED_FETCHED"], "delta": d,
                             "gained": gd, "lost": ls, "p": p,
                             "sig": bool(p < 0.05 and abs(d) > floor)})
        deltas = [r["delta"] for r in rows_out]
        regs = [r["ds"] for r in rows_out if r["sig"] and r["delta"] < 0]
        vs_f6[m] = {"per_corpus": rows_out, "worst_delta": round(min(deltas), 4),
                   "macro_delta": round(float(np.mean(deltas)), 4), "sig_regressions": regs,
                   "PROMOTABLE": bool(not regs and (min(deltas) > 0 or float(np.mean(deltas)) > 0))}

    missing = {ds: rec["MISSING_UNDER_F6"] for ds, rec in recs.items()}
    out = {"VS_F6": vs_f6, "MISSING_UNDER_F6": missing, "n_corpora": len(recs), "corpora": sorted(recs)}
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    json.dump(out, open("%s/ceiling/S_GATE.json" % OUT, "w"), indent=1)
    return out


def report():
    out = gate()
    if not out:
        return out
    for m in ("G1", "G2", "G3", "G4", "G5", "SP0", "SP1", "SP2", "SP3"):
        print("\n=== %s vs F6 ===" % m)
        for r in out["VS_F6"][m]["per_corpus"]:
            print("  %-16s F6=%.4f %s=%.4f delta=%+.4f gained=%-4d lost=%-4d p=%-10s sig=%s"
                  % (r["ds"], r["F6"], m, r[m], r["delta"], r["gained"], r["lost"], r["p"], r["sig"]))
        v = out["VS_F6"][m]
        print("  worst=%+.4f macro=%+.4f sig_regressions=%s PROMOTABLE=%s"
              % (v["worst_delta"], v["macro_delta"], v["sig_regressions"], v["PROMOTABLE"]))
    print("\n=== MISSING_UNDER_F6 (rescue by method) ===")
    for ds, h in out["MISSING_UNDER_F6"].items():
        print("  %-16s n_missing=%d n_q=%d rescue=%s"
              % (ds, h["n_missing_nodes"], h["n_missing_queries"], h["query_rescue_conversion"]))
        if h.get("by_hop"):
            for hk, v in h["by_hop"].items():
                print("      %s n=%d rescue=%s" % (hk, v["n_missing_nodes"], v["query_rescue_conversion"]))
    return out


# ============================================================ PART I -- feature-separability audit
# Conditional: only if oracle >= target but the best tested ranker (G1-G6) falls well short.  Pools
# (query, candidate) rows over the identified hard residual and reports per-feature AUC/enrichment
# for distinguishing still-missing REQUIRED candidates from the rest of the pool.  Gold is used only
# to measure separability -- no model is trained here.
PART_I_FEATS = ("f2", "cnt", "g1", "g2", "g3", "qr_dense", "qr_splade", "qr_rrf",
               "deg", "best_core_rank", "n_families", "has_struct")


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def part_i(ds="metaqa", hops_filter=(3,), log=log):
    E = _setup(ds, log)
    hard, npart, N = E["hard"], E["npart"], E["N"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    SEL = E["f650"]

    fam_bptr, fam_bidx = {}, {}
    for fam in FAMS:
        fp_, fm_ = OV.boundary_mass(ds, hard, TAG, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = _part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = _hyperedge_csrs(ds, N)
    HW = np.zeros(n_edges, np.float64)

    z, meta, C, ctxs, base50, f650_, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    ret_rrf = np.asarray(z["ret_rrf"])
    ret_dense = np.asarray(z["ret_dense"])
    ret_splade = np.asarray(z["ret_splade"])

    h = np.asarray(hops)[:nq] if hops is not None else np.full(nq, -1)
    qidxs = [qi for qi in range(nq) if (hops_filter is None or int(h[qi]) in hops_filter) and need[qi]]
    log("  %s: part_i scoped to %d queries (hops_filter=%s)" % (ds, len(qidxs), hops_filter))

    feat_vals = {f: [] for f in PART_I_FEATS}
    labels = []
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for ii, qi in enumerate(qidxs):
        if ii and ii % 100 == 0:
            log("  %s: part_i %d/%d, %.1fs" % (ds, ii, len(qidxs), time.time() - t0))
        nd = need[qi]
        nd_set = set(nd)
        S = SEL[qi]; Ss = set(S)
        selmask[:] = False; selmask[np.asarray(S, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}
        req_rem = nd_set - got_core
        if not req_rem:
            continue

        cv, cj, cm, crk = [], [], [], []
        for rk_, j in enumerate(S):
            s, e = bptr[j], bptr[j + 1]
            if e > s:
                cv.append(bidx[s:e]); cj.append(np.full(e - s, j, np.int64))
                cm.append(bmass[s:e]); crk.append(np.full(e - s, rk_, np.int64))
        if not cv:
            continue
        V = np.concatenate(cv); J = np.concatenate(cj); Mm = np.concatenate(cm); Rk = np.concatenate(crk)
        keep = ~selmask[hard[V.astype(np.int64)]]
        V, J, Mm, Rk = V[keep], J[keep], Mm[keep], Rk[keep]
        if not len(V):
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        best_core_rank = np.minimum.reduceat(Rks, start)
        nU = len(uniq_v)

        fam_sets = {}
        for fam in FAMS:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in S if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        has_struct = np.isin(uniq_v, fam_sets["O1_STRUCT"])
        n_families = (has_struct.astype(np.int64) + np.isin(uniq_v, fam_sets["O2_NERX"])
                     + np.isin(uniq_v, fam_sets["O3_KNN"]))

        def _rankmap(row):
            return {int(row[r]): r for r in range(len(row)) if row[r] >= 0}
        qrank_rrf = _rankmap(ret_rrf[qi]); qrank_dense = _rankmap(ret_dense[qi])
        qrank_splade = _rankmap(ret_splade[qi])

        def _recip(rm):
            return np.array([1.0 / (K0 + rm[int(v)]) if int(v) in rm else 0.0 for v in uniq_v.tolist()])
        qr_rrf, qr_dense, qr_splade = _recip(qrank_rrf), _recip(qrank_dense), _recip(qrank_splade)

        U_all = np.concatenate([part_idx[part_ptr[j]:part_ptr[j + 1]] for j in S])
        w_u_num = np.array([1.0 / (K0 + qrank_rrf[int(u)]) if int(u) in qrank_rrf else 0.0
                            for u in U_all.tolist()])
        deg_u = np.maximum(udeg[U_all].astype(np.float64), 1.0)
        contrib_u = w_u_num / deg_u
        _, nbrs, sizes_n = _ragged_gather(U_all, uptr, uidx)
        contrib_rep = np.repeat(contrib_u, sizes_n)
        keep_n = ~selmask[hard[nbrs.astype(np.int64)]]
        nbrs_k, contrib_k = nbrs[keep_n], contrib_rep[keep_n]
        pos = np.searchsorted(uniq_v, nbrs_k)
        valid = (pos < nU) & (uniq_v[np.minimum(pos, nU - 1)] == nbrs_k)
        g1_score = np.bincount(pos[valid], weights=contrib_k[valid], minlength=nU)

        g2_terms = (Ms / bmax[Js]) / (K0 + Rks)
        g2_score = np.add.reduceat(g2_terms, start)

        _, e_h, sizes_h = _ragged_gather(U_all, hptr, hidx)
        touched_e = np.zeros(0, np.int64)
        if len(e_h):
            w_rep_h = np.repeat(w_u_num, sizes_h)
            touched_e, inv = np.unique(e_h, return_inverse=True)
            HW[touched_e] = np.bincount(inv, weights=w_rep_h, minlength=len(touched_e))
        _, e_v, sizes_v = _ragged_gather(uniq_v, hptr, hidx)
        if len(e_v):
            cand_idx_rep = np.repeat(np.arange(nU, dtype=np.int64), sizes_v)
            g3_score = np.bincount(cand_idx_rep, weights=HW[e_v], minlength=nU)
        else:
            g3_score = np.zeros(nU)
        if len(touched_e):
            HW[touched_e] = 0.0

        req_arr = np.fromiter(req_rem, np.int64, count=len(req_rem))
        label_v = np.isin(uniq_v, req_arr).astype(np.int64)

        feat_vals["f2"].append(f2_score); feat_vals["cnt"].append(cnt.astype(np.float64))
        feat_vals["g1"].append(g1_score); feat_vals["g2"].append(g2_score); feat_vals["g3"].append(g3_score)
        feat_vals["qr_dense"].append(qr_dense); feat_vals["qr_splade"].append(qr_splade)
        feat_vals["qr_rrf"].append(qr_rrf)
        feat_vals["deg"].append(udeg[uniq_v].astype(np.float64))
        feat_vals["best_core_rank"].append(1.0 / (K0 + best_core_rank))
        feat_vals["n_families"].append(n_families.astype(np.float64))
        feat_vals["has_struct"].append(has_struct.astype(np.float64))
        labels.append(label_v)
    log("  %s: part_i loop done, %.1fs" % (ds, time.time() - t0))

    labels = np.concatenate(labels)
    npos, nneg = int(labels.sum()), int(len(labels) - labels.sum())
    log("  %s: pooled candidates=%d pos(required-missing)=%d neg=%d" % (ds, len(labels), npos, nneg))

    aucs, enrich = {}, {}
    for f in PART_I_FEATS:
        sc = np.concatenate(feat_vals[f])
        r = _rankdata(sc)
        auc = (r[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
        aucs[f] = round(float(auc), 4)
        thresh = np.quantile(sc, 0.90)
        top10 = sc >= thresh
        base_rate = npos / len(labels)
        top_rate = labels[top10].mean() if top10.any() else 0.0
        enrich[f] = round(float(top_rate / base_rate), 3) if base_rate > 0 else None

    rec = {"ds": ds, "hops_filter": list(hops_filter) if hops_filter else None,
          "n_queries": len(qidxs), "n_pairs": len(labels), "n_pos": npos, "n_neg": nneg,
          "AUC": dict(sorted(aucs.items(), key=lambda kv: -kv[1])),
          "ENRICHMENT_TOP10PCT": enrich}
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    tag = "".join(map(str, hops_filter)) if hops_filter else "all"
    fp = "%s/ceiling/I_%s_hop%s.json" % (OUT, ds, tag)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "oracle":
        oracle(a[1])
    elif a and a[0] == "oracle_all":
        for d in HH.DS:
            oracle(d)
    elif a and a[0] == "oracle_report":
        oracle_report()
    elif a and a[0] == "part_i":
        ds_ = a[1] if len(a) > 1 else "metaqa"
        hf_ = tuple(int(x) for x in a[2]) if len(a) > 2 else (3,)
        part_i(ds_, hf_)
    elif a and a[0] == "run":
        run(a[1], int(a[2]) if len(a) > 2 else None)
    elif a and a[0] == "run_all":
        for d in HH.DS:
            run(d)
    elif a and a[0] == "gate":
        gate()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
