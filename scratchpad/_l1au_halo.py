"""PHASE 4/5/6 -- halo family audit under SP1, family-contribution provenance, ranker x family-set
interaction matrix.  One per-query loop, parametrized by (core_lane, pool_variant, kq_mode),
computing methods {F2, F6, G5, SP1} (Phase 6's P0-P3) plus, for the H0_FULL_C arm only, per-candidate
family-provenance bookkeeping (Phase 5).  G1/G2/G3's underlying graph/hyperedge SUBSTRATE stays the
FULL union graph / full SK hyperedges in every arm -- Phase 4/6 ask "does restricting POOL ADMISSION
hurt", not "does restricting the ranking SIGNAL hurt" (that is Phase 7's separate, explicit question).
SP1's struct_supported/knn_only tiering IS family-gated per arm (FAMS_IN_PLAY) so a family excluded
from the pool cannot silently keep steering the ranker through its tiering signal either.

Backward elimination per compute policy: SAFE-lane (shipped) + MATCHED exposure (4B) is the primary,
decision-relevant cell and runs first.  BASE-lane and NATURAL-pool (4A) are added only if Phase 2
says SAFE survives and the primary result leaves something unresolved.  H4_MINUS_STRUCT is explicitly
diagnostic-only (natural mode, no matched variant, per spec).

Pool variants:
  H0_FULL_C        STRUCT+NERX+KNN  (shipped -- reuses existing ceiling/S_{ds}.json, not rerun)
  H1_MINUS_NERX    STRUCT+KNN
  H2_MINUS_KNN     STRUCT+NERX
  H3_STRUCT_ONLY   STRUCT
  H4_MINUS_STRUCT  NERX+KNN          (diagnostic)

  python scratchpad/_l1au_halo.py run <ds> <core_lane> <pool_variant> <kq_mode>
  python scratchpad/_l1au_halo.py run_grid <ds>       # primary sweep: SAFE+MATCHED, H1-H4
  python scratchpad/_l1au_halo.py report
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
import _l1hu_e as HE
import _l1hu_s as S

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
K0 = RTR.K0
AOUT = OUT + "/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = HH.DS
METHODS4 = ("F2", "F6", "G5", "SP1")

POOL_LABEL = {"H1_MINUS_NERX": "STRUCT_KNN", "H2_MINUS_KNN": "STRUCT_NERX",
             "H3_STRUCT_ONLY": "STRUCT_ONLY", "H4_MINUS_STRUCT": "NERX_KNN"}
FAMS_IN_PLAY = {"H0_FULL_C": ("O1_STRUCT", "O2_NERX", "O3_KNN"),
               "H1_MINUS_NERX": ("O1_STRUCT", "O3_KNN"), "H2_MINUS_KNN": ("O1_STRUCT", "O2_NERX"),
               "H3_STRUCT_ONLY": ("O1_STRUCT",), "H4_MINUS_STRUCT": ("O2_NERX", "O3_KNN")}


def _pool_bm(ds, hard, N, npart, pool_variant, log=log, tag=None):
    tag = tag or TAG
    if pool_variant == "H0_FULL_C":
        return OV.boundary_mass(ds, hard, tag, "O4_FULL_C", N, npart, log)
    _, S_, K_, X_ = KS.keysets(ds, log)
    keymap = {"STRUCT_KNN": np.union1d(S_, K_), "STRUCT_NERX": np.union1d(S_, X_),
              "STRUCT_ONLY": S_, "NERX_KNN": np.union1d(K_, X_)}
    label = POOL_LABEL[pool_variant]
    return HE.boundary_mass_keys(ds, hard, N, npart, keymap[label], label, log, tag=tag)


def _setup4(ds, core_lane, pool_variant, kq_mode, log=log, hard=None, npart=None, tag=None):
    """hard/npart/tag let a caller substitute a NON-shipped partition assignment (e.g. the
    P0_S/P2_SN/P3_SKN factorial builds) -- default (all three None) is bit-identical to the
    original shipped-partition-only behavior.  `tag` MUST be distinct per alternate `hard` or the
    boundary-mass caches in OV/HE (keyed by tag, not by the array's contents) will silently return
    another variant's cached mass -- see boundary_mass_keys's docstring.
    """
    if hard is None:
        hard, npart, N = GL.load_core(ds)
        tag = TAG
    else:
        assert tag and tag != TAG, \
            "an explicit hard override needs its own distinct tag (not the shipped TAG/None) " \
            "or the boundary-mass caches will silently return another variant's cached mass"
        hard = np.asarray(hard, np.int64)
        N = len(hard)
        npart = npart if npart is not None else int(hard.max()) + 1
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(nq)]

    pairs, mass = _pool_bm(ds, hard, N, npart, pool_variant, log, tag=tag)
    bptr, bidx, bmass = GL.block_mass_csr(pairs, mass, npart, N)
    bmax = GL.block_max_mass(bptr, bmass, npart)

    if kq_mode == "MATCHED" or pool_variant == "H0_FULL_C":
        kq_pairs, kq_mass = OV.boundary_mass(ds, hard, tag, "O4_FULL_C", N, npart, log)
    else:
        kq_pairs, kq_mass = pairs, mass
    f0_pairs = OV.bounded_pairs(kq_pairs, kq_mass, hard, npart, N, BETA_REF)
    f0_bptr_, f0_bidx_ = OV.to_block_csr(f0_pairs, npart, N)

    ret_rrf = np.asarray(z["ret_rrf"])
    SEL = base50 if core_lane == "BASE" else f650
    return dict(hard=hard, npart=npart, N=N, tag=tag, SEL=SEL, need=need, hops=hops, nq=nq,
               bptr=bptr, bidx=bidx, bmass=bmass, bmax=bmax,
               f0_bptr_=f0_bptr_, f0_bidx_=f0_bidx_, ret_rrf=ret_rrf)


def run4(ds, core_lane, pool_variant, kq_mode, log=log, hard=None, npart=None, tag=None):
    E = _setup4(ds, core_lane, pool_variant, kq_mode, log, hard=hard, npart=npart, tag=tag)
    hard, npart, N, tag = E["hard"], E["npart"], E["N"], E["tag"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["SEL"]
    in_play = FAMS_IN_PLAY[pool_variant]

    fam_bptr, fam_bidx = {}, {}
    for fam in in_play:
        fp_, fm_ = OV.boundary_mass(ds, hard, tag, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = S._part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = S._hyperedge_csrs(ds, N)
    HW = np.zeros(n_edges, np.float64)
    log("  %s[%s/%s/%s]: setup done, entering per-query loop" % (ds, core_lane, pool_variant, kq_mode))

    track_prov = (pool_variant == "H0_FULL_C")
    SIGS = ("STRUCT", "NERX", "KNN", "STRUCT+NERX", "STRUCT+KNN", "NERX+KNN", "STRUCT+NERX+KNN")
    prov = {"req_fetched": {s: 0 for s in SIGS}, "req_total_in_pool": {s: 0 for s in SIGS},
           "distractor_fetched": {s: 0 for s in SIGS}, "distractor_in_pool": {s: 0 for s in SIGS},
           "unique_rescue_queries": {"STRUCT": set(), "NERX": set(), "KNN": set()}}

    rows_ = {m: np.zeros(nq, np.int8) for m in METHODS4}
    cand_count = np.zeros(nq, np.int64)
    kq_arr = np.zeros(nq, np.int64)
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 400 == 0:
            log("  %s[%s/%s/%s]: %d/%d, %.1fs" % (ds, core_lane, pool_variant, kq_mode, qi, nq,
                                                   time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        Sl = SEL[qi]; Ss = set(Sl)
        selmask[:] = False; selmask[np.asarray(Sl, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}

        cv, cj, cm, crk = [], [], [], []
        for rk_, j in enumerate(Sl):
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
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in Sl]))
            K_new = u0[~selmask[hard[u0.astype(np.int64)]]]
        else:
            K_new = np.zeros(0, np.int64)
        K_q = len(K_new)
        kq_arr[qi] = K_q

        if K_q == 0 or not len(V):
            for m in METHODS4:
                rows_[m][qi] = int(len(got_core) == len(nd))
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)
        k = min(K_q, nU)
        cand_count[qi] = nU

        fam_sets = {}
        for fam in in_play:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in Sl if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        has_s = np.isin(uniq_v, fam_sets["O1_STRUCT"]) if "O1_STRUCT" in in_play else np.zeros(nU, bool)
        has_x = np.isin(uniq_v, fam_sets["O2_NERX"]) if "O2_NERX" in in_play else np.zeros(nU, bool)
        has_k = np.isin(uniq_v, fam_sets["O3_KNN"]) if "O3_KNN" in in_play else np.zeros(nU, bool)
        struct_supported = has_s
        knn_only = has_k & ~has_s & ~has_x

        rrf_row = ret_rrf[qi]
        qrank_of = {int(rrf_row[r]): r for r in range(len(rrf_row)) if rrf_row[r] >= 0}
        qr = np.array([qrank_of.get(int(v), -1) for v in uniq_v], np.int64)

        order3 = np.lexsort((uniq_v, -f2_score, -cnt))
        rank3 = S._rank_of(order3)
        rrf6 = 1.0 / (K0 + rank3) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        rank6 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf6)))
        rank_f2 = S._rank_of(np.lexsort((uniq_v, -f2_score)))

        g2_terms = (Ms / bmax[Js]) / (K0 + Rks)
        g2_score = np.add.reduceat(g2_terms, start)
        rank_g2 = S._rank_of(np.lexsort((uniq_v, -f2_score, -g2_score)))

        U_all = np.concatenate([part_idx[part_ptr[j]:part_ptr[j + 1]] for j in Sl])
        w_u_num = np.array([1.0 / (K0 + qrank_of[int(u)]) if int(u) in qrank_of else 0.0
                            for u in U_all.tolist()])
        deg_u = np.maximum(udeg[U_all].astype(np.float64), 1.0)
        contrib_u = w_u_num / deg_u
        _, nbrs, sizes_n = S._ragged_gather(U_all, uptr, uidx)
        contrib_rep = np.repeat(contrib_u, sizes_n)
        keep_n = ~selmask[hard[nbrs.astype(np.int64)]]
        nbrs_k, contrib_k = nbrs[keep_n], contrib_rep[keep_n]
        pos = np.searchsorted(uniq_v, nbrs_k)
        valid = (pos < nU) & (uniq_v[np.minimum(pos, nU - 1)] == nbrs_k)
        g1_score = np.bincount(pos[valid], weights=contrib_k[valid], minlength=nU)
        rank_g1 = S._rank_of(np.lexsort((uniq_v, -f2_score, -g1_score)))

        _, e_h, sizes_h = S._ragged_gather(U_all, hptr, hidx)
        touched_e = np.zeros(0, np.int64)
        if len(e_h):
            w_rep_h = np.repeat(w_u_num, sizes_h)
            touched_e, inv = np.unique(e_h, return_inverse=True)
            HW[touched_e] = np.bincount(inv, weights=w_rep_h, minlength=len(touched_e))
        _, e_v, sizes_v = S._ragged_gather(uniq_v, hptr, hidx)
        if len(e_v):
            cand_idx_rep = np.repeat(np.arange(nU, dtype=np.int64), sizes_v)
            g3_score = np.bincount(cand_idx_rep, weights=HW[e_v], minlength=nU)
        else:
            g3_score = np.zeros(nU)
        if len(touched_e):
            HW[touched_e] = 0.0
        rank_g3 = S._rank_of(np.lexsort((uniq_v, -f2_score, -g3_score)))

        rrf4 = 1.0 / (K0 + rank_g1) + 1.0 / (K0 + rank_g2) + 1.0 / (K0 + rank_g3)
        rank_g4 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf4)))
        rrf5 = 1.0 / (K0 + rank_g4) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        rank_g5 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf5)))

        tier1 = (~struct_supported).astype(np.int64)
        rank_sp1 = S._rank_of(np.lexsort((rank_g5, tier1)))

        ranks = {"F2": rank_f2, "F6": rank6, "G5": rank_g5, "SP1": rank_sp1}
        top_sets = {}
        for m, rk in ranks.items():
            top = uniq_v[rk < k]
            top_sets[m] = set(top.tolist())
            gotX = got_core | (top_sets[m] & nd_set)
            rows_[m][qi] = int(len(gotX) == len(nd))

        if track_prov:
            sig_of = np.where(has_s & has_x & has_k, "STRUCT+NERX+KNN",
                     np.where(has_s & has_x, "STRUCT+NERX", np.where(has_s & has_k, "STRUCT+KNN",
                     np.where(has_x & has_k, "NERX+KNN", np.where(has_s, "STRUCT",
                     np.where(has_x, "NERX", np.where(has_k, "KNN", "")))))))
            sp1_top = top_sets["SP1"]
            req_fetched_this = set()
            for i_, v_ in enumerate(uniq_v.tolist()):
                sig = sig_of[i_]
                if not sig:
                    continue
                is_req = v_ in nd_set
                is_fetched = v_ in sp1_top
                if is_req:
                    prov["req_total_in_pool"][sig] += 1
                    if is_fetched:
                        prov["req_fetched"][sig] += 1
                        req_fetched_this.add((v_, sig))
                else:
                    prov["distractor_in_pool"][sig] += 1
                    if is_fetched:
                        prov["distractor_fetched"][sig] += 1
            for v_, sig in req_fetched_this:
                for fam_key in ("STRUCT", "NERX", "KNN"):
                    if sig == fam_key:
                        prov["unique_rescue_queries"][fam_key].add(qi)
    log("  %s[%s/%s/%s]: done %d queries in %.1fs" % (ds, core_lane, pool_variant, kq_mode, nq,
                                                       time.time() - t0))

    ok = np.array([bool(x) for x in need])
    h = np.asarray(hops)[:nq] if ds == "metaqa" and hops is not None else None
    cells = {}
    for m in METHODS4:
        rec_ = {"ALL_REQUIRED_FETCHED": round(float(rows_[m][ok].mean()), 4), "_ind_ALL": rows_[m].tolist()}
        if h is not None:
            for hk in (1, 2, 3):
                mm = (h == hk) & ok
                if mm.any():
                    rec_["hop%d_ALL_REQUIRED_FETCHED" % hk] = round(float(rows_[m][mm].mean()), 4)
        cells[m] = rec_
    rec = {"ds": ds, "core_lane": core_lane, "pool_variant": pool_variant, "kq_mode": kq_mode,
          "tag": tag, "nq": nq,
          "CELLS": cells,
          "MEAN_CANDIDATES_PER_QUERY": round(float(cand_count[ok].mean()), 2),
          "MEAN_Kq_PER_QUERY": round(float(kq_arr[ok].mean()), 2),
          "runtime_s": round(time.time() - t0, 1), "runtime_ms_per_query": round(
              1000.0 * (time.time() - t0) / max(nq, 1), 3)}
    if track_prov:
        rec["PROVENANCE"] = {
            "req_fetched": prov["req_fetched"], "req_total_in_pool": prov["req_total_in_pool"],
            "distractor_fetched": prov["distractor_fetched"], "distractor_in_pool": prov["distractor_in_pool"],
            "UNIQUE_QUERY_RESCUES_STRUCT": len(prov["unique_rescue_queries"]["STRUCT"]),
            "UNIQUE_QUERY_RESCUES_NERX": len(prov["unique_rescue_queries"]["NERX"]),
            "UNIQUE_QUERY_RESCUES_KNN": len(prov["unique_rescue_queries"]["KNN"])}
    if tag == TAG:
        os.makedirs("%s/halo_family" % AOUT, exist_ok=True)
        fp = "%s/halo_family/P4_%s__%s__%s__%s.json" % (AOUT, ds, core_lane, pool_variant, kq_mode)
    else:
        os.makedirs("%s/halo_family_factorial" % AOUT, exist_ok=True)
        fp = "%s/halo_family_factorial/P12_%s__%s__%s__%s__%s.json" % (
            AOUT, ds, tag, core_lane, pool_variant, kq_mode)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


PRIMARY_GRID = [("H1_MINUS_NERX", "MATCHED"), ("H2_MINUS_KNN", "MATCHED"), ("H3_STRUCT_ONLY", "MATCHED"),
                ("H0_FULL_C", "MATCHED")]


def run_grid(ds, core_lane="SAFE", log=log):
    for pv, kq in PRIMARY_GRID:
        run4(ds, core_lane, pv, kq, log)


def report():
    rows = []
    for ds in DS:
        cell_fp = {}
        for pv, kq in PRIMARY_GRID:
            fp = "%s/halo_family/P4_%s__SAFE__%s__%s.json" % (AOUT, ds, pv, kq)
            if os.path.exists(fp):
                cell_fp[pv] = json.load(open(fp))
        if "H0_FULL_C" not in cell_fp:
            continue
        ref = cell_fp["H0_FULL_C"]["CELLS"]["SP1"]
        row = {"ds": ds, "H0_FULL_C": ref["ALL_REQUIRED_FETCHED"]}
        for pv in ("H1_MINUS_NERX", "H2_MINUS_KNN", "H3_STRUCT_ONLY"):
            if pv not in cell_fp:
                continue
            v = cell_fp[pv]["CELLS"]["SP1"]
            g_, l_, p_ = HH.mcnemar(ref["_ind_ALL"], v["_ind_ALL"])
            d = round(v["ALL_REQUIRED_FETCHED"] - ref["ALL_REQUIRED_FETCHED"], 4)
            floor = HH.FLOOR.get(ds, 0.0)
            row[pv] = {"ALL_REQUIRED": v["ALL_REQUIRED_FETCHED"], "delta_vs_H0": d,
                      "gained": g_, "lost": l_, "p": p_, "sig": bool(p_ < 0.05 and abs(d) > floor),
                      "mean_candidates": cell_fp[pv]["MEAN_CANDIDATES_PER_QUERY"],
                      "runtime_ms_per_query": cell_fp[pv]["runtime_ms_per_query"]}
        if "PROVENANCE" in cell_fp["H0_FULL_C"]:
            row["PROVENANCE"] = cell_fp["H0_FULL_C"]["PROVENANCE"]
        row["interaction_matrix"] = {pv: {m: cell_fp[pv]["CELLS"][m]["ALL_REQUIRED_FETCHED"]
                                          for m in METHODS4} for pv in cell_fp}
        rows.append(row)
    os.makedirs(AOUT, exist_ok=True)
    fp = "%s/PHASE4_5_6_HALO_FAMILY_REPORT.json" % AOUT
    json.dump(rows, open(fp, "w"), indent=1)
    for r in rows:
        print("\n== %s ==  H0(FULL_C,SP1)=%.4f" % (r["ds"], r["H0_FULL_C"]))
        for pv in ("H1_MINUS_NERX", "H2_MINUS_KNN", "H3_STRUCT_ONLY"):
            if pv not in r:
                continue
            v = r[pv]
            print("  %-18s SP1=%.4f delta=%+.4f gained=%-3d lost=%-3d p=%-10s sig=%s  "
                  "cand/q=%.1f  ms/q=%.3f" % (pv, v["ALL_REQUIRED"], v["delta_vs_H0"], v["gained"],
                                              v["lost"], v["p"], v["sig"], v["mean_candidates"],
                                              v["runtime_ms_per_query"]))
        if "PROVENANCE" in r:
            p = r["PROVENANCE"]
            print("  PROVENANCE (H0 pool, SAFE+SP1): required-fetched by signature: %s" % p["req_fetched"])
            print("  UNIQUE_QUERY_RESCUES: STRUCT=%d NERX=%d KNN=%d" % (
                p["UNIQUE_QUERY_RESCUES_STRUCT"], p["UNIQUE_QUERY_RESCUES_NERX"], p["UNIQUE_QUERY_RESCUES_KNN"]))
        print("  interaction matrix (ALL_REQUIRED_FETCHED by pool x ranker):")
        for pv, ms in r["interaction_matrix"].items():
            print("    %-18s %s" % (pv, {m: ms[m] for m in METHODS4}))
    print("\nwrote", fp)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run4(a[1], a[2], a[3], a[4])
    elif a and a[0] == "run_grid":
        run_grid(a[1], a[2] if len(a) > 2 else "SAFE")
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
