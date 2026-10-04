"""PHASE 7 -- hyperedge coherence (G3) dependence on hyperedge SOURCE family.

The shipped H4 hyperedge export is famset="SK" (STRUCT+KNN only -- see _l1au.py header; NER was
never part of hyperedge construction).  Hyperedges are emitted in family order then anchor order
(_l1hu_build.py docstring + build() loop), so hyperedge ids [0, struct_count) are STRUCT-anchored
and [struct_count, total) are KNN-anchored -- exact split point read from the existing
hypergraph_build/BUILD_MANIFEST.json per_family.STRUCT.hyperedges_out, no rebuild needed.

  C0 full G3 (both families)      C1 STRUCT-derived hyperedges only      C3 KNN-derived only
  C2 NER-derived -- NOT MEASURABLE from the current SK export (no NER hyperedges exist on disk);
     reported as such, not silently computed as zero-and-unremarked.

Masking is done on the SOURCE side only (which hyperedges a query-weighted core member's membership
list is allowed to populate HW from) -- a candidate's own hyperedge list needs no separate mask,
since HW for any hyperedge never populated by an in-family core member stays at its default 0.  Pool
stays FULL_C, cores stay SAFE-selected, fixed exactly as shipped -- only G3's own substrate varies.

Two evaluation modes per C-variant: G3 ALONE (raw signal quality) and the full SP1 pipeline with
G3 computed under that variant (does the final ranker need the richer substrate).

  python scratchpad/_l1au_p7.py run <ds>
  python scratchpad/_l1au_p7.py run_all
  python scratchpad/_l1au_p7.py report
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
import _l1hu_s as S

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
K0 = RTR.K0
AOUT = OUT + "/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = HH.DS
GDIR = "scratchpad/_l1hu/graphs"
VARIANTS = ("C0_FULL", "C1_STRUCT_ONLY", "C3_KNN_ONLY")


def _struct_edge_count(ds):
    m = json.load(open("%s/hypergraph_build/BUILD_MANIFEST.json" % OUT))
    return int(m["%s__H4_SPLIT_PRESERVE__SK" % ds]["per_family"]["STRUCT"]["hyperedges_out"])


def run(ds, log=log):
    E = S._setup(ds, log)
    hard, npart, N = E["hard"], E["npart"], E["N"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["f650"]

    fam_bptr, fam_bidx = {}, {}
    for fam in S.FAMS:
        fp_, fm_ = OV.boundary_mass(ds, hard, TAG, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = S._part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = S._hyperedge_csrs(ds, N)
    struct_split = _struct_edge_count(ds)
    HW = np.zeros(n_edges, np.float64)
    log("  %s: setup done (%s hyperedges, struct_split=%s), entering per-query loop"
        % (ds, format(n_edges, ","), format(struct_split, ",")))

    rows_ = {"G3_%s" % v: np.zeros(nq, np.int8) for v in VARIANTS}
    rows_.update({"SP1_%s" % v: np.zeros(nq, np.int8) for v in VARIANTS})
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 400 == 0:
            log("  %s: %d/%d, %.1fs" % (ds, qi, nq, time.time() - t0))
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

        if K_q == 0 or not len(V):
            for m in rows_:
                rows_[m][qi] = int(len(got_core) == len(nd))
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)
        k = min(K_q, nU)

        fam_sets = {}
        for fam in S.FAMS:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in Sl if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        struct_supported = np.isin(uniq_v, fam_sets["O1_STRUCT"])

        rrf_row = ret_rrf[qi]
        qrank_of = {int(rrf_row[r]): r for r in range(len(rrf_row)) if rrf_row[r] >= 0}
        qr = np.array([qrank_of.get(int(v), -1) for v in uniq_v], np.int64)

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
        _, e_v, sizes_v = S._ragged_gather(uniq_v, hptr, hidx)
        w_rep_h_full = np.repeat(w_u_num, sizes_h) if len(e_h) else np.zeros(0)
        cand_idx_rep = np.repeat(np.arange(nU, dtype=np.int64), sizes_v) if len(e_v) else np.zeros(0, np.int64)

        for variant in VARIANTS:
            if variant == "C0_FULL":
                mask_h = np.ones(len(e_h), bool)
            elif variant == "C1_STRUCT_ONLY":
                mask_h = e_h < struct_split
            else:
                mask_h = e_h >= struct_split
            touched_e = np.zeros(0, np.int64)
            if mask_h.any():
                eh_m, wh_m = e_h[mask_h], w_rep_h_full[mask_h]
                touched_e, inv = np.unique(eh_m, return_inverse=True)
                HW[touched_e] = np.bincount(inv, weights=wh_m, minlength=len(touched_e))
            g3_score = (np.bincount(cand_idx_rep, weights=HW[e_v], minlength=nU)
                       if len(e_v) else np.zeros(nU))
            if len(touched_e):
                HW[touched_e] = 0.0
            rank_g3v = S._rank_of(np.lexsort((uniq_v, -f2_score, -g3_score)))

            top_g3 = uniq_v[rank_g3v < k]
            gotG3 = got_core | (set(top_g3.tolist()) & nd_set)
            rows_["G3_%s" % variant][qi] = int(len(gotG3) == len(nd))

            rrf4 = 1.0 / (K0 + rank_g1) + 1.0 / (K0 + rank_g2) + 1.0 / (K0 + rank_g3v)
            rank_g4 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf4)))
            rrf5 = 1.0 / (K0 + rank_g4) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
            rank_g5 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf5)))
            tier1 = (~struct_supported).astype(np.int64)
            rank_sp1 = S._rank_of(np.lexsort((rank_g5, tier1)))
            top_sp1 = uniq_v[rank_sp1 < k]
            gotSP1 = got_core | (set(top_sp1.tolist()) & nd_set)
            rows_["SP1_%s" % variant][qi] = int(len(gotSP1) == len(nd))
    log("  %s: done %d queries in %.1fs" % (ds, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    cells = {m: {"ALL_REQUIRED_FETCHED": round(float(rows_[m][ok].mean()), 4),
               "_ind_ALL": rows_[m].tolist()} for m in rows_}
    rec = {"ds": ds, "nq": nq, "struct_edge_count": struct_split, "knn_edge_count": n_edges - struct_split,
          "NER_NOTE": "C2 (NER-derived hyperedge coherence) is not measurable: the shipped H4 export "
                     "is famset=SK (STRUCT+KNN only), no NER hyperedges exist on disk. Not computed, "
                     "not assumed zero.",
          "CELLS": cells}
    os.makedirs("%s/hyperedge_family" % AOUT, exist_ok=True)
    fp = "%s/hyperedge_family/P7_%s.json" % (AOUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def run_all():
    for ds in DS:
        run(ds)


def report():
    rows = []
    for ds in DS:
        fp = "%s/hyperedge_family/P7_%s.json" % (AOUT, ds)
        if not os.path.exists(fp):
            continue
        rec = json.load(open(fp))
        c = rec["CELLS"]
        row = {"ds": ds}
        for base in ("G3", "SP1"):
            ref = c["%s_C0_FULL" % base]
            for v in ("C1_STRUCT_ONLY", "C3_KNN_ONLY"):
                cur = c["%s_%s" % (base, v)]
                g_, l_, p_ = HH.mcnemar(ref["_ind_ALL"], cur["_ind_ALL"])
                d = round(cur["ALL_REQUIRED_FETCHED"] - ref["ALL_REQUIRED_FETCHED"], 4)
                floor = HH.FLOOR.get(ds, 0.0)
                row["%s_%s" % (base, v)] = {"val": cur["ALL_REQUIRED_FETCHED"], "delta_vs_C0": d,
                                            "gained": g_, "lost": l_, "p": p_,
                                            "sig": bool(p_ < 0.05 and abs(d) > floor)}
            row["%s_C0_FULL" % base] = ref["ALL_REQUIRED_FETCHED"]
        rows.append(row)
    os.makedirs(AOUT, exist_ok=True)
    fp = "%s/PHASE7_HYPEREDGE_FAMILY_REPORT.json" % AOUT
    json.dump(rows, open(fp, "w"), indent=1)
    for r in rows:
        print("\n== %s ==" % r["ds"])
        for base in ("G3", "SP1"):
            print("  %s: C0_FULL=%.4f" % (base, r["%s_C0_FULL" % base]))
            for v in ("C1_STRUCT_ONLY", "C3_KNN_ONLY"):
                e = r["%s_%s" % (base, v)]
                print("    %-16s val=%.4f delta=%+.4f gained=%-3d lost=%-3d p=%-10s sig=%s"
                      % (v, e["val"], e["delta_vs_C0"], e["gained"], e["lost"], e["p"], e["sig"]))
    print("\nwrote", fp)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
