"""PHASE 13 -- membership vs score-mass: does the online ranking SIGNAL (G1 neighbour substrate, G3
hyperedge substrate) actually need NERX/KNN, independent of whether those families are admitted to
the candidate POOL at all?

Two prior phases each answered half of this and neither combined the two axes:
  Phase 4/6 (H3_STRUCT_ONLY, _l1au_halo.py)  restricts pool MEMBERSHIP to STRUCT only, but G1/G3's
      scoring substrate stays the FULL union in every arm BY DESIGN (see that file's own docstring:
      "G1/G2/G3's underlying graph/hyperedge SUBSTRATE stays the FULL union graph / full SK
      hyperedges in every arm").  So the existing H3_STRUCT_ONLY/SP1 result never actually tested a
      fully-STRUCT-only online path -- only a STRUCT-only-reachable one, scored with full-family
      signal.
  Phase 7 (_l1au_p7.py) restricts G3's hyperedge SOURCE (STRUCT-anchored vs KNN-anchored -- NER has
      no hyperedges in the shipped SK export) but holds pool at FULL_C throughout, and never touches
      G1 at all.
G1's scoring substrate has therefore never been ablated by anything before this phase, and no phase
has combined membership-restriction with scoring-restriction. This phase clears both gaps with one
clean 2x2-ish design:

  M0_FULL_FULL           FULL_C membership, FULL(STRUCT+NERX+KNN) G1/G3 scoring.  Identical to the
                          existing H0_FULL_C/SP1 cell -- REUSED from halo_family/P4_*.json, not
                          recomputed (bit-identical by construction, see run4()'s own H0_FULL_C arm).
  M1_FULL_STRUCT_SCORE    FULL_C membership, G1 neighbour-substrate + G3 hyperedge-substrate BOTH
                          restricted to STRUCT only (membership unchanged from shipped).
  M2_STRUCT_STRUCT        STRUCT_ONLY membership (H3_STRUCT_ONLY pool) + STRUCT-only G1/G3 scoring
                          -- the fully-honest pure-STRUCT online path Phase 4 never actually tested.
  M3_FULL_SIMPLE_STRUCT   FULL_C membership, ranking collapsed to F2 + SP1's own STRUCT-tiering only
                          (drops G1/G2/G3/G4/G5's entire signal-stacking machinery).  "Simplest
                          already-validated STRUCT-based ranking stack": F2 is Phase 6's long-
                          validated base score, STRUCT-tiering is SP1's own core idea.  NOTE: M3 is
                          reported for completeness but was NOT part of the M0/M1/M2 interpretation
                          triangle that motivated this phase -- the promotion decision rests on
                          M0/M1/M2 alone.

G2 (core-rank-weighted boundary mass) has no separate family axis to restrict: it is already fully
determined by whichever pool membership is active (boundary mass over the admitted candidates), so
it is left as-is in every cell -- there is nothing beyond membership-gating to apply to it.

Interpretation (M0 vs M1 vs M2), gated exactly like every other phase in this program (McNemar
p<0.05 AND |delta|>corpus noise floor):
  M1 ~= M0  and  M2 <  M0   -> NERX/KNN matter for candidate REACH (membership), not scoring.
  M1 <  M0  and  M2 ~= M1   -> NERX/KNN's value is score-mass/corroboration, not reach.
  M1 ~= M2 ~= M0            -> NERX/KNN can disappear from the online path entirely.
  M1 <  M0  but  M2 >= M1   -> unexpected interaction -- inspect before concluding anything.

NERX_UNIQUE_MEMBERSHIP_REQUIRED / KNN_UNIQUE_MEMBERSHIP_REQUIRED are NOT recomputed here -- they are
pure membership questions already answered by Phase 4's H1_MINUS_NERX / H2_MINUS_KNN cells (pulled
from the existing halo_family/P4_*.json cache).  This phase only adds the two NEW flags Phase 4/7
could not answer: AUX_FAMILY_SCORE_MASS_REQUIRED (M1 vs M0) and STRUCT_ONLY_ONLINE_SAFE (M2 vs M0).

No new graph search, no new partition, no new embeddings: reuses the exact shipped SK cores / SAFE
core-selection / FULL_C+STRUCT_ONLY boundary-mass caches already on disk via _l1au_halo._setup4, and
Phase 7's struct_split hyperedge-source-masking technique for G3 (SK's hyperedge ids are STRUCT-
anchored first then KNN-anchored; split point read from BUILD_MANIFEST.json, no rebuild).

  python scratchpad/_l1au_p13_ms.py run <ds>
  python scratchpad/_l1au_p13_ms.py run_all
  python scratchpad/_l1au_p13_ms.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS
import _l1hu_hard as HH
import _l1hu_s as S
import _l1ps_router as RTR
import _l1au_halo as AH
import _l1au_p7 as P7

OUT = HH.OUT
K0 = RTR.K0
AOUT = OUT + "/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = HH.DS

CELLS_SPEC = {
    "M1_FULL_STRUCT_SCORE": dict(pool_variant="H0_FULL_C", score_scope="STRUCT", simple=False),
    "M2_STRUCT_STRUCT":     dict(pool_variant="H3_STRUCT_ONLY", score_scope="STRUCT", simple=False),
    "M3_FULL_SIMPLE_STRUCT": dict(pool_variant="H0_FULL_C", score_scope="FULL", simple=True),
}

# PER-FAMILY membership decomposition under FIXED STRUCT-only scoring.  Deliberately kept OUT of
# CELLS_SPEC so run()/run_all()/report() -- and therefore the already-published Part A six-corpus
# table -- are byte-for-byte unaffected.  Reachable only via the `cell` CLI verb.
# Together with the existing M1 (FULL membership) and M2 (STRUCT membership) these give the full
# ladder the user asked for, all four arms sharing one fixed STRUCT-only scoring substrate:
#     FULL / STRUCT-score      = M1_FULL_STRUCT_SCORE      (already run)
#     -NERX / STRUCT-score     = M1a_MINUS_NERX_STRUCT_SCORE
#     -KNN  / STRUCT-score     = M1b_MINUS_KNN_STRUCT_SCORE
#     STRUCT / STRUCT-score    = M2_STRUCT_STRUCT           (already run)
# Intended for metaqa ONLY (the sole corpus with a significant joint membership effect).  Do NOT
# launch this as another six-corpus legacy sweep.
CELLS_EXTRA = {
    "M1a_MINUS_NERX_STRUCT_SCORE": dict(pool_variant="H1_MINUS_NERX", score_scope="STRUCT", simple=False),
    "M1b_MINUS_KNN_STRUCT_SCORE":  dict(pool_variant="H2_MINUS_KNN",  score_scope="STRUCT", simple=False),
}
ALL_CELLS = dict(CELLS_SPEC, **CELLS_EXTRA)


def _m0_cached(ds):
    fp = "%s/halo_family/P4_%s__SAFE__H0_FULL_C__MATCHED.json" % (AOUT, ds)
    rec = json.load(open(fp))
    sp1 = rec["CELLS"]["SP1"]
    out = {"ALL_REQUIRED_FETCHED": sp1["ALL_REQUIRED_FETCHED"], "_ind_ALL": sp1["_ind_ALL"],
          "MEAN_CANDIDATES_PER_QUERY": rec["MEAN_CANDIDATES_PER_QUERY"],
          "runtime_ms_per_query": rec["runtime_ms_per_query"]}
    for hk in (1, 2, 3):
        k = "hop%d_ALL_REQUIRED_FETCHED" % hk
        if k in sp1:
            out[k] = sp1[k]
    return out


def run_cell(ds, cell, log=log):
    spec = ALL_CELLS[cell]
    pool_variant, score_scope, simple = spec["pool_variant"], spec["score_scope"], spec["simple"]
    E = AH._setup4(ds, "SAFE", pool_variant, "MATCHED", log)
    hard, npart, N, tag = E["hard"], E["npart"], E["N"], E["tag"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["SEL"]

    import _l1ov_core as OV
    fp_s, fm_s = OV.boundary_mass(ds, hard, tag, "O1_STRUCT", N, npart, log)
    struct_bptr, struct_bidx = OV.to_block_csr(fp_s, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = S_ if score_scope == "STRUCT" else np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = S._part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = S._hyperedge_csrs(ds, N)
    struct_split = P7._struct_edge_count(ds) if score_scope == "STRUCT" else n_edges
    HW = np.zeros(n_edges, np.float64)
    log("  %s[%s]: setup done (score_scope=%s simple=%s), entering per-query loop"
        % (ds, cell, score_scope, simple))

    rows = np.zeros(nq, np.int8)
    cand_count = np.zeros(nq, np.int64)
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 400 == 0:
            log("  %s[%s]: %d/%d, %.1fs" % (ds, cell, qi, nq, time.time() - t0))
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
            rows[qi] = int(len(got_core) == len(nd))
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)
        k = min(K_q, nU)
        cand_count[qi] = nU
        rank_f2 = S._rank_of(np.lexsort((uniq_v, -f2_score)))

        sparts = [struct_bidx[struct_bptr[j]:struct_bptr[j + 1]] for j in Sl
                 if struct_bptr[j + 1] > struct_bptr[j]]
        struct_set = np.unique(np.concatenate(sparts)) if sparts else np.zeros(0, np.int64)
        struct_supported = np.isin(uniq_v, struct_set)
        tier1 = (~struct_supported).astype(np.int64)

        if simple:
            rank_sp1 = S._rank_of(np.lexsort((rank_f2, tier1)))
        else:
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
            touched_e = np.zeros(0, np.int64)
            if len(e_h):
                mask_h = e_h < struct_split
                w_rep_h = np.repeat(w_u_num, sizes_h)
                if mask_h.any():
                    eh_m, wh_m = e_h[mask_h], w_rep_h[mask_h]
                    touched_e, inv = np.unique(eh_m, return_inverse=True)
                    HW[touched_e] = np.bincount(inv, weights=wh_m, minlength=len(touched_e))
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
            rank_sp1 = S._rank_of(np.lexsort((rank_g5, tier1)))

        top = uniq_v[rank_sp1 < k]
        gotX = got_core | (set(top.tolist()) & nd_set)
        rows[qi] = int(len(gotX) == len(nd))
    log("  %s[%s]: done %d queries in %.1fs" % (ds, cell, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    h = np.asarray(hops)[:nq] if ds == "metaqa" and hops is not None else None
    rec = {"ds": ds, "cell": cell, "spec": spec, "nq": nq,
          "ALL_REQUIRED_FETCHED": round(float(rows[ok].mean()), 4), "_ind_ALL": rows.tolist(),
          "MEAN_CANDIDATES_PER_QUERY": round(float(cand_count[ok].mean()), 2),
          "runtime_s": round(time.time() - t0, 1),
          "runtime_ms_per_query": round(1000.0 * (time.time() - t0) / max(nq, 1), 3)}
    if h is not None:
        for hk in (1, 2, 3):
            mm = (h == hk) & ok
            if mm.any():
                rec["hop%d_ALL_REQUIRED_FETCHED" % hk] = round(float(rows[mm].mean()), 4)
    os.makedirs("%s/membership_score" % AOUT, exist_ok=True)
    fp = "%s/membership_score/P13_%s__%s.json" % (AOUT, ds, cell)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def run(ds, log=log):
    out = {"M0_FULL_FULL": _m0_cached(ds)}
    for cell in CELLS_SPEC:
        out[cell] = run_cell(ds, cell, log)
    return out


def run_all():
    for ds in DS:
        run(ds)


def _load(ds, cell):
    if cell == "M0_FULL_FULL":
        return _m0_cached(ds)
    fp = "%s/membership_score/P13_%s__%s.json" % (AOUT, ds, cell)
    return json.load(open(fp))


def report():
    rows = []
    for ds in DS:
        try:
            m0 = _load(ds, "M0_FULL_FULL")
        except FileNotFoundError:
            continue
        row = {"ds": ds, "M0_FULL_FULL": m0["ALL_REQUIRED_FETCHED"]}
        floor = HH.FLOOR.get(ds, 0.0)
        for cell in CELLS_SPEC:
            try:
                v = _load(ds, cell)
            except FileNotFoundError:
                continue
            g_, l_, p_ = HH.mcnemar(m0["_ind_ALL"], v["_ind_ALL"])
            d = round(v["ALL_REQUIRED_FETCHED"] - m0["ALL_REQUIRED_FETCHED"], 4)
            row[cell] = {"ALL_REQUIRED": v["ALL_REQUIRED_FETCHED"], "delta_vs_M0": d,
                        "gained": g_, "lost": l_, "p": p_,
                        "sig": bool(p_ < 0.05 and abs(d) > floor),
                        "mean_candidates": v["MEAN_CANDIDATES_PER_QUERY"],
                        "runtime_ms_per_query": v["runtime_ms_per_query"]}
            if ds == "metaqa":
                for hk in (1, 2, 3):
                    k = "hop%d_ALL_REQUIRED_FETCHED" % hk
                    if k in v and k in m0:
                        row[cell][k] = v[k]
                        row[cell]["hop%d_delta_vs_M0" % hk] = round(v[k] - m0[k], 4)
        rows.append(row)

    os.makedirs(AOUT, exist_ok=True)
    fp = "%s/PHASE13_MEMBERSHIP_SCORE_REPORT.json" % AOUT
    for r in rows:
        print("\n== %s ==  M0_FULL_FULL(SP1)=%.4f" % (r["ds"], r["M0_FULL_FULL"]))
        for cell in CELLS_SPEC:
            if cell not in r:
                continue
            v = r[cell]
            print("  %-22s val=%.4f delta=%+.4f gained=%-3d lost=%-3d p=%-10s sig=%s  "
                  "cand/q=%.1f  ms/q=%.3f" % (cell, v["ALL_REQUIRED"], v["delta_vs_M0"],
                                              v["gained"], v["lost"], v["p"], v["sig"],
                                              v["mean_candidates"], v["runtime_ms_per_query"]))
            if r["ds"] == "metaqa":
                for hk in (1, 2, 3):
                    k = "hop%d_ALL_REQUIRED_FETCHED" % hk
                    if k in v:
                        print("      hop%d val=%.4f delta=%+.4f" % (hk, v[k], v["hop%d_delta_vs_M0" % hk]))

    def _flag(cell):
        regs = [r["ds"] for r in rows if cell in r and r[cell]["sig"] and r[cell]["delta_vs_M0"] < 0]
        return {"sig_regressions": regs, "REQUIRED": bool(regs)}

    m1_flag = _flag("M1_FULL_STRUCT_SCORE")
    m2_flag = _flag("M2_STRUCT_STRUCT")
    verdict = {
        "AUX_FAMILY_SCORE_MASS_REQUIRED": m1_flag["REQUIRED"],
        "AUX_FAMILY_SCORE_MASS_sig_regressions": m1_flag["sig_regressions"],
        "STRUCT_ONLY_ONLINE_SAFE": not m2_flag["REQUIRED"],
        "STRUCT_ONLY_ONLINE_sig_regressions": m2_flag["sig_regressions"],
        "NOTE_membership_flags_not_recomputed": (
            "NERX_UNIQUE_MEMBERSHIP_REQUIRED / KNN_UNIQUE_MEMBERSHIP_REQUIRED are pure membership "
            "questions, unchanged by this phase's scoring-substrate axis -- read them from the "
            "existing halo_family/P4_{ds}__SAFE__H1_MINUS_NERX__MATCHED.json (NERX) and "
            "H2_MINUS_KNN (KNN) SP1 cells vs H0_FULL_C, already computed by _l1au_halo.py.")}
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print("  %s: %s" % (k, v))

    out = {"rows": rows, "verdict": verdict}
    json.dump(out, open(fp, "w"), indent=1)
    print("\nwrote", fp)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "cell":
        run_cell(a[1], a[2])
    elif a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
