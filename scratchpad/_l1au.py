"""FULL L1 ARCHITECTURE + EDGE-FAMILY AUDIT -- Phases 0-2 (foundation).

Governing question: which of {H4 partition-build, SAFE core-swap, FULL_C halo, SP1 ranking} are
carrying the system, and does STRUCT alone suffice for any/all of them?  Compute policy: LOCAL,
CACHE-FIRST, backward elimination, no repartition/Modal until a later explicit gate says so.

RECON FINDING (binds every later phase): the shipped H4 hypergraph is famset="SK" = STRUCT+KNN
ONLY (scratchpad/_l1hu_build.py FAMSETS).  NER was never part of hyperedge construction -- disk has
only __SK.npz for all 6 corpora.  So "H4 FULL" in the informal sense actually means STRUCT+KNN, not
STRUCT+NER+KNN; a true 3-family build (SKN) has never been run.  This is reported, not silently
corrected -- Phase 9's history table and the final writeup must say this explicitly.

PHASE 0 -- artifact manifest.  Existence + size + mtime + a cheap partial-content fingerprint (first
64KB) for every artifact the later phases depend on.  Not full-file sha1 (some caches are 100s of MB
and are at-rest, not being rewritten under us) -- this is a staleness/presence check, not a
cryptographic integrity audit.

PHASE 1 -- the 8-cell ladder.  A0-A3 and A4/A5 (user's numbering) are IDENTICAL to existing
attribution/A_LADDER.json's A0-A3/A6-A7 (see _l1hu_attr.py CELLS) -- no recompute, pure relabel.
User's A6 (H4 + BASE-selected cores + SP1) is new: SP1 has only ever been run on SAFE(F6)-selected
cores.  User's A7 (H4 + SAFE + SP1) already sits in ceiling/S_{ds}.json's F6/SP1 cells (SAFE lane).
So the ENTIRE new-compute surface for Phase 1 is: rerun the G1-G6/SP0-3 loop with core_lane=BASE.

PHASE 2 -- does SAFE still earn its keep under SP1?  S0=A6(BASE+SP1), S1=A7(SAFE+SP1) -- same
computation as Phase 1's new cell, no separate run needed.  Also reports core-set churn (% P50
changed, swaps/query) and SAFE-vs-halo rescue-overlap classification (reuses base50 vs f650, both
already returned by OVE.selected_blocks).

  python scratchpad/_l1au.py phase0
  python scratchpad/_l1au.py ladder <ds>          # new BASE-lane SP1 run for one corpus
  python scratchpad/_l1au.py ladder_all
  python scratchpad/_l1au.py ladder_report        # assemble full 8-cell table + Phase 2 SAFE gate
"""
import os, sys, json, time, hashlib
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

# ============================================================================= PHASE 0 -- manifest
GLOBAL_ARTIFACTS = [
    "%s/halo/CORES.json" % OUT, "%s/partitions/HYPER_GATE.json" % OUT,
    "%s/attribution/A_LADDER.json" % OUT, "%s/attribution/B_QUERY_CLASSES.json" % OUT,
    "%s/halo_family/E_REPORT.json" % OUT, "%s/hypergraph_build/BUILD_MANIFEST.json" % OUT,
    "%s/ceiling/S_GATE.json" % OUT, "%s/ceiling/O_REPORT.json" % OUT,
    "scratchpad/_l1kb_core.py", "scratchpad/_l1ps_router.py", "scratchpad/_l1hu_build.py",
]
PER_DS_TEMPLATES = {
    "metis_base_map":   "scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json",
    "metis_base_parts":  "scratchpad/_l1ep/parts/{ds}__P2_METIS_STRONG.npy",
    "h4_hyperedges_sk":  "scratchpad/_l1hu/graphs/{ds}__H4_SPLIT_PRESERVE__SK.npz",
    "h4_parts_raw":      "scratchpad/_l1hu/parts/{ds}__H4_SPLIT_PRESERVE__SK.npy",
    "h4_parts_shipped":  "scratchpad/_l1ep/parts/{ds}__C1_HYPER_UNIVERSAL.npy",
    "safe_selector_cache": "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{ds}.npz",
    "ks_keysets_cache":  "scratchpad/_l1kn/keys_{ds}.npz",
    "struct_topo_cache": "scratchpad/_ta/{ds}_topo.npz",
    "raw_knn_graph_pt":  "data/ukb_storage/{ds}/gte_qwen/graph.pt",
    "raw_ner_edges":     "data/ukb_storage/{ds}/ner_edges_w_df25.pkl",
    "oracle_json":       "%s/ceiling/O_{ds}.json" % OUT,
    "sp1_json":          "%s/ceiling/S_{ds}.json" % OUT,
    "coverage_json":     "%s/halo/COVERAGE_{ds}.json" % OUT,
    "e_family_json":     "%s/halo_family/E_{ds}.json" % OUT,
}
PER_DS_FAM_TEMPLATE = "scratchpad/_l1ov/halo_{ds}__%s__{fam}.npy" % TAG
HALO_FAMS = ("O1_STRUCT", "O2_NERX", "O3_KNN", "O4_FULL_C")


def _fingerprint(fp, head=65536):
    if not os.path.exists(fp):
        return {"exists": False}
    st = os.stat(fp)
    with open(fp, "rb") as fh:
        h = hashlib.sha1(fh.read(head)).hexdigest()[:16]
    return {"exists": True, "bytes": st.st_size, "mtime": time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)), "head64k_sha1": h}


def phase0():
    rec = {"NOTE": "existence+size+mtime+head-64KB-sha1, not full-file hash (staleness check, "
                   "not crypto integrity)", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "GLOBAL": {}, "PER_DATASET": {}}
    for fp in GLOBAL_ARTIFACTS:
        rec["GLOBAL"][fp] = _fingerprint(fp)
    missing = []
    for ds in DS:
        d = {}
        for key, tmpl in PER_DS_TEMPLATES.items():
            fp = tmpl.format(ds=ds)
            fpr = _fingerprint(fp)
            d[key] = dict(fpr, path=fp)
            if not fpr["exists"]:
                missing.append((ds, key, fp))
        d["halo_pairs_by_family"] = {}
        for fam in HALO_FAMS:
            fp = PER_DS_FAM_TEMPLATE.format(ds=ds, fam=fam)
            fpr = _fingerprint(fp)
            d["halo_pairs_by_family"][fam] = dict(fpr, path=fp)
            if not fpr["exists"]:
                missing.append((ds, "halo_" + fam, fp))
        rec["PER_DATASET"][ds] = d
        log("  %s: %d/%d per-ds artifacts present" % (
            ds, sum(1 for v in d.values() if isinstance(v, dict) and v.get("exists")), len(PER_DS_TEMPLATES)))
    rec["MISSING"] = [{"ds": m[0], "key": m[1], "path": m[2]} for m in missing]
    rec["N_MISSING"] = len(missing)
    os.makedirs(AOUT, exist_ok=True)
    fp = "%s/AUDIT_CACHE_MANIFEST.json" % AOUT
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote %s (%d missing artifacts)" % (fp, len(missing)))
    for m in missing:
        log("  MISSING: %s / %s -> %s" % m)
    return rec


# ==================================================================== PHASE 1/2 -- ladder + SAFE gate
def _setup2(ds, core_lane="SAFE", log=log):
    """_l1hu_s._setup, plus exposing BOTH base50 and f650 so either core-selection lane can be run."""
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
    SEL = base50 if core_lane == "BASE" else f650
    return dict(hard=hard, npart=npart, N=N, base50=base50, f650=f650, SEL=SEL, core_lane=core_lane,
               need=need, hops=hops, nq=nq,
               bptr=bptr, bidx=bidx, bmass=bmass, bmax=bmax,
               f0_bptr_=f0_bptr_, f0_bidx_=f0_bidx_, ret_rrf=ret_rrf)


def run_lane(ds, core_lane, log=log):
    """S.run()'s G1-G6/SP0-3 loop, verbatim, but against an arbitrary core-selection lane (BASE or
    SAFE) instead of always SAFE(f650).  Reused unmodified for Phase 1's new A6 cell AND Phase 2's
    S0.  (SAFE-lane == S.run()'s existing output == user's A7/S1; not recomputed here.)"""
    E = _setup2(ds, core_lane, log)
    hard, npart, N = E["hard"], E["npart"], E["N"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["SEL"]

    fam_bptr, fam_bidx = {}, {}
    for fam in S.FAMS:
        fp_, fm_ = OV.boundary_mass(ds, hard, TAG, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = S._part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = S._hyperedge_csrs(ds, N)
    HW = np.zeros(n_edges, np.float64)
    log("  %s[%s]: setup done, entering per-query loop" % (ds, core_lane))

    METHODS = S.METHODS
    rows_ = {m: {"allf": np.zeros(nq, np.int8)} for m in METHODS}
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 400 == 0:
            log("  %s[%s]: %d/%d queries, %.1fs elapsed" % (ds, core_lane, qi, nq, time.time() - t0))
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

        fam_sets = {}
        for fam in S.FAMS:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in Sl if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        struct_supported = np.isin(uniq_v, fam_sets["O1_STRUCT"])
        knn_only = (np.isin(uniq_v, fam_sets["O3_KNN"]) & ~struct_supported
                   & ~np.isin(uniq_v, fam_sets["O2_NERX"]))

        rrf_row = ret_rrf[qi]
        qrank_of = {int(rrf_row[r]): r for r in range(len(rrf_row)) if rrf_row[r] >= 0}
        qr = np.array([qrank_of.get(int(v), -1) for v in uniq_v], np.int64)

        order3 = np.lexsort((uniq_v, -f2_score, -cnt))
        rank3 = S._rank_of(order3)
        rrf6 = 1.0 / (K0 + rank3) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        rank6 = S._rank_of(np.lexsort((uniq_v, -f2_score, -rrf6)))

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

        rank_sp0 = rank_g5
        tier1 = (~struct_supported).astype(np.int64)
        rank_sp1 = S._rank_of(np.lexsort((rank_g5, tier1)))
        multi_struct = cnt * struct_supported
        tier2 = (multi_struct < 2).astype(np.int64)
        rank_sp2 = S._rank_of(np.lexsort((rank_g5, tier2)))
        max_struct_support = int(cnt[struct_supported].max()) if struct_supported.any() else 0
        demote = (knn_only & (cnt <= max_struct_support)).astype(np.int64)
        rank_sp3 = S._rank_of(np.lexsort((rank_g5, demote)))

        ranks = {"F6": rank6, "G1": rank_g1, "G2": rank_g2, "G3": rank_g3, "G4": rank_g4,
                "G5": rank_g5, "SP0": rank_sp0, "SP1": rank_sp1, "SP2": rank_sp2, "SP3": rank_sp3}
        for m, rk in ranks.items():
            top = uniq_v[rk < k]
            gotX = got_core | (set(top.tolist()) & nd_set)
            rows_[m]["allf"][qi] = int(len(gotX) == len(nd))
    log("  %s[%s]: done %d queries in %.1fs" % (ds, core_lane, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    cells = {m: {"ALL_REQUIRED_FETCHED": round(float(rows_[m]["allf"][ok].mean()), 4),
               "_ind_ALL": rows_[m]["allf"].tolist()} for m in METHODS}
    rec = {"ds": ds, "core_lane": core_lane, "nq": nq, "CELLS": cells}
    os.makedirs("%s/ceiling" % OUT, exist_ok=True)
    fp = "%s/ceiling/LANE_%s_%s.json" % (OUT, core_lane, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def ladder(ds, log=log):
    return run_lane(ds, "BASE", log)


def ladder_all():
    for ds in DS:
        ladder(ds)


def _core_churn(ds, log=log):
    hard, npart, N = GL.load_core(ds)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    nq = meta["n_dev_queries"]
    swaps = np.zeros(nq, np.int64)
    for qi in range(nq):
        swaps[qi] = len(set(base50[qi]) - set(f650[qi]))
    return {"mean_swaps_per_query": round(float(swaps.mean()), 3),
           "median_swaps_per_query": float(np.median(swaps)),
           "frac_queries_with_any_swap": round(float((swaps > 0).mean()), 4),
           "max_swaps": int(swaps.max())}


def ladder_report():
    """Assembles the full user-numbered A0-A7 ladder (relabelled from the existing A_LADDER.json
    where possible) plus Phase 2's SAFE-necessity gate, using S0=A6(BASE+SP1) / S1=A7(SAFE+SP1)."""
    old = json.load(open("%s/attribution/A_LADDER.json" % OUT)) if os.path.exists(
        "%s/attribution/A_LADDER.json" % OUT) else {}
    sp1_safe = {}
    for ds in DS:
        fp = "%s/ceiling/S_%s.json" % (OUT, ds)
        if os.path.exists(fp):
            sp1_safe[ds] = json.load(open(fp))["CELLS"]

    report = {}
    for ds in DS:
        row = {}
        if ds in old:
            o = old[ds]
            row["A0_METIS_BASE_noHalo"] = o["A0"]
            row["A1_METIS_SAFE_noHalo"] = o["A1"]
            row["A2_H4_BASE_noHalo"] = o["A2"]
            row["A3_H4_SAFE_noHalo"] = o["A3"]
            row["A4_H4_BASE_currentHalo_F6"] = o["A6"]
            row["A5_H4_SAFE_currentHalo_F6"] = o["A7"]
        base_lane_fp = "%s/ceiling/LANE_BASE_%s.json" % (OUT, ds)
        if os.path.exists(base_lane_fp):
            lb = json.load(open(base_lane_fp))["CELLS"]
            row["A6_H4_BASE_SP1"] = lb["SP1"]["ALL_REQUIRED_FETCHED"]
            row["A6_full_methods"] = {m: lb[m]["ALL_REQUIRED_FETCHED"] for m in lb}
        if ds in sp1_safe:
            row["A7_H4_SAFE_SP1"] = sp1_safe[ds]["SP1"]["ALL_REQUIRED_FETCHED"]
            row["A7_full_methods"] = {m: sp1_safe[ds][m]["ALL_REQUIRED_FETCHED"] for m in sp1_safe[ds]}
        if "A6_H4_BASE_SP1" in row and "A7_H4_SAFE_SP1" in row:
            gd, ls, p = HH.mcnemar(np.array(json.load(open(base_lane_fp))["CELLS"]["SP1"]["_ind_ALL"]),
                                   np.array(sp1_safe[ds]["SP1"]["_ind_ALL"]))
            floor = HH.FLOOR.get(ds, 0.0)
            d = round(row["A7_H4_SAFE_SP1"] - row["A6_H4_BASE_SP1"], 4)
            row["PHASE2_SAFE_ON_SP1_EFFECT"] = {
                "S1_minus_S0": d, "gained_by_SAFE": gd, "lost_by_SAFE": ls, "p": p,
                "sig": bool(p < 0.05 and abs(d) > floor)}
            row["PHASE2_CORE_CHURN"] = _core_churn(ds)
        report[ds] = row

    os.makedirs(AOUT, exist_ok=True)
    fp = "%s/PHASE1_LADDER_PHASE2_SAFE_GATE.json" % AOUT
    json.dump(report, open(fp, "w"), indent=1)
    print("\n%-16s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s" % (
        "ds", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"))
    for ds, row in report.items():
        vals = [row.get(k) for k in ("A0_METIS_BASE_noHalo", "A1_METIS_SAFE_noHalo",
                                     "A2_H4_BASE_noHalo", "A3_H4_SAFE_noHalo",
                                     "A4_H4_BASE_currentHalo_F6", "A5_H4_SAFE_currentHalo_F6",
                                     "A6_H4_BASE_SP1", "A7_H4_SAFE_SP1")]
        print("%-16s " % ds + " ".join("%-8s" % (("%.4f" % v) if v is not None else "-") for v in vals))
        if "PHASE2_SAFE_ON_SP1_EFFECT" in row:
            e = row["PHASE2_SAFE_ON_SP1_EFFECT"]; c = row["PHASE2_CORE_CHURN"]
            print("    PHASE2: S1-S0=%+.4f gained=%d lost=%d p=%.3g sig=%s | "
                  "mean_swaps/q=%.2f frac_q_any_swap=%.4f"
                  % (e["S1_minus_S0"], e["gained_by_SAFE"], e["lost_by_SAFE"], e["p"], e["sig"],
                     c["mean_swaps_per_query"], c["frac_queries_with_any_swap"]))
    print("\nwrote", fp)
    return report


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "phase0":
        phase0()
    elif a and a[0] == "ladder":
        ladder(a[1])
    elif a and a[0] == "ladder_all":
        ladder_all()
    elif a and a[0] == "ladder_report":
        ladder_report()
    else:
        print(__doc__)
