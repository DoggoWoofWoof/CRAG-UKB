"""LOCAL RESIDUAL 1-HOP RANKING PHASE (R0-R6) -- rank the F2 candidate pool better, not wider.

Part H found the residual bottleneck is the RANKING, not budget or graph reach: 52-93% of every
corpus's missing nodes are already valid 1-hop F2 candidates, ranked too low even at 2x budget
(bucket H2).  This phase stays entirely inside that fact: SAME 50 selected cores, SAME uncapped
FULL_C candidate universe, SAME per-query budget K_q (F0's own new-halo-node count) -- only the
SCORE used to cut the top-K_q changes.  No repartition, no Modal, no 2-hop, no learned weight.

  F2_SUM_NORMALIZED (reference)  sum boundary_mass(v,core_j)/max_mass(core_j) over supporting j
  F3_SUPPORT_FIRST   (R1)        lexicographic: support_count desc, F2 desc, node id asc
  F4_QUERY_RRF        (R2)       RRF(F2's own within-pool rank, node's rank in the CACHED
                                  Dense+SPLADE ret_rrf[qi] node continuation), K0=60 (same
                                  constant _l1ps_router.py already uses).  A candidate absent from
                                  ret_rrf's top-200 contributes 0 to that channel -- the same
                                  per-query masking convention _l1kb_core.f6_select already uses,
                                  not a fallback rank.  No new encoder, no new embedding.
  F5_FAMILY_SUPPORT  (R3)        lexicographic: family_count(STRUCT/NERX/KNN present among the
                                  selected cores) desc, support_count desc, F2 desc, node id asc
  F6_SUPPORT_QUERY_RRF (R4)      RRF(F3's own within-pool rank, ret_rrf rank), same K0/masking

Every ranker is computed in the SAME per-query pass (support_count and family_count are needed for
the R1/R3 enrichment diagnostics regardless of whether F3/F5 end up promotable, so building their
ranking alongside is free -- same efficiency call as merging Part I's oracle into Part H's pass).
What stays conditional, per the spec, is REPORTING/PROMOTING them: R1 says "if enrichment is not
directionally useful, do not create F3" and R3 the same for F5; R4 says F6 only counts as a real
answer if F3 AND F4 both independently clear R5's bar first.  gate()/report() apply those calls
explicitly against the measured numbers rather than skipping computation speculatively.

R6 (H2-targeted) re-derives H2 membership inline using the exact _l1hu_h.py threshold (1-hop F2
candidate, still cut even at 2x budget) instead of re-running or extending that script -- H2 status
is fully determined by data this script already computes per query (cand_score, K_q), so no extra
corpus-level pass (webqsp/hotpot cost 200-700s there) is needed.

  python scratchpad/_l1hu_r.py run <ds>
  python scratchpad/_l1hu_r.py run_all
  python scratchpad/_l1hu_r.py gate
  python scratchpad/_l1hu_r.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as OVE
import _l1ep_pu as PU
import _l1hu_hard as HH
import _l1hu_global as GL
import _l1ps_router as RTR

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
K0 = RTR.K0
FAMS = ("O1_STRUCT", "O2_NERX", "O3_KNN")
METHODS = ("F2", "F3", "F4", "F5", "F6")
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def _rank_of(order):
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order))
    return r


def run(ds, log=log):
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

    fam_bptr, fam_bidx = {}, {}
    for fam in FAMS:
        fp_, fm_ = OV.boundary_mass(ds, hard, TAG, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    ret_rrf = np.asarray(z["ret_rrf"])
    log("  %s: setup done, N=%d npart=%d nq=%d, entering per-query loop" % (ds, N, npart, nq))

    SEL = f650
    rows_ = {m: {"allf": np.zeros(nq, np.int8), "expo": np.zeros(nq, np.int64)} for m in METHODS}
    supp_tot = {k: 0 for k in (1, 2, 3, 4)}
    supp_req = {k: 0 for k in (1, 2, 3, 4)}
    fam_tot = {k: 0 for k in (1, 2, 3)}
    fam_req = {k: 0 for k in (1, 2, 3)}
    h2 = {"n": 0, "moved": {m: 0 for m in METHODS}, "pct_sum": {m: 0.0 for m in METHODS},
          "queries": set()}
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 200 == 0:
            log("  %s: %d/%d queries, %.1fs elapsed" % (ds, qi, nq, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_arr = np.asarray(nd, np.int64)
        nd_set = set(nd)
        S = SEL[qi]
        Ss = set(S)
        selmask[:] = False
        selmask[np.asarray(S, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}

        # ---- uncapped FULL_C candidate pool (identical gather to Part F/G/H) ----
        cv, cj, cm = [], [], []
        for j in S:
            s, e = bptr[j], bptr[j + 1]
            if e > s:
                cv.append(bidx[s:e]); cj.append(np.full(e - s, j, np.int64)); cm.append(bmass[s:e])
        if cv:
            V = np.concatenate(cv); J = np.concatenate(cj); Mm = np.concatenate(cm)
            keep = ~selmask[hard[V.astype(np.int64)]]
            V, J, Mm = V[keep], J[keep], Mm[keep]
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
        Vs, Ms, Js = V[order], Mm[order], J[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)

        # ---- family_count: STRUCT/NERX/KNN support among blocks in S, per candidate ----
        family_count = np.zeros(nU, np.int64)
        for fam in FAMS:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in S if fb[j + 1] > fb[j]]
            fset = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
            if len(fset):
                family_count += np.isin(uniq_v, fset)

        # ---- query relevance rank: cached Dense+SPLADE RRF node continuation, zero new compute ----
        rrf_row = ret_rrf[qi]
        qrank_of = {int(rrf_row[r]): r for r in range(len(rrf_row)) if rrf_row[r] >= 0}
        qr = np.array([qrank_of.get(int(v), -1) for v in uniq_v], np.int64)

        # ---- five deterministic rankings, all lexicographic, tertiary = ascending node id ----
        order2 = np.lexsort((uniq_v, -f2_score))
        rank2 = _rank_of(order2)

        order3 = np.lexsort((uniq_v, -f2_score, -cnt))
        rank3 = _rank_of(order3)

        rrf4 = 1.0 / (K0 + rank2) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        order4 = np.lexsort((uniq_v, -f2_score, -rrf4))
        rank4 = _rank_of(order4)

        order5 = np.lexsort((uniq_v, -f2_score, -cnt, -family_count))
        rank5 = _rank_of(order5)

        rrf6 = 1.0 / (K0 + rank3) + np.where(qr >= 0, 1.0 / (K0 + np.maximum(qr, 0)), 0.0)
        order6 = np.lexsort((uniq_v, -f2_score, -rrf6))
        rank6 = _rank_of(order6)

        ranks = {"F2": rank2, "F3": rank3, "F4": rank4, "F5": rank5, "F6": rank6}
        k = min(K_q, nU)
        for m, rk in ranks.items():
            top = uniq_v[rk < k]
            gotX = got_core | (set(top.tolist()) & nd_set)
            rows_[m]["allf"][qi] = int(len(gotX) == len(nd))
            rows_[m]["expo"][qi] = k

        # ---- R1 / R3 enrichment diagnostics: every candidate this query, gold-labelled ----
        is_req = np.isin(uniq_v, nd_arr)
        sb = np.minimum(cnt, 4)
        for kk in (1, 2, 3, 4):
            m_ = sb == kk
            supp_tot[kk] += int(m_.sum()); supp_req[kk] += int((m_ & is_req).sum())
        for kk in (1, 2, 3):
            m_ = family_count == kk
            fam_tot[kk] += int(m_.sum()); fam_req[kk] += int((m_ & is_req).sum())

        # ---- R6: H2-targeted (inline re-derivation, exact _l1hu_h.py 2x-budget threshold) ----
        f2_top = uniq_v[rank2 < k]
        got_f2 = got_core | (set(f2_top.tolist()) & nd_set)
        missing_f2 = [x for x in nd if x not in got_f2]
        if missing_f2:
            score_of = dict(zip(uniq_v.tolist(), f2_score.tolist()))
            idx_of = {int(v): i for i, v in enumerate(uniq_v.tolist())}
            sorted_scores = np.sort(f2_score)
            any_h2 = False
            for x in missing_f2:
                if x not in score_of:
                    continue                                      # H3/H4/H5, out of scope for R6
                sc = score_of[x]
                better = nU - np.searchsorted(sorted_scores, sc, side="right")
                if better < 2 * K_q:
                    continue                                      # H1, not H2
                any_h2 = True
                h2["n"] += 1
                i_ = idx_of[x]
                for m, rk in ranks.items():
                    h2["moved"][m] += int(rk[i_] < k)
                    h2["pct_sum"][m] += 1.0 - rk[i_] / max(nU - 1, 1)
            if any_h2:
                h2["queries"].add(qi)

    log("  %s: done %d queries in %.1fs" % (ds, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    hops_arg = hops if ds == "metaqa" else None
    cells = {}
    for m in METHODS:
        allf = rows_[m]["allf"]
        rec = {"ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4),
              "MEAN_EXPOSURE_USED": round(float(rows_[m]["expo"][ok].mean()), 1),
              "_ind_ALL": allf.tolist()}
        if hops_arg is not None:
            h = np.asarray(hops_arg)[:nq]
            for hk in (2, 3):
                mm = (h == hk) & ok
                if mm.any():
                    rec["hop%d_ALL_REQUIRED_FETCHED" % hk] = round(float(allf[mm].mean()), 4)
        cells[m] = rec

    h2_queries = sorted(h2["queries"])
    h2_rescue = {m: (round(float(np.mean([cells[m]["_ind_ALL"][qi] for qi in h2_queries])), 4)
                     if h2_queries else None) for m in METHODS}

    rec = {"ds": ds, "N": N, "npart": npart, "nq": nq, "CELLS": cells,
          "R1_SUPPORT_ENRICHMENT": {str(k): {"n": supp_tot[k], "req": supp_req[k],
                                             "req_rate": (supp_req[k] / supp_tot[k]
                                                          if supp_tot[k] else None),
                                             "req_per_10k": (round(1e4 * supp_req[k] / supp_tot[k], 4)
                                                            if supp_tot[k] else None)}
                                    for k in (1, 2, 3, 4)},
          "R3_FAMILY_ENRICHMENT": {str(k): {"n": fam_tot[k], "req": fam_req[k],
                                            "req_rate": (fam_req[k] / fam_tot[k]
                                                         if fam_tot[k] else None),
                                            "req_per_10k": (round(1e4 * fam_req[k] / fam_tot[k], 4)
                                                           if fam_tot[k] else None)}
                                   for k in (1, 2, 3)},
          "R6_H2_TARGETED": {"n_h2_nodes": h2["n"], "n_h2_queries": len(h2_queries),
                             "frac_moved_into_Kq": {m: (round(h2["moved"][m] / h2["n"], 4)
                                                        if h2["n"] else None) for m in METHODS},
                             "mean_rank_percentile": {m: (round(h2["pct_sum"][m] / h2["n"], 4)
                                                          if h2["n"] else None) for m in METHODS},
                             "query_rescue_conversion": h2_rescue}}
    os.makedirs("%s/ranking_refine" % OUT, exist_ok=True)
    fp = "%s/ranking_refine/R_%s.json" % (OUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def r0_parity(ds, rec, log=log):
    """R0 -- this script's own F2 (lexsort-cut, explicit tie-break) vs _l1hu_global.py's cached
    F2 (argpartition-cut, incidental tie-break).  Same score, same K_q, same candidate universe --
    only a genuine floating-point tie at the cut boundary could ever disagree."""
    fp = "%s/global_halo/F_%s.json" % (OUT, ds)
    if not os.path.exists(fp):
        return {"status": "NO_REFERENCE"}
    ref = json.load(open(fp))["CELLS"]["F6"]["F2"]["_ind_ALL"]
    mine = rec["CELLS"]["F2"]["_ind_ALL"]
    if len(ref) != len(mine):
        return {"status": "LENGTH_MISMATCH", "ref_n": len(ref), "mine_n": len(mine)}
    diffs = [i for i, (a, b) in enumerate(zip(ref, mine)) if a != b]
    return {"status": "EXACT" if not diffs else "MISMATCH", "n_diff": len(diffs),
           "diff_qids": diffs[:20], "ref_rate": round(float(np.mean(ref)), 4),
           "mine_rate": round(float(np.mean(mine)), 4)}


def gate():
    recs = {}
    for ds in HH.DS:
        fp = "%s/ranking_refine/R_%s.json" % (OUT, ds)
        if os.path.exists(fp):
            recs[ds] = json.load(open(fp))
    if not recs:
        print("no ranking_refine results yet"); return {}

    parity = {ds: r0_parity(ds, rec) for ds, rec in recs.items()}

    vs_f2 = {}
    for m in ("F3", "F4", "F5", "F6"):
        rows_out = []
        for ds, rec in recs.items():
            f2 = rec["CELLS"]["F2"]; mm = rec["CELLS"][m]
            gd, ls, p = HH.mcnemar(f2["_ind_ALL"], mm["_ind_ALL"])
            floor = HH.FLOOR.get(ds, 0.0)
            d = round(mm["ALL_REQUIRED_FETCHED"] - f2["ALL_REQUIRED_FETCHED"], 4)
            rows_out.append({"ds": ds, "F2": f2["ALL_REQUIRED_FETCHED"],
                             m: mm["ALL_REQUIRED_FETCHED"], "delta": d,
                             "gained": gd, "lost": ls, "p": p,
                             "sig": bool(p < 0.05 and abs(d) > floor)})
        deltas = [r["delta"] for r in rows_out]
        regs = [r["ds"] for r in rows_out if r["sig"] and r["delta"] < 0]
        vs_f2[m] = {"per_corpus": rows_out, "worst_delta": round(min(deltas), 4),
                    "macro_delta": round(float(np.mean(deltas)), 4),
                    "sig_regressions": regs,
                    "PROMOTABLE": bool(not regs and (min(deltas) > 0 or
                                                     float(np.mean(deltas)) > 0))}

    r1 = {ds: rec["R1_SUPPORT_ENRICHMENT"] for ds, rec in recs.items()}
    r3 = {ds: rec["R3_FAMILY_ENRICHMENT"] for ds, rec in recs.items()}
    r6 = {ds: rec["R6_H2_TARGETED"] for ds, rec in recs.items()}

    out = {"PARITY_R0": parity, "VS_F2": vs_f2, "R1_SUPPORT_ENRICHMENT": r1,
          "R3_FAMILY_ENRICHMENT": r3, "R6_H2_TARGETED": r6,
          "n_corpora": len(recs), "corpora": sorted(recs)}
    os.makedirs("%s/ranking_refine" % OUT, exist_ok=True)
    json.dump(out, open("%s/ranking_refine/R_GATE.json" % OUT, "w"), indent=1)
    return out


def report():
    out = gate()
    if not out:
        return out
    print("=== R0 PARITY ===")
    for ds, p in out["PARITY_R0"].items():
        print("  %-16s %s" % (ds, p))
    for m in ("F3", "F4", "F5", "F6"):
        print("\n=== %s vs F2 ===" % m)
        for r in out["VS_F2"][m]["per_corpus"]:
            print("  %-16s F2=%.4f %s=%.4f delta=%+.4f gained=%-4d lost=%-4d p=%-10s sig=%s"
                  % (r["ds"], r["F2"], m, r[m], r["delta"], r["gained"], r["lost"], r["p"], r["sig"]))
        v = out["VS_F2"][m]
        print("  worst=%+.4f macro=%+.4f sig_regressions=%s PROMOTABLE=%s"
              % (v["worst_delta"], v["macro_delta"], v["sig_regressions"], v["PROMOTABLE"]))
    print("\n=== R1 support-count enrichment (required per 10k candidates, by support bucket) ===")
    for ds, e in out["R1_SUPPORT_ENRICHMENT"].items():
        print("  %-16s %s" % (ds, {k: (v["req_per_10k"], v["n"]) for k, v in e.items()}))
    print("\n=== R3 family-count enrichment (required per 10k candidates, by family bucket) ===")
    for ds, e in out["R3_FAMILY_ENRICHMENT"].items():
        print("  %-16s %s" % (ds, {k: (v["req_per_10k"], v["n"]) for k, v in e.items()}))
    print("\n=== R6 H2-targeted ===")
    for ds, h in out["R6_H2_TARGETED"].items():
        print("  %-16s n_h2=%d n_h2_q=%d moved=%s rescue=%s"
              % (ds, h["n_h2_nodes"], h["n_h2_queries"], h["frac_moved_into_Kq"],
                 h["query_rescue_conversion"]))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        for d in HH.DS:
            run(d)
    elif a and a[0] == "gate":
        gate()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
