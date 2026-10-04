"""PHASE 14 -- hotpotqa_clean A/B/C/D latency MECHANISM, measured not hypothesised.

Step4's six-corpus table found SKN+STRUCT_ONLY online (D) is ~48.5% SLOWER than shipped SK+FULL_C
(A) on hotpotqa_clean specifically, the opposite direction from every other corpus.  The narrative
written into PHASE11_DECISION_GATE.json attributed this to hotpot's high raw-NER density inflating
the STRUCT-only candidate set -- but that is an unverified hypothesis, not a measurement, and a
cheap sanity check argues against it in its simplest form: the six-corpus table's own
MEAN_CANDIDATES_PER_QUERY for hotpot's 4 cells are all within ~2% of each other (A=453009, B=449747,
C=458834, D=455680) -- nowhere near big enough a gap to explain a 48.5%/79.9% latency swing by
candidate-set SIZE alone.  So if NER density matters here, it must be acting on something upstream
of the final deduped candidate count (e.g. raw pre-dedup traversal volume, or hyperedge/neighbour
pin density) -- or the real cause is something else entirely (e.g. provenance-tracking's per-
candidate PYTHON loop, which _l1au_halo.run4() only runs when pool_variant=="H0_FULL_C", i.e. for
cells A and C but not B/D -- a pure bookkeeping-cost confound unrelated to any accuracy question).

This phase instruments an EXACT, unmodified copy of run4()'s per-query loop (same computation, same
semantics, nothing about the ranking changed) with counters and stage timers, run on hotpotqa_clean
across the same 4 (hard, tag, pool_variant) combinations Phase 12 already used for A/B/C/D:

  A = (H4FAM_SK,  H0_FULL_C)      B = (H4FAM_SK,  H3_STRUCT_ONLY)
  C = (H4FAM_SKN, H0_FULL_C)      D = (H4FAM_SKN, H3_STRUCT_ONLY)

Reuses the exact staged partitions Phase 12 already wrote to scratchpad/_l1ep/parts (no restage, no
new partition/graph search) and whatever boundary-mass caches already exist under each tag.

Per-query metrics accumulated (means reported, not stored per-query -- these are aggregate/mechanism
numbers, not another accuracy cell):
  struct_boundary_degree   selected-core STRUCT boundary size (sum over the 50 selected cores of
                            their STRUCT-family boundary-adjacency list length) -- how much STRUCT
                            structure the selected cores expose under each partition.
  raw_candidate_pairs      len(V) -- total (candidate, core) pair rows gathered, BEFORE dedup.
  dedup_candidates         nU -- deduped candidate-pool size (== MEAN_CANDIDATES_PER_QUERY elsewhere).
  Kq                       the query's fetch budget.
  g1_edges_gathered        len(nbrs_k) -- raw neighbour-edge instances gathered for G1, before the
                            uniq_v validity filter.
  g1_edges_applied         valid.sum() -- neighbour-edge instances that actually landed on a pool
                            candidate and contributed to g1_score.
  g3_pins_in               len(e_h) -- hyperedge-pin instances touched from the selected-core side
                            (U_all) that populate HW.
  g3_pins_out              len(e_v) -- hyperedge-pin instances gathered back out on the candidate
                            (uniq_v) side to build g3_score.
  sort_size_pool           len(V) -- size of the initial np.argsort over raw pool rows.
  sort_size_rank           nU -- size of every downstream lexsort (f2/g1/g2/g3/g4/g5/sp1 ranks).

Stage timers (wall-clock seconds accumulated over the whole run, reported as ms/query):
  gather_dedup    candidate gather + filter + dedup + F2.
  family          family-set / struct_supported / knn_only bookkeeping.
  g1g2            query-rank map, G2, and G1 (neighbour-substrate) computation.
  g3              G3 (hyperedge-substrate) computation.
  combine_rank    G4/G5/SP1 combination + top-k extraction for all 4 methods.
  provenance      the H0_FULL_C-only per-candidate Python bookkeeping loop (cells A, C only; 0 for
                  B, D by construction) -- isolated specifically because it is a NON-substantive
                  bookkeeping cost that happens to correlate with pool_variant, not an accuracy-
                  relevant computation, and could alone explain part of any A/C vs B/D timing gap.

  python scratchpad/_l1au_p14_hotpot_mech.py run_all
  python scratchpad/_l1au_p14_hotpot_mech.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1kn_sub as KS
import _l1hu_hard as HH
import _l1hu_s as S
import _l1ps_router as RTR
import _l1au_halo as AH

OUT = HH.OUT
K0 = RTR.K0
AOUT = OUT + "/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = "hotpotqa_clean"
EPARTS = "scratchpad/_l1ep/parts"

CELLS = {
    "A_SK_FULL_C":        dict(fam_tag="H4FAM_SK",  pool_variant="H0_FULL_C"),
    "B_SK_STRUCT_ONLY":   dict(fam_tag="H4FAM_SK",  pool_variant="H3_STRUCT_ONLY"),
    "C_SKN_FULL_C":       dict(fam_tag="H4FAM_SKN", pool_variant="H0_FULL_C"),
    "D_SKN_STRUCT_ONLY":  dict(fam_tag="H4FAM_SKN", pool_variant="H3_STRUCT_ONLY"),
}
STAGES = ("gather_dedup", "family", "g1g2", "g3", "combine_rank", "provenance")
tock = time.perf_counter


def run_cell(cell, log=log):
    spec = CELLS[cell]
    fam_tag, pool_variant = spec["fam_tag"], spec["pool_variant"]
    hard = np.load("%s/%s__%s.npy" % (EPARTS, DS, fam_tag))
    hard = np.asarray(hard, np.int64)
    npart = int(hard.max()) + 1

    E = AH._setup4(DS, "SAFE", pool_variant, "MATCHED", log, hard=hard, npart=npart, tag=fam_tag)
    hard, npart, N, tag = E["hard"], E["npart"], E["N"], E["tag"]
    need, hops, nq = E["need"], E["hops"], E["nq"]
    bptr, bidx, bmass, bmax = E["bptr"], E["bidx"], E["bmass"], E["bmax"]
    f0_bptr_, f0_bidx_ = E["f0_bptr_"], E["f0_bidx_"]
    ret_rrf = E["ret_rrf"]
    SEL = E["SEL"]
    in_play = AH.FAMS_IN_PLAY[pool_variant]

    fam_bptr, fam_bidx = {}, {}
    for fam in in_play:
        fp_, fm_ = OV.boundary_mass(DS, hard, tag, fam, N, npart, log)
        fam_bptr[fam], fam_bidx[fam] = OV.to_block_csr(fp_, npart, N)

    _, S_, K_, X_ = KS.keysets(DS, log)
    union_keys = np.union1d(np.union1d(S_, K_), X_)
    uptr, uidx, udeg = KS._csr(union_keys, N)
    part_ptr, part_idx = S._part_csr(hard, npart, N)
    eptr, eidx, hptr, hidx, n_edges = S._hyperedge_csrs(DS, N)
    HW = np.zeros(n_edges, np.float64)
    track_prov = (pool_variant == "H0_FULL_C")
    lim = int(os.environ.get("P14_LIMIT", "0"))
    nq_full = nq
    if lim > 0:
        nq = min(nq, lim)
        need = need[:nq]
    log("  %s[%s]: setup done, entering per-query loop (nq=%d%s)" %
        (DS, cell, nq, " SMOKE of %d" % nq_full if lim > 0 else ""))

    rows = np.zeros(nq, np.int8)
    cnt_ = {k: np.zeros(nq, np.float64) for k in
           ("struct_boundary_degree", "raw_candidate_pairs", "dedup_candidates", "Kq",
            "g1_edges_gathered", "g1_edges_applied", "g3_pins_in", "g3_pins_out")}
    stage_t = {k: 0.0 for k in STAGES}
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 200 == 0:
            log("  %s[%s]: %d/%d, %.1fs" % (DS, cell, qi, nq, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        Sl = SEL[qi]; Ss = set(Sl)
        selmask[:] = False; selmask[np.asarray(Sl, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}

        t_a = tock()
        cnt_["struct_boundary_degree"][qi] = sum(
            fam_bidx["O1_STRUCT"][fam_bptr["O1_STRUCT"][j]:fam_bptr["O1_STRUCT"][j + 1]].shape[0]
            for j in Sl) if "O1_STRUCT" in fam_bidx else 0.0

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
        cnt_["raw_candidate_pairs"][qi] = len(V)

        if len(f0_bidx_):
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in Sl]))
            K_new = u0[~selmask[hard[u0.astype(np.int64)]]]
        else:
            K_new = np.zeros(0, np.int64)
        K_q = len(K_new)
        cnt_["Kq"][qi] = K_q

        if K_q == 0 or not len(V):
            rows[qi] = int(len(got_core) == len(nd))
            stage_t["gather_dedup"] += tock() - t_a
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js, Rks = V[order], Mm[order], J[order], Rk[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs))).astype(np.int64)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)
        nU = len(uniq_v)
        k = min(K_q, nU)
        cnt_["dedup_candidates"][qi] = nU
        stage_t["gather_dedup"] += tock() - t_a

        t_b = tock()
        fam_sets = {}
        for fam in in_play:
            fb, fx = fam_bptr[fam], fam_bidx[fam]
            parts = [fx[fb[j]:fb[j + 1]] for j in Sl if fb[j + 1] > fb[j]]
            fam_sets[fam] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        has_s = np.isin(uniq_v, fam_sets["O1_STRUCT"]) if "O1_STRUCT" in in_play else np.zeros(nU, bool)
        has_x = np.isin(uniq_v, fam_sets["O2_NERX"]) if "O2_NERX" in in_play else np.zeros(nU, bool)
        has_k = np.isin(uniq_v, fam_sets["O3_KNN"]) if "O3_KNN" in in_play else np.zeros(nU, bool)
        struct_supported = has_s
        stage_t["family"] += tock() - t_b

        t_c = tock()
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
        cnt_["g1_edges_gathered"][qi] = len(nbrs_k)
        pos = np.searchsorted(uniq_v, nbrs_k)
        valid = (pos < nU) & (uniq_v[np.minimum(pos, nU - 1)] == nbrs_k)
        cnt_["g1_edges_applied"][qi] = int(valid.sum())
        g1_score = np.bincount(pos[valid], weights=contrib_k[valid], minlength=nU)
        rank_g1 = S._rank_of(np.lexsort((uniq_v, -f2_score, -g1_score)))
        stage_t["g1g2"] += tock() - t_c

        t_d = tock()
        _, e_h, sizes_h = S._ragged_gather(U_all, hptr, hidx)
        cnt_["g3_pins_in"][qi] = len(e_h)
        touched_e = np.zeros(0, np.int64)
        if len(e_h):
            w_rep_h = np.repeat(w_u_num, sizes_h)
            touched_e, inv = np.unique(e_h, return_inverse=True)
            HW[touched_e] = np.bincount(inv, weights=w_rep_h, minlength=len(touched_e))
        _, e_v, sizes_v = S._ragged_gather(uniq_v, hptr, hidx)
        cnt_["g3_pins_out"][qi] = len(e_v)
        if len(e_v):
            cand_idx_rep = np.repeat(np.arange(nU, dtype=np.int64), sizes_v)
            g3_score = np.bincount(cand_idx_rep, weights=HW[e_v], minlength=nU)
        else:
            g3_score = np.zeros(nU)
        if len(touched_e):
            HW[touched_e] = 0.0
        rank_g3 = S._rank_of(np.lexsort((uniq_v, -f2_score, -g3_score)))
        stage_t["g3"] += tock() - t_d

        t_e = tock()
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
        sp1_top = top_sets["SP1"]
        gotX = got_core | (sp1_top & nd_set)
        rows[qi] = int(len(gotX) == len(nd))
        stage_t["combine_rank"] += tock() - t_e

        if track_prov:
            t_f = tock()
            sig_of = np.where(has_s & has_x & has_k, "STRUCT+NERX+KNN",
                     np.where(has_s & has_x, "STRUCT+NERX", np.where(has_s & has_k, "STRUCT+KNN",
                     np.where(has_x & has_k, "NERX+KNN", np.where(has_s, "STRUCT",
                     np.where(has_x, "NERX", np.where(has_k, "KNN", "")))))))
            for i_, v_ in enumerate(uniq_v.tolist()):
                sig = sig_of[i_]
                if not sig:
                    continue
                _ = (v_ in nd_set, v_ in sp1_top)
            stage_t["provenance"] += tock() - t_f
    log("  %s[%s]: done %d queries in %.1fs" % (DS, cell, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    rec = {"ds": DS, "cell": cell, "spec": spec, "nq": nq,
          "ALL_REQUIRED_FETCHED": round(float(rows[ok].mean()), 4),
          "runtime_s": round(time.time() - t0, 1),
          "runtime_ms_per_query": round(1000.0 * (time.time() - t0) / max(nq, 1), 3),
          "MEANS": {k: round(float(v[ok].mean()), 2) for k, v in cnt_.items()},
          "STAGE_MS_PER_QUERY": {k: round(1000.0 * v / max(nq, 1), 4) for k, v in stage_t.items()},
          "STAGE_PCT_OF_TOTAL": {k: round(100.0 * v / max(sum(stage_t.values()), 1e-9), 1)
                                 for k, v in stage_t.items()}}
    os.makedirs("%s/hotpot_mech" % AOUT, exist_ok=True)
    fp = "%s/hotpot_mech/P14_%s%s.json" % (AOUT, cell, "__SMOKE%d" % nq if lim > 0 else "")
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def run_all(log=log):
    for cell in CELLS:
        run_cell(cell, log)


def report():
    rows = {}
    for cell in CELLS:
        fp = "%s/hotpot_mech/P14_%s.json" % (AOUT, cell)
        if os.path.exists(fp):
            rows[cell] = json.load(open(fp))
    if not rows:
        print("no P14 results yet"); return {}

    print("== hotpotqa_clean A/B/C/D mechanism ==")
    for cell, r in rows.items():
        print("\n%-20s ALL_REQUIRED=%.4f  total_ms/q=%.2f" %
              (cell, r["ALL_REQUIRED_FETCHED"], r["runtime_ms_per_query"]))
        print("  MEANS: %s" % r["MEANS"])
        print("  STAGE ms/q: %s" % r["STAGE_MS_PER_QUERY"])
        print("  STAGE pct:  %s" % r["STAGE_PCT_OF_TOTAL"])

    if "A_SK_FULL_C" in rows and "D_SKN_STRUCT_ONLY" in rows:
        a, d = rows["A_SK_FULL_C"], rows["D_SKN_STRUCT_ONLY"]
        print("\n=== A vs D (the observed +48.5%% latency gap) ===")
        print("  total ms/q: A=%.2f D=%.2f ratio=%.3f" %
              (a["runtime_ms_per_query"], d["runtime_ms_per_query"],
               d["runtime_ms_per_query"] / max(a["runtime_ms_per_query"], 1e-9)))
        for k in a["MEANS"]:
            av, dv = a["MEANS"][k], d["MEANS"][k]
            ratio = dv / av if av else None
            print("  %-24s A=%-12.1f D=%-12.1f D/A=%s" % (k, av, dv, "%.3f" % ratio if ratio else "n/a"))
        print("  NOTE: D has provenance-tracking OFF (H3_STRUCT_ONLY) while A has it ON (H0_FULL_C)")
        print("  -- if D is still slower than A despite skipping that stage entirely, the cause is")
        print("  in one of the other stages/counts above, not bookkeeping overhead.")

    os.makedirs(AOUT, exist_ok=True)
    json.dump(rows, open("%s/PHASE14_HOTPOT_MECH_REPORT.json" % AOUT, "w"), indent=1)
    print("\nwrote %s/PHASE14_HOTPOT_MECH_REPORT.json" % AOUT)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run_cell(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
