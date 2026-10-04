"""TRIPLET PHASE -- STEPS 8, 9, 10, 11.

STEP 8   complementarity BEFORE fusion.  Per-query error vectors of the surviving signals, their
         disagreement, and the union ceiling.  Only if two signals are genuinely complementary is
         ONE parameter-free combination allowed: equal rank fusion, no weights, no lambda, no
         threshold, no grid.
STEP 9   the surviving triplet evidence mapped onto the EXISTING C partitions -- no repartitioning.
         The comparison that matters: CURRENT NODE-S4  vs  TRIPLET-S4 / PATH-S4.
STEP 10  through the unchanged boundary mechanism at the frozen contract: P = 50, B = 6, M_ret = 32.
         Reported at full depth AND depth-matched, because the previous phase established that a
         longer candidate list costs coverage through F6 by dilution alone; without the matched
         control the two effects are confounded.
STEP 11  the candidate-availability budget curve, 50 / 64 / 80 / 100 / 128 (256 diagnostic only).

  python scratchpad/_l1tp_p50.py <ds>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1sr_eval as EV
import _l1tp_core as TP
import _l1tp_build as TB
import _l1tp_run as TR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
PBUD = [6, 12, 20, 50]
BUDG = [50, 64, 80, 100, 128, 256]

# The orderings carried into the exact-P50 contract.  Each is one quantity over the Pi->Pj table
# aggregated to Pj, except FUSE_* which is the single permitted parameter-free combination.
CAND0 = ["C0_SINGLE_TRIPLET", "TR_EDGE_SUM", "CORE_EXIT_OFFSET", "CORE_EXIT_ONLY",
        "TRIPLET_S4", "PATH_S4"]
# STEP 8 allows exactly ONE parameter-free combination, and only between signals shown to be
# genuinely complementary.  Equal rank fusion of the frozen node ordering with the one
# transition-native signal.  No weights, no lambda, no threshold, no grid.
FUSE = "FUSE_NODE_S4_X_CORE_EXIT"
CAND = CAND0 + [FUSE]


def equal_rank_fusion(a, b):
    """one equal-weight RRF over two orderings, K0 = 60.  Absent -> no contribution.  Ties broken
    by the first ordering rank, then partition id.  Exactly the F6 fusion shape, no parameters."""
    ra = {int(p): i for i, p in enumerate(np.asarray(a))}
    rb = {int(p): i for i, p in enumerate(np.asarray(b))}
    MISS = 10 ** 6
    u = list(dict.fromkeys([int(p) for p in np.asarray(a)] + [int(p) for p in np.asarray(b)]))
    sc = [(-(1.0 / (EV.K0 + ra[p]) if p in ra else 0.0)
           - (1.0 / (EV.K0 + rb[p]) if p in rb else 0.0), ra.get(p, MISS), p) for p in u]
    return np.array([p for _, _, p in sorted(sc)], np.int64)


def orderings(r, prot, base50):
    """all candidate orderings of the target partitions for one query."""
    pi = r["pi"].astype(np.int64); pj = r["pj"].astype(np.int64)
    up, ip = np.unique(pj, return_inverse=True)
    nk = len(up)
    first = np.full(nk, 1 << 30, np.int64); np.minimum.at(first, ip, r["first_jj"].astype(np.int64))
    mx = np.full(nk, -np.inf); np.maximum.at(mx, ip, r["max_T0"])
    sm = np.bincount(ip, weights=r["sum_T0"], minlength=nk)
    cnt = np.bincount(ip, weights=r["n_tgt"].astype(np.float64), minlength=nk)
    mh = np.full(nk, 1 << 20, np.float64); np.minimum.at(mh, ip, r["min_T3"].astype(np.float64))
    ps = np.full(nk, -np.inf); np.maximum.at(ps, ip, r["max_psum"])

    # ---- the one signal only the TRANSITION representation can express: a transition that LEAVES
    # the protected core.  A filter on the existing protected set -- no weight, no threshold.
    inc = np.fromiter((int(p) in prot for p in pi), bool, len(pi))
    ce = np.full(nk, -np.inf)
    if inc.any():
        np.maximum.at(ce, ip[inc], r["max_T0"][inc])
    ce_seen = np.zeros(nk, bool)
    if inc.any():
        ce_seen[np.unique(ip[inc])] = True

    def order(val, sgn=-1.0):
        return up[np.lexsort((up, first, sgn * val))]

    def rrf(keys):
        """equal rank fusion, K0 = 60, ties by arrival then partition id.  No weights."""
        acc = np.zeros(nk)
        for val, sgn in keys:
            o = np.lexsort((up, first, sgn * val))
            rk = np.empty(nk, np.int64); rk[o] = np.arange(nk)
            acc += 1.0 / (EV.K0 + rk)
        return up[np.lexsort((up, first, -acc))]

    O = {"C0_SINGLE_TRIPLET": order(mx),
         "TR_EDGE_SUM": order(sm),
         "CORE_EXIT_OFFSET": order(np.where(ce_seen, ce, -np.inf)),
         # core-exit first (by offset), then everything else in the C0 order
         "CORE_EXIT_ONLY": np.concatenate([
             up[np.lexsort((up, first, -ce))][:int(ce_seen.sum())],
             np.array([p for p in order(mx) if not ce_seen[np.searchsorted(up, p)]], np.int64)])
             if ce_seen.any() else order(mx),
         # STEP 9: the frozen NODE-S4 shape with the TRIPLET unit substituted for the node unit
         "TRIPLET_S4": rrf([(first.astype(float), +1.0), (cnt, -1.0), (mh, +1.0), (mx, -1.0)]),
         # ... and with the CHAIN score in place of the single-transition offset
         "PATH_S4": rrf([(first.astype(float), +1.0), (cnt, -1.0), (mh, +1.0), (ps, -1.0)])}
    return O, up


def run(ds, log=log):
    T = dict(np.load(f"{TP.TPD}/data/tr_{ds}.npz", allow_pickle=False))
    nq = int(T["nq"][0])
    S = EV.substrate(ds)
    hard = S["hard"].astype(np.int64)
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S["hops"] is not None else np.full(nq, -1)
    BL = [("ALL", np.ones(nq, bool))]
    if len(hops) and int(np.min(hops)) >= 0:
        for h in sorted(set(int(x) for x in hops)):
            BL.append((f"hop{h}", hops == h))
    NEED = T["NEEDN"]; has = NEED > 0

    frozen = [np.asarray(EV.sf_from_cache(S["z"], qi, hard, "P0_S4")) for qi in range(S["nq"])]
    K = [len(frozen[qi]) for qi in range(nq)]
    ORD = {c: [] for c in CAND}
    REC = {c: np.zeros((nq, len(PBUD))) for c in list(CAND) + ["NODE_S4_FROZEN"]}
    COV6 = {c: np.zeros(nq, np.int8) for c in list(CAND) + ["NODE_S4_FROZEN"]}
    for qi in range(nq):
        c = ctxs[qi]
        need = set(int(p) for p in goldp[qi]) - c["prot_set"]
        O, up = orderings(TR.rows(T, qi), c["prot_set"], c["base50"])
        O[FUSE] = equal_rank_fusion(frozen[qi], O["CORE_EXIT_OFFSET"])
        for k in CAND:
            ORD[k].append(np.asarray(O[k], np.int64))
        for k in list(CAND) + ["NODE_S4_FROZEN"]:
            o = frozen[qi] if k == "NODE_S4_FROZEN" else O[k]
            REC[k][qi] = TR._rec(np.asarray(o), need, PBUD)
            COV6[k][qi] = int(bool(need) and set(int(x) for x in np.asarray(o)[:6]) >= need)

    OUT = {"ds": ds, "nq": nq,
           "frozen_ordering_len_mean": round(float(np.mean(K)), 1),
           "triplet_ordering_len_mean": round(float(np.mean([len(ORD[CAND[0]][q])
                                                             for q in range(nq)])), 1),
           "STEP9_PARTITION_RECALL": {}, "STEP8_COMPLEMENTARITY": {},
           "STEP10_EXACT_P50": {}, "STEP10_EXACT_P50_DEPTH_MATCHED": {}, "STEP11_BUDGET": {}}
    for k in list(CAND) + ["NODE_S4_FROZEN"]:
        OUT["STEP9_PARTITION_RECALL"][k] = {
            b: {f"R@{c2}": round(float(REC[k][mk & has, i].mean()), 4)
                for i, c2 in enumerate(PBUD)} for b, mk in BL}

    # ---------------- STEP 8: complementarity, measured BEFORE any combination
    base = COV6["NODE_S4_FROZEN"]
    for k in CAND:
        a = COV6[k]
        OUT["STEP8_COMPLEMENTARITY"][f"{k}_vs_NODE_S4_FROZEN"] = {
            "solved_by_both": int(((a == 1) & (base == 1)).sum()),
            "solved_only_by_triplet": int(((a == 1) & (base == 0)).sum()),
            "solved_only_by_node": int(((a == 0) & (base == 1)).sum()),
            "union_ceiling": round(float(((a == 1) | (base == 1)).mean()), 4),
            "triplet_alone": round(float(a.mean()), 4),
            "node_alone": round(float(base.mean()), 4)}

    # ---------------- STEP 10: through the unchanged F6 at exact P = 50, B = 6
    i_safe, _ = EV.evaluate(S, frozen, "A_F6", (EV.P,))
    i_safe = i_safe[EV.P][:nq]
    OUT["SAFE"] = {b: round(float(i_safe[mk].mean()), 4) for b, mk in BL}
    log(f"[{ds}] SAFE " + "  ".join(f"{b} {OUT['SAFE'][b]:.4f}" for b, _ in BL) +
        f"   frozen list {OUT['frozen_ordering_len_mean']}  "
        f"triplet list {OUT['triplet_ordering_len_mean']}")
    for tag, cut in (("STEP10_EXACT_P50", None), ("STEP10_EXACT_P50_DEPTH_MATCHED", K)):
        for k in CAND:
            SF = [(ORD[k][qi] if cut is None else ORD[k][qi][:cut[qi]]) for qi in range(nq)]
            SF = SF + frozen[nq:]
            ind, churn = EV.evaluate(S, SF, "A_F6", (EV.P,))
            ind = ind[EV.P][:nq]
            row = {b: round(float(ind[mk].mean()), 4) for b, mk in BL}
            row["churn"] = round(float(churn[:nq].mean()), 3)
            row["vs_SAFE"] = PP.mcnemar(ind, i_safe)
            for b, mk in BL[1:]:
                row[f"vs_SAFE_{b}"] = PP.mcnemar(ind[mk], i_safe[mk])
            OUT[tag][k] = row
            m = row["vs_SAFE"]
            log(f"[{ds}] {'FULL ' if cut is None else 'MATCH'}/{k:20s} " +
                "  ".join(f"{b} {row[b]:.4f}" for b, _ in BL) +
                f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")

    # ---------------- STEP 11: candidate availability curve.  50 is the contract; the rest show
    # whether the required-partition curve moves LEFT toward it.  256 is reference only.
    ib, meta = EV.evaluate_budget(S, frozen, "A_F6", BUDG)
    OUT["STEP11_BUDGET"]["NODE_S4_FROZEN"] = {
        f"k{k}": {b: round(float(ib[k][:nq][mk].mean()), 4) for b, mk in BL} for k in BUDG}
    best = max(CAND, key=lambda k: OUT["STEP10_EXACT_P50_DEPTH_MATCHED"][k]["ALL"])
    for k in [best] + (["C0_SINGLE_TRIPLET"] if best != "C0_SINGLE_TRIPLET" else []):
        SF = [ORD[k][qi][:K[qi]] for qi in range(nq)] + frozen[nq:]
        ibk, _ = EV.evaluate_budget(S, SF, "A_F6", BUDG)
        OUT["STEP11_BUDGET"][f"{k}_DEPTH_MATCHED"] = {
            f"k{k2}": {b: round(float(ibk[k2][:nq][mk].mean()), 4) for b, mk in BL} for k2 in BUDG}
    for k, v in OUT["STEP11_BUDGET"].items():
        log(f"[{ds}] STEP11 {k:34s} " + "  ".join(f"{kk} {vv['ALL']:.4f}" for kk, vv in v.items()))

    json.dump(OUT, open(f"{TP.TPD}/diag/p50_{ds}.json", "w"), indent=1)
    log(f"[{ds}] wrote diag/p50_{ds}.json")
    return OUT


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["metaqa"]):
        run(d)
