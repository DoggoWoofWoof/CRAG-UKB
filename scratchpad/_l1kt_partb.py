"""L1 CAPACITY PHASE -- PART B: the TRANSFORMATION EVICTION AUDIT.

The admission phase proved that at fixed K(q) every gain and every loss is an EVICTION.  So the
decision the machinery actually has to make is not "admit or not" but "is this proposed SRC
candidate b worth more than the SAFE candidate a it displaces?".  PART B asks whether the
transformation algebra supplies NEW mathematical information for exactly that decision.

Forced pairs, deterministic, no learning: at identical K(q) the SRC pool evicts a set of SAFE
candidates and admits an equal number of new ones; the i-th evicted is paired with the i-th
admitted in pool order.  Labels are EVALUATION ONLY and are never fed to any scorer:

    GOOD_SWAP  b is required-and-missing, a is not
    BAD_SWAP   a is required-and-missing, b is not

Partition scores (B3), each tested on its own, no combination grid:

    V0_NOTRANSFORM  control -- max over edges into p of cos(z0, x_v)
    V1_SINGLE       max over edges into p of cos(T_e(z0), x_v)
    V2_CHAIN        max over edges into p of cos(z_e, x_v), z_e composed along the real parent chain
    V3_MAXVIEW      max(V0, V2) -- the multi-state view, states not collapsed to one scalar
    V4_SRC_EXIT     V2 restricted to edges whose SOURCE node lies in the protected core

This module also supplies the NO_TRANSFORM control for the STEP-3 marginal partition rank that the
transformation module reported without one.

  python scratchpad/_l1kt_partb.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1tp_core as TP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1bc_diag as DGX
import _l1ca_admit as AD
import _l1kt_tf as TF

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
FAMS, ZS, EPS = TF.FAMS, TF.ZS, TF.EPS
VARS = ["V1_SINGLE", "V2_CHAIN", "V3_MAXVIEW", "V4_SRC_EXIT"]
PLS = ["A1_SRC", "A2_HYB"]
KS = [6, 12, 20, 50]
REF = ["FROZEN_SAFE", "SRC_ASSIGN", "OFFSET_RAW"]
FLOOR = -2.0
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def pbest(tc, pix, npart, sub=None):
    b = np.full(npart, -np.inf, np.float64)
    if sub is None:
        np.maximum.at(b, pix, tc)
    elif sub.any():
        np.maximum.at(b, pix[sub], tc[sub])
    return b


def auc(p, n):
    return DGX._auc(np.asarray(p, float), np.asarray(n, float)) if len(p) and len(n) else None


def run(ds, stride=1, log=log):
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    Xn, hard = C["Xn"], C["hard"]
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    qs = list(range(0, nq, stride))
    SC = ["V0_NOTRANSFORM"] + [f"{z}/{f}/{v}" for z in ZS for f in FAMS for v in VARS]
    M = {(pl, s): {"GOOD": [], "BAD": [], "GOOD_h": [], "BAD_h": [], "GOOD_ev": [], "BAD_ev": []}
         for pl in PLS for s in SC}
    prank0 = []
    RK = REF + SC
    rec = {(nm, k): [] for nm in RK for k in KS}
    rech = {(nm, k, h): [] for nm in RK for k in KS for h in (1, 2, 3)}
    npair = {pl: 0 for pl in PLS}
    nlab = {pl: {"GOOD": 0, "BAD": 0, "TIE": 0, "POSITIONAL": 0} for pl in PLS}
    par_ok, ms, nseen = 0, 0.0, 0
    for qi in qs:
        t = time.perf_counter()
        E, rq, par, st = TP.replay(C, qi)
        par_ok += int(par)
        u, v = E["u"], E["v"]
        if not len(u):
            continue
        nseen += 1
        c, r = ctxs[qi], BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        cpos = c["cpos"]
        Xsel, cands, sc = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(pp): i for i, (_, _, pp) in enumerate(sc)}
        K = len(cands)
        REQ = set(int(p) for p in goldp[qi])
        need = REQ - pset
        a1 = AD.a1_pool(r, cpos, K, pset)
        pools = {"A1_SRC": a1,
                 "A2_HYB": AD.a2_pool(r, ixp, cands, cpos, K, prot, a1)[0]}
        pairs = {}
        for pl, npool in pools.items():
            ns, cs = set(npool), set(cands)
            ev = [p for p in cands if p not in ns]
            ad = [p for p in npool if p not in cs]
            # every evicted SAFE candidate could have been kept in place of every admitted SRC
            # candidate, so the forced-swap population is the labelled cross product.  Only pairs
            # with exactly one required-and-missing side carry a label; the rest are ties.
            evn = [p for p in ev if p in need]
            evo = [p for p in ev if p not in need]
            adn = [p for p in ad if p in need]
            ado = [p for p in ad if p not in need]
            pr = [(a, b, "GOOD") for a in evo for b in adn]
            pr += [(a, b, "BAD") for a in evn for b in ado]
            nlab[pl]["GOOD"] += len(evo) * len(adn)
            nlab[pl]["BAD"] += len(evn) * len(ado)
            nlab[pl]["TIE"] += len(ev) * len(ad) - len(evo) * len(adn) - len(evn) * len(ado)
            nlab[pl]["POSITIONAL"] += min(len(ev), len(ad))
            npair[pl] += len(ev) * len(ad)
            pairs[pl] = pr
        if not any(pairs.values()):
            ms += time.perf_counter() - t
            continue
        tp = hard[v]
        upart = np.unique(tp[tp >= 0])
        pinv = {int(p): i for i, p in enumerate(upart)}
        pix = np.array([pinv.get(int(p), -1) for p in tp], np.int64)
        keep = pix >= 0
        U, V = gx(u), gx(v)
        cuv = (U * V).sum(1)
        hop = E["T3_HOP"].astype(np.int64)
        hopmasks = [(h, np.nonzero(hop == h)[0]) for h in sorted(set(int(x) for x in hop))]
        hopmasks = [(h, m) for h, m in hopmasks if len(m)]
        is_parent = np.zeros(len(u), bool)
        is_parent[E["parent_tid"][E["parent_tid"] >= 0]] = True
        exitm = np.array([int(p) in pset for p in hard[u]])
        one = np.ones(len(u), np.float32)
        Z0 = {"q_raw": C["Qm"][qi].astype(np.float32), "r_q": rq.astype(np.float32)}
        MISSREQ = need - set(int(p) for p in Xsel)
        pos_mask = np.array([int(p) in MISSREQ for p in upart]) if len(upart) else np.zeros(0, bool)
        tab = {}
        z0r = Z0["q_raw"] / max(float(np.linalg.norm(Z0["q_raw"])), EPS)
        b0 = V @ z0r
        tab["V0_NOTRANSFORM"] = pbest(b0[keep], pix[keep], len(upart))
        if MISSREQ and pos_mask.any():
            o = np.argsort(-tab["V0_NOTRANSFORM"], kind="mergesort")
            rr = np.nonzero(pos_mask[o])[0] / max(len(upart) - 1, 1)
            prank0.append(float(rr.mean()))
        for z in ZS:
            z0 = Z0[z] / max(float(np.linalg.norm(Z0[z])), EPS)
            uz0, vz0 = U @ z0, V @ z0
            for f in FAMS:
                tv, n2, _ = TF.scalar_apply(f, uz0, vz0, one, one, cuv, uz0, vz0)
                s1 = tv / np.sqrt(np.maximum(n2, EPS))
                tv, n2, _ = TF.transport(U, V, z0, hopmasks, E["parent_tid"], is_parent,
                                         f, cuv, uz0, vz0)
                s2 = tv / np.sqrt(np.maximum(n2, EPS))
                p1 = pbest(s1[keep], pix[keep], len(upart))
                p2 = pbest(s2[keep], pix[keep], len(upart))
                tab[f"{z}/{f}/V1_SINGLE"] = p1
                tab[f"{z}/{f}/V2_CHAIN"] = p2
                tab[f"{z}/{f}/V3_MAXVIEW"] = np.maximum(tab["V0_NOTRANSFORM"], p2)
                tab[f"{z}/{f}/V4_SRC_EXIT"] = pbest(s2[keep], pix[keep], len(upart),
                                                    sub=exitm[keep])
        # ---- STEP 7: rank partitions ALREADY INSIDE FULL_VISITED, missing-required recall @k
        if MISSREQ:
            univ = sorted(set(int(x) for x in r["pid"]) - pset)
            uix = np.array([pinv.get(x, -1) for x in univ], np.int64)
            miss = np.array([x in MISSREQ for x in univ])
            den = len(MISSREQ)
            a50 = AD.a1_pool(r, cpos, max(KS), pset)
            so = {x: i for i, x in enumerate(a50)}
            offp = pbest(E["T0_OFFSET"].astype(np.float64)[keep], pix[keep], len(upart))
            ref = {"FROZEN_SAFE": np.array([-safe_ord.get(x, 10 ** 6) for x in univ], float),
                   "SRC_ASSIGN": np.array([-so.get(x, 10 ** 6) for x in univ], float),
                   "OFFSET_RAW": np.where(uix >= 0, offp[np.maximum(uix, 0)], FLOOR)}
            for nm in RK:
                a = ref[nm] if nm in ref else np.where(uix >= 0, tab[nm][np.maximum(uix, 0)], FLOOR)
                a = np.where(np.isfinite(a), a, FLOOR)
                h = miss[np.argsort(-a, kind="mergesort")]
                for k in KS:
                    rec[(nm, k)].append(float(h[:k].sum()) / den)
                    if int(hops[qi]) in (1, 2, 3):
                        rech[(nm, k, int(hops[qi]))].append(float(h[:k].sum()) / den)
        hq = int(hops[qi])
        for pl, pr in pairs.items():
            for a, b, lab in pr:
                ia, ib = pinv.get(a, -1), pinv.get(b, -1)
                for s in SC:
                    tb = tab[s]
                    sa = float(tb[ia]) if ia >= 0 and np.isfinite(tb[ia]) else FLOOR
                    sb = float(tb[ib]) if ib >= 0 and np.isfinite(tb[ib]) else FLOOR
                    M[(pl, s)][lab].append(sb - sa)
                    M[(pl, s)][lab + "_h"].append(hq)
                    if sa > FLOOR and sb > FLOOR:
                        M[(pl, s)][lab + "_ev"].append(sb - sa)
        ms += time.perf_counter() - t
        if nseen % 200 == 0:
            log(f"   {ds} {nseen}/{len(qs)}")
    return dict(ds=ds, nq=nseen, nqs=len(qs), par=par_ok, M=M, SC=SC, RK=RK, rec=rec,
                rech=rech, prank0=prank0, npair=npair, nlab=nlab, ms=ms)


def report(R):
    ds = R["ds"]
    out = {"ds": ds, "n_queries": R["nq"], "parity": f"{R['par']}/{R['nqs']}",
           "STEP3_partition_rank_NO_TRANSFORM": (round(float(np.mean(R["prank0"])), 4)
                                                 if R["prank0"] else None),
           "pairs": {pl: dict(R["nlab"][pl], total=R["npair"][pl]) for pl in PLS},
           "STEP7_missing_required_admission_recall": {
               nm: {f"@{k}": (round(float(np.mean(R["rec"][(nm, k)])), 4)
                              if R["rec"][(nm, k)] else None) for k in KS}
               for nm in R["RK"]},
           "STEP7_n_queries_with_missing_required": len(R["rec"][(R["RK"][0], KS[0])]),
           "STEP7_by_hop": {nm: {f"hop{h}": {f"@{k}": (round(float(np.mean(R["rech"][(nm, k, h)])), 4)
                                                       if R["rech"][(nm, k, h)] else None)
                                             for k in KS} for h in (1, 2, 3)}
                            for nm in R["RK"]} if ds == "metaqa" else {},
           "B4_swap_margin": {}, "B4_by_hop": {},
           "latency_ms_per_q": round(1000 * R["ms"] / max(R["nq"], 1), 2)}
    for pl in PLS:
        for s in R["SC"]:
            d = R["M"][(pl, s)]
            g, b = d["GOOD"], d["BAD"]
            sgn = ((sum(1 for x in g if x > 0) + sum(1 for x in b if x < 0))
                   / max(len(g) + len(b), 1))
            out["B4_swap_margin"][f"{pl}/{s}"] = {
                "n_good": len(g), "n_bad": len(b),
                "mean_good": round(float(np.mean(g)), 4) if g else None,
                "mean_bad": round(float(np.mean(b)), 4) if b else None,
                "AUC": round(auc(g, b), 4) if g and b else None,
                "sign_acc": round(float(sgn), 4),
                "AUC_both_evid": (round(auc(d["GOOD_ev"], d["BAD_ev"]), 4)
                                  if d["GOOD_ev"] and d["BAD_ev"] else None),
                "n_good_ev": len(d["GOOD_ev"]), "n_bad_ev": len(d["BAD_ev"])}
            if ds == "metaqa":
                bh = {}
                for h in (1, 2, 3):
                    gh = [x for x, y in zip(g, d["GOOD_h"]) if y == h]
                    bb = [x for x, y in zip(b, d["BAD_h"]) if y == h]
                    bh[f"hop{h}"] = {"n_good": len(gh), "n_bad": len(bb),
                                       "AUC": round(auc(gh, bb), 4) if gh and bb else None}
                out["B4_by_hop"][f"{pl}/{s}"] = bh
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    rp = report(run(ds, stride))
    json.dump(rp, open(f"{KTD}/diag/pb_{ds}.json", "w"), indent=1)
    log(ds, "pairs", json.dumps(rp["pairs"]))
    log(ds, "NO_TRANSFORM prank", rp["STEP3_partition_rank_NO_TRANSFORM"])
