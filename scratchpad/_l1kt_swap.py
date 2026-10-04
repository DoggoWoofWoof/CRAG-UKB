"""PART B5 / STEPS 9-10: fixed-K candidate REPLACEMENT driven by the transformation score.

The frozen selector is algebraically closed to APPENDED candidates (see closure.json), so the only
way transformation evidence can reach the output at all is by REPLACING pool members.  This module
does exactly that, at the frozen candidate depth and with no threshold anywhere:

    universe = SAFE pool ++ SRC-assignment candidates not already in it   (both at K(q))
    pool     = top K(q) of the universe ordered by  (-transform_score, frozen_safe_rank, pid)

|pool| is exactly K(q), the ordering is total and deterministic, nothing is learned or tuned, and the
final output is still exactly P = 50 with B = 6.  delta = the frozen SAFE pool is recovered whenever
the transformation score agrees with the frozen order, which is the built-in sanity anchor.

  python scratchpad/_l1kt_swap.py <ds> <score_key> [score_key ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1pp_core as PP
import _l1tp_core as TP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1ca_admit as AD
import _l1kt_tf as TF
import _l1kt_partb as PB

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
P, B, MISS, EPS = BC.P, BC.B, BC.MISS, TF.EPS
SELS = ["F6", "G4"]
FLOOR = PB.FLOOR
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def run(ds, scores, stride=1, log=log):
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    Xn, hard = C["Xn"], C["hard"]
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    qs = list(range(0, nq, stride))
    need_zf = sorted({tuple(k.split("/")[:2]) for k in scores if "/" in k})
    hit = {(s, sel): np.zeros(len(qs), np.int8) for s in scores for sel in SELS}
    base = np.zeros(len(qs), np.int8)
    hq = np.zeros(len(qs), np.int32)
    changed = {s: 0 for s in scores}
    nexit = [0, 0]
    par_ok, ms = 0, 0.0
    for j, qi in enumerate(qs):
        E, rq, par, st = TP.replay(C, qi)
        par_ok += int(par)
        c, r = ctxs[qi], BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        Xsel, cands, sc = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        K = len(cands)
        REQ = set(int(p) for p in goldp[qi])
        hq[j] = int(hops[qi])
        base[j] = int(REQ <= (pset | set(int(p) for p in Xsel)))
        a1 = AD.a1_pool(r, cpos, K, pset)
        uni = list(cands) + [p for p in a1 if p not in set(cands)]
        u, v = E["u"], E["v"]
        t = time.perf_counter()
        tab = {}
        if len(u):
            tp = hard[v]
            upart = np.unique(tp[tp >= 0])
            pinv = {int(p): i for i, p in enumerate(upart)}
            pix = np.array([pinv.get(int(p), -1) for p in tp], np.int64)
            keep = pix >= 0
            U, V = gx(u), gx(v)
            cuv = (U * V).sum(1)
            hop = E["T3_HOP"].astype(np.int64)
            hm = [(h, np.nonzero(hop == h)[0]) for h in sorted(set(int(x) for x in hop))]
            hm = [(h, m) for h, m in hm if len(m)]
            isp = np.zeros(len(u), bool)
            isp[E["parent_tid"][E["parent_tid"] >= 0]] = True
            exitm = np.array([int(p) in pset for p in hard[u]])
            one = np.ones(len(u), np.float32)
            Z0 = {"q_raw": C["Qm"][qi].astype(np.float32), "r_q": rq.astype(np.float32)}
            z0r = Z0["q_raw"] / max(float(np.linalg.norm(Z0["q_raw"])), EPS)
            v0 = PB.pbest((V @ z0r)[keep], pix[keep], len(upart))
            tab["V0_NOTRANSFORM"] = v0
            tab["OFFSET_RAW"] = PB.pbest(E["T0_OFFSET"].astype(np.float64)[keep],
                                         pix[keep], len(upart))
            # provenance control: NO transformation at all -- how well the query matches the
            # PROTECTED-CORE node that points at p.  Under the orthogonal-degeneracy identity
            # cos(T z0, x_v) = cos(z0, x_u) this is exactly what V4_SRC_EXIT reduces to on the
            # hop-1 core-exit edges that dominate that view.
            tab["V6_EXIT_SRC_SIM"] = PB.pbest((gx(u) @ z0r)[keep], pix[keep], len(upart),
                                              sub=exitm[keep])
            nexit[0] += int(exitm.sum()); nexit[1] += int((exitm & (hop == 1)).sum())
            for z, f in need_zf:
                z0 = Z0[z] / max(float(np.linalg.norm(Z0[z])), EPS)
                uz0, vz0 = U @ z0, V @ z0
                tv, n2, _ = TF.scalar_apply(f, uz0, vz0, one, one, cuv, uz0, vz0)
                s1 = tv / np.sqrt(np.maximum(n2, EPS))
                tv, n2, _ = TF.transport(U, V, z0, hm, E["parent_tid"], isp, f, cuv, uz0, vz0)
                s2 = tv / np.sqrt(np.maximum(n2, EPS))
                tab[f"{z}/{f}/V1_SINGLE"] = PB.pbest(s1[keep], pix[keep], len(upart))
                tab[f"{z}/{f}/V2_CHAIN"] = PB.pbest(s2[keep], pix[keep], len(upart))
                tab[f"{z}/{f}/V3_MAXVIEW"] = np.maximum(v0, tab[f"{z}/{f}/V2_CHAIN"])
                tab[f"{z}/{f}/V4_SRC_EXIT"] = PB.pbest(s2[keep], pix[keep], len(upart),
                                                       sub=exitm[keep])
        else:
            pinv = {}
        for s in scores:
            tb = tab.get(s)
            def sco(p):
                i = pinv.get(p, -1)
                return float(tb[i]) if (tb is not None and i >= 0 and np.isfinite(tb[i])) else FLOOR
            pool = sorted(uni, key=lambda p: (-sco(p), safe_ord.get(p, MISS), p))[:K]
            changed[s] += int(sorted(pool) != sorted(cands))
            for sel in SELS:
                pick = (AD.f6_on_pool(pool, spos, rpos, cpos) if sel == "F6"
                        else AD.g4_on_pool(r, ixp, prot, pool, safe_ord)[0])[:B]
                fs = pset | set(pick)
                assert len(fs) == P
                hit[(s, sel)][j] = int(REQ <= fs)
        ms += time.perf_counter() - t
        if (j + 1) % 400 == 0:
            log(f"   {ds} {j+1}/{len(qs)}")
    return dict(ds=ds, nq=len(qs), par=par_ok, hit=hit, base=base, hq=hq, scores=scores,
                changed=changed, ms=ms, nexit=nexit)


def report(R):
    ds, nq = R["ds"], R["nq"]
    out = {"ds": ds, "nq": nq, "parity": f"{R['par']}/{nq}",
           "queries_with_a_changed_pool": {s: R["changed"][s] for s in R["scores"]},
           "p50": {}, "transform_overhead_ms_per_q": round(1000 * R["ms"] / max(nq, 1), 3),
           "core_exit_edges": R["nexit"][0],
           "core_exit_edges_at_hop1": R["nexit"][1],
           "frac_core_exit_at_hop1": (round(R["nexit"][1] / R["nexit"][0], 4)
                                      if R["nexit"][0] else None)}
    for nm, m in LG.masks(ds, R["hq"], nq):
        out["p50"][nm] = {"FROZEN": round(float(R["base"][m].mean()), 4)}
        for s in R["scores"]:
            for sel in SELS:
                cur = R["hit"][(s, sel)]
                st = PP.mcnemar(cur[m], R["base"][m])
                out["p50"][nm][f"{s}/{sel}"] = {
                    "acc": round(float(cur[m].mean()), 4), "net": st["net"],
                    "gained": st["gained"], "lost": st["lost"],
                    "p": round(st["mcnemar_p"], 4), "sig": bool(st["sig"])}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    scores = sys.argv[2:] or ["q_raw/F3_RANK1_TRANSPORT/V2_CHAIN"]
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    rp = report(run(ds, scores))
    json.dump(rp, open(f"{KTD}/diag/swap_{ds}.json", "w"), indent=1)
    log(ds, json.dumps(rp["p50"].get("ALL", {}), indent=1))
