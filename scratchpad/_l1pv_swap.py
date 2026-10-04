"""CORE-EXIT PROVENANCE AUDIT -- STEPS 5-7: TRUSTED_SRC_ASSIGNMENT at exact P50.

Same fixed-K replacement framework as `_l1kt_swap.py`, with the transformation algebra REMOVED.
Every score here is pure provenance -- `max over a family-restricted edge set of cos(q, x_u)`:

    universe = SAFE pool ++ SRC-assignment candidates not already in it   (both at K(q))
    pool     = top K(q) of the universe ordered by  (-provenance_score, frozen_safe_rank, pid)

K = frozen K(q), B = 6, P = 50.  No transformation, no weight, no learned gate, no dataset branch,
no threshold: a family either supplies evidence for a candidate or it does not.

`SRC_ALL` is an exact parity anchor -- it is `V6_EXIT_SRC_SIM` from the prior phase, so its p50
numbers must reproduce `L1_CAPACITY/diag/swap_<ds>.json` byte for byte.

  python scratchpad/_l1pv_swap.py <ds> <score_key> [score_key ...]
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
import _l1kt_partb as PB
import _l1pv_fam as FM

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PROVENANCE"
P, B, MISS, EPS, FLOOR = BC.P, BC.B, BC.MISS, 1e-12, PB.FLOOR
SELS = ["F6", "G4"]
SC = ["SRC_ALL", "SRC_NER", "SRC_STRUCT_ONLY", "ALL_NER", "ALL_STRUCT_ONLY"]
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
    FAM = FM.Fam(ds, log)
    NERBIT = FM.BIT["NER"]
    qs = list(range(0, nq, stride))
    hit = {(s, sel): np.zeros(len(qs), np.int8) for s in scores for sel in SELS}
    base = np.zeros(len(qs), np.int8)
    hq = np.zeros(len(qs), np.int32)
    changed = {s: 0 for s in scores}
    cov = {s: [0, 0] for s in scores}          # candidates with ANY evidence under this score
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
            fam, _ = FAM.of(u, v)
            nerm = (fam & NERBIT) > 0
            exitm = np.array([int(p) in pset for p in hard[u]])
            z0r = C["Qm"][qi].astype(np.float32)
            z0r = z0r / max(float(np.linalg.norm(z0r)), EPS)
            uz0 = gx(u) @ z0r
            subs = {"SRC_ALL": exitm, "SRC_NER": exitm & nerm, "SRC_STRUCT_ONLY": exitm & ~nerm,
                    "ALL_NER": nerm, "ALL_STRUCT_ONLY": ~nerm}
            for s in scores:
                tab[s] = PB.pbest(uz0[keep], pix[keep], len(upart), sub=subs[s][keep])
        else:
            pinv = {}
        for s in scores:
            tb = tab.get(s)

            def sco(p):
                i = pinv.get(p, -1)
                return float(tb[i]) if (tb is not None and i >= 0 and np.isfinite(tb[i])) else FLOOR

            cov[s][0] += sum(1 for p in uni if sco(p) > FLOOR)
            cov[s][1] += len(uni)
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
                changed=changed, cov=cov, ms=ms)


def report(R):
    ds, nq = R["ds"], R["nq"]
    out = {"ds": ds, "nq": nq, "parity": f"{R['par']}/{nq}",
           "queries_with_a_changed_pool": {s: R["changed"][s] for s in R["scores"]},
           "candidate_evidence_coverage": {s: round(R["cov"][s][0] / max(R["cov"][s][1], 1), 4)
                                           for s in R["scores"]},
           "p50": {}, "score_overhead_ms_per_q": round(1000 * R["ms"] / max(nq, 1), 3)}
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
    scores = sys.argv[2:] or SC
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    rp = report(run(ds, scores))
    fp = f"{KTD}/diag/pvswap_{ds}.json"
    json.dump(rp, open(fp, "w"), indent=1)
    log("wrote", fp)
    print(json.dumps(rp["p50"].get("ALL", {}), indent=1))
