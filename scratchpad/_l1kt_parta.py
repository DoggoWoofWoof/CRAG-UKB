"""L1 CAPACITY PHASE -- PART A: the micro-K capacity ladder.

The admission phase closed with one unambiguous mechanism: at fixed K(q) every gain and every loss
is an EVICTION (75 gains of which 1 was a newly admitted partition; 162 losses, all of them SAFE
candidates removed from the pool).  PART A tests the first of the two remaining explanations --
that K is simply slightly too small -- by never evicting anything at all.

    pool(delta) = ENTIRE frozen SAFE pool, in frozen order,
                  ++ SRC-assignment candidates not already present,
                  truncated to K(q) + delta,  delta in {0, 4, 8, 16, 32}

delta = 0 must reproduce the frozen SAFE pool exactly; that is the parity gate.  Graph work is
byte-identical (the same replayed substrate), the final output is still EXACTLY P = 50 with B = 6,
and nothing here is learned or tuned.  This is an internal candidate depth, not a wider final P.

  python scratchpad/_l1kt_parta.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1pp_core as PP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1ca_admit as AD

P, B, K0, MISS = BC.P, BC.B, BC.K0, BC.MISS
KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
DELTAS = [0, 4, 8, 16, 32]
SELS = ["F6", "G4"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def append_pool(cands, a1, K):
    """the entire SAFE pool in frozen order, then SRC candidates not already in it, capped at K."""
    out, seen = list(cands), set(cands)
    for p in a1:
        if len(out) >= K:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:K]


def run(ds, log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    par = 0
    orc = {d: np.zeros(nq, np.int8) for d in DELTAS}
    hit = {(d, s): np.zeros(nq, np.int8) for d in DELTAS for s in SELS}
    size = {d: np.zeros(nq) for d in DELTAS}
    ms = {d: 0.0 for d in DELTAS}
    ms_sel = {(d, s): 0.0 for d in DELTAS for s in SELS}
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        X, cands, sc = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        K = len(cands)
        REQ = set(int(p) for p in goldp[qi])
        a1 = AD.a1_pool(r, cpos, K + max(DELTAS), pset)
        for d in DELTAS:
            t = time.perf_counter()
            pool = append_pool(cands, a1, K + d)
            ms[d] += time.perf_counter() - t
            size[d][qi] = len(pool)
            if d == 0:
                assert pool == cands, "delta=0 must be the frozen SAFE pool"
            need = REQ - pset
            orc[d][qi] = int(len(need) <= B and need <= set(pool))
            for s in SELS:
                t = time.perf_counter()
                pick = (AD.f6_on_pool(pool, spos, rpos, cpos) if s == "F6"
                        else AD.g4_on_pool(r, ixp, prot, pool, safe_ord)[0])[:B]
                ms_sel[(d, s)] += time.perf_counter() - t
                fs = pset | set(pick)
                assert len(fs) == P
                hit[(d, s)][qi] = int(REQ <= fs)
        # the frozen replay gate: delta=0 with F6 must be the frozen system
        par += int(hit[(0, "F6")][qi] == int(REQ <= (pset | set(X))))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}")
    assert par == nq, f"F6 parity {par}/{nq}"
    return dict(ds=ds, nq=nq, hops=hops, orc=orc, hit=hit, size=size, ms=ms, ms_sel=ms_sel, par=par)


def report(R):
    ds, nq = R["ds"], R["nq"]
    out = {"ds": ds, "nq": nq, "F6_PARITY": f"{R['par']}/{nq}", "pool_size": {}, "oracle": {},
           "p50": {}, "latency_ms": {}}
    base = R["hit"][(0, "F6")]
    for nm, m in LG.masks(ds, R["hops"], nq):
        out["oracle"][nm] = {f"K0+{d}": round(float(R["orc"][d][m].mean()), 4) for d in DELTAS}
        out["p50"][nm] = {}
        for d in DELTAS:
            for s in SELS:
                cur = R["hit"][(d, s)]
                st = PP.mcnemar(cur[m], base[m])
                out["p50"][nm][f"K0+{d}/{s}"] = {"acc": round(float(cur[m].mean()), 4),
                                                 "net": st["net"], "gained": st["gained"],
                                                 "lost": st["lost"], "p": round(st["mcnemar_p"], 4),
                                                 "sig": bool(st["sig"])}
    out["pool_size"] = {f"K0+{d}": round(float(R["size"][d].mean()), 2) for d in DELTAS}
    out["latency_ms"] = {"admission": {f"K0+{d}": round(1000 * R["ms"][d] / nq, 4) for d in DELTAS},
                         "selector": {f"K0+{d}/{s}": round(1000 * R["ms_sel"][(d, s)] / nq, 4)
                                      for d in DELTAS for s in SELS}}
    return out


if __name__ == "__main__":
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    for ds in (sys.argv[1:] or ["metaqa"]):
        rp = report(run(ds))
        json.dump(rp, open(f"{KTD}/diag/parta_{ds}.json", "w"), indent=1)
        log(ds, json.dumps({k: rp[k] for k in ("F6_PARITY", "pool_size", "oracle")}, indent=1))
        log(ds, "p50 ALL", json.dumps(rp["p50"]["ALL"], indent=1))
