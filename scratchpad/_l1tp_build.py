"""TRIPLET PHASE -- one replay pass per corpus.

Produces, per query:
  * STEP 3/4  per-signal diagnostics over TRIPLETS, each signal measured on its own, against the
              MARGINAL_USEFUL label (target partition is a required gold partition NOT already in
              the canonical/protected core).  Gold never enters any score.
  * STEP 5    the partition-TRANSITION table  Pi -> Pj, aggregated per query, cached.
  * STEP 6    the chain columns (path_min / path_sum / distinct-ancestor support) carried on the
              same table, so the three controls C0/C1/C2 are read off it without a second pass.

The NODE baselines are computed in the same loop from the frozen structural read, so the triplet
numbers and the numbers they must beat come from one identical replay.

  python scratchpad/_l1tp_build.py <ds> [nq]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1tp_core as TP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
BUD = [16, 32, 64, 128]

# STEP 3: the four triplet signals, plus the two composable-chain scores of STEP 6.  Each is a
# single quantity, oriented so that LARGER = ranked first.  No combination anywhere in this file.
SIG = ["T0_OFFSET", "T1_SOURCE", "T2_TARGET", "T3_HOP", "C1_PATH_MIN", "C1_PATH_SUM"]
# the NODE units the triplet units have to beat, at the same budget K
NSIG = ["NODE_FROZEN_ARRIVAL", "NODE_MAX_SDIR"]
TCOL = ["pi", "pj", "n", "first_jj", "max_T0", "sum_T0", "max_T1", "max_T2", "min_T3",
        "n_src", "n_tgt", "n_seed", "n_chain", "max_psum", "max_pmin"]


def _ndist(inv, val, nk):
    """distinct `val` per group `inv`."""
    if len(inv) == 0:
        return np.zeros(nk, np.int64)
    pr = np.unique(np.stack([inv.astype(np.int64), val.astype(np.int64)]), axis=1)
    return np.bincount(pr[0], minlength=nk)


def _rec(order, pj, need):
    """needed-partition recall of the top-K units under one ordering, K in BUD."""
    out = np.zeros(len(BUD))
    if not need or len(order) == 0:
        return out
    for i, k in enumerate(BUD):
        got = set(int(x) for x in pj[order[:k]]) & need
        out[i] = len(got) / len(need)
    return out


def build(ds, nq_max=None, log=log):
    C = TP.ctx(ds)
    nq = C["nq"] if nq_max is None else min(int(nq_max), C["nq"])
    hard, npart, S = C["hard"], C["npart"], C["S"]
    ctxs, goldp = S["ctxs"], S["goldp"]
    Xn, Qm = C["Xn"], C["Qm"]
    fz_node = C["z"]["s_node"]; fz_sdir = C["z"]["s_sdir"]

    REC = {k: np.zeros((nq, len(BUD))) for k in SIG + NSIG}
    PCT = {k: np.full(nq, np.nan) for k in SIG}          # marginal-useful percentile rank
    NEEDN = np.zeros(nq, np.int32)                        # |need|
    NMU = np.zeros(nq, np.int32)                          # marginal-useful triplets
    NE = np.zeros(nq, np.int32); NTR = np.zeros(nq, np.int32)
    NPJ = np.zeros(nq, np.int32); NPATH = np.zeros(nq, np.int32); NV = np.zeros(nq, np.int32)
    REACH = np.zeros(nq, np.int8)                         # is every needed partition a transition tgt
    cols = {k: [] for k in TCOL}
    qptr = [0]; par_ok = 0

    for qi in range(nq):
        E, rq, par, st = TP.replay(C, qi)
        par_ok += int(par)
        c = ctxs[qi]
        need = set(int(p) for p in goldp[qi]) - c["prot_set"]
        NEEDN[qi] = len(need)

        u, v = E["u"], E["v"]
        pi_, pj_ = hard[u], hard[v]
        m = (pi_ >= 0) & (pj_ >= 0)
        u, v, pi_, pj_ = u[m], v[m], pi_[m], pj_[m]
        tid = E["tid"][m]
        t0 = E["T0_OFFSET"][m]; t1 = E["T1_SOURCE"][m]; t3 = E["T3_HOP"][m].astype(np.int64)
        psum = E["path_sum"][m]; pmin = E["path_min"][m]
        seed = E["seed"][m]; ptid = E["parent_tid"][m]
        NE[qi] = len(v); NV[qi] = len(np.unique(v))

        uv, iv = np.unique(v, return_inverse=True)
        qh = Qm[qi].astype(np.float64); qh = qh / (np.linalg.norm(qh) + 1e-9)
        t2 = (np.asarray(Xn[uv], np.float64) @ qh)[iv] if len(uv) else np.zeros(0)

        # ---- STEP 4 label: does the TARGET partition of this triplet supply a required partition
        # that the canonical/protected core does not already hold?  Never an input anywhere.
        mu = np.fromiter((int(p) in need for p in pj_), bool, len(pj_))
        NMU[qi] = int(mu.sum())
        vals = {"T0_OFFSET": t0, "T1_SOURCE": t1, "T2_TARGET": t2, "T3_HOP": -t3.astype(float),
                "C1_PATH_MIN": pmin, "C1_PATH_SUM": psum}
        for k, s in vals.items():
            o = np.lexsort((tid, -s))                    # ties -> frozen arrival order, deterministic
            REC[k][qi] = _rec(o, pj_, need)
            if mu.any():
                rk = np.empty(len(o), np.float64); rk[o] = np.arange(len(o))
                PCT[k][qi] = float((1.0 - rk[mu] / max(len(o) - 1, 1)).mean())

        # ---- NODE baselines at the SAME budget
        sn = np.asarray(fz_node[qi]); sd = np.asarray(fz_sdir[qi], np.float64)
        ok = sn >= 0
        sn, sd = sn[ok], sd[ok]
        pn = hard[sn]
        REC["NODE_FROZEN_ARRIVAL"][qi] = _rec(np.arange(len(sn)), pn, need)
        REC["NODE_MAX_SDIR"][qi] = _rec(np.lexsort((np.arange(len(sn)), -sd)), pn, need)

        # ---- STEP 5: the Pi -> Pj transition table
        key = pi_.astype(np.int64) * (npart + 1) + pj_
        uk, ik = np.unique(key, return_inverse=True)
        nk = len(uk)
        NTR[qi] = nk; NPJ[qi] = len(np.unique(pj_))
        REACH[qi] = int(bool(need) and need <= set(int(x) for x in np.unique(pj_)))
        cols["pi"].append((uk // (npart + 1)).astype(np.int32))
        cols["pj"].append((uk % (npart + 1)).astype(np.int32))
        cols["n"].append(np.bincount(ik, minlength=nk).astype(np.int32))
        fj = np.full(nk, 1 << 30, np.int64); np.minimum.at(fj, ik, tid)
        cols["first_jj"].append(fj.astype(np.int32))
        for nm, arr in (("max_T0", t0), ("max_T1", t1), ("max_T2", t2),
                        ("max_psum", psum), ("max_pmin", pmin)):
            a = np.full(nk, -np.inf); np.maximum.at(a, ik, arr)
            cols[nm].append(a)
        cols["sum_T0"].append(np.bincount(ik, weights=t0, minlength=nk))
        mh = np.full(nk, 1 << 20, np.int64); np.minimum.at(mh, ik, t3)
        cols["min_T3"].append(mh.astype(np.int16))
        cols["n_src"].append(_ndist(ik, u, nk).astype(np.int32))
        cols["n_tgt"].append(_ndist(ik, v, nk).astype(np.int32))
        cols["n_seed"].append(_ndist(ik, seed, nk).astype(np.int32))
        cols["n_chain"].append(_ndist(ik, ptid, nk).astype(np.int32))
        qptr.append(qptr[-1] + nk)
        NPATH[qi] = (len(np.unique(np.stack([seed, ptid, pj_]), axis=1)[0]) if len(pj_) else 0)
        if (qi + 1) % 250 == 0:
            log(f"   {ds} {qi+1}/{nq}  parity {par_ok}/{qi+1}")

    assert par_ok == nq, f"PARITY BROKEN {par_ok}/{nq} -- nothing downstream is trustworthy"
    D = {k: np.concatenate(cols[k]) for k in TCOL}
    D["qptr"] = np.asarray(qptr, np.int64)
    D["nq"] = np.array([nq]); D["npart"] = np.array([npart])
    D["NEEDN"] = NEEDN; D["NMU"] = NMU; D["NE"] = NE; D["NTR"] = NTR
    D["NPJ"] = NPJ; D["NPATH"] = NPATH; D["NV"] = NV; D["REACH"] = REACH
    for k in SIG + NSIG:
        D[f"REC_{k}"] = REC[k]
    for k in SIG:
        D[f"PCT_{k}"] = PCT[k]
    os.makedirs(f"{TP.TPD}/data", exist_ok=True)
    os.makedirs(f"{TP.TPD}/diag", exist_ok=True)
    np.savez_compressed(f"{TP.TPD}/data/tr_{ds}.npz", **D)
    log(f"[{ds}] PARITY EXACT {par_ok}/{nq}  transitions {len(D['pi']):,}  "
        f"edges/q {NE.mean():.0f}  transitions/q {NTR.mean():.0f}  Pj/q {NPJ.mean():.0f}")
    return D


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
          int(sys.argv[2]) if len(sys.argv) > 2 else None)
