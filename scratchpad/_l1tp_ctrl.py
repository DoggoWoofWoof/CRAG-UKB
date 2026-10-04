"""TRIPLET PHASE -- STEP 5 decisive control: what does preserving the EDGE actually buy?

The frozen search collapses every edge arriving at a node with `np.maximum.at(best_s, inv, S)`.
This phase exists to stop doing that.  So the question has to be asked exactly:

  For a partition score built by MAX, the collapse is ALGEBRAICALLY LOSSLESS --
      max over edges into Pj  ==  max over nodes v in Pj of (max over edges into v)
  because max is associative.  That is asserted here, not argued.

  For SUM / COUNT aggregations the two differ: a node reached by seven edges counts once at the
  node level and seven times at the edge level.  That difference is the ONLY thing the triplet
  representation adds over the node representation for a target-side score, and it is measured.

  The one genuinely new axis is the SOURCE side: Pi.  A node score cannot express "this partition
  is reached FROM k distinct partitions".  Measured as well.

Everything is scored against MARGINAL_USEFUL: a required gold partition the protected core does not
already hold.  Gold is never an input.

  python scratchpad/_l1tp_ctrl.py <ds>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1tp_core as TP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
PBUD = [6, 12, 20, 50]
NAMES = ["EDGE_MAX", "NODE_MAX", "EDGE_SUM", "NODE_SUM", "EDGE_COUNT", "NODE_COUNT",
         "SOURCE_PARTITION_COUNT", "SOURCE_NODE_COUNT"]


def _rec(order, need):
    o = np.zeros(len(PBUD))
    if not need:
        return o
    for i, k in enumerate(PBUD):
        o[i] = len(set(int(x) for x in order[:k]) & need) / len(need)
    return o


def run(ds, nq_max=None, log=log):
    C = TP.ctx(ds)
    nq = C["nq"] if nq_max is None else min(int(nq_max), C["nq"])
    hard, S = C["hard"], C["S"]
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S["hops"] is not None else np.full(nq, -1)
    BL = [("ALL", np.ones(nq, bool))]
    if len(hops) and int(np.min(hops)) >= 0:
        for h in sorted(set(int(x) for x in hops)):
            BL.append((f"hop{h}", hops == h))

    REC = {n: np.zeros((nq, len(PBUD))) for n in NAMES}
    NEED = np.zeros(nq, np.int32)
    ident = 0                       # queries where EDGE_MAX and NODE_MAX give the identical order
    ident_score = 0                 # queries where the two SCORE VECTORS are bit-identical
    for qi in range(nq):
        E, rq, par, st = TP.replay(C, qi)
        assert par, f"parity broken at {qi}"
        need = set(int(p) for p in goldp[qi]) - ctxs[qi]["prot_set"]
        NEED[qi] = len(need)
        u, v, t0 = E["u"], E["v"], E["T0_OFFSET"]
        pi_, pj_ = hard[u], hard[v]
        m = (pi_ >= 0) & (pj_ >= 0)
        u, v, t0, pi_, pj_ = u[m], v[m], t0[m], pi_[m], pj_[m]
        if len(v) == 0:
            continue
        up, ip = np.unique(pj_, return_inverse=True)
        nk = len(up)
        first = np.full(nk, 1 << 30, np.int64)
        np.minimum.at(first, ip, E["tid"][m])

        # ---- target side, edge unit
        e_max = np.full(nk, -np.inf); np.maximum.at(e_max, ip, t0)
        e_sum = np.bincount(ip, weights=t0, minlength=nk)
        e_cnt = np.bincount(ip, minlength=nk).astype(np.float64)
        # ---- target side, NODE unit: the frozen collapse, then the same aggregation over nodes
        uv, iv = np.unique(v, return_inverse=True)
        nmax = np.full(len(uv), -np.inf); np.maximum.at(nmax, iv, t0)     # == frozen best_s
        pv = hard[uv]
        jp = np.searchsorted(up, pv)
        n_max = np.full(nk, -np.inf); np.maximum.at(n_max, jp, nmax)
        n_sum = np.bincount(jp, weights=nmax, minlength=nk)
        n_cnt = np.bincount(jp, minlength=nk).astype(np.float64)
        # ---- source side: only the transition representation can express these
        sp = np.bincount(np.unique(np.stack([ip, pi_.astype(np.int64)]), axis=1)[0],
                         minlength=nk).astype(np.float64)
        sn = np.bincount(np.unique(np.stack([ip, u.astype(np.int64)]), axis=1)[0],
                         minlength=nk).astype(np.float64)

        ident_score += int(np.array_equal(e_max, n_max))
        vals = {"EDGE_MAX": e_max, "NODE_MAX": n_max, "EDGE_SUM": e_sum, "NODE_SUM": n_sum,
                "EDGE_COUNT": e_cnt, "NODE_COUNT": n_cnt,
                "SOURCE_PARTITION_COUNT": sp, "SOURCE_NODE_COUNT": sn}
        ords = {}
        for n, val in vals.items():
            ords[n] = up[np.lexsort((up, first, -val))]
            REC[n][qi] = _rec(ords[n], need)
        ident += int(np.array_equal(ords["EDGE_MAX"], ords["NODE_MAX"]))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}")

    has = NEED > 0
    O = {"ds": ds, "nq": nq,
         "MAX_COLLAPSE_IS_LOSSLESS": {
             "queries": nq,
             "identical_partition_score_vectors": ident_score,
             "identical_partition_orderings": ident,
             "frac_identical": round(ident / max(nq, 1), 4)},
         "RECALL": {n: {b: {f"R@{k}": round(float(REC[n][mk & has, i].mean()), 4)
                            for i, k in enumerate(PBUD)} for b, mk in BL} for n in NAMES}}
    json.dump(O, open(f"{TP.TPD}/diag/ctrl_{ds}.json", "w"), indent=1)
    b0 = "hop3" if any(b == "hop3" for b, _ in BL) else "ALL"
    log(f"[{ds}] MAX collapse lossless on {ident}/{nq} queries "
        f"(identical score vectors {ident_score}/{nq})")
    for n in NAMES:
        log(f"[{ds}] CTRL {n:24s} " + "  ".join(f"{a} {x}" for a, x in O["RECALL"][n][b0].items()))
    return O


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["metaqa"]):
        run(d)
