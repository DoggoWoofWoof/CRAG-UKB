"""PART G -- support-multiplicity diagnostic: WHY does F2_SUM_NORMALIZED beat F1_MAX_BOUNDARY?

F1 scores a candidate node v by its single best-supporting core's boundary mass (max over j in S).
F2 scores v by the SUM of v's (normalized) boundary mass across every selected core that supports
it.  The two rules agree exactly when every candidate has exactly one supporting core; they can
only diverge on candidates supported by >=2 selected cores (F2 rewards that multiplicity, F1 is
blind to it).  This part measures whether that is in fact the mechanism behind Part F's result,
by comparing SUPPORT COUNT (number of distinct selected cores j in S with v in core j's O4_FULL_C
boundary-mass pairs) across four disjoint groups of REQUIRED (gold) candidate nodes on the F6 lane:

  BOTH      required node v is in both F1's and F2's per-query top-K_q cut
  F2_ONLY   required node v is in F2's cut but NOT F1's  (F2's net wins)
  F1_ONLY   required node v is in F1's cut but NOT F2's  (F2's net losses)
  NEITHER   required node v is a halo candidate (1-hop from some j in S) but in neither cut

Plus a POOL baseline: support-count distribution over EVERY halo candidate (gold or not), so the
required-node groups can be read relative to the ambient rate of multi-core support, not in
isolation.  If F2_ONLY's mean support count is clearly above BOTH/POOL and F1_ONLY's is clearly
below, that confirms multiplicity (not some other correlate of the sum-vs-max scoring rule) is
what F2 is actually rewarding.  Purely diagnostic -- no promotion decision attaches to Part G.

  python scratchpad/_l1hu_g.py run <ds>
  python scratchpad/_l1hu_g.py report
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

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
GROUPS = ["BOTH", "F2_ONLY", "F1_ONLY", "NEITHER"]
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


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

    SEL = f650
    support = {g_: [] for g_ in GROUPS}       # required-node support counts, by group
    pool_support = []                          # ambient support counts, ALL candidates (subsampled)
    selmask = np.zeros(npart, bool)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 200 == 0:
            log("  %s: %d/%d queries, %.1fs elapsed" % (ds, qi, nq, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        S = SEL[qi]
        Ss = set(S)
        selmask[:] = False
        selmask[np.asarray(S, np.int64)] = True
        got_core = {x for x in nd if int(hard[x]) in Ss}

        if len(f0_bidx_):
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in S]))
            K_q = int(np.count_nonzero(~selmask[hard[u0.astype(np.int64)]]))
        else:
            K_q = 0
        if K_q == 0:
            continue

        cv, cj, cm = [], [], []
        for j in S:
            s, e = bptr[j], bptr[j + 1]
            if e > s:
                cv.append(bidx[s:e]); cj.append(np.full(e - s, j, np.int64)); cm.append(bmass[s:e])
        if not cv:
            continue
        V = np.concatenate(cv); J = np.concatenate(cj); Mm = np.concatenate(cm)
        keep = ~selmask[hard[V.astype(np.int64)]]
        V, J, Mm = V[keep], J[keep], Mm[keep]
        if not len(V):
            continue

        order = np.argsort(V, kind="stable")
        Vs, Ms, Js = V[order], Mm[order], J[order]
        uniq_v, start = np.unique(Vs, return_index=True)
        cnt = np.diff(np.append(start, len(Vs)))                  # support count per unique v
        f1_score = np.maximum.reduceat(Ms, start)
        f2_score = np.add.reduceat(Ms / bmax[Js], start)

        k = min(K_q, len(uniq_v))
        top1 = set(uniq_v[np.argpartition(-f1_score, k - 1)[:k]].tolist()) if k < len(uniq_v) \
            else set(uniq_v.tolist())
        top2 = set(uniq_v[np.argpartition(-f2_score, k - 1)[:k]].tolist()) if k < len(uniq_v) \
            else set(uniq_v.tolist())

        support_of = dict(zip(uniq_v.tolist(), cnt.tolist()))
        for x in nd:
            if x in got_core or x not in support_of:
                continue
            in1, in2 = x in top1, x in top2
            grp = "BOTH" if (in1 and in2) else "F2_ONLY" if in2 else "F1_ONLY" if in1 else "NEITHER"
            support[grp].append(support_of[x])

        if len(uniq_v):
            samp = uniq_v if len(uniq_v) <= 200 else np.random.RandomState(qi).choice(
                uniq_v, 200, replace=False)
            pool_support.extend(int(support_of[int(s_)]) for s_ in samp)

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "median": None, "frac_multi": None}
        a = np.asarray(xs, np.float64)
        return {"n": len(a), "mean": round(float(a.mean()), 3), "median": float(np.median(a)),
               "frac_multi": round(float((a >= 2).mean()), 4)}

    rec = {"ds": ds, "nq": nq, "groups": {g_: stats(support[g_]) for g_ in GROUPS},
          "pool_baseline": stats(pool_support)}
    log("  %s: done in %.1fs -- %s" % (ds, time.time() - t0,
        {g_: rec["groups"][g_]["n"] for g_ in GROUPS}))

    os.makedirs("%s/support_multiplicity" % OUT, exist_ok=True)
    fp = "%s/support_multiplicity/G_%s.json" % (OUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def report():
    rows = []
    for ds in HH.DS:
        fp = "%s/support_multiplicity/G_%s.json" % (OUT, ds)
        if not os.path.exists(fp):
            continue
        rows.append(json.load(open(fp)))
    json.dump(rows, open("%s/support_multiplicity/G_REPORT.json" % OUT, "w"), indent=1)
    for r in rows:
        print("\n== %s ==" % r["ds"])
        print("  POOL_BASELINE      %s" % r["pool_baseline"])
        for g_ in GROUPS:
            print("  %-18s %s" % (g_, r["groups"][g_]))
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
