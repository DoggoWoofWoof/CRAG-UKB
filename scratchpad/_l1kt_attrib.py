"""ATTRIBUTION: is the V4_SRC_EXIT effect transformation geometry, or plain source provenance?

V4_SRC_EXIT(p) = max over CORE-EXIT edges (u -> v) with hard[v] == p  of  cos(T_chain(z0), x_v)
V6_EXIT_SRC_SIM(p) = max over the SAME edges                          of  cos(z0, x_u)

For an ORTHOGONAL family with T(u) = v the two are EXACTLY equal on hop-1 edges
(<T z0, v> = <T z0, T u> = <z0, u>).  Beyond hop 1 the chain state is z_{h-1}, not z0, so they can
diverge.  This measures how far they actually diverge, on the population that decides the pool:

  * hop histogram of the CORE-EXIT edge set and of the per-partition ARGMAX edge
  * agreement of the two partition score vectors (Spearman, Kendall-tau-like, argmax match)
  * whether the two produce the IDENTICAL top-K(q) pool

  python scratchpad/_l1kt_attrib.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1tp_core as TP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1ca_admit as AD
import _l1kt_tf as TF
import _l1kt_partb as PB

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
MISS, EPS, FLOOR = BC.MISS, TF.EPS, PB.FLOOR
FAM = "F2_MIN_ROTATION"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main(ds, stride=7):
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    Xn, hard = C["Xn"], C["hard"]
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    hop_all = np.zeros(8, np.int64)
    hop_exit = np.zeros(8, np.int64)
    hop_argmax = np.zeros(8, np.int64)
    sp, am, samepool, nq_used, nboth = [], [0, 0], [0, 0], 0, 0
    for qi in range(0, nq, stride):
        E, rq, par, st = TP.replay(C, qi)
        u, v = E["u"], E["v"]
        if not len(u):
            continue
        c, r = S["ctxs"][qi], BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        pset = set(int(p) for p in c["prot"])
        Xsel, cands, sc = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        K = len(cands)
        a1 = AD.a1_pool(r, c["cpos"], K, pset)
        uni = list(cands) + [p for p in a1 if p not in set(cands)]
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
        z0 = C["Qm"][qi].astype(np.float32)
        z0 = z0 / max(float(np.linalg.norm(z0)), EPS)
        uz0, vz0 = U @ z0, V @ z0
        tv, n2, _ = TF.transport(U, V, z0, hm, E["parent_tid"], isp, FAM, cuv, uz0, vz0)
        s4 = tv / np.sqrt(np.maximum(n2, EPS))
        s6 = uz0
        for h in range(1, 8):
            hop_all[h] += int((hop == h).sum())
            hop_exit[h] += int((exitm & (hop == h)).sum())
        sub = exitm & keep
        if not sub.any():
            continue
        nq_used += 1
        # per-partition argmax edge under each score, restricted to core-exit edges
        b4 = np.full(len(upart), -np.inf); b6 = np.full(len(upart), -np.inf)
        i4 = np.full(len(upart), -1, np.int64); i6 = np.full(len(upart), -1, np.int64)
        idx = np.nonzero(sub)[0]
        o4 = np.argsort(s4[idx], kind="stable"); o6 = np.argsort(s6[idx], kind="stable")
        for o, bb, ii, ss in ((o4, b4, i4, s4), (o6, b6, i6, s6)):
            j = idx[o]
            bb[pix[j]] = ss[j]
            ii[pix[j]] = j
        ok = i4 >= 0
        am[0] += int((i4[ok] == i6[ok]).sum()); am[1] += int(ok.sum())
        for h in range(1, 8):
            hop_argmax[h] += int((hop[i4[ok]] == h).sum())
        f = np.isfinite(b4) & np.isfinite(b6)
        if f.sum() > 2:
            r4 = np.argsort(np.argsort(b4[f])); r6 = np.argsort(np.argsort(b6[f]))
            r4 = r4 - r4.mean(); r6 = r6 - r6.mean()
            dn = float(np.sqrt((r4 ** 2).sum() * (r6 ** 2).sum()))
            if dn > 0:
                sp.append(float((r4 * r6).sum() / dn)); nboth += 1

        def pool(b):
            def sco(p):
                i = pinv.get(p, -1)
                return float(b[i]) if (i >= 0 and np.isfinite(b[i])) else FLOOR
            return sorted(uni, key=lambda p: (-sco(p), safe_ord.get(p, MISS), p))[:K]
        samepool[0] += int(pool(b4) == pool(b6)); samepool[1] += 1
    tot = max(int(hop_exit.sum()), 1)
    return {"ds": ds, "stride": stride, "n_queries": nq_used, "family": FAM,
            "edges_by_hop": {str(h): int(hop_all[h]) for h in range(1, 8) if hop_all[h]},
            "core_exit_edges_by_hop": {str(h): int(hop_exit[h]) for h in range(1, 8) if hop_exit[h]},
            "core_exit_share_by_hop": {str(h): round(float(hop_exit[h]) / tot, 4)
                                       for h in range(1, 8) if hop_exit[h]},
            "argmax_edge_hop_hist": {str(h): int(hop_argmax[h]) for h in range(1, 8) if hop_argmax[h]},
            "V4_V6_same_argmax_edge": f"{am[0]}/{am[1]}",
            "V4_V6_same_argmax_frac": round(am[0] / max(am[1], 1), 4),
            "V4_V6_spearman_mean": round(float(np.mean(sp)), 4) if sp else None,
            "V4_V6_spearman_min": round(float(np.min(sp)), 4) if sp else None,
            "V4_V6_identical_pool": f"{samepool[0]}/{samepool[1]}",
            "V4_V6_identical_pool_frac": round(samepool[0] / max(samepool[1], 1), 4)}


if __name__ == "__main__":
    ds = sys.argv[1]
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    o = main(ds, st)
    json.dump(o, open(f"{KTD}/diag/attrib_{ds}.json", "w"), indent=1)
    print(json.dumps(o, indent=1))
