"""L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- STEP 8 (kNN as its own evidence family) + STEP 11 (cost).

The kNN family is NEVER merged into `s_dir`.  It is measured as a distinct channel, with the three
inference-safe controls the directive names, computed on the T2_FULL_UNION traversal -- the only
substrate where all families compete for the same beam slots, so a kNN edge has to earn its place:

  K0_REACHED        the partition is reached by >= 1 KNN edge
  K1_BEST_KNN_SIM   max over those edges of  cos(x_u, x_v)   -- the neighbour similarity that put
                    the edge in the kNN graph in the first place (existing embeddings, no encoder)
  K2_MULTI_SOURCE   number of DISTINCT frozen retrieval seeds reaching the partition through a KNN
                    edge (and, separately, distinct SOURCE PARTITIONS -- the core-exit analogue)

Control, because Dense is built from the SAME Qwen space:
  K1_QUERY_SIM      max over those edges of cos(q_hat, x_v).  If this separates needed from nuisance
                    but K1_BEST_KNN_SIM does not, the channel is Dense wearing a graph costume.

STEP 11 rides along: `edges` is the adjacency volume inspected, `scored` the edges that survived
DEG_CAP and actually entered the matvec, and `ms` the wall clock, all measured under IDENTICAL
instrumentation for every substrate (the main run enables `want_edges` only for T2, which biases its
timing; these are the numbers the cost table uses).

  python scratchpad/_l1kn_k8.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1ss_core as SS
import _l1bc_ledger as LG
import _l1kn_sub as SUB

KND = SUB.KND
SUBS = SUB.SUBS
KBIT = SUB.FAMBIT["KNN"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def auc(pos, neg):
    """rank-sum AUC, ties at 0.5.  Returns None when either side is empty."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return None
    a = np.concatenate([pos, neg])
    r = np.empty(len(a), np.float64)
    o = np.argsort(a, kind="stable")
    s = a[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def run(ds, stride=1, log=log):
    S = EV.substrate(ds)
    z = S["z"]; nq = S["nq"]
    hard = np.asarray(z["hard"], np.int64)
    ctxs, goldp = S["ctxs"], S["goldp"]
    hops = np.asarray(S["hops"]) if S.get("hops") is not None else np.zeros(nq, np.int32)
    import _l1bm_run as RUN
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    TOPO = {t: SUB.substrate(ds, t, log)[:3] for t in SUBS}
    LAB = SUB.Lab(ds, log)
    qs = list(range(0, nq, stride))
    nQ = len(qs)

    COST = {t: {"edges": 0.0, "scored": 0.0, "ms": 0.0, "scope": 0.0} for t in SUBS}
    F = {k: {"need": [], "nuis": [], "miss": []} for k in
         ["K1_BEST_KNN_SIM", "K1_QUERY_SIM", "K2_SEEDS", "K2_SRC_PARTS", "K2_EDGES"]}
    K0R = {"needed_reached": 0, "needed_total": 0, "missed_reached": 0, "missed_total": 0,
           "parts_reached": 0, "queries": 0}
    for j, qi in enumerate(qs):
        c = ctxs[qi]
        sd0 = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd0, Xn)
        REQ = set(int(p) for p in goldp[qi])
        Xsel, _, _ = LG.safe_pick(c)
        fin = c["prot_set"] | set(int(p) for p in Xsel)
        MISS = REQ - fin
        for t in SUBS:
            ap, ai, dg = TOPO[t]
            t0 = time.perf_counter()
            added, vmeta, st, FE = SS.expand_feat(sd0, rq, ap, ai, dg, Xn, W, want_future=False,
                                                  want_edges=True)
            el = time.perf_counter() - t0
            COST[t]["edges"] += st["edges"]; COST[t]["ms"] += 1000 * el
            COST[t]["scope"] += st["scope"]; COST[t]["scored"] += len(FE["EDGES"]["u"])
            if t != "T2_FULL_UNION":
                continue
            E = FE["EDGES"]
            if not len(E["u"]):
                continue
            fam, _ = LAB.of(E["u"], E["v"])
            km = (fam & KBIT) > 0
            if not km.any():
                continue
            u, v = E["u"][km], E["v"][km]
            sdix = E["seed"][km]
            tp = hard[v]
            ok = tp >= 0
            u, v, sdix, tp = u[ok], v[ok], sdix[ok], tp[ok]
            if not len(tp):
                continue
            Xu, Xv = gx(u), gx(v)
            esim = np.einsum("ij,ij->i", Xu, Xv)
            qh = Qm[qi].astype(np.float32)
            qh = qh / max(float(np.linalg.norm(qh)), 1e-12)
            qsim = Xv @ qh
            spart = hard[u]
            agg = {}
            for i in range(len(tp)):
                p = int(tp[i])
                a = agg.get(p)
                if a is None:
                    agg[p] = a = [-1e30, -1e30, set(), set(), 0]
                a[0] = max(a[0], float(esim[i])); a[1] = max(a[1], float(qsim[i]))
                a[2].add(int(sdix[i])); a[3].add(int(spart[i])); a[4] += 1
            K0R["queries"] += 1
            K0R["parts_reached"] += len(agg)
            K0R["needed_total"] += len(REQ); K0R["missed_total"] += len(MISS)
            K0R["needed_reached"] += len(REQ & set(agg))
            K0R["missed_reached"] += len(MISS & set(agg))
            for p, a in agg.items():
                w = "need" if p in REQ else "nuis"
                for k, val in (("K1_BEST_KNN_SIM", a[0]), ("K1_QUERY_SIM", a[1]),
                               ("K2_SEEDS", len(a[2])), ("K2_SRC_PARTS", len(a[3])),
                               ("K2_EDGES", a[4])):
                    F[k][w].append(val)
                    if p in MISS:
                        F[k]["miss"].append(val)
        if (j + 1) % 200 == 0:
            log(f"   {ds} k8 {j+1}/{nQ}")

    out = {"ds": ds, "nq": nQ, "stride": stride,
           "COST": {t: {"edges_per_q": round(COST[t]["edges"] / nQ, 1),
                        "scored_per_q": round(COST[t]["scored"] / nQ, 1),
                        "scope_per_q": round(COST[t]["scope"] / nQ, 1),
                        "ms_per_q": round(COST[t]["ms"] / nQ, 3)} for t in SUBS},
           "K0": dict(K0R), "K8": {}}
    for k in F:
        out["K8"][k] = {
            "n_needed": len(F[k]["need"]), "n_nuisance": len(F[k]["nuis"]),
            "mean_needed": round(float(np.mean(F[k]["need"])), 4) if F[k]["need"] else None,
            "mean_nuisance": round(float(np.mean(F[k]["nuis"])), 4) if F[k]["nuis"] else None,
            "AUC_needed_vs_nuisance": (round(auc(F[k]["need"], F[k]["nuis"]), 4)
                                       if auc(F[k]["need"], F[k]["nuis"]) is not None else None),
            "AUC_missed_vs_nuisance": (round(auc(F[k]["miss"], F[k]["nuis"]), 4)
                                       if auc(F[k]["miss"], F[k]["nuis"]) is not None else None),
            "n_missed": len(F[k]["miss"])}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(f"{KND}/diag", exist_ok=True)
    R = run(ds, stride)
    fp = f"{KND}/diag/k8_{ds}.json"
    json.dump(R, open(fp, "w"), indent=1)
    log("wrote", fp)
    for t in SUBS:
        c = R["COST"][t]
        log(f"  {t:20s} edges/q {c['edges_per_q']:10,.0f}  scored/q {c['scored_per_q']:10,.0f}"
            f"  scope/q {c['scope_per_q']:8,.0f}  {c['ms_per_q']:8.2f} ms")
    log("  K0 " + json.dumps(R["K0"]))
    for k, v in R["K8"].items():
        log(f"  {k:18s} AUC(need vs nuis) {v['AUC_needed_vs_nuisance']}  "
            f"AUC(missed vs nuis) {v['AUC_missed_vs_nuisance']}  "
            f"mean {v['mean_needed']} vs {v['mean_nuisance']}  n={v['n_needed']}/{v['n_nuisance']}")
