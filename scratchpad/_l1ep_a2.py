"""FINAL L1 EDGE-SUBSTRATE PROGRAM -- second traversal pass: the substrates the kNN audit lacked.

The kNN audit already replayed E0_STRUCT / E1_NERX_ONLY / E2_KNN_ONLY / E4_STRUCT_KNN /
E6_TOPOLOGY_C / M2_STRUCT_KNN_MATCHED.  The A5 attribution and the A11 table additionally need

    E3_STRUCT_NERX          S u N          the +NERX arm at full work
    M1_STRUCT_NERX_MATCHED  S/N matched    the +NERX arm at E0's own budget
    M3_TOPOLOGY_C_MATCHED   S/(NuK) matched the full-C arm at E0's own budget

so the interaction  M(SuNuK) - M(SuN) - M(SuK) + M(S)  is computable at BOTH work levels.
E0_STRUCT is re-run in the same pass as the parity anchor -- identical seeds, identical residual,
identical query order -- so the two passes can be merged query-by-query without alignment risk.

A6 rides along on E3: every scored NERX edge is measured for
    EDGE_SIM   cos(x_u, x_v)   the neighbour similarity the family itself asserts
    QUERY_SIM  cos(q_hat, x_v) what Dense already knows
If QUERY_SIM separates needed partitions from nuisance ones and EDGE_SIM does not, the family is
Dense wearing a graph costume -- the same test the kNN audit ran on E4.

  python scratchpad/_l1ep_a2.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1ss_core as SS
import _l1bm_core as BM
import _l1bc_ledger as LG
import _l1kn_sub as KS
import _l1kn_run as KR
import _l1ep_sub as EP

OUT = EP.OUT
CACHE = "scratchpad/_l1ep"
P, B, K0, MSTR = EV.P, EV.B, EV.K0, EV.M_STRUCT
NEW = ["E0_STRUCT", "E3_STRUCT_NERX", "M1_STRUCT_NERX_MATCHED", "M3_TOPOLOGY_C_MATCHED"]
NXBIT = KS.FAMBIT["NERX"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def auc(pos, neg):
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return None
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="stable"); s = a[o]
    r = np.empty(len(a), np.float64); i = 0
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
    TOPO = {t: EP.substrate(ds, t, log)[:3] for t in NEW}
    LAB = KS.Lab(ds, log)
    qs = list(range(0, nq, stride)); nQ = len(qs)

    A = {t: {k: np.zeros(nQ, np.float64) for k in
             ["vis_parts", "add_parts", "read_parts", "edges", "scope", "ms", "p50", "orac",
              "npool", "need_VIS", "need_ADD", "need_READ"]} for t in NEW}
    need_tot = np.zeros(nQ, np.float64); base = np.zeros(nQ, np.int8); hq = np.zeros(nQ, np.int32)
    PAR = {"spos_exact": 0, "p50_exact": 0}
    A6 = {k: {"need": [], "nuis": []} for k in ["NERX_EDGE_SIM", "NERX_QUERY_SIM"]}

    for j, qi in enumerate(qs):
        c = ctxs[qi]
        sd0 = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd0, Xn)
        REQ = set(int(p) for p in goldp[qi])
        hq[j] = int(hops[qi]); need_tot[j] = len(REQ)
        Xsel, _, _ = LG.safe_pick(c)
        fin = c["prot_set"] | set(int(p) for p in Xsel)
        base[j] = int(REQ <= fin)
        cpos, rpos = c["cpos"], c["rpos"]
        RF = [int(p) for p in c["_RF"]]
        b50, bnd = c["base50"], c["bnd"]
        for t in NEW:
            ap, ai, dg = TOPO[t]
            t0 = time.perf_counter()
            added, vmeta, st, FE = SS.expand_feat(sd0, rq, ap, ai, dg, Xn, W, want_future=False,
                                                  want_visited=True,
                                                  want_edges=(t == "E3_STRUCT_NERX"))
            el = time.perf_counter() - t0
            vis = np.concatenate([np.asarray(sd0, np.int64), FE["node"], FE["VIS_PRUNED"]])
            vp = np.unique(hard[vis]); vp = vp[vp >= 0]
            adp = np.unique(hard[FE["node"]]) if len(FE["node"]) else np.zeros(0, np.int64)
            adp = adp[adp >= 0]
            sn, sh, sdd, scn = BM.arrays_of(added, vmeta)
            SF = KR._sf(sn, sh, sdd, scn, hard)
            spos = {p: r for r, p in enumerate(SF)}
            chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
            pick, pool = KR._f6(bnd, chal, spos, rpos, cpos)
            fs = c["prot_set"] | set(pick)
            assert len(fs) == P
            a = A[t]
            a["vis_parts"][j] = len(vp); a["add_parts"][j] = len(adp); a["read_parts"][j] = len(SF)
            a["edges"][j] = st["edges"]; a["scope"][j] = st["scope"]; a["ms"][j] = 1000 * el
            a["p50"][j] = int(REQ <= fs); a["orac"][j] = KR._oracle(c["prot_set"], pool, REQ)
            a["npool"][j] = len(pool)
            a["need_VIS"][j] = len(REQ & set(int(x) for x in vp))
            a["need_ADD"][j] = len(REQ & set(int(x) for x in adp))
            a["need_READ"][j] = len(REQ & set(spos))
            if t == "E0_STRUCT":
                PAR["spos_exact"] += int(spos == {int(k2): int(v2) for k2, v2 in c["spos"].items()})
                PAR["p50_exact"] += int(a["p50"][j] == base[j])
            if t == "E3_STRUCT_NERX" and FE.get("EDGES") is not None and len(FE["EDGES"]["u"]):
                E = FE["EDGES"]
                fam, _ = LAB.of(E["u"], E["v"])
                m = (fam & NXBIT) > 0
                if m.any():
                    u, v = E["u"][m], E["v"][m]
                    tp = hard[v]; ok = tp >= 0
                    u, v, tp = u[ok], v[ok], tp[ok]
                    if len(tp):
                        es = np.einsum("ij,ij->i", gx(u), gx(v))
                        qh = Qm[qi].astype(np.float32)
                        qh = qh / max(float(np.linalg.norm(qh)), 1e-12)
                        qs_ = gx(v) @ qh
                        agg = {}
                        for i2 in range(len(tp)):
                            p = int(tp[i2]); g = agg.get(p)
                            if g is None:
                                agg[p] = g = [-1e30, -1e30]
                            g[0] = max(g[0], float(es[i2])); g[1] = max(g[1], float(qs_[i2]))
                        for p, g in agg.items():
                            w = "need" if p in REQ else "nuis"
                            A6["NERX_EDGE_SIM"][w].append(g[0])
                            A6["NERX_QUERY_SIM"][w].append(g[1])
        if (j + 1) % 200 == 0:
            log(f"   {ds} {j+1}/{nQ}")

    out = {"ds": ds, "nq": nQ, "stride": stride, "hops": hq.tolist(),
           "need_tot": need_tot.tolist(), "base_p50": base.tolist(),
           "PARITY": {"spos_exact": PAR["spos_exact"], "p50_exact": PAR["p50_exact"], "nq": nQ,
                      "E0_PARITY": "EXACT" if (PAR["spos_exact"] == nQ and PAR["p50_exact"] == nQ)
                      else "MISMATCH"},
           "A6_NERX": {k: {"n_needed": len(v["need"]), "n_nuisance": len(v["nuis"]),
                           "AUC_needed_vs_nuisance": (round(auc(v["need"], v["nuis"]), 4)
                                                      if auc(v["need"], v["nuis"]) is not None
                                                      else None),
                           "mean_needed": (round(float(np.mean(v["need"])), 4) if v["need"]
                                           else None),
                           "mean_nuisance": (round(float(np.mean(v["nuis"])), 4) if v["nuis"]
                                             else None)} for k, v in A6.items()},
           "subs": {t: {k: v.tolist() for k, v in A[t].items()} for t in NEW}}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    R = run(ds, stride)
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/a2_{ds}.json"
    json.dump(R, open(fp, "w"))
    log(f"wrote {fp} nq={R['nq']}")
    log("  PARITY " + json.dumps(R["PARITY"]))
    for t in NEW:
        a = R["subs"][t]
        nt = np.sum(R["need_tot"])
        log(f"  {t:24s} edges/q {np.mean(a['edges']):9,.0f}  visP {np.mean(a['vis_parts']):7.1f}  "
            f"readP {np.mean(a['read_parts']):6.1f}  needVIS {np.sum(a['need_VIS'])/max(nt,1):.4f}  "
            f"needREAD {np.sum(a['need_READ'])/max(nt,1):.4f}  p50 {np.mean(a['p50']):.4f}  "
            f"orac {np.mean(a['orac']):.4f}  {np.mean(a['ms']):7.1f} ms")
    log("  A6_NERX " + json.dumps(R["A6_NERX"]))
