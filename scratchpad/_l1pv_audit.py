"""CORE-EXIT PROVENANCE AUDIT -- STEPS 1-4.

The surviving hypothesis from the transformation phase: what converted on MetaQA was the CORE-EXIT
RESTRICTION (which partitions get evidence at all), not the geometry that ranks them.  This module
asks whether the already-existing EDGE PROVENANCE of those core-exit transitions explains why the
same operation helps MetaQA and regresses the text corpora.

No new graph work.  Families come from `_l1pv_fam` and are exactly the ones already on disk:
STRUCT (the traversal adjacency itself) and NER (`ner_edges_w_df25.pkl`).  KNN is structurally
ABSENT from the traversal on every corpus, so the live axis is STRUCT_ONLY vs STRUCT_AND_NER.

Scores (all pure provenance -- max over a restricted edge set of cos(q, x_u); no transformation):

  SRC_ALL           control: all CORE-EXIT edges                    (= V6_EXIT_SRC_SIM, prior phase)
  SRC_NER           core-exit edges that ALSO carry an NER edge
  SRC_STRUCT_ONLY   core-exit edges that carry NO NER edge
  ALL_NER           every edge that carries NER      (core-exit restriction OFF -- STEP-4 contrast)
  ALL_STRUCT_ONLY   every edge with no NER           (core-exit restriction OFF -- STEP-4 contrast)

GOOD_SWAP / BAD_SWAP are the PRIOR PHASE's definitions, copied verbatim from `_l1kt_partb.run`,
on the SAME family-independent pools (`AD.a1_pool` / `AD.a2_pool`), so every family is scored on an
identical pair population.

  python scratchpad/_l1pv_audit.py <ds> [stride]
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
import _l1kt_partb as PB
import _l1pv_fam as FM

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PROVENANCE"
EPS, FLOOR, MISS = 1e-12, PB.FLOOR, BC.MISS
PLS = ["A1_SRC", "A2_HYB"]
SC = ["SRC_ALL", "SRC_NER", "SRC_STRUCT_ONLY", "ALL_NER", "ALL_STRUCT_ONLY"]
# exclusive-support buckets for STEP 3, by which families supply a candidate's core-exit evidence
BUCK = ["ONLY_NER", "ONLY_STRUCT", "MULTI_FAMILY", "NO_CORE_EXIT_EVIDENCE"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def cbin(rank):
    """canonical-rank bucket of a source partition under the FROZEN contract (prot 0..43, bnd
    44..49, everything else outside P50).  Fixed edges, no search."""
    if rank >= 50:
        return "OUT_OF_P50"
    if rank >= 44:
        return "BND"
    return f"PROT_Q{rank // 11}"


def qbin(x, q):
    """quartile index of x against precomputed quartile cuts q (existing values, no search)."""
    return int(np.searchsorted(q, x, side="right"))


def run(ds, stride=1, log=log):
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

    # ---- STEP 1 ledger -------------------------------------------------------------------------
    L1 = {"core_exit": [], "core_exit_NER": [], "core_exit_STRUCT_ONLY": [], "all_edges": [],
          "all_NER": [], "frac_core_exit_NER": []}
    L1h = {(h, k): [] for h in (1, 2, 3) for k in ("core_exit", "core_exit_NER")}
    ehop = {f: np.zeros(8, np.int64) for f in ("core_exit", "core_exit_NER")}
    famiss = [0, 0]

    # ---- STEP 2 margins ------------------------------------------------------------------------
    M = {(pl, s): {"GOOD": [], "BAD": [], "GOOD_ev": [], "BAD_ev": [], "GOOD_h": [], "BAD_h": []}
         for pl in PLS for s in SC}
    npair = {pl: 0 for pl in PLS}
    nlab = {pl: {"GOOD": 0, "BAD": 0, "TIE": 0} for pl in PLS}

    # ---- STEP 3 exclusive support --------------------------------------------------------------
    B3 = {(pl, b): {"GOOD": 0, "BAD": 0} for pl in PLS for b in BUCK}
    # ---- STEP 4 source quality: P(admitted candidate is required-and-missing | bin) -------------
    S4 = {}

    def s4add(fam, axis, b, good):
        k = (fam, axis, str(b))
        d = S4.setdefault(k, [0, 0])
        d[0] += int(good)
        d[1] += 1

    par_ok, nseen, ms = 0, 0, 0.0
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

        fam, fok = FAM.of(u, v)
        famiss[0] += int((~fok).sum()); famiss[1] += len(u)
        nerm = (fam & NERBIT) > 0
        srcp = hard[u]
        exitm = np.array([int(p) in pset for p in srcp])
        hop = E["T3_HOP"].astype(np.int64)
        hq = int(hops[qi])

        ce = exitm
        L1["all_edges"].append(int(len(u)))
        L1["all_NER"].append(int(nerm.sum()))
        L1["core_exit"].append(int(ce.sum()))
        L1["core_exit_NER"].append(int((ce & nerm).sum()))
        L1["core_exit_STRUCT_ONLY"].append(int((ce & ~nerm).sum()))
        L1["frac_core_exit_NER"].append(float((ce & nerm).sum()) / max(int(ce.sum()), 1))
        if hq in (1, 2, 3):
            L1h[(hq, "core_exit")].append(int(ce.sum()))
            L1h[(hq, "core_exit_NER")].append(int((ce & nerm).sum()))
        for h in range(1, 8):
            hm = hop == h
            ehop["core_exit"][h] += int((ce & hm).sum())
            ehop["core_exit_NER"][h] += int((ce & nerm & hm).sum())

        pairs = {}
        for pl, npool in pools.items():
            ns, cs = set(npool), set(cands)
            ev = [p for p in cands if p not in ns]
            ad = [p for p in npool if p not in cs]
            evn = [p for p in ev if p in need]
            evo = [p for p in ev if p not in need]
            adn = [p for p in ad if p in need]
            ado = [p for p in ad if p not in need]
            pr = [(a, b, "GOOD") for a in evo for b in adn]
            pr += [(a, b, "BAD") for a in evn for b in ado]
            nlab[pl]["GOOD"] += len(evo) * len(adn)
            nlab[pl]["BAD"] += len(evn) * len(ado)
            nlab[pl]["TIE"] += len(ev) * len(ad) - len(evo) * len(adn) - len(evn) * len(ado)
            npair[pl] += len(ev) * len(ad)
            pairs[pl] = (pr, ad)

        tp = hard[v]
        upart = np.unique(tp[tp >= 0])
        pinv = {int(p): i for i, p in enumerate(upart)}
        pix = np.array([pinv.get(int(p), -1) for p in tp], np.int64)
        keep = pix >= 0
        z0 = C["Qm"][qi].astype(np.float32)
        z0 = z0 / max(float(np.linalg.norm(z0)), EPS)
        uz0 = gx(u) @ z0
        subs = {"SRC_ALL": ce, "SRC_NER": ce & nerm, "SRC_STRUCT_ONLY": ce & ~nerm,
                "ALL_NER": nerm, "ALL_STRUCT_ONLY": ~nerm}
        tab = {s: PB.pbest(uz0[keep], pix[keep], len(upart), sub=subs[s][keep]) for s in SC}

        # per-candidate exclusive support, from core-exit edges only
        supN = PB.pbest(uz0[keep], pix[keep], len(upart), sub=(ce & nerm)[keep])
        supS = PB.pbest(uz0[keep], pix[keep], len(upart), sub=(ce & ~nerm)[keep])

        def bucket(p):
            i = pinv.get(p, -1)
            if i < 0:
                return "NO_CORE_EXIT_EVIDENCE"
            n, s = np.isfinite(supN[i]), np.isfinite(supS[i])
            if n and s:
                return "MULTI_FAMILY"
            if n:
                return "ONLY_NER"
            if s:
                return "ONLY_STRUCT"
            return "NO_CORE_EXIT_EVIDENCE"

        # STEP 4 source-quality bins.  Existing information only; bin edges come from the FROZEN
        # contract (P=50, B=6 => prot = ranks 0..43, bnd = 44..49), not from any search.
        cq = np.array([cpos.get(int(p), 10 ** 6) for p in srcp], np.int64)
        t1 = E["T1_SOURCE"].astype(np.float64)
        t1q = np.quantile(t1, [0.25, 0.5, 0.75]) if len(t1) > 3 else np.array([0.0, 0.0, 0.0])

        for pl, (pr, ad) in pairs.items():
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
                B3[(pl, bucket(b))][lab] += 1
            if pl == "A1_SRC":
                # STEP 4 is over the ADMITTED candidates themselves (one row each), attributed to
                # the best supporting core-exit edge in each family.  Diagnostic only.
                for b in ad:
                    ib = pinv.get(b, -1)
                    if ib < 0:
                        continue
                    good = b in need
                    # masks are NOT core-exit-restricted here: `core_source` is one of the STEP-4
                    # axes, so it has to be able to take both values.
                    for fname, m in (("NER", nerm), ("STRUCT_ONLY", ~nerm)):
                        sel = m & keep & (pix == ib)
                        if not sel.any():
                            continue
                        j = int(np.nonzero(sel)[0][np.argmax(uz0[np.nonzero(sel)[0]])])
                        s4add(fname, "core_source", int(exitm[j]), good)
                        s4add(fname, "hop", int(hop[j]), good)
                        s4add(fname, "src_cpos_quartile", cbin(int(cq[j])), good)
                        s4add(fname, "src_conf_quartile", qbin(t1[j], t1q), good)
        ms += time.perf_counter() - t
        if nseen % 250 == 0:
            log(f"   {ds} {nseen}/{len(qs)}")
    return dict(ds=ds, nq=nseen, nqs=len(qs), par=par_ok, L1=L1, L1h=L1h, ehop=ehop,
                famiss=famiss, M=M, npair=npair, nlab=nlab, B3=B3, S4=S4, ms=ms,
                famstats=FAM.stats)


def _st(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    return {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p90": round(float(np.quantile(a, 0.9)), 4), "max": round(float(a.max()), 4)}


def report(R):
    ds, L1 = R["ds"], R["L1"]
    out = {"ds": ds, "n_queries": R["nq"], "parity": f"{R['par']}/{R['nqs']}",
           "family_edge_lookup_miss": f"{R['famiss'][0]}/{R['famiss'][1]}",
           "corpus_family_stats": {k: R["famstats"][k] for k in
                                   ("traversal_undirected_edges", "traversal_edges_also_NER",
                                    "traversal_edges_also_KNN", "frac_traversal_also_NER",
                                    "traversal_is_subset_of_A")},
           "STEP1_transitions_per_query": {k: _st(v) for k, v in L1.items()},
           "STEP1_core_exit_by_hop": {str(h): int(R["ehop"]["core_exit"][h]) for h in range(1, 8)
                                      if R["ehop"]["core_exit"][h]},
           "STEP1_core_exit_NER_by_hop": {str(h): int(R["ehop"]["core_exit_NER"][h])
                                          for h in range(1, 8) if R["ehop"]["core_exit"][h]},
           "STEP1_core_exit_NER_frac_by_hop": {
               str(h): round(float(R["ehop"]["core_exit_NER"][h]) /
                             max(int(R["ehop"]["core_exit"][h]), 1), 4)
               for h in range(1, 8) if R["ehop"]["core_exit"][h]},
           "STEP1_by_query_hop": {f"hop{h}": {k: _st(R["L1h"][(h, k)])
                                              for k in ("core_exit", "core_exit_NER")}
                                  for h in (1, 2, 3) if R["L1h"][(h, "core_exit")]},
           "pairs": {pl: dict(R["nlab"][pl], total=R["npair"][pl]) for pl in PLS},
           "STEP2_swap": {}, "STEP2_swap_by_hop": {}, "STEP3_exclusive_support": {},
           "STEP4_source_quality": {},
           "latency_ms_per_q": round(1000 * R["ms"] / max(R["nq"], 1), 2)}
    for pl in PLS:
        for s in SC:
            d = R["M"][(pl, s)]
            g, b = d["GOOD"], d["BAD"]
            sgn = ((sum(1 for x in g if x > 0) + sum(1 for x in b if x < 0))
                   / max(len(g) + len(b), 1))
            out["STEP2_swap"][f"{pl}/{s}"] = {
                "n_good": len(g), "n_bad": len(b),
                "mean_good": round(float(np.mean(g)), 4) if g else None,
                "mean_bad": round(float(np.mean(b)), 4) if b else None,
                "AUC": (round(PB.auc(g, b), 4) if g and b else None),
                "sign_acc": round(sgn, 4) if (g or b) else None,
                "n_good_ev": len(d["GOOD_ev"]), "n_bad_ev": len(d["BAD_ev"]),
                "AUC_both_evid": (round(PB.auc(d["GOOD_ev"], d["BAD_ev"]), 4)
                                  if d["GOOD_ev"] and d["BAD_ev"] else None)}
            if ds == "metaqa":
                gh, bh = np.asarray(d["GOOD_h"]), np.asarray(d["BAD_h"])
                ga, ba = np.asarray(g), np.asarray(b)
                for h in (1, 2, 3):
                    gg, bb = ga[gh == h].tolist(), ba[bh == h].tolist()
                    out["STEP2_swap_by_hop"][f"{pl}/{s}/hop{h}"] = {
                        "n_good": len(gg), "n_bad": len(bb),
                        "mean_good": round(float(np.mean(gg)), 4) if gg else None,
                        "mean_bad": round(float(np.mean(bb)), 4) if bb else None,
                        "AUC": (round(PB.auc(gg, bb), 4) if gg and bb else None),
                        "sign_acc": round(((sum(1 for x in gg if x > 0)
                                            + sum(1 for x in bb if x < 0))
                                           / max(len(gg) + len(bb), 1)), 4)
                        if (gg or bb) else None}
    for pl in PLS:
        for bk in BUCK:
            d = R["B3"][(pl, bk)]
            out["STEP3_exclusive_support"][f"{pl}/{bk}"] = {
                "good_swaps": d["GOOD"], "bad_swaps": d["BAD"],
                "gain_loss_ratio": (round(d["GOOD"] / d["BAD"], 4) if d["BAD"] else
                                    (None if not d["GOOD"] else float("inf")))}
    for (fam, axis, b), (ngood, n) in sorted(R["S4"].items()):
        out["STEP4_source_quality"].setdefault(fam, {}).setdefault(axis, {})[b] = {
            "n": n, "n_required": ngood, "p_required": round(ngood / max(n, 1), 4)}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    o = report(run(ds, stride))
    json.dump(o, open(f"{KTD}/diag/pv_{ds}.json", "w"), indent=1)
    log("wrote", f"{KTD}/diag/pv_{ds}.json")
    print(json.dumps({k: o[k] for k in ("ds", "n_queries", "parity", "pairs",
                                        "STEP1_transitions_per_query")}, indent=1))
