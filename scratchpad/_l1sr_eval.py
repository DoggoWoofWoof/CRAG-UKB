"""STEPS 4-7 and 9 -- node scorers, node->partition aggregations, EXACT P50, budget curve, swap.

Only the QUALITY OF THE STRUCTURAL RANKING changes.  The protected core, B=6, the retrieval channel,
the canonical channel and the exactly-50 output are the frozen contract and are untouched:

    SF   = the structural partition ranking            <- the ONLY thing a mechanism replaces
    chal = [p in SF not in base50] + [p in RF not in base50]
    spos = rank map of SF ; rpos, cpos frozen
    X    = f6_select(bnd, chal, spos, rpos, cpos, B=6)
    final = prot(44) | X                                -> EXACTLY 50 partitions, always

PARITY GATE.  N0 x P0 must reproduce SAFE exactly.  N0 is the beam with the residual update switched
off, which was verified to reproduce the frozen static node set on 300/300 queries, and P0 is the
frozen S4 aggregation, so any deviation is a harness bug and is reported as one.

STEP 7 keeps the same contract shape and varies only the budget: final = prot(44) | top-(k-44) of
the same score order, so k=50 is exactly SAFE.
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1kb_core as KB
import _l1cv_core as CV
import _l1sr_seq as SQ

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
SRD = SQ.SRD
P, K0, B = PP.P, PP.K0, CV.B
M_STRUCT = KB.BASE_CFG["M_struct"]
VARIANTS = [("N0", False, "term"), ("N1", True, "term"), ("N2", True, "minv"), ("N3", True, "geo")]
KEYOF = {"N0": "term", "N1": "term", "N2": "minv", "N3": "geo"}
AGGS = ["P0_S4", "P1_MAX_NODE", "P2_TOP2_SUPPORT", "P3_BEST_PATH", "P4_MULTI_SEED_PATH_SUPPORT"]
BUDGETS = [50, 64, 80, 100, 128, 256]


# ------------------------------------------------------------------ STEP 3/4: the beams
def beams(ds, S, log=log, cache=True):
    """one beam per node scorer, with path provenance preserved (STEP 5 consumes it)."""
    os.makedirs(f"{SRD}/cache", exist_ok=True)
    fp = f"{SRD}/cache/beam_{ds}.pkl"
    if cache and os.path.exists(fp):
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    z = S["z"]
    _, _, _, (adjp, adji), deg, _ = TA.load_topology(ds, lambda *a: None)
    adjp = np.asarray(adjp); adji = np.asarray(adji); deg = np.asarray(deg)
    Xn, Qm = SQ.load_emb(ds, z)
    nq = S["nq"]
    OUT = {v: [] for v, _, _ in VARIANTS}
    OUT["_parity_static"] = 0
    OUT["_edges"] = {v: 0 for v, _, _ in VARIANTS}
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        r0 = TA.residual(Qm[qi].astype(np.float64), sd, Xn).astype(np.float32)
        a, _, _ = TA.expand_dir(sd, r0, adjp, adji, deg, Xn, SQ.BEAM)
        stat = set(int(v) for v in a)
        for nm, up, cr in VARIANTS:
            rec, e = SQ.expand_seq(sd, r0, adjp, adji, deg, Xn, SQ.BEAM, update=up, crit=cr)
            OUT["_edges"][nm] += e
            if nm == "N0":
                OUT["_parity_static"] += int(set(rec) == stat)
            OUT[nm].append({int(v): (float(r["term"]), float(r["minv"]), float(r["geo"]),
                                     int(r["hop"]), int(r["sup"]), len(r["seeds"]))
                            for v, r in rec.items()})
        if (qi + 1) % 400 == 0:
            log(f"   beam {ds} {qi+1}/{nq}  static-parity {OUT['_parity_static']}/{qi+1}")
    if cache:
        with open(fp, "wb") as fh:
            pickle.dump(OUT, fh, protocol=5)
    log(f"[beam] {ds} N0-vs-frozen-static node-set parity {OUT['_parity_static']}/{nq}  "
        f"edges " + " ".join(f"{k}={v:,}" for k, v in OUT["_edges"].items()))
    return OUT


# ------------------------------------------------------------------ STEP 5: node -> partition
IDX = {"term": 0, "minv": 1, "geo": 2}


def sf_of(rec, key, hard, aggname, depth=M_STRUCT):
    """structural PARTITION ranking from one beam under one aggregation."""
    ki = IDX[key]
    order = sorted(rec, key=lambda v: (-rec[v][ki], v))[:depth]
    agg = {}
    for r, v in enumerate(order):
        t, mn, ge, hp, sup, nsd = rec[v]
        p = int(hard[v])
        if p < 0:
            continue
        sc = (t, mn, ge)[ki]
        a = agg.get(p)
        if a is None:
            agg[p] = [r, 1, 1.0 / (K0 + r), hp, sc, sup, ge, {v: nsd}, [sc]]
        else:
            a[1] += 1; a[2] += 1.0 / (K0 + r); a[3] = min(a[3], hp)
            a[4] = max(a[4], sc); a[5] += sup; a[6] = max(a[6], ge)
            a[7][v] = nsd; a[8].append(sc)
    if aggname == "P0_S4":
        return RT.order_struct({p: a[:6] for p, a in agg.items()}, "S4")
    if aggname == "P1_MAX_NODE":
        return sorted(agg, key=lambda p: (-agg[p][4], agg[p][0], p))
    if aggname == "P2_TOP2_SUPPORT":
        return sorted(agg, key=lambda p: (-sum(sorted(agg[p][8], reverse=True)[:2]),
                                          agg[p][0], p))
    if aggname == "P3_BEST_PATH":
        return sorted(agg, key=lambda p: (-agg[p][6], agg[p][0], p))
    if aggname == "P4_MULTI_SEED_PATH_SUPPORT":
        return sorted(agg, key=lambda p: (-max(agg[p][7].values()), -agg[p][4], agg[p][0], p))
    raise ValueError(aggname)


def sf_from_cache(z, qi, hard, aggname, depth=M_STRUCT):
    """P0/P1/P2 straight from the FROZEN structural cache -- no re-expansion, no node embeddings.

    s_node/s_hop/s_sdir/s_cnt are exactly the quantities struct_aggregate_full reads, so P0 here is
    the frozen S4 by construction and P1/P2 are the same aggregation family over the same nodes.
    P3/P4 need path provenance the frozen cache never stored and are not available on this path."""
    sn = z["s_node"][qi]; sh = z["s_hop"][qi]; sd = z["s_sdir"][qi]; sc = z["s_cnt"][qi]
    agg = {}
    for jj in range(min(len(sn), depth)):
        v = int(sn[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0:
            continue
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (K0 + jj), int(sh[jj]), float(sd[jj]), int(sc[jj]),
                      [float(sd[jj])]]
        else:
            a[1] += 1; a[2] += 1.0 / (K0 + jj); a[3] = min(a[3], int(sh[jj]))
            a[4] = max(a[4], float(sd[jj])); a[5] += int(sc[jj]); a[6].append(float(sd[jj]))
    if aggname == "P0_S4":
        return RT.order_struct({p: a[:6] for p, a in agg.items()}, "S4")
    if aggname == "P1_MAX_NODE":
        return sorted(agg, key=lambda p: (-agg[p][4], agg[p][0], p))
    if aggname == "P2_TOP2_SUPPORT":
        return sorted(agg, key=lambda p: (-sum(sorted(agg[p][6], reverse=True)[:2]), agg[p][0], p))
    raise ValueError(f"{aggname} needs path provenance; use sf_of on a rebuilt beam")


# ------------------------------------------------------------------ STEP 6/7/9: exact P50
def order_for(c, SF, rule="A_F6", cont=False):
    """the boundary score order under one swap rule.  Returns the sorted candidate list."""
    b50, bnd, cpos, rpos = c["base50"], c["bnd"], c["cpos"], c["rpos"]
    RF = c["_RF"]
    spos = {p: r for r, p in enumerate(SF)}
    chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
    if cont:
        chal = chal + [p for p in c["_CONT"] if p not in b50]
    if rule == "A_F6":
        _, sc = KB.f6_select(bnd, chal, spos, rpos, cpos, B)
        return sc, spos
    cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
    if rule == "B_STRUCT_DIRECT":
        # the strongest structural challengers replace the boundary outright
        sc = sorted(((spos.get(p, 10 ** 6), cpos.get(p, 10 ** 6), p) for p in cands))
        return [(-1.0 / (1 + a), b, p) for a, b, p in sc], spos
    if rule == "C_PAIRWISE_EVIDENCE":
        # one deterministic pairwise rule, no weights and no thresholds: a challenger outranks an
        # incumbent iff it carries strictly more evidence CHANNELS (canonical/structural/retrieval).
        def ch(p):
            return (int(p in cpos) + int(p in spos) + int(p in rpos))
        sc = sorted(((-ch(p), spos.get(p, 10 ** 6), cpos.get(p, 10 ** 6), p) for p in cands))
        return [(a, c2, p) for a, _, c2, p in sc], spos
    raise ValueError(rule)


def evaluate(S, SFs, rule="A_F6", budgets=(P,)):
    """exact-P50 (and, for STEP 7, exact-k) coverage under one mechanism."""
    ctxs, goldp, nq = S["ctxs"], S["goldp"], S["nq"]
    out = {k: np.zeros(nq, np.int8) for k in budgets}
    churn = np.zeros(nq, np.int32)
    for qi, c in enumerate(ctxs):
        sc, _ = order_for(c, SFs[qi], rule)
        picks = [p for _, _, p in sc]
        for k in budgets:
            X = picks[:k - (P - B)]
            fs = c["prot_set"] | set(X)
            if k == P:
                assert len(fs) == P, f"{len(fs)} != {P}"
                churn[qi] = len(fs - c["base50"])
            out[k][qi] = int(goldp[qi] <= fs)
    return out, churn


def evaluate_budget(S, SFs, rule="A_F6", budgets=BUDGETS):
    """STEP 7 -- the same contract shape with only the budget varied.

    final_k = prot(44) | top-(k-44) of the SAME score order.  k = 50 is exactly the P50 output, so
    the curve is anchored on SAFE and nothing else changes.  The candidate pool is extended with the
    canonical continuation base_rank[50:] (the I_CANON_CONT family, already part of the frozen
    substrate) purely so a budget of 128 can be FILLED; a continuation entry has canonical rank >= 50
    and therefore always scores below every boundary incumbent, so it cannot alter k = 50."""
    ctxs, goldp, nq = S["ctxs"], S["goldp"], S["nq"]
    out = {k: np.zeros(nq, np.int8) for k in budgets}
    short = {k: 0 for k in budgets}
    size = {k: np.zeros(nq, np.int32) for k in budgets}
    for qi, c in enumerate(ctxs):
        sc, _ = order_for(c, SFs[qi], rule, cont=True)
        picks = [p for _, _, p in sc]
        for k in budgets:
            X = picks[:k - (P - B)]
            short[k] += int(len(X) < k - (P - B))
            fs = c["prot_set"] | set(X)
            size[k][qi] = len(fs)
            out[k][qi] = int(goldp[qi] <= fs)
    return out, {k: {"underfilled": short[k], "mean_partitions": round(float(size[k].mean()), 1)}
                 for k in budgets}


def by_hop(ind, hops):
    """ALL, plus a per-hop split only where the substrate actually carries one (metaqa)."""
    d = {"ALL": round(float(ind.mean()), 4)}
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    if hs and min(hs) >= 0:
        for h in hs:
            m = np.asarray(hops) == h
            d[f"hop{h}"] = round(float(ind[m].mean()), 4)
    return d


def substrate(ds):
    S = CV.substrate(ds)
    if hasattr(S["z"], "files"):
        S["z"] = {k: S["z"][k] for k in S["z"].files}
    C, _ = KB.substrate(ds, S["z"], S["meta"], __import__("_l1kb_router").build_groups)
    cfg = KB.BASE_CFG
    for qi, c in enumerate(S["ctxs"]):
        c["_RF"] = C["rfull"][cfg["M_ret"]][qi]
        c["_CONT"] = [p for p, _ in sorted(c["cpos"].items(), key=lambda kv: kv[1])][P:]
    S["hard"] = np.asarray(S["z"]["hard"])
    return S
