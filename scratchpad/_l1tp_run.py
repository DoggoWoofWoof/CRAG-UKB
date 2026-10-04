"""TRIPLET PHASE -- STEPS 3, 4, 5, 6, 12 read off the cached transition table.

STEP 3  each triplet signal measured INDIVIDUALLY (no combination anywhere in this file).
STEP 4  the target of every measurement is MARGINAL_USEFUL: a required gold partition the
        canonical/protected core does not already hold.  Generic gold-node recall is never the
        primary number.
STEP 5  the partition-TRANSITION representation Pi -> Pj against the collapse to Pj, and against
        the plain "Pj was structurally reached" control inside the same universe.
STEP 6  exactly three controls -- C0_SINGLE_TRIPLET, C1_BEST_COMPOSABLE_CHAIN,
        C2_MULTI_PATH_SUPPORT.  No scorer grid.
STEP 12 saturation: partitions reached per query vs transitions retained vs coherent chains.

  python scratchpad/_l1tp_run.py <ds>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1tp_core as TP
import _l1tp_build as TB

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
PBUD = [6, 12, 20, 50]        # partition budgets, as in the calibration audit

# STEP 5/6 partition-level orderings.  Each is ONE quantity over the Pi->Pj table, aggregated to
# the target partition Pj.  The first is the "Pj was structurally reached" control.
PORD = [
    ("R0_REACHED_ARRIVAL",      "first_jj",  +1),   # control: reached, and how early
    ("R1_REACHED_COUNT",        "n_tgt",     -1),   # control: how many nodes of Pj were reached
    ("C0_SINGLE_TRIPLET",       "max_T0",    -1),   # STEP 6 C0: best single real transition
    ("T1_SOURCE",               "max_T1",    -1),
    ("T2_TARGET",               "max_T2",    -1),
    ("T3_HOP",                  "min_T3",    +1),
    ("C1_BEST_CHAIN_MIN",       "max_pmin",  -1),   # STEP 6 C1: best composable chain, weakest link
    ("C1_BEST_CHAIN_SUM",       "max_psum",  -1),   # STEP 6 C1: best composable chain, total
    ("C2_MULTI_PATH_SUPPORT",   "n_chain",   -1),   # STEP 6 C2: distinct ancestor triplets
    ("C2_MULTI_SEED_SUPPORT",   "n_seed",    -1),
    ("TR_SOURCE_PARTITIONS",    "_nsrcpart", -1),   # transition-only: how many distinct Pi feed Pj
    ("TR_TRANSITION_COUNT",     "_ntrans",   -1),   # transition-only: how many distinct Pi->Pj
]


def rows(T, qi):
    a, b = int(T["qptr"][qi]), int(T["qptr"][qi + 1])
    return {k: T[k][a:b] for k in TB.TCOL}


def pj_orderings(r):
    """collapse the transition table to one ordering of target partitions per named signal."""
    pj = r["pj"].astype(np.int64)
    up, ip = np.unique(pj, return_inverse=True)
    nk = len(up)
    first = np.full(nk, 1 << 30, np.int64)
    np.minimum.at(first, ip, r["first_jj"].astype(np.int64))
    out = {}
    for name, col, sgn in PORD:
        if col == "_nsrcpart":
            val = np.bincount(np.unique(np.stack([ip, r["pi"].astype(np.int64)]), axis=1)[0],
                              minlength=nk).astype(np.float64)
        elif col == "_ntrans":
            val = np.bincount(ip, minlength=nk).astype(np.float64)
        elif col in ("n_tgt", "n_chain", "n_seed"):
            val = np.bincount(ip, weights=r[col].astype(np.float64), minlength=nk)
        elif col == "first_jj":
            val = first.astype(np.float64)
        elif col == "min_T3":
            val = np.full(nk, 1 << 20, np.float64); np.minimum.at(val, ip, r[col].astype(np.float64))
        else:
            val = np.full(nk, -np.inf); np.maximum.at(val, ip, r[col].astype(np.float64))
        # deterministic: signal, then arrival inside the frozen universe, then partition id
        out[name] = up[np.lexsort((up, first, sgn * val))]
    return out, up, first


def _rec(order, need, buds):
    o = np.zeros(len(buds))
    if not need:
        return o
    for i, k in enumerate(buds):
        o[i] = len(set(int(x) for x in order[:k]) & need) / len(need)
    return o


def _auc(score, lab):
    """rank AUC of one signal against the marginal-useful label, pooled per query then averaged."""
    n1 = int(lab.sum()); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.empty(len(score)); r[np.argsort(score, kind="stable")] = np.arange(len(score))
    # average ranks for ties
    o = np.argsort(score, kind="stable"); s = score[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return (r[lab].sum() - n1 * (n1 - 1) / 2.0) / (n1 * n0)


def run(ds, log=log):
    fp = f"{TP.TPD}/data/tr_{ds}.npz"
    T = dict(np.load(fp, allow_pickle=False))
    nq = int(T["nq"][0])
    S = EV.substrate(ds)
    hard = S["hard"].astype(np.int64)
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S["hops"] is not None else np.full(nq, -1)
    BL = [("ALL", np.ones(nq, bool))]
    if len(hops) and int(np.min(hops)) >= 0:
        for h in sorted(set(int(x) for x in hops)):
            BL.append((f"hop{h}", hops == h))

    NEED = T["NEEDN"]
    has = NEED > 0
    O = {"ds": ds, "nq": nq,
         "STEP12_SATURATION": {}, "STEP3_PERCENTILE": {}, "STEP4_TRIPLET_RECALL": {},
         "STEP4_NODE_RECALL": {}, "STEP5_PARTITION_RECALL": {}, "STEP5_AUC": {},
         "MARGINAL_PREVALENCE": {}}

    # ---------------- STEP 12: what each representation actually retains, per query
    npart = int(T["npart"][0])
    for b, mk in BL:
        s = mk & has
        O["STEP12_SATURATION"][b] = {
            "queries": int(mk.sum()), "queries_with_a_needed_partition": int(s.sum()),
            "needed_partitions_per_query": round(float(NEED[mk].mean()), 3),
            "NODE_partitions_reached_per_query": round(float(T["NPJ"][mk].mean()), 1),
            "NODE_partition_saturation": round(float(T["NPJ"][mk].mean() / npart), 4),
            "TRIPLET_transitions_per_query": round(float(T["NTR"][mk].mean()), 1),
            "PATH_coherent_chains_per_query": round(float(T["NPATH"][mk].mean()), 1),
            "edges_per_query": round(float(T["NE"][mk].mean()), 1),
            "unique_target_nodes_per_query": round(float(T["NV"][mk].mean()), 1),
            "transition_per_partition_ratio": round(float(T["NTR"][mk].sum() /
                                                         max(T["NPJ"][mk].sum(), 1)), 3),
            "needed_partitions_all_reachable_as_a_transition_target":
                round(float(T["REACH"][s].mean()), 4) if s.any() else None,
            "marginal_useful_triplets_per_query": round(float(T["NMU"][mk].mean()), 1),
            "marginal_useful_triplet_prevalence":
                round(float(T["NMU"][mk].sum() / max(T["NE"][mk].sum(), 1)), 5)}

    # ---------------- STEP 3 + STEP 4: per-signal, per-TRIPLET, individually
    for k in TB.SIG:
        P = T[f"PCT_{k}"]
        O["STEP3_PERCENTILE"][k] = {b: (round(float(np.nanmean(P[mk])), 4)
                                        if np.isfinite(P[mk]).any() else None) for b, mk in BL}
    for k in TB.SIG:
        R = T[f"REC_{k}"]
        O["STEP4_TRIPLET_RECALL"][k] = {b: {f"R@{c}": round(float(R[mk & has, i].mean()), 4)
                                            for i, c in enumerate(TB.BUD)} for b, mk in BL}
    for k in TB.NSIG:
        R = T[f"REC_{k}"]
        O["STEP4_NODE_RECALL"][k] = {b: {f"R@{c}": round(float(R[mk & has, i].mean()), 4)
                                         for i, c in enumerate(TB.BUD)} for b, mk in BL}

    # ---------------- STEP 5 + STEP 6: at the PARTITION budget, transition evidence vs reach
    REC = {n: np.zeros((nq, len(PBUD))) for n, _, _ in PORD}
    AUC = {n: np.full(nq, np.nan) for n, _, _ in PORD}
    NODE = {"NODE_S4_FROZEN": np.zeros((nq, len(PBUD)))}
    for qi in range(nq):
        need = set(int(p) for p in goldp[qi]) - ctxs[qi]["prot_set"]
        r = rows(T, qi)
        ords, up, first = pj_orderings(r)
        lab = np.fromiter((int(p) in need for p in up), bool, len(up))
        for n, _, _ in PORD:
            REC[n][qi] = _rec(ords[n], need, PBUD)
        if need:
            pos = {n: {int(p): i for i, p in enumerate(ords[n])} for n, _, _ in PORD}
            for n, _, _ in PORD:
                AUC[n][qi] = _auc(-np.array([pos[n][int(p)] for p in up], float), lab)
        NODE["NODE_S4_FROZEN"][qi] = _rec(
            np.asarray(EV.sf_from_cache(S["z"], qi, hard, "P0_S4")), need, PBUD)
    for n, _, _ in PORD:
        O["STEP5_PARTITION_RECALL"][n] = {b: {f"R@{c}": round(float(REC[n][mk & has, i].mean()), 4)
                                              for i, c in enumerate(PBUD)} for b, mk in BL}
        O["STEP5_AUC"][n] = {b: (round(float(np.nanmean(AUC[n][mk])), 4)
                                 if np.isfinite(AUC[n][mk]).any() else None) for b, mk in BL}
    O["STEP5_PARTITION_RECALL"]["NODE_S4_FROZEN"] = {
        b: {f"R@{c}": round(float(NODE["NODE_S4_FROZEN"][mk & has, i].mean()), 4)
            for i, c in enumerate(PBUD)} for b, mk in BL}

    json.dump(O, open(f"{TP.TPD}/diag/tp_{ds}.json", "w"), indent=1)
    b0 = "hop3" if any(b == "hop3" for b, _ in BL) else "ALL"
    log(f"[{ds}] STEP12 {b0}: Pj/q {O['STEP12_SATURATION'][b0]['NODE_partitions_reached_per_query']} "
        f"transitions/q {O['STEP12_SATURATION'][b0]['TRIPLET_transitions_per_query']} "
        f"chains/q {O['STEP12_SATURATION'][b0]['PATH_coherent_chains_per_query']} "
        f"MU prevalence {O['STEP12_SATURATION'][b0]['marginal_useful_triplet_prevalence']}")
    for k in TB.SIG:
        log(f"[{ds}] STEP3/4 {k:14s} pct {O['STEP3_PERCENTILE'][k][b0]}  " +
            "  ".join(f"{a} {v}" for a, v in O["STEP4_TRIPLET_RECALL"][k][b0].items()))
    for k in TB.NSIG:
        log(f"[{ds}] STEP4 NODE {k:14s}      " +
            "  ".join(f"{a} {v}" for a, v in O["STEP4_NODE_RECALL"][k][b0].items()))
    for n, _, _ in PORD:
        log(f"[{ds}] STEP5/6 {n:24s} AUC {O['STEP5_AUC'][n][b0]}  " +
            "  ".join(f"{a} {v}" for a, v in O["STEP5_PARTITION_RECALL"][n][b0].items()))
    log(f"[{ds}] STEP5/6 {'NODE_S4_FROZEN':24s}           " +
        "  ".join(f"{a} {v}" for a, v in O["STEP5_PARTITION_RECALL"]["NODE_S4_FROZEN"][b0].items()))
    log(f"[{ds}] wrote diag/tp_{ds}.json")
    return O


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["metaqa"]):
        run(d)
