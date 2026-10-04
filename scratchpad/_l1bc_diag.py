"""L1 BACKWARD CAUSAL PHASE -- STEP 4 (Pareto diagnostic) + STEP 5 (novelty diagnostic).

Both are pure diagnostics on the frozen SAFE state: they ask whether the information needed to
prefer a MISSING required partition over the nuisance challengers is present in the evidence at all,
before any method is allowed to use it.

STEP 4.  Every one of the 10 rank columns is oriented smaller-is-better, so p is DOMINATED iff some
candidate q is at least as good in every column and strictly better in at least one.  A dominated
missing partition cannot be rescued by ANY monotone scoring rule over these columns -- not by a
different fusion, not by a set objective.  MISSING_GOLD_ON_PARETO_FRONT is therefore the honest
ceiling on the whole idea, and the front SIZE is the price: everything on the front is a candidate
a dominance filter would keep.

STEP 5.  Novelty of p against the current final set S:
    nov_atoms = number of evidence atoms p carries that S covers not at all
    nov_mass  = sum_e max(0, W[p, e] - max_{q in S} W[q, e])
Reported for MISSING REQUIRED vs NUISANCE challengers (pool members that are neither required nor
selected).  The discriminative question is not whether missing partitions are novel -- it is whether
they are MORE novel than the nuisance population, which is what the AUC measures.

  python scratchpad/_l1bc_diag.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1bc_core as BC
import _l1bc_ledger as LG

P, B, MISS = BC.P, BC.B, BC.MISS
RCOL = BC.RCOL
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def front_ranks(M):
    """iterated Pareto peeling.  rank 0 = the front itself."""
    n = len(M)
    rk = np.full(n, -1, np.int32)
    live = np.ones(n, bool)
    d = 0
    while live.any() and d < 64:
        idx = np.flatnonzero(live)
        nd, _ = LG.pareto_front(M[idx])
        rk[idx[nd]] = d
        live[idx[nd]] = False
        d += 1
    rk[rk < 0] = d
    return rk


def _auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.empty(len(a))
    o = np.argsort(a, kind="mergesort")
    s = a[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(ds, log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    rec = []           # one row per POOL candidate of a query that has a missing required partition
    per_q = []
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        X, cands, sc = LG.safe_pick(c)
        pool = [int(p) for p in cands if int(p) in ixp]
        REQ = set(int(p) for p in goldp[qi])
        sel = pset | set(int(p) for p in X)
        missing = REQ - sel
        pix = np.array([ixp[p] for p in pool], np.int64)
        M = np.stack([r[k][pix].astype(np.float64) for k in RCOL], 1)
        fr = front_ranks(M)
        n_atom = r["n_atom"]
        Wd = np.zeros((max(n_atom, 1), r["n"]))
        if n_atom:
            Wd[r["A_aid"], r["A_pix"]] = r["A_w"]
        selix = np.array([ixp[p] for p in sel if p in ixp], np.int64)
        cov = Wd[:, selix].max(1) if len(selix) else np.zeros(len(Wd))
        Wp = Wd[:, pix]
        nov_a = ((Wp > 0) & (cov[:, None] <= 0)).sum(0)
        nov_m = np.maximum(Wp - cov[:, None], 0).sum(0)
        per_q.append((int(hops[qi]), len(missing), int((fr == 0).sum()), len(pool),
                      len(missing & set(pool))))
        if not missing:
            continue
        for j, p in enumerate(pool):
            kind = 2 if p in missing else (1 if p in sel else 0)   # 0 nuisance 1 selected 2 missing
            rec.append((int(hops[qi]), kind, int(fr[j]), int(nov_a[j]), float(nov_m[j])))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}")
    return dict(ds=ds, nq=nq, rec=np.array(rec, np.float64) if rec else np.zeros((0, 5)),
                per_q=np.array(per_q, np.float64), hops=hops)


def report(R):
    ds, nq = R["ds"], R["nq"]
    rc, pq = R["rec"], R["per_q"]
    MS = LG.masks(ds, R["hops"], nq)
    out = {"ds": ds, "nq": nq, "pareto": {}, "novelty": {}}
    for nm, m in MS:
        h = np.asarray(R["hops"])[m]
        hs = set(int(x) for x in h)
        sub = rc[np.isin(rc[:, 0], list(hs))] if ds == "metaqa" and nm != "ALL" else rc
        qsub = pq[np.isin(pq[:, 0], list(hs))] if ds == "metaqa" and nm != "ALL" else pq
        miss = sub[sub[:, 1] == 2]
        nuis = sub[sub[:, 1] == 0]
        selr = sub[sub[:, 1] == 1]
        nm_tot = float(qsub[:, 1].sum())
        out["pareto"][nm] = {
            "missing_in_pool": int(len(miss)),
            "missing_total": int(nm_tot),
            "MISSING_GOLD_ON_PARETO_FRONT": round(float((miss[:, 2] == 0).mean()), 4) if len(miss) else 0.0,
            "nuisance_on_front": round(float((nuis[:, 2] == 0).mean()), 4) if len(nuis) else 0.0,
            "selected_on_front": round(float((selr[:, 2] == 0).mean()), 4) if len(selr) else 0.0,
            "front_size_per_q": round(float(qsub[:, 2].mean()), 2),
            "pool_per_q": round(float(qsub[:, 3].mean()), 1),
            "missing_front_rank_mean": round(float(miss[:, 2].mean()), 3) if len(miss) else 0.0,
            "nuisance_front_rank_mean": round(float(nuis[:, 2].mean()), 3) if len(nuis) else 0.0,
            "front_rank_AUC_missing_vs_nuisance":
                round(_auc(-miss[:, 2], -nuis[:, 2]), 4) if len(miss) and len(nuis) else 0.0}
        out["novelty"][nm] = {
            "nov_atoms_missing": round(float(miss[:, 3].mean()), 3) if len(miss) else 0.0,
            "nov_atoms_nuisance": round(float(nuis[:, 3].mean()), 3) if len(nuis) else 0.0,
            "nov_mass_missing": round(float(miss[:, 4].mean()), 5) if len(miss) else 0.0,
            "nov_mass_nuisance": round(float(nuis[:, 4].mean()), 5) if len(nuis) else 0.0,
            "AUC_nov_atoms": round(_auc(miss[:, 3], nuis[:, 3]), 4) if len(miss) and len(nuis) else 0.0,
            "AUC_nov_mass": round(_auc(miss[:, 4], nuis[:, 4]), 4) if len(miss) and len(nuis) else 0.0,
            "frac_missing_zero_novelty": round(float((miss[:, 3] == 0).mean()), 4) if len(miss) else 0.0,
            "frac_nuisance_zero_novelty": round(float((nuis[:, 3] == 0).mean()), 4) if len(nuis) else 0.0}
    return out


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["metaqa"]):
        R = run(ds)
        rp = report(R)
        with open(f"{BC.BCD}/diag/diag_{ds}.json", "w") as fh:
            json.dump(rp, fh, indent=1)
        log(ds, json.dumps(rp, indent=1))
