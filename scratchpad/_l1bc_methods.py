"""L1 BACKWARD CAUSAL PHASE -- STEPS 4-10 and 12: the four SET-SELECTION methods.

Every method produces a real-valued score over prot | pool and nothing else changes:
    B6   final = prot(44) | top-6 of (pool - prot)        -> EXACTLY 50, the frozen contract
    B50  final = top-50 of (prot | pool)                  -> EXACTLY 50, diagnostic only
Ties are always broken by the frozen SAFE priority, so a method that carries no information
reproduces SAFE exactly.  No learned weight, no tuning knob, no threshold.

    G0_SAFE        the frozen equal-RRF F6 score
    G1_PARETO      strictly dominated candidates removed, SAFE order preserved among survivors
    G2_NOVELTY     greedy on the COUNT of evidence atoms the working set does not cover at all
    G3_SUBMODULAR  greedy on the weighted coverage F(S) = sum_e max_{p in S} W[p, e]
    G4_ASSIGNMENT  each atom credited to its argmax partition over prot | pool; score = won mass

STEP 8 depth matching is not optional.  DEPTH_MATCHED runs every method on EXACTLY the frozen F6
candidate list (same per-query length as SAFE); FULL_AVAILABLE runs on the whole visited universe.
A longer list alone costs coverage, so only the depth-matched column is a statement about ranking.

  python scratchpad/_l1bc_methods.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1pp_core as PP
import _l1bc_core as BC
import _l1bc_ledger as LG

P, B, K0, MISS = BC.P, BC.B, BC.K0, BC.MISS
RCOL = BC.RCOL
METHODS = ["G0_SAFE", "G1_PARETO", "G2_NOVELTY", "G3_SUBMODULAR", "G4_ASSIGNMENT"]
DEPTHS = ["DEPTH_MATCHED", "FULL_AVAILABLE"]
RK = [6, 12, 20, 50]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def _cov(Wd, idx):
    return Wd[:, idx].max(1) if len(idx) else np.zeros(len(Wd))


def _greedy(Wd, base_cov, pool_ix, k, tie, count_mode):
    """greedy over the pool.  count_mode: new atoms covered at all.  else: weighted coverage mass.

    Only the first k picks are produced -- everything past the deepest budget we score is left to the
    frozen SAFE tie-break, which is what makes an uninformative method reproduce SAFE exactly."""
    cov = base_cov.copy()
    left = list(range(len(pool_ix)))
    out = []
    for _ in range(min(k, len(left))):
        W = Wd[:, pool_ix[left]]
        if count_mode:
            g = ((W > 0) & (cov[:, None] <= 0)).sum(0).astype(np.float64)
        else:
            g = np.maximum(W - cov[:, None], 0).sum(0)
        j = int(np.lexsort((tie[left], -g))[0])
        out.append(left[j])
        cov = np.maximum(cov, Wd[:, pool_ix[left[j]]])
        left.pop(j)
    return out


def scores(r, ixp, prot, pool, safe_ord, method, k=P):
    """method score over `pool` (higher better) plus the frozen tie key.  pool excludes prot."""
    n_pool = len(pool)
    tie = np.array([safe_ord.get(int(p), 10 ** 6) for p in pool], np.int64)
    if method == "G0_SAFE":
        return -tie.astype(np.float64), tie
    pix = np.array([ixp[int(p)] for p in pool], np.int64)
    if method == "G1_PARETO":
        M = np.stack([r[kk][pix].astype(np.float64) for kk in RCOL], 1)
        nd, _ = LG.pareto_front(M)
        return nd.astype(np.float64), tie
    n_atom = r["n_atom"]
    Wd = np.zeros((max(n_atom, 1), r["n"]))
    if n_atom:
        Wd[r["A_aid"], r["A_pix"]] = r["A_w"]
    prot_ix = np.array([ixp[int(p)] for p in prot if int(p) in ixp], np.int64)
    if method == "G4_ASSIGNMENT":
        allix = np.concatenate([prot_ix, pix])
        Wa = Wd[:, allix]
        win = np.asarray(np.argmax(Wa, 1))
        best = Wa[np.arange(len(Wa)), win]
        s = np.zeros(n_pool)
        for e in range(len(Wa)):
            if best[e] <= 0 or win[e] < len(prot_ix):
                continue
            s[win[e] - len(prot_ix)] += best[e]
        return s, tie
    base = _cov(Wd, prot_ix)
    order = _greedy(Wd, base, pix, k, tie, method == "G2_NOVELTY")
    s = np.full(n_pool, -float(k + 1))
    for rk, j in enumerate(order):
        s[j] = -float(rk)
    return s, tie


def run(ds, depth="DEPTH_MATCHED", log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    hit = {m: {b: np.zeros(nq, np.int8) for b in ("B6", "B50")} for m in METHODS}
    rep = {m: {k: np.zeros(nq, np.float64) for k in RK} for m in METHODS}
    rep_n = np.zeros(nq, np.float64)
    ms = {m: 0.0 for m in METHODS}
    poolsz = np.zeros(nq, np.int32)
    picks = {m: [] for m in METHODS}
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        X, cands, sc = LG.safe_pick(c)
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        if depth == "DEPTH_MATCHED":
            pool = [int(p) for p in cands]
        else:
            pool = [int(p) for p in cands]
            pool += [int(p) for p in r["pid"] if int(p) not in pset and int(p) not in set(pool)]
        pool = [p for p in pool if p in ixp]
        poolsz[qi] = len(pool)
        REQ = set(int(p) for p in goldp[qi])
        safe_final = pset | set(int(p) for p in X)
        missing = REQ - safe_final
        rep_n[qi] = len(missing)
        for m in METHODS:
            t = time.perf_counter()
            s, tie = scores(r, ixp, prot, pool, safe_ord, m)
            ordr = np.lexsort((tie, -s))
            ms[m] += (time.perf_counter() - t) * 1000.0
            ch = [pool[j] for j in ordr]
            picks[m].append(ch[:B])
            f6 = pset | set(ch[:B])
            assert len(f6) == P, f"{len(f6)} != {P}"
            hit[m]["B6"][qi] = int(REQ <= f6)
            # B50: the same score over prot | pool, protected core no longer protected
            allp = prot + pool
            if m == "G0_SAFE":
                key = np.array([c["cpos"].get(p, MISS) for p in allp], np.float64)
                o50 = np.lexsort((np.asarray(allp), key))[:P]
            else:
                sa, ta = scores(r, ixp, [], allp, safe_ord, m)
                ta = np.array([c["cpos"].get(p, MISS) for p in allp], np.int64)
                o50 = np.lexsort((np.asarray(allp), ta, -sa))[:P]
            f50 = set(allp[j] for j in o50)
            hit[m]["B50"][qi] = int(REQ <= f50)
            if missing:
                for k in RK:
                    rep[m][k][qi] = len(missing & set(ch[:k]))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {depth} {qi+1}/{nq}")
    return dict(ds=ds, depth=depth, nq=nq, hops=hops, hit=hit, rep=rep, rep_n=rep_n,
                ms=ms, poolsz=poolsz, picks=picks)


def report(R):
    ds, nq = R["ds"], R["nq"]
    MS = LG.masks(ds, R["hops"], nq)
    out = {"ds": ds, "depth": R["depth"], "nq": nq,
           "pool_per_q": round(float(R["poolsz"].mean()), 1),
           "select_ms_per_q": {m: round(R["ms"][m] / nq, 3) for m in METHODS},
           "p50": {}, "repair": {}}
    base = R["hit"]["G0_SAFE"]["B6"]
    for nm, m in MS:
        out["p50"][nm] = {}
        for me in METHODS:
            row = {}
            for b in ("B6", "B50"):
                row[b] = round(float(R["hit"][me][b][m].mean()), 4)
            st = PP.mcnemar(R["hit"][me]["B6"][m], base[m])
            row.update({"net": st["net"], "gained": st["gained"], "lost": st["lost"],
                        "p": round(st["mcnemar_p"], 4), "sig": bool(st["sig"])})
            out["p50"][nm][me] = row
        den = float(R["rep_n"][m].sum())
        out["repair"][nm] = {"missing_total": int(den)}
        for me in METHODS:
            out["repair"][nm][me] = {f"REPAIR@{k}": round(float(R["rep"][me][k][m].sum()) / den, 4)
                                     if den else 0.0 for k in RK}
    return out


if __name__ == "__main__":
    dss = sys.argv[1:] or ["metaqa"]
    for ds in dss:
        for depth in DEPTHS:
            R = run(ds, depth)
            rp = report(R)
            with open(f"{BC.BCD}/diag/meth_{ds}_{depth}.json", "w") as fh:
                json.dump(rp, fh, indent=1)
            log(ds, depth, json.dumps(rp["p50"].get("ALL", rp["p50"]), indent=1))
