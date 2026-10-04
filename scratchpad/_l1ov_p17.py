"""PHASE 17 -- OPTIONAL halo-aware ranking.

Phase 9 kept the halo out of ranking on purpose: cores are scored exactly as today and the halo
only changes what a selected block PAYS OUT, so the gain cannot be a ranking artefact.  Phase 17
asks the follow-up the program allows once Phase 9 passes: if halo nodes are ALSO allowed to
vote for a block, is that universally better?

The only thing that changes is the membership map the router votes through:

    core-scored (P9)   v votes for hard[v] and the blocks of its out-neighbours  <- frozen rule
    halo-aware  (P17)  ... plus every block j whose halo contains v

The fetch rule is identical in both arms, so any difference is ranking and nothing else.

Two guards, because this touches the frozen path:
  * the membership builder is swapped in for the duration of one call and restored in a finally,
  * with an EMPTY halo the swap must reproduce the frozen selection exactly (PARITY below).

  python scratchpad/_l1ov_p17.py <ds> [cell ...]
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as EV
import _l1ep_pu as PU
import _l1ep_c as EC

OUT, log = OV.OUT, OV.log
CELLS = [("O4_FULL_C", 0.5), ("O4_FULL_C", 0.25)]


def mem_with_halo(ds, hard, pairs, npart, N):
    """the frozen membership, unioned with the halo pairs.  Same (mem_ptr, mem_flat) contract."""
    mp, mf = EC.mem_from(ds, hard)
    nptr, nidx = OV.to_node_csr(pairs, npart, N)
    out_ptr = np.zeros(N + 1, np.int64)
    flat = []
    for v in range(N):
        a = mf[mp[v]:mp[v + 1]].astype(np.int32)
        b = nidx[nptr[v]:nptr[v + 1]]
        u = np.union1d(a, b.astype(np.int32)) if len(b) else a
        flat.append(u)
        out_ptr[v + 1] = out_ptr[v] + len(u)
    return out_ptr, (np.concatenate(flat).astype(np.int32) if flat else np.zeros(0, np.int32))


def select_with(ds, hard, npart, mem):
    """run the frozen selection with a substituted membership map, then put the frozen one back."""
    orig = EC.mem_from
    EC.mem_from = lambda _ds, _hard: mem
    try:
        return EV.selected_blocks(ds, hard, npart, log)
    finally:
        EC.mem_from = orig


def coverage(SEL, hard, g, gptr, nq, nptr, nidx, bptr, bidx, npart, core_sizes):
    """the Phase-10 quantity for one selection: did we fetch every required NODE."""
    allf = np.zeros(nq, np.int8)
    allcore = np.zeros(nq, np.int8)
    rn_num = np.zeros(nq)
    rn_den = np.zeros(nq)
    expo = np.zeros(nq, np.int64)
    expo_core = np.zeros(nq, np.int64)
    selmask = np.zeros(npart, bool)
    for qi in range(nq):
        S = SEL[qi]
        Ss = set(S)
        selmask[:] = False
        selmask[np.asarray(S, np.int64)] = True
        need = sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]})
        if not need:
            rn_den[qi] = np.nan
            continue
        incore = [x for x in need if int(hard[x]) in Ss]
        got = set(incore)
        for x in need:
            if x in got:
                continue
            bs = nidx[nptr[x]:nptr[x + 1]]
            if len(bs) and Ss.intersection(bs.tolist()):
                got.add(x)
        rn_num[qi] = len(got)
        rn_den[qi] = len(need)
        allf[qi] = int(len(got) == len(need))
        allcore[qi] = int(len(incore) == len(need))
        expo_core[qi] = int(core_sizes[S].sum())
        if len(bidx):
            u = np.unique(np.concatenate([bidx[bptr[j]:bptr[j + 1]] for j in S]))
            expo[qi] = expo_core[qi] + int(np.count_nonzero(~selmask[hard[u.astype(np.int64)]]))
        else:
            expo[qi] = expo_core[qi]
    ok = ~np.isnan(rn_den)
    return {"ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4),
            "ALL_REQUIRED_CORE_ONLY": round(float(allcore[ok].mean()), 4),
            "REQUIRED_NODE_RECALL": round(float((rn_num[ok] / rn_den[ok]).mean()), 4),
            "UNIQUE_EXPOSURE_P50": round(float(expo[ok].mean()), 1),
            "EXPOSURE_MULTIPLIER": round(float(expo[ok].mean() /
                                               max(expo_core[ok].mean(), 1e-9)), 4),
            "_ind": allf.tolist()}


def mcnemar(a, b):
    a = np.asarray(a, np.int8)
    b = np.asarray(b, np.int8)
    gained = int(((b == 1) & (a == 0)).sum())
    lost = int(((b == 0) & (a == 1)).sum())
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0
    k = min(gained, lost)
    return gained, lost, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def run(ds, cells=CELLS, log=log):
    hard, npart = PU.load_assignment(ds, "CURRENT")
    hard = np.asarray(hard, np.int64)
    N = len(hard)
    g, gptr, rows, hops = PU.gold_rows(ds)
    core_sizes = np.bincount(hard, minlength=npart).astype(np.int64)

    # GUARD: an empty halo must reproduce the frozen selection exactly through the swapped path
    t0 = time.time()
    z, meta, C, ctxs, base50, f650, ind_base, PAR = select_with(
        ds, hard, npart, EC.mem_from(ds, hard))
    nq = meta["n_dev_queries"]
    ref = json.load(open(f"{OUT}/overlap/COVERAGE_{ds}.json"))["CURRENT"]
    p9 = ref["O0_CORE"]["F6"]["ALL_REQUIRED_FETCHED"]
    parity = {"P17_EMPTY_HALO_F6_ALL": PAR["F6_ALL_P50"], "P9_O0_CORE_F6_ALL": p9,
              "EXACT": bool(abs(PAR["F6_ALL_P50"] - p9) < 1e-9)}
    log(f"  {ds}: P17 path parity {parity['EXACT']} "
        f"({parity['P17_EMPTY_HALO_F6_ALL']} vs {p9})  [{time.time()-t0:.1f}s]")

    R = {"ds": ds, "PARITY": parity, "CELLS": {}}
    for fam, beta in cells:
        name = f"{fam}_b{beta}"
        t0 = time.time()
        OV.halo_pairs(ds, hard, "CURRENT", fam, N, log)
        bm_k, mass = OV.boundary_mass(ds, hard, "CURRENT", fam, N, npart, log)
        pairs = OV.bounded_pairs(bm_k, mass, hard, npart, N, beta)
        nptr, nidx = OV.to_node_csr(pairs, npart, N)
        bptr, bidx = OV.to_block_csr(pairs, npart, N)
        # ARM A -- core-scored (Phase 9): frozen selection, halo only pays out
        A = coverage(f650, hard, g, gptr, nq, nptr, nidx, bptr, bidx, npart, core_sizes)
        # ARM B -- halo-aware: the halo also votes, same fetch rule
        mem = mem_with_halo(ds, hard, pairs, npart, N)
        _, _, _, _, b50h, f650h, _, PARh = select_with(ds, hard, npart, mem)
        B = coverage(f650h, hard, g, gptr, nq, nptr, nidx, bptr, bidx, npart, core_sizes)
        gained, lost, p = mcnemar(A["_ind"], B["_ind"])
        same = int(sum(1 for i in range(nq) if set(f650[i]) == set(f650h[i])))
        R["CELLS"][name] = {
            "CORE_SCORED": {k: v for k, v in A.items() if k != "_ind"},
            "HALO_AWARE": {k: v for k, v in B.items() if k != "_ind"},
            "DELTA": round(B["ALL_REQUIRED_FETCHED"] - A["ALL_REQUIRED_FETCHED"], 4),
            "EXPOSURE_DELTA": round(B["EXPOSURE_MULTIPLIER"] - A["EXPOSURE_MULTIPLIER"], 4),
            "gained": gained, "lost": lost, "p": float(f"{p:.3g}"),
            "identical_selections": same, "frac_identical": round(same / nq, 4),
            "_ind_core": A["_ind"], "_ind_halo": B["_ind"],
            "seconds": round(time.time() - t0, 1)}
        if ds == "metaqa" and hops is not None:
            h = np.asarray(hops)[:nq]
            R["CELLS"][name]["BY_HOP"] = {
                f"hop{k}": {"CORE_SCORED": round(float(np.asarray(A["_ind"])[h == k].mean()), 4),
                            "HALO_AWARE": round(float(np.asarray(B["_ind"])[h == k].mean()), 4)}
                for k in (1, 2, 3) if (h == k).any()}
        x = R["CELLS"][name]
        log(f"  {ds:16s} {name:16s} core {A['ALL_REQUIRED_FETCHED']:.4f} -> halo-aware "
            f"{B['ALL_REQUIRED_FETCHED']:.4f}  ({x['DELTA']:+.4f}, p={x['p']:.3g}, "
            f"{gained} gained / {lost} lost, {x['frac_identical']:.1%} identical sets, "
            f"expo {x['EXPOSURE_DELTA']:+.4f})")
    os.makedirs(f"{OUT}/diagnostics", exist_ok=True)
    fp = f"{OUT}/diagnostics/P17_HALO_AWARE_RANKING.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    rec[ds] = R
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return R


if __name__ == "__main__":
    ds = sys.argv[1]
    sel = [(c.split("_b")[0], float(c.split("_b")[1])) for c in sys.argv[2:]] or CELLS
    run(ds, sel)
