"""STEP 4 done honestly -- separating the HOP PRIOR from WITHIN-STRATUM discrimination.

The first STEP 4 pass measured each signal's percentile rank over the whole candidate list, which
conflates two very different things: where a signal places the hop-1 block as a whole, and whether it
can tell gold from non-gold *inside* a hop.  A constant like `HOP` scores brilliantly on the first and
carries literally zero of the second.

So everything here is computed twice:

  * CROSS  -- gold density per (query hop x node hop) cell.  This is the prior a hop-conditioned rule
    would be exploiting, and whether it is a MetaQA artifact or a structural fact is decided by
    whether the cell pattern holds inside every query-hop block.
  * WITHIN -- rank quality computed among candidates of the SAME node hop in the SAME query.  A signal
    with no within-stratum AUC cannot rescue anything once the stratum order is fixed.

  python scratchpad/_l1ss_sep2.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ss_core as SS
import _l1ss_sep as SEP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def within_auc(val, sign, key, gold):
    """Mann-Whitney AUC inside each group of `key`, averaged over groups containing both classes."""
    if not len(val):
        return None, 0
    v = sign * np.nan_to_num(val, nan=-np.inf, neginf=-1e300, posinf=1e300)
    gi = np.searchsorted(np.unique(key), key)
    ng_ = int(gi.max()) + 1
    cnt = np.bincount(gi, minlength=ng_)
    first = np.zeros(ng_, np.int64); first[1:] = np.cumsum(cnt)[:-1]
    o = np.lexsort((np.arange(len(v)), -v, gi))          # group asc, then signal desc, then stable
    pos = np.empty(len(v), np.int64)
    pos[o] = np.arange(len(v)) - first[gi[o]]
    ng = np.bincount(gi, weights=gold, minlength=ng_)
    sp = np.bincount(gi, weights=pos * gold, minlength=ng_)
    nn = cnt - ng
    ok = (ng > 0) & (nn > 0)
    if not ok.any():
        return None, 0
    auc = 1.0 - (sp[ok] - ng[ok] * (ng[ok] - 1) / 2.0) / (ng[ok] * nn[ok])
    return round(float(auc.mean()), 4), int(ok.sum())


def main(ds="metaqa"):
    D = SEP.load(ds)
    nq = len(D["ngold"]); nrow = len(D["gold"])
    g = D["gold"].astype(np.float64)
    nh = D["HOP"].astype(np.int64)
    qh = np.asarray(D["hops"])[D["qidx"]]
    OUT = {"ds": ds, "CROSS": {}, "WITHIN": {}, "STRATUM_ORACLE": {}}

    # ---- CROSS: the hop prior, cell by cell ----------------------------------------------------
    for q in sorted(set(int(x) for x in D["hops"])):
        row = {}
        for h in sorted(set(nh.tolist())):
            m = (qh == q) & (nh == h)
            row[f"node_hop{h}"] = {"rows": int(m.sum()), "gold": int(g[m].sum()),
                                   "gold_rate": round(float(g[m].mean()), 5) if m.any() else None}
        OUT["CROSS"][f"query_hop{q}"] = row
    row = {}
    for h in sorted(set(nh.tolist())):
        m = nh == h
        row[f"node_hop{h}"] = {"rows": int(m.sum()), "gold": int(g[m].sum()),
                               "gold_rate": round(float(g[m].mean()), 5)}
    OUT["CROSS"]["ALL"] = row

    # ---- WITHIN: is there any discrimination left once the stratum is fixed? -------------------
    key = D["qidx"].astype(np.int64) * 8 + nh
    for sg in SEP.SIGNALS:
        a, n = within_auc(D[sg], SS.SIGN[sg], key, g)
        OUT["WITHIN"][sg] = {"auc_within_query_and_nodehop": a, "groups": n}
        per = {}
        for h in sorted(set(nh.tolist())):
            m = nh == h
            ah, nn = within_auc(D[sg][m], SS.SIGN[sg], D["qidx"][m].astype(np.int64), g[m])
            per[f"node_hop{h}"] = {"auc": ah, "groups": nn}
        OUT["WITHIN"][sg]["by_node_hop"] = per
        log(f"  {sg:28s} within-stratum AUC {a}   " +
            "  ".join(f"h{h[-1]} {per[h]['auc']}" for h in per))

    # ---- what a perfect stratum ORDER alone would buy, with the frozen key inside each stratum --
    # (upper bound on any purely hop-conditioned reordering that does not touch within-hop order)
    ncand = np.bincount(D["qidx"], minlength=nq)
    for name, order in (("hop_asc", [1, 2, 3]), ("frozen_global", None)):
        if order is None:
            continue
        pos = np.empty(nrow, np.int64)
        o = np.lexsort((np.arange(nrow), D["ADMIT_RANK"], nh, D["qidx"]))
        pos[o] = np.arange(nrow) - D["qptr"][D["qidx"][o]]
        hit = int((D["gold"].astype(bool) & (pos < 64)).sum())
        OUT["STRATUM_ORACLE"]["hop_block_then_admit_rank"] = {
            "recall@64": round(hit / max(1, int(D["ngold"].sum())), 4),
            "cond@64": round(hit / max(1, int(D["gold"].sum())), 4)}
    OUT["STRATUM_ORACLE"]["ceiling_all_candidate_gold"] = round(
        float(D["gold"].sum() / D["ngold"].sum()), 4)
    OUT["STRATUM_ORACLE"]["mean_candidates_per_query"] = round(float(ncand.mean()), 1)

    fp = f"{SS.SSD}/diag/sep2_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["metaqa"]))
