"""STEPS 2-5 -- single-signal separability at the read bottleneck.

Every signal re-ranks the SAME candidate set (the admitted non-seed nodes the `M_struct=64` read is
applied to).  Nothing about the search changes, so any difference is the ranking.

Two denominators are reported and they answer different questions:

  * absolute recall  = gold in the top k / ALL required gold nodes.  Directly comparable to the
    funnel: at k=64 under the frozen key it must reproduce 0.1334 on MetaQA.
  * conditional recall = gold in the top k / gold present in the candidate set.  Pure ranking
    quality, with discovery held constant.

Reporting is by MetaQA QUERY hop, which is an evaluation annotation.  Conditioning (STEP 4) is by
NODE hop, which is the only one of the two a rule may see at inference time.

  python scratchpad/_l1ss_sep.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1ss_core as SS

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
KS = (16, 32, 64, 128)
# STEP 2 signal list, in the directive's order.  PATH_SUM/PATH_MIN are the two parameter-free
# aggregations of G; CS_ADMIT is the frozen ordering key and is the control.
SIGNALS = ["CS_ADMIT", "STATIC_SDIR", "QSIM", "RSIM", "HOP", "SEED_RANK", "PARENT_SCORE",
           "PATH_SUM", "PATH_MIN", "DISTINCT_SEED_SUPPORT", "PATH_SUPPORT", "RR_PARENT_SUPPORT",
           "PARTITION_CANONICAL_PRIOR", "PARTITION_RETRIEVAL_PRIOR", "FUTURE_MAX", "ADMIT_RANK"]


def load(ds):
    z = np.load(f"{SS.SSD}/data/{ds}.npz")
    return {k: z[k] for k in z.files}


def ranks_of(val, sign, qidx, qptr, nrow):
    """0-based rank of every row inside its own query, ties broken by frozen `added` order."""
    key = (-sign) * np.nan_to_num(val, nan=-np.inf, neginf=-1e300, posinf=1e300)
    o = np.lexsort((np.arange(nrow), key, qidx))
    pos = np.empty(nrow, np.int64)
    pos[o] = np.arange(nrow) - qptr[qidx[o]]
    return pos


def metrics(pos, D, nq, mask_q, gold_total):
    """absolute/conditional recall, percentile rank and per-query AUC on one block of queries."""
    g = D["gold"].astype(bool)
    inb = mask_q[D["qidx"]]
    n_in_q = (D["qptr"][1:] - D["qptr"][:-1]).astype(np.float64)
    out = {"gold_in_candidates": int((g & inb).sum()), "gold_total": int(gold_total)}
    for k in KS:
        hit = int((g & inb & (pos < k)).sum())
        out[f"recall@{k}"] = round(hit / max(1, gold_total), 4)
        out[f"cond@{k}"] = round(hit / max(1, int((g & inb).sum())), 4)
    den = np.maximum(n_in_q[D["qidx"]] - 1.0, 1.0)
    sel = g & inb
    out["gold_pctrank"] = round(float((pos[sel] / den[sel]).mean()), 4) if sel.any() else None
    # per-query AUC, averaged over queries that actually contain both classes
    ng = np.zeros(nq); np.add.at(ng, D["qidx"], D["gold"])
    sp = np.zeros(nq); np.add.at(sp, D["qidx"][g], pos[g].astype(np.float64))
    nn = n_in_q - ng
    ok = mask_q & (ng > 0) & (nn > 0)
    if ok.any():
        auc = 1.0 - (sp[ok] - ng[ok] * (ng[ok] - 1) / 2.0) / (ng[ok] * nn[ok])
        out["auc"] = round(float(auc.mean()), 4)
        out["auc_queries"] = int(ok.sum())
    return out


def main(ds="metaqa"):
    D = load(ds)
    nq = len(D["ngold"]); nrow = len(D["gold"])
    hops = D["hops"]
    blocks = [("ALL", np.ones(nq, bool))]
    hs = sorted(set(int(x) for x in hops))
    if hs and min(hs) >= 0:
        blocks += [(f"hop{h}", np.asarray(hops) == h) for h in hs]

    # gold nodes that are retrieval SEEDS are in scope but never in the candidate set; the funnel
    # counts them, so the read-efficiency denominator has to be built the same way
    S = EV.substrate(ds)
    gr, _ = DG.gold_rows_of(ds, S["z"])
    seedg = np.zeros(nq, np.int64)
    for qi in range(nq):
        sd = set(int(s) for s in S["z"]["seeds"][qi] if s >= 0)
        seedg[qi] = len(sd & set(int(x) for x in gr[qi]))
    ncand = np.zeros(nq, np.int64); np.add.at(ncand, D["qidx"], 1)
    gcand = np.zeros(nq, np.int64); np.add.at(gcand, D["qidx"], D["gold"])

    OUT = {"ds": ds, "nq": nq, "rows": nrow, "signals": SIGNALS, "blocks": [b for b, _ in blocks],
           "STEP2": {}, "STEP4": {}, "STEP5": {}, "REF": {}}
    for nm, mk in blocks:
        gt = int(D["ngold"][mk].sum())
        OUT["REF"][nm] = {
            "queries": int(mk.sum()), "gold_total": gt,
            "candidates_per_query": round(float(ncand[mk].mean()), 1),
            "gold_in_candidates": int(gcand[mk].sum()),
            "gold_in_candidates_frac": round(float(gcand[mk].sum() / max(1, gt)), 4),
            "gold_in_scope_frac": round(float((gcand[mk].sum() + seedg[mk].sum()) / max(1, gt)), 4)}

    for sg in SIGNALS:
        pos = ranks_of(D[sg], SS.SIGN[sg], D["qidx"], D["qptr"], nrow)
        OUT["STEP2"][sg] = {}
        for nm, mk in blocks:
            m = metrics(pos, D, nq, mk, int(D["ngold"][mk].sum()))
            m["read_efficiency"] = round(m["recall@64"] / max(1e-9,
                                         OUT["REF"][nm]["gold_in_scope_frac"]), 4)
            OUT["STEP2"][sg][nm] = m
        a = OUT["STEP2"][sg]["ALL"]
        log(f"  {sg:28s} R@64 {a['recall@64']:.4f}  cond@64 {a['cond@64']:.4f}  "
            f"pct {a['gold_pctrank']:.4f}  auc {a['auc']:.4f}  eff {a['read_efficiency']:.4f}")

    # ---- STEP 4: does the best signal change with NODE hop (the inference-safe conditioner)? ----
    nh = D["HOP"].astype(int)
    for h in sorted(set(nh.tolist())):
        hm = nh == h
        OUT["STEP4"][f"node_hop{h}"] = {"rows": int(hm.sum()),
                                        "gold_rows": int(D["gold"][hm].sum()), "signals": {}}
        for sg in SIGNALS:
            pos = ranks_of(D[sg], SS.SIGN[sg], D["qidx"], D["qptr"], nrow)
            g = D["gold"].astype(bool) & hm
            den = np.maximum((D["qptr"][1:] - D["qptr"][:-1]).astype(np.float64)[D["qidx"]] - 1, 1)
            OUT["STEP4"][f"node_hop{h}"]["signals"][sg] = {
                "gold_pctrank": round(float((pos[g] / den[g]).mean()), 4) if g.any() else None,
                "gold_in_top64": int((g & (pos < 64)).sum()),
                "nongold_in_top64": int((~D["gold"].astype(bool) & hm & (pos < 64)).sum())}

    # ---- STEP 5: support mechanism audit -- gold vs non-gold, effect size ------------------------
    g = D["gold"].astype(bool)
    for sg in SIGNALS:
        a, b = D[sg][g], D[sg][~g]
        a = a[np.isfinite(a) & (a > SS.NEG / 2)]; b = b[np.isfinite(b) & (b > SS.NEG / 2)]
        if not len(a) or not len(b):
            continue
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / max(1, len(a) + len(b) - 2))
        OUT["STEP5"][sg] = {"gold_mean": round(float(a.mean()), 4),
                            "nongold_mean": round(float(b.mean()), 4),
                            "gold_median": round(float(np.median(a)), 4),
                            "nongold_median": round(float(np.median(b)), 4),
                            "cohen_d": round(float((a.mean() - b.mean()) / max(1e-12, sd)), 4),
                            "defined_gold": int(len(a)), "defined_nongold": int(len(b))}
    fp = f"{SS.SSD}/diag/sep_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["metaqa"]))
