"""ROUND 5 -- IS THERE A QUERY-LOCAL SIGNAL FOR MODALITY ALLOCATION?  (STEP 6; message-1 STEP 2/3)

OBSERVED FAILURE (round 4):
  Round 3 de-duplication maximises the KB axis (MetaQA hop3 +0.0135, hop2 +0.0255) and costs
  HotpotQA -0.0175.  Round 4's modality-balanced slot split protects text (ALT_STRLEAD_SEED_B6:
  hotpot +0.0165 > F6 +0.0160, 2wiki/squad exactly F6) but only reaches hop3 +0.0030.  The two
  corpora want OPPOSITE allocations of the same six slots -- MetaQA wants essentially all of
  them structural, HotpotQA wants them lexical -- so any FIXED split, at any ratio, lands
  between the two regimes and gets neither.  A fixed rule is therefore exhausted.

WHAT THIS ROUND TESTS -- and it is a test, not a family:
  Does a QUERY-LOCAL, label-free, parameter-free signal separate the queries where dropping the
  redundant lexical vote HELPS from the ones where it HURTS?  If such a signal exists it must
  separate WITHIN a corpus, not merely between corpora: a feature that only tells MetaQA from
  HotpotQA is dataset identity wearing a disguise, which the contract forbids.  So every
  candidate is scored by its within-corpus separation, pooled separation is reported only as a
  contrast, and the whole thing is repeated on the DISCOVERY fold and the VALIDATION fold.

  Gold labels the OUTCOME of a swap.  It is never an input to any feature.

CANDIDATE QUERY-LOCAL FEATURES (all rank/count/percentile, no corpus statistics, no thresholds)
  f_concord     fraction of concordant (canonical, retrieval) ordered pairs among out-of-P50
                partitions both channels rank -- how far the two lexical granularities agree
  f_ret_redund  fraction of retrieval nominations the canonical channel already ranks
  f_jaccard     |S n R| / |S u R| over the two channels' out-of-P50 nominations
  f_deep        count of structure-only challengers reached at traversal hop >= 2
  f_deep_frac   that count as a fraction of all structural nominations
  f_seed        distinct retrieval seed slots supporting the best structural chain
  f_topgap      rank of the best structural challenger inside the merged candidate order
  f_ret_top_c   canonical rank of the top retrieval challenger, minus the worst incumbent's
  f_bnd_struct  how many of the B incumbents the structural channel also supports

LABEL (analysis only): for each query, compare the frozen F6 selection with the CANON de-dup
  selection under the SAME B and record GOOD / BAD / NEUTRAL by whether ALL-gold coverage
  improved, degraded, or was unchanged.

  python scratchpad/_l1kb_round5.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR
import _l1kb_round3 as R3

INF = JR.INF
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
FEATS = ["f_concord", "f_ret_redund", "f_jaccard", "f_deep", "f_deep_frac", "f_seed",
         "f_topgap", "f_ret_top_c", "f_bnd_struct"]


def features(c, qi, GRP):
    cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
    b50 = c["base50"]; bnd = c["bnd"]
    S = [p for p in spos if p not in b50]
    R = [p for p in rpos if p not in b50]
    Ss, Rs = set(S), set(R)
    both = [p for p in R if p in cpos]
    conc = disc = 0
    for i in range(len(both)):
        for j in range(i + 1, len(both)):
            a, b = both[i], both[j]
            if (cpos[a] < cpos[b]) == (rpos[a] < rpos[b]):
                conc += 1
            else:
                disc += 1
    uni = len(Ss | Rs)
    deep = R3.deep_struct_only(c)
    seed = 0
    for g in GRP[qi]:
        if any(p not in b50 for p in g["parts"]):
            seed = max(seed, len(g["seedslots"]))
    cands = bnd + [p for p in dict.fromkeys(c["chal"]) if p not in bnd]
    tot = {}
    for p in cands:
        tot[p] = (1.0 / (KB.K0 + cpos[p]) if p in cpos else 0.0) \
               + (1.0 / (KB.K0 + rpos[p]) if p in rpos else 0.0) \
               + (1.0 / (KB.K0 + spos[p]) if p in spos else 0.0)
    order = sorted(cands, key=lambda p: (-tot[p], cpos.get(p, INF), p))
    topgap = next((i for i, p in enumerate(order) if p in Ss), len(order))
    rtc = min((cpos.get(p, INF) for p in R), default=INF)
    worst_inc = max(cpos.get(p, 0) for p in bnd) if bnd else 0
    return {"f_concord": conc / max(1, conc + disc),
            "f_ret_redund": sum(1 for p in R if p in cpos) / max(1, len(R)),
            "f_jaccard": len(Ss & Rs) / max(1, uni),
            "f_deep": float(len(deep)),
            "f_deep_frac": len(deep) / max(1, len(S)),
            "f_seed": float(seed),
            "f_topgap": float(topgap),
            "f_ret_top_c": float(min(rtc - worst_inc, 1000)),
            "f_bnd_struct": float(sum(1 for p in bnd if p in spos))}


def auc(x, y):
    """rank AUC of feature x for binary label y; 0.5 == no separation.  None if a class is empty."""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = np.empty(len(x))
    o = np.argsort(x, kind="mergesort")
    xs = x[o]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    OUT = {}
    POOL = {f: {"x": [], "y": []} for f in FEATS}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        goldp = KB.goldparts(z, meta)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ind_base = C["ind_base"]; fold = JR.folds(ds, z)
        ctxs = KB.contexts(z, meta, C, 6)
        dsel = R3.make_dsel("CANON", "ALWAYS", None, False, GRP)
        rows = []
        for qi, c in enumerate(ctxs):
            f6 = set(KB.sel_f6(c, qi, None)) | c["prot_set"]
            dd = set(dsel(c, qi, None)) | c["prot_set"]
            a = int(goldp[qi] <= f6); b = int(goldp[qi] <= dd)
            lab = "NEUTRAL" if a == b else ("GOOD" if b > a else "BAD")
            if f6 == dd:
                lab = "SAME"
            # higher-powered label: signed change in the number of GOLD partitions retained.
            # ALL-coverage flips on only ~1-2% of queries, which cannot support a separability
            # claim either way; the gold-count delta moves on an order of magnitude more.
            dg = len(goldp[qi] & dd) - len(goldp[qi] & f6)
            r = features(c, qi, GRP); r["label"] = lab; r["fold"] = int(fold[qi])
            r["dgold"] = int(dg)
            rows.append(r)
        chg = [r for r in rows if r["label"] in ("GOOD", "BAD")]
        n_same = sum(1 for r in rows if r["label"] == "SAME")
        o = {"n": len(rows), "n_same_selection": n_same,
             "n_changed_selection": len(rows) - n_same,
             "GOOD": sum(1 for r in rows if r["label"] == "GOOD"),
             "BAD": sum(1 for r in rows if r["label"] == "BAD"),
             "NEUTRAL": sum(1 for r in rows if r["label"] == "NEUTRAL"), "AUC": {}}
        gch = [r for r in rows if r["dgold"] != 0]
        o["GOLD_UP"] = sum(1 for r in gch if r["dgold"] > 0)
        o["GOLD_DOWN"] = sum(1 for r in gch if r["dgold"] < 0)
        o["AUC_GOLD"] = {}
        for f in FEATS:
            x = [r[f] for r in chg]; y = [r["label"] == "GOOD" for r in chg]
            o["AUC"][f] = {"all": auc(x, y)}
            for nm, fv in (("DISCOVERY", 0), ("VALIDATION", 1)):
                xs = [r[f] for r in chg if r["fold"] == fv]
                ys = [r["label"] == "GOOD" for r in chg if r["fold"] == fv]
                o["AUC"][f][nm] = auc(xs, ys)
            gx = [r[f] for r in gch]; gy = [r["dgold"] > 0 for r in gch]
            o["AUC_GOLD"][f] = {"all": auc(gx, gy)}
            for nm, fv in (("DISCOVERY", 0), ("VALIDATION", 1)):
                xs = [r[f] for r in gch if r["fold"] == fv]
                ys = [r["dgold"] > 0 for r in gch if r["fold"] == fv]
                o["AUC_GOLD"][f][nm] = auc(xs, ys)
            POOL[f]["x"] += gx; POOL[f]["y"] += gy
        OUT[ds] = o
        log("%-15s ALLlabel GOOD %3d BAD %3d | GOLDlabel up %4d down %4d  " % (
            ds, o["GOOD"], o["BAD"], o["GOLD_UP"], o["GOLD_DOWN"])
            + " ".join("%s %s" % (f.replace("f_", ""),
                                  "  n/a" if o["AUC_GOLD"][f]["all"] is None
                                  else "%.3f" % o["AUC_GOLD"][f]["all"]) for f in FEATS))
    OUT["_POOLED"] = {"AUC_GOLD": {f: auc(POOL[f]["x"], POOL[f]["y"]) for f in FEATS},
                      "n": len(POOL[FEATS[0]]["x"])}
    log("POOLED n=%d (contrast only)  " % OUT["_POOLED"]["n"] + " ".join(
        "%s %s" % (f.replace("f_", ""), "n/a" if OUT["_POOLED"]["AUC_GOLD"][f] is None
                   else "%.3f" % OUT["_POOLED"]["AUC_GOLD"][f]) for f in FEATS))
    # a feature is only usable if it separates INSIDE a corpus and does so on BOTH folds with
    # the SAME sign.  Anything else is dataset identity or fold noise.
    OUT["_WITHIN"] = {}
    for f in FEATS:
        v = [OUT[d]["AUC_GOLD"][f] for d in KB.DSETS if OUT[d]["AUC_GOLD"][f]["all"] is not None]
        if not v:
            continue
        sg = [np.sign(x["all"] - 0.5) for x in v]
        st = [x for x in v if x["DISCOVERY"] is not None and x["VALIDATION"] is not None
              and np.sign(x["DISCOVERY"] - 0.5) == np.sign(x["VALIDATION"] - 0.5) != 0]
        OUT["_WITHIN"][f] = {"mean_abs_auc_minus_half": round(
            float(np.mean([abs(x["all"] - 0.5) for x in v])), 4),
            "n_corpora": len(v), "sign_agreement": int(abs(sum(sg))),
            "n_corpora_fold_stable": len(st)}
    log("within-corpus: " + " ".join(
        "%s(|d|%.3f sign%d/%d fold%d)" % (f.replace("f_", ""),
                                          OUT["_WITHIN"][f]["mean_abs_auc_minus_half"],
                                          OUT["_WITHIN"][f]["sign_agreement"],
                                          OUT["_WITHIN"][f]["n_corpora"],
                                          OUT["_WITHIN"][f]["n_corpora_fold_stable"])
        for f in FEATS if f in OUT["_WITHIN"]))
    os.makedirs(f"{KB.KBD}/diag", exist_ok=True)
    fp = f"{KB.KBD}/diag/query_local_separability.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote", fp)


if __name__ == "__main__":
    main()
