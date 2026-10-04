"""FINAL ANALYSIS -- paired-vs-SAFE-BASELINE significance, STEP 13 headroom, STEP 14 hop3 decomposition.

Every scoreboard number so far is paired against BASE (the un-routed P50).  The promotion
decision is a different question: does a candidate beat the SAFE UNIVERSAL BASELINE
B6_S4_F6_Ms64_Mr32?  So this recomputes per-query indicators for a shortlist and pairs them
against F6 directly, per corpus and per MetaQA hop.

STEP 13  RECOVERED_HOP_HEADROOM(h) = (NEW_h - BASE_h) / (DIRECT_COMBINED_h - BASE_h)
STEP 14  every hop3 query whose coverage changed is classed RECOVERED / DAMAGED / PARTIAL /
         NEUTRAL, and the still-uncovered hop3 queries are decomposed by failure mode.

  python scratchpad/_l1kb_final.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR
import _l1kb_round3 as R3
import _l1kb_round4 as R4

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

SHORT = [
    ("F6_B6", lambda G: KB.sel_f6, 6),
    ("CANON_DEEP_B6", lambda G: R3.make_dsel("CANON", "DEEP", None, False, G), 6),
    ("CANON_B4", lambda G: R3.make_dsel("CANON", "ALWAYS", None, False, G), 4),
    ("MINRANK_DEEP_B6", lambda G: R3.make_dsel("MINRANK", "DEEP", None, False, G), 6),
    ("MINRANK_SEED_B6", lambda G: R3.make_dsel("MINRANK", "SEED", None, False, G), 6),
    ("CANON_SEED_B6", lambda G: R3.make_dsel("CANON", "SEED", None, False, G), 6),
    ("ALT_STRLEAD_SEED_B6", lambda G: R4.make_alt("ALT", "STR", False, "SEED", None, G), 6),
    ("ALT_STRLEAD_DEEP_B6", lambda G: R4.make_alt("ALT", "STR", False, "DEEP", None, G), 6),
    ("ES_B6", lambda G: JR.make_selector("ES", None, G, "S4"), 6),
    ("J1_B6", lambda G: JR.make_selector("J1", None, G, "S4"), 6),
]


def main():
    DIAG = json.load(open(f"{KB.KBD}/diag/hop_diagnosis.json"))["RESULTS"]
    OUT = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        hops = z["hops"]; goldp = KB.goldparts(z, meta)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ind_base = C["ind_base"]; fold = JR.folds(ds, z)
        ctxs = {B: KB.contexts(z, meta, C, B) for B in (4, 6)}
        runs = {}
        for nm, mk, B in SHORT:
            runs[nm] = KB.run_selector(ctxs[B], mk(GRP), goldp, ind_base)
        f6 = runs["F6_B6"]
        o = {"n": len(ind_base), "BASE_ALL": round(float(ind_base.mean()), 4), "configs": {}}
        for nm, _, B in SHORT:
            r = runs[nm]; ind = r["ind"]
            e = {"ALL": round(float(ind.mean()), 4),
                 "dALL_vs_BASE": round(float(ind.mean() - ind_base.mean()), 4),
                 "dALL_vs_F6": round(float(ind.mean() - f6["ind"].mean()), 4),
                 "vs_F6": RT.mcnemar(ind, f6["ind"]),
                 "vs_BASE": RT.mcnemar(ind, ind_base),
                 "churn_mean": round(float(r["churn"].mean()), 3),
                 "abstain_frac": round(float((r["churn"] == 0).mean()), 4),
                 "churn_hist": {str(k): int(v) for k, v in
                                zip(*np.unique(r["churn"], return_counts=True))},
                 "gold_admitted": r["gold_admitted"], "gold_evicted": r["gold_evicted"]}
            if (hops >= 0).any():
                e["per_hop"] = {}
                for h in sorted(set(int(x) for x in hops if x >= 0)):
                    m = hops == h
                    bh = float(ind_base[m].mean())
                    dc = DIAG[ds]["metaqa_hop_table"]["direct_COMBINED_32_32"][str(h)] \
                        if "metaqa_hop_table" in DIAG.get(ds, {}) else None
                    ee = {"n": int(m.sum()), "BASE": round(bh, 4),
                          "ALL": round(float(ind[m].mean()), 4),
                          "dALL_vs_BASE": round(float(ind[m].mean() - bh), 4),
                          "dALL_vs_F6": round(float(ind[m].mean() - f6["ind"][m].mean()), 4),
                          "vs_F6": RT.mcnemar(ind[m], f6["ind"][m]),
                          "vs_BASE": RT.mcnemar(ind[m], ind_base[m])}
                    if dc is not None and dc > bh:
                        ee["RECOVERED_HOP_HEADROOM"] = round((float(ind[m].mean()) - bh)
                                                            / (dc - bh), 4)
                        ee["DIRECT_COMBINED"] = dc
                    e["per_hop"][str(h)] = ee
            o["configs"][nm] = e
        OUT[ds] = o
        log("%-15s " % ds + "  ".join(
            "%s %+.4f(p%.3f)" % (nm, o["configs"][nm]["dALL_vs_F6"],
                                 o["configs"][nm]["vs_F6"]["mcnemar_p"])
            for nm, _, _ in SHORT if nm != "F6_B6"))

    # ---------------- STEP 14: MetaQA hop3 outcome classes + residual decomposition
    ds = "metaqa"
    z, meta = KB.load(ds); hops = z["hops"]; goldp = KB.goldparts(z, meta)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    ind_base = C["ind_base"]
    ctxs = {B: KB.contexts(z, meta, C, B) for B in (4, 6)}
    f6r = KB.run_selector(ctxs[6], KB.sel_f6, goldp, ind_base)
    S14 = {}
    for nm, mk, B in SHORT:
        if nm == "F6_B6":
            continue
        r = KB.run_selector(ctxs[B], mk(GRP), goldp, ind_base)
        cls = {"RECOVERED": 0, "DAMAGED": 0, "PARTIAL": 0, "NEUTRAL": 0}
        resid = {"CAPACITY": 0, "REACH": 0, "REACH_AT_ROUTER_M": 0, "EVICTION": 0,
                 "CO_SELECT": 0, "RANKING": 0}
        for qi in range(len(ind_base)):
            if hops[qi] != 3:
                continue
            new, old = r["finals"][qi], f6r["finals"][qi]
            if new == old:
                continue
            gn, go = len(goldp[qi] & new), len(goldp[qi] & old)
            an, ao = int(goldp[qi] <= new), int(goldp[qi] <= old)
            if an > ao:
                cls["RECOVERED"] += 1
            elif an < ao:
                cls["DAMAGED"] += 1
            elif gn != go:
                cls["PARTIAL"] += 1
            else:
                cls["NEUTRAL"] += 1
        c6 = ctxs[B]
        for qi in range(len(ind_base)):
            if hops[qi] != 3 or goldp[qi] <= r["finals"][qi]:
                continue
            c = c6[qi]
            nd = goldp[qi] - c["prot_set"]
            pool = set(c["bnd"]) | set(c["chal"])
            if len(nd) > B:
                resid["CAPACITY"] += 1
            elif not nd <= pool:
                resid["REACH"] += 1
            elif goldp[qi] & (c["base50"] - r["finals"][qi]):
                resid["EVICTION"] += 1
            else:
                got = goldp[qi] & r["finals"][qi]
                resid["CO_SELECT" if 0 < len(got) < len(nd) else "RANKING"] += 1
        S14[nm] = {"hop3_changed_classes": cls, "hop3_still_uncovered": resid}
        log("hop3 %-22s R %3d D %3d P %3d N %4d | residual %s"
            % (nm, cls["RECOVERED"], cls["DAMAGED"], cls["PARTIAL"], cls["NEUTRAL"],
               " ".join("%s:%d" % (k, v) for k, v in resid.items() if v)))
    OUT["_STEP14_metaqa_hop3"] = S14
    fp = f"{KB.KBD}/final_analysis.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote", fp)


if __name__ == "__main__":
    main()
