"""PHASE 3 -- dismantle SAFE (the F6 partition-selector) into STRUCT-branch-only (F7) vs
retrieval-continuation-only (F8) vs neither (BASE/S0) vs both (F6/S1, the shipped SAFE).

This tests SAFE at the SELECTOR level (does the chosen 50-partition set cover all gold
PARTITIONS), the level _l1ps_router.evaluate/_l1kb_core.py were built for -- upstream of and
distinct from the L1-hypergraph-halo ALL_REQUIRED_FETCHED metric used in Phase 1/2/4 (which
tests the DOWNSTREAM effect of choosing SAFE's 50 partitions as the halo's candidate-generating
cores). F7/F8 are pre-existing, already-validated code paths in _l1ps_router.evaluate (use_s/
use_r flags) -- no new selector code, only a new comparison driver.

Per the spec's "no B/M sweeps" instruction, B/M_struct/M_ret/agg are held fixed at the shipped
BASE_CFG (B=6, M_struct=64, M_ret=32, agg=S4) throughout -- only the fusion family (which
channels compete) varies. The finer sub-decomposition (s_dir/max-collapse/S4-vs-simpler-agg/
M_struct-truncation/M_ret/B6/F6-boundary) is explicitly DEPRIORITIZED relative to Phases 4-11
(the audit's actual title question, edge-FAMILY, not selector-internals) -- flagged, not silently
dropped, in the Phase 3 report's SCOPE_NOTE.

  python scratchpad/_l1au_p3.py run <ds>
  python scratchpad/_l1au_p3.py run_all
  python scratchpad/_l1au_p3.py report
"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1kb_core as KB
import _l1ps_router as RT

KBD = KB.KBD
AOUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = RT.DSETS
SCOPE_NOTE = ("Finer sub-decomposition (s_dir / max-collapse / S4-vs-simpler-agg / "
             "M_struct-truncation / M_ret / B6 / F6-boundary) is deprioritized relative to "
             "Phases 4-11 (the audit's actual EDGE-FAMILY question) -- would require new "
             "aggregation-mode code and the spec explicitly rules out B/M parameter sweeps. "
             "Not run; flagged here rather than silently dropped.")


def run(ds, log=log):
    z, meta = KB.load(ds)
    fp = "%s/ctx/sub_%s_Ms64_Mr32_S4.pkl" % (KBD, ds)
    with open(fp, "rb") as fh:
        o = pickle.load(fh)
    C = o["C"]
    base_all = C["BASE_ALL"]
    cells = {"S0_BASE": {"ALL": round(base_all, 4)}}
    for fam, key in (("F6", "S1_F6_BOTH"), ("F7", "S2_STRUCT_ONLY"), ("F8", "S3_RET_ONLY")):
        cfg = dict(KB.BASE_CFG); cfg["fusion"] = fam
        r = RT.evaluate(ds, z, meta, cfg, C)
        cells[key] = {"ALL": r["ALL"], "delta_vs_BASE": round(r["ALL"] - base_all, 4),
                     "gained_vs_BASE": r["newly_covered"], "lost_vs_BASE": r["newly_uncovered"],
                     "p_vs_BASE": r["mcnemar_p"], "sig_vs_BASE": r["sig"],
                     "churn_per_query": r["churn_per_query"],
                     "decisive_channel_for_newly_covered": r["decisive_channel_for_newly_covered"]}
        log("  %s[%s]: ALL=%.4f delta_vs_BASE=%+.4f p=%.4g sig=%s" % (
            ds, key, r["ALL"], r["ALL"] - base_all, r["mcnemar_p"], r["sig"]))
    f6 = cells["S1_F6_BOTH"]["ALL"]; f7 = cells["S2_STRUCT_ONLY"]["ALL"]; f8 = cells["S3_RET_ONLY"]["ALL"]
    ind_f7 = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG, fusion="F7"), C)["_ind"]
    ind_f8 = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG, fusion="F8"), C)["_ind"]
    ind_f6 = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG, fusion="F6"), C)["_ind"]
    m_78 = RT.mcnemar(ind_f7, ind_f8)
    m_76 = RT.mcnemar(ind_f7, ind_f6)
    m_86 = RT.mcnemar(ind_f8, ind_f6)
    cells["STRUCT_vs_RET"] = {"F7_minus_F8": round(f7 - f8, 4), "p": m_78["mcnemar_p"], "sig": m_78["sig"]}
    cells["STRUCT_vs_F6"] = {"F7_minus_F6": round(f7 - f6, 4), "p": m_76["mcnemar_p"], "sig": m_76["sig"]}
    cells["RET_vs_F6"] = {"F8_minus_F6": round(f8 - f6, 4), "p": m_86["mcnemar_p"], "sig": m_86["sig"]}
    rec = {"ds": ds, "CELLS": cells, "SCOPE_NOTE": SCOPE_NOTE}
    os.makedirs("%s/safe_dismantle" % AOUT, exist_ok=True)
    fpo = "%s/safe_dismantle/P3_%s.json" % (AOUT, ds)
    json.dump(rec, open(fpo, "w"), indent=1)
    log("wrote", fpo)
    return rec


def run_all():
    for ds in DS:
        run(ds)


def report():
    rows = []
    for ds in DS:
        fp = "%s/safe_dismantle/P3_%s.json" % (AOUT, ds)
        if os.path.exists(fp):
            rows.append(json.load(open(fp)))
    os.makedirs(AOUT, exist_ok=True)
    json.dump(rows, open("%s/PHASE3_SAFE_DISMANTLE_REPORT.json" % AOUT, "w"), indent=1)
    for r in rows:
        c = r["CELLS"]
        print("\n== %s ==  S0(BASE)=%.4f  S1(F6,both)=%.4f" % (r["ds"], c["S0_BASE"]["ALL"], c["S1_F6_BOTH"]["ALL"]))
        for k in ("S2_STRUCT_ONLY", "S3_RET_ONLY"):
            v = c[k]
            print("  %-16s ALL=%.4f delta_vs_BASE=%+.4f gained=%-3d lost=%-3d p=%-10s sig=%s  churn/q=%.3f" % (
                k, v["ALL"], v["delta_vs_BASE"], v["gained_vs_BASE"], v["lost_vs_BASE"], v["p_vs_BASE"],
                v["sig_vs_BASE"], v["churn_per_query"]))
        print("  STRUCT_vs_RET: F7-F8=%+.4f p=%s sig=%s | STRUCT_vs_F6: F7-F6=%+.4f p=%s sig=%s | "
              "RET_vs_F6: F8-F6=%+.4f p=%s sig=%s" % (
              c["STRUCT_vs_RET"]["F7_minus_F8"], c["STRUCT_vs_RET"]["p"], c["STRUCT_vs_RET"]["sig"],
              c["STRUCT_vs_F6"]["F7_minus_F6"], c["STRUCT_vs_F6"]["p"], c["STRUCT_vs_F6"]["sig"],
              c["RET_vs_F6"]["F8_minus_F6"], c["RET_vs_F6"]["p"], c["RET_vs_F6"]["sig"]))
    print("\nwrote %s/PHASE3_SAFE_DISMANTLE_REPORT.json" % AOUT)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
