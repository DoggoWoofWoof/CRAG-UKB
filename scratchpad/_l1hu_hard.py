"""PHASE A7-A10 -- replay the frozen L1 contract on each hypergraph core and gate it.

A8 says BASE FIRST, and it is the right instruction: the previous programs found that partition
gains originate in the Dense+SPLADE partition RANKING, not in the frozen selector on top of it.
So every cell here reports three numbers, in this order:

  BASE   Dense+SPLADE partition ranking at P=50, no selector          <- where the effect lives
  SAFE   the frozen B6_S4_F6_Ms64_Mr32 selector, mechanically reaggregated, nothing retuned
  CORR   SAFE - BASE, the selector's own correction

Significance is exact two-sided McNemar on the paired per-query indicator, and a delta must also
exceed the measured reseed noise floor for that corpus.  Those floors were measured on METIS
reseeds; Mt-KaHyPar at a fixed seed is deterministic, so using them here is conservative in the
only direction that matters -- it makes claiming an effect harder, never easier.

A partition is INELIGIBLE regardless of accuracy if it leaves the production balance envelope
(max block / mean block <= 1.05, every block used, same node universe), per A7.

  python scratchpad/_l1hu_hard.py pull
  python scratchpad/_l1hu_hard.py scoreall
  python scratchpad/_l1hu_hard.py score <ds> <tag ...>
  python scratchpad/_l1hu_hard.py gate
"""
import os, sys, json, math, subprocess, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_pu as PU
import _l1ep_c as EC
import _l1kn_sub as KS

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL"
PARTS = "scratchpad/_l1hu/parts"
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "webqsp", "hotpotqa_clean"]
FLOOR = {"metaqa": 0.0021, "2wiki_clean": 0.0025, "musique_clean": 0.0029,
         "squad_clean": 0.0013, "hotpotqa_clean": 0.0025, "webqsp": 0.0077}
MAX_OVER_MEAN = 1.05
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def mcnemar(a, b):
    """exact two-sided McNemar on paired per-query indicators."""
    a, b = np.asarray(a, np.int8), np.asarray(b, np.int8)
    g = int(((b == 1) & (a == 0)).sum())
    l = int(((b == 0) & (a == 1)).sum())
    n = g + l
    if n == 0:
        return g, l, 1.0
    k = min(g, l)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return g, l, float("%.3g" % p)


def _env():
    return dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "swathihrao28"),
                PYTHONUTF8="1", MSYS_NO_PATHCONV="1")


def _modal(*a):
    return subprocess.run([sys.executable, "-m", "modal", *a], capture_output=True, text=True,
                          env=_env())


def pull():
    """One directory-level get; per-file gets cost a CLI start-up each and do not scale."""
    os.makedirs(PARTS, exist_ok=True)
    tmp = "%s/_dl" % PARTS
    _modal("volume", "get", "crag-partition", "hparts", tmp, "--force")
    got = []
    for root, _, files in os.walk(tmp):
        for f in files:
            if not f.endswith(".npy"):
                continue
            bits = f[:-4].split("__")
            if len(bits) < 2:
                log("  skip unparseable %s" % f)
                continue
            os.replace(os.path.join(root, f), "%s/%s" % (PARTS, f))
            got.append([bits[0], "__".join(bits[1:])])
    subprocess.run(["rm", "-rf", tmp]) if os.path.isdir(tmp) else None
    log("pulled %d hypergraph assignments" % len(got))
    os.makedirs("%s/partitions" % OUT, exist_ok=True)
    json.dump(sorted(got), open("%s/partitions/PULLED.json" % OUT, "w"), indent=1)
    return got


def cut_stats(ds, hard):
    N, S, K, X = KS.keysets(ds, lambda *a: None)
    C = np.unique(np.concatenate([S, X, K]))
    u = (C // np.int64(N)).astype(np.int64)
    v = (C % np.int64(N)).astype(np.int64)
    cross = hard[u] != hard[v]
    bnd = np.zeros(len(hard), bool)
    bnd[u[cross]] = True
    bnd[v[cross]] = True
    return {"edges": int(len(C)), "edge_cut": int(cross.sum()),
            "edge_cut_fraction": round(float(cross.mean()), 4),
            "boundary_fraction": round(float(bnd.mean()), 4)}


def score(ds, tags):
    os.makedirs("%s/partitions" % OUT, exist_ok=True)
    fp = "%s/partitions/REPLAY_%s.json" % (OUT, ds)
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for tag in tags:
        if tag in rec:
            log("  %s %s cached" % (ds, tag))
            continue
        if tag == "CURRENT":
            hard, npart = PU.load_assignment(ds, "CURRENT")
        else:
            p = "%s/%s__%s.npy" % (PARTS, ds, tag)
            if not os.path.exists(p):
                log("  %s %s MISSING" % (ds, tag))
                continue
            hard = np.load(p)
            npart = int(np.asarray(hard).max()) + 1
        hard = np.asarray(hard, np.int64)
        ref_hard, ref_npart = PU.load_assignment(ds, "CURRENT")
        same_universe = bool(len(hard) == len(np.asarray(ref_hard)) and npart == int(ref_npart))
        sizes = np.bincount(hard, minlength=npart).astype(np.int64)
        t0 = time.time()
        R = EC.replay(ds, hard, npart, tag, log)
        mom = round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4)
        R["BALANCE"] = {"npart": int(npart), "blocks_used": int((sizes > 0).sum()),
                        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                        "size_median": float(np.median(sizes[sizes > 0])),
                        "size_mean": round(float(sizes.mean()), 2),
                        "size_p90": float(np.percentile(sizes[sizes > 0], 90)),
                        "size_p99": float(np.percentile(sizes[sizes > 0], 99)),
                        "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4),
                        "max_over_mean": mom, "same_node_universe": same_universe,
                        "ELIGIBLE": bool(mom <= MAX_OVER_MEAN and bool((sizes > 0).all())
                                         and same_universe)}
        R["CUT"] = cut_stats(ds, hard)
        R["seconds"] = round(time.time() - t0, 1)
        rec[tag] = R
        json.dump(rec, open(fp, "w"), indent=1)
        log("  %-16s %-28s BASE %.4f  SAFE %.4f  corr %+.4f  max/mean %.3f  cut %.4f"
            % (ds, tag, R["BASE_ALL_P50"], R["F6_ALL_P50"],
               R["F6_ALL_P50"] - R["BASE_ALL_P50"], mom, R["CUT"]["edge_cut_fraction"]))
    return rec


def gate():
    cells, have = {}, [d for d in DS
                       if os.path.exists("%s/partitions/REPLAY_%s.json" % (OUT, d))]
    for ds in have:
        rec = json.load(open("%s/partitions/REPLAY_%s.json" % (OUT, ds)))
        ref = rec.get("CURRENT")
        if not ref:
            continue
        fl = FLOOR.get(ds, 0.003)
        for tag, R in rec.items():
            if tag == "CURRENT":
                continue
            row = {"noise_floor": fl, "max_over_mean": R["BALANCE"]["max_over_mean"],
                   "ELIGIBLE": R["BALANCE"]["ELIGIBLE"],
                   "edge_cut_fraction": R["CUT"]["edge_cut_fraction"],
                   "boundary_fraction": R["CUT"]["boundary_fraction"]}
            for lab, key, ind in (("BASE", "BASE_ALL_P50", "_ind_BASE"),
                                  ("SAFE", "F6_ALL_P50", "_ind_F6")):
                g, l, p = mcnemar(ref[ind], R[ind])
                d = round(R[key] - ref[key], 4)
                row[lab] = {"value": R[key], "ref": ref[key], "delta": d, "gained": g,
                            "lost": l, "p": p, "sig": bool(p < 0.05 and abs(d) > fl)}
            row["CORR"] = round(R["F6_ALL_P50"] - R["BASE_ALL_P50"], 4)
            row["CORR_ref"] = round(ref["F6_ALL_P50"] - ref["BASE_ALL_P50"], 4)
            if "by_hop" in R:
                row["by_hop"] = {h: {"BASE": v["BASE_ALL"], "SAFE": v["F6_ALL"],
                                     "BASE_delta": round(v["BASE_ALL"]
                                                         - ref["by_hop"][h]["BASE_ALL"], 4),
                                     "SAFE_delta": round(v["F6_ALL"]
                                                         - ref["by_hop"][h]["F6_ALL"], 4)}
                                 for h, v in R["by_hop"].items()}
            cells.setdefault(tag, {})[ds] = row

    tab = {}
    for tag, per in cells.items():
        ds_list = sorted(per)
        s = {"corpora": ds_list, "n_corpora": len(ds_list),
             "ALL_ELIGIBLE": all(per[d]["ELIGIBLE"] for d in ds_list),
             "worst_max_over_mean": round(max(per[d]["max_over_mean"] for d in ds_list), 4)}
        for lab in ("BASE", "SAFE"):
            dl = [per[d][lab]["delta"] for d in ds_list]
            regs = [d for d in ds_list if per[d][lab]["sig"] and per[d][lab]["delta"] < 0]
            s[lab] = {"WORST_DELTA": round(min(dl), 4),
                      "MACRO_DELTA": round(float(np.mean(dl)), 4),
                      "SIG_REGRESSIONS": len(regs), "regressed_on": regs,
                      "n_sig_gains": sum(1 for d in ds_list
                                         if per[d][lab]["sig"] and per[d][lab]["delta"] > 0),
                      "per_corpus": {d: per[d][lab]["delta"] for d in ds_list}}
        bh = (per.get("metaqa") or {}).get("by_hop") or {}
        s["METAQA_H2_SAFE"] = (bh.get("hop2") or {}).get("SAFE")
        s["METAQA_H3_SAFE"] = (bh.get("hop3") or {}).get("SAFE")
        s["METAQA_H3_BASE"] = (bh.get("hop3") or {}).get("BASE")
        tab[tag] = s

    full = [t for t in tab if tab[t]["n_corpora"] == len(have)]
    order = sorted(tab, key=lambda t: (tab[t]["SAFE"]["SIG_REGRESSIONS"],
                                       -tab[t]["SAFE"]["WORST_DELTA"],
                                       -tab[t]["SAFE"]["MACRO_DELTA"],
                                       -(tab[t]["METAQA_H3_SAFE"] or 0)))
    passed = [t for t in order if t in full and tab[t]["ALL_ELIGIBLE"]
              and tab[t]["SAFE"]["SIG_REGRESSIONS"] == 0
              and tab[t]["BASE"]["SIG_REGRESSIONS"] == 0
              and tab[t]["SAFE"]["n_sig_gains"] > 0]
    res = {"CORPORA": have, "MAX_OVER_MEAN_ENVELOPE": MAX_OVER_MEAN, "CELLS": cells,
           "SUMMARY": tab, "ORDER": order, "UNIVERSAL_HYPERGRAPH_GATE_PASSED": passed,
           "HYPER_CROSS_CORPUS_SAFE": bool(passed),
           "BEST_UNIVERSAL_HYPERGRAPH_CORE": passed[0] if passed else None,
           "BEST_DIAGNOSTIC_HYPERGRAPH_CORE": (order[0] if order else None)}
    os.makedirs("%s/partitions" % OUT, exist_ok=True)
    json.dump(res, open("%s/partitions/HYPER_GATE.json" % OUT, "w"), indent=1)

    hdr = ("%-24s %8s %8s %6s %8s %8s %6s %5s %7s %7s %6s"
           % ("REPRESENTATION", "bWORST", "bMACRO", "bREG", "sWORST", "sMACRO", "sREG",
              "GAIN", "H3base", "H3safe", "bal"))
    print("corpora scored: %s" % have)
    print("  b = BASE (Dense+SPLADE partition ranking, P=50)   s = SAFE (frozen selector)")
    print(hdr)
    print("-" * len(hdr))
    for t in order:
        v = tab[t]
        print("%-24s %+8.4f %+8.4f %6d %+8.4f %+8.4f %6d %5d %7.4f %7.4f %6s"
              % (t, v["BASE"]["WORST_DELTA"], v["BASE"]["MACRO_DELTA"],
                 v["BASE"]["SIG_REGRESSIONS"], v["SAFE"]["WORST_DELTA"],
                 v["SAFE"]["MACRO_DELTA"], v["SAFE"]["SIG_REGRESSIONS"],
                 v["SAFE"]["n_sig_gains"], v["METAQA_H3_BASE"] or 0,
                 v["METAQA_H3_SAFE"] or 0, "ok" if v["ALL_ELIGIBLE"] else "OVER"))
    print("\nUNIVERSAL_HYPERGRAPH_GATE_PASSED = %s" % passed)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if cmd == "pull":
        pull()
    elif cmd == "scoreall":
        byds = collections.defaultdict(list)
        for f in sorted(os.listdir(PARTS)) if os.path.isdir(PARTS) else []:
            if not f.endswith(".npy"):
                continue
            bits = f[:-4].split("__")
            byds[bits[0]].append("__".join(bits[1:]))
        for ds in DS:
            if byds.get(ds):
                score(ds, ["CURRENT"] + sorted(byds[ds]))
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3:])
    else:
        gate()
