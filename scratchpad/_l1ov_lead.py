"""PHASES 14/19 -- family contribution and the universal leaderboard.

Significance is exact and needs no permutation: a halo can only ADD fetched nodes, so every
query that changes does so in one direction.  With `r` queries rescued and 0 lost, the exact
two-sided McNemar p is 2^(1-r).  r >= 6 is significant at 0.05, r >= 8 at 0.01.

The universal rule is applied verbatim and lexicographically:
  1. zero significant cross-corpus regressions   2. max worst-corpus delta
  3. max macro delta   4. max MetaQA hop3   5. min exposure multiplier

  python scratchpad/_l1ov_lead.py
"""
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "webqsp", "hotpotqa_clean"]


def mcnemar_one_way(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    # exact binomial, two-sided
    k = min(gained, lost)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, p)


def mcnemar_pair(a, b):
    """exact two-sided McNemar for two paired binary indicator vectors."""
    gained = int(((b == 1) & (a == 0)).sum())
    lost = int(((b == 0) & (a == 1)).sum())
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0
    k = min(gained, lost)
    return gained, lost, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def boot_ci(x, n=4000, seed=0):
    """paired bootstrap over queries for a difference-in-differences mean."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return round(float(lo), 4), round(float(hi), 4), round(float(min(1.0, p)), 4)


def load():
    cov, mat = {}, {}
    for ds in DS:
        fp = f"{OUT}/overlap/COVERAGE_{ds}.json"
        if os.path.exists(fp):
            cov[ds] = json.load(open(fp)).get("CURRENT", {})
        fp2 = f"{OUT}/exposure/MATCHED_{ds}.json"
        if os.path.exists(fp2):
            mat[ds] = json.load(open(fp2)).get("CURRENT", {}).get("MATCHED", {})
    return cov, mat


def cell_stats(cov, mat):
    """delta vs O0_CORE + exact significance, per (corpus, cell)."""
    rows = {}
    for ds, d in cov.items():
        base = d.get("O0_CORE", {}).get("F6")
        if not base:
            continue
        b_ind = np.asarray(base["_ind_ALL"], np.int8)
        for name, rec in d.items():
            if name.startswith("_") or name == "O0_CORE" or "F6" not in rec:
                continue
            f = rec["F6"]
            ind = np.asarray(f["_ind_ALL"], np.int8)
            gained = int(((ind == 1) & (b_ind == 0)).sum())
            lost = int(((ind == 0) & (b_ind == 1)).sum())
            p = mcnemar_one_way(gained, lost)
            r = {"delta": round(f["ALL_REQUIRED_FETCHED"] - base["ALL_REQUIRED_FETCHED"], 4),
                 "ALL_REQUIRED": f["ALL_REQUIRED_FETCHED"],
                 "REQUIRED_NODE_RECALL": f["REQUIRED_NODE_RECALL"],
                 "gained": gained, "lost": lost, "p": float(f"{p:.3g}"), "sig": bool(p < 0.05),
                 "EXPOSURE_MULTIPLIER": f["EXPOSURE_MULTIPLIER"],
                 "R": rec["REPLICATION_FACTOR"],
                 "MIN_BLOCK_COVER_median": (rec.get("MIN_BLOCK_COVER") or {}).get("median")}
            if "BY_HOP" in rec:
                bh = rec["BY_HOP"]; bb = d["O0_CORE"].get("BY_HOP", {})
                for h in ("hop1", "hop2", "hop3"):
                    if h in bh:
                        r[h] = bh[h]["ALL_REQUIRED_FETCHED"]
                        if h in bb:
                            r[f"{h}_delta"] = round(bh[h]["ALL_REQUIRED_FETCHED"]
                                                    - bb[h]["ALL_REQUIRED_FETCHED"], 4)
            m = (mat.get(ds) or {}).get(name)
            if m:
                # O0_CORE IS hard P=50, yet it scores above the P'=50 control, because the
                # control ranks by base_rank (the BASE lane) while every overlap cell is
                # selected by F6.  That selector gap inflates every raw matched delta, so the
                # cell's own gain is the EXCESS over what O0_CORE already gets for free.
                m0 = (mat.get(ds) or {}).get("O0_CORE")
                if m0:
                    adj = round(m["OVERLAP_MINUS_MATCHED_HARD"]
                                - m0["OVERLAP_MINUS_MATCHED_HARD"], 4)
                    r["SELECTOR_GAP_AT_P50"] = m0["OVERLAP_MINUS_MATCHED_HARD"]
                    r["MATCHED_HARD_DELTA_ADJ"] = adj
                    r["BEATS_MATCHED_HARD_ADJ"] = bool(adj > 0)
                # is the overlap-vs-read-deeper difference itself significant?  Both are
                # per-query binary indicators over the same queries, so McNemar applies.
                ih = m.get("_ind_matched_hard")
                if ih is not None:
                    gd, ls, pv = mcnemar_pair(np.asarray(ih, np.int8), ind)
                    r["vsHARD_RAW_gained"] = gd; r["vsHARD_RAW_lost"] = ls
                    r["vsHARD_RAW_p"] = float(f"{pv:.3g}"); r["vsHARD_RAW_sig"] = bool(pv < 0.05)
                # The adjusted quantity is a difference-in-differences -- (cell - its matched
                # hard) minus (O0_CORE - hard at P'=50) -- so McNemar does not apply to it.
                # Test it by paired bootstrap over queries instead.
                ih0 = (m0 or {}).get("_ind_matched_hard")
                if ih0 is not None and m0:
                    per_q = ((ind.astype(np.int16) - np.asarray(ih, np.int16))
                             - (b_ind.astype(np.int16) - np.asarray(ih0, np.int16)))
                    lo, hi, pboot = boot_ci(per_q)
                    r["vsHARD_ADJ_ci"] = [lo, hi]
                    r["vsHARD_ADJ_p"] = pboot
                    r["vsHARD_ADJ_sig"] = bool(lo > 0 or hi < 0)
                r["MATCHED_HARD_DELTA"] = m["OVERLAP_MINUS_MATCHED_HARD"]
                r["BEATS_MATCHED_HARD"] = m["BEATS_MATCHED_HARD"]
                r["EXPOSURE_MATCHED"] = m.get("EXPOSURE_MATCHED")
                r["matched_hard_depth"] = m.get("matched_hard_depth")
            rows.setdefault(name, {})[ds] = r
    return rows


def leaderboard(rows, need_all=True):
    have = sorted({ds for v in rows.values() for ds in v})
    out = []
    for name, per in rows.items():
        if need_all and len(per) < len(have):
            continue
        ds_list = sorted(per)
        deltas = [per[d]["delta"] for d in ds_list]
        regs = [d for d in ds_list if per[d]["sig"] and per[d]["delta"] < 0]
        mh = [per[d].get("MATCHED_HARD_DELTA_ADJ") for d in ds_list]
        # only cells whose exposure actually matched can support a matched-exposure claim
        mh_ok = [per[d].get("BEATS_MATCHED_HARD_ADJ") for d in ds_list
                 if per[d].get("BEATS_MATCHED_HARD_ADJ") is not None
                 and per[d].get("EXPOSURE_MATCHED") is not False]
        out.append({
            "METHOD": name,
            "WORST_DELTA": round(min(deltas), 4),
            "MACRO_DELTA": round(float(np.mean(deltas)), 4),
            "SIG_REGRESSIONS": len(regs), "regressed_on": regs,
            "n_sig_gains": sum(1 for d in ds_list if per[d]["sig"] and per[d]["delta"] > 0),
            "METAQA_H2": per.get("metaqa", {}).get("hop2"),
            "METAQA_H3": per.get("metaqa", {}).get("hop3"),
            "METAQA_H3_DELTA": per.get("metaqa", {}).get("hop3_delta"),
            "MAX_EXPOSURE_MULT": round(max(per[d]["EXPOSURE_MULTIPLIER"] for d in ds_list), 3),
            "MEAN_EXPOSURE_MULT": round(float(np.mean([per[d]["EXPOSURE_MULTIPLIER"]
                                                       for d in ds_list])), 3),
            "MAX_R": round(max(per[d]["R"] for d in ds_list), 3),
            "BEATS_MATCHED_HARD_ON": sum(1 for x in mh_ok if x),
            "MATCHED_HARD_CELLS": len(mh_ok),
            "WORST_MATCHED_HARD_DELTA_ADJ": (round(min(x for x in mh if x is not None), 4)
                                         if any(x is not None for x in mh) else None),
            "per_corpus": {d: per[d]["delta"] for d in ds_list}})
    # the universal selection rule, verbatim and lexicographic
    out.sort(key=lambda r: (r["SIG_REGRESSIONS"], -r["WORST_DELTA"], -r["MACRO_DELTA"],
                            -(r["METAQA_H3"] or 0), r["MEAN_EXPOSURE_MULT"]))
    return out, have


def main():
    cov, mat = load()
    rows = cell_stats(cov, mat)
    lb, have = leaderboard(rows)
    part = leaderboard(rows, need_all=False)[0]
    os.makedirs(f"{OUT}/universal_leaderboard", exist_ok=True)
    json.dump({"CORPORA_COMPLETE": have, "CELLS": rows, "LEADERBOARD": lb,
               "LEADERBOARD_PARTIAL": part},
              open(f"{OUT}/universal_leaderboard/LEADERBOARD.json", "w"), indent=1)
    print(f"corpora with coverage: {have}\n")
    hdr = (f"{'METHOD':20s} {'WORST':>8s} {'MACRO':>8s} {'SIGREG':>6s} {'GAINS':>5s} "
           f"{'H2':>6s} {'H3':>6s} {'EXPO':>6s} {'R':>7s} {'vsHARD':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in (lb or part):
        print(f"{r['METHOD']:20s} {r['WORST_DELTA']:+8.4f} {r['MACRO_DELTA']:+8.4f} "
              f"{r['SIG_REGRESSIONS']:6d} {r['n_sig_gains']:5d} "
              f"{(r['METAQA_H2'] or 0):6.3f} {(r['METAQA_H3'] or 0):6.3f} "
              f"{r['MEAN_EXPOSURE_MULT']:6.3f} {r['MAX_R']:7.2f} "
              f"{r['BEATS_MATCHED_HARD_ON']:3d}/{r['MATCHED_HARD_CELLS']:<3d}")
    return lb


if __name__ == "__main__":
    main()
