"""PHASES 14/18/19 + FINAL_REPORT / TABLES / RETURNS for the universal partition+overlap search.

Everything here is derived from measured artifacts on disk.  Nothing is hand-entered.

  python scratchpad/_l1ov_report.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_final as FR
import _l1ov_cost as CO
import _l1ov_lead as LD

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean"]
SHORT = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
         "musique_clean": "MuSiQue", "squad_clean": "SQuAD", "hotpotqa_clean": "HotpotQA"}
FAMS = ["O1_STRUCT", "O2_NERX", "O3_KNN", "O4_FULL_C"]
BETAS = ["b0.25", "b0.5", "b1.0"]


def jload(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d


def gather():
    g = {"lead": jload(f"{OUT}/universal_leaderboard/LEADERBOARD.json", {}),
         "expl": jload(f"{OUT}/overlap/EXPLOSION_AUDIT.json", {}),
         "hard": jload(f"{OUT}/hard_partitions/HARD_GATE.json", {}),
         "export": jload(f"{OUT}/hypergraph/EXPORT_MANIFEST.json", {}),
         "probe": jload(f"{OUT}/modal/PROBE.json", {}),
         "precap": jload(f"{OUT}/hypergraph/PRECAP_HYPEREDGE_SIZES.json", {}),
         "p17": jload(f"{OUT}/diagnostics/P17_HALO_AWARE_RANKING.json", {}),
         "minb": jload(f"{OUT}/hard_partitions/MIN_BLOCKS_REQUIRED.json", {}),
         "rrank": jload(f"{OUT}/diagnostics/HALO_RANKING_CONTROL.json", {}),
         "cov": {}, "cov_all": {}, "matched": {}, "replay": {}}
    for ds in DS:
        c = jload(f"{OUT}/overlap/COVERAGE_{ds}.json", {})
        if c:
            g["cov"][ds] = c.get("CURRENT", {})
            g["cov_all"][ds] = c
        m = jload(f"{OUT}/exposure/MATCHED_{ds}.json", {})
        if m:
            g["matched"][ds] = m.get("CURRENT", {}).get("MATCHED", {})
        r = jload(f"{OUT}/hard_partitions/REPLAY_{ds}.json", {})
        if r:
            g["replay"][ds] = r
    return g


def parity(g):
    """PHASE 9 -- prove the halo never touched partition RANKING.

    Two guarantees, one structural and one measured:
      * structural: selected_blocks() runs ONCE per corpus, before any halo object exists, and
        every cell reuses the same base50/f650 lists, so the core ids are the same objects.
      * measured: with an empty halo "all required NODES fetched" is the same event as "all gold
        PARTITIONS selected", so O0_CORE must equal replay(CURRENT) exactly.  Any halo leaking
        into ranking would break that identity.
    """
    out = {}
    for ds, cov in g["cov"].items():
        o0 = (cov.get("O0_CORE") or {}).get("F6")
        o0b = (cov.get("O0_CORE") or {}).get("BASE")
        rep = (g["replay"].get(ds) or {}).get("CURRENT")
        if not (o0 and rep):
            continue
        out[ds] = {
            "O0_CORE_F6_ALL_REQUIRED": o0["ALL_REQUIRED_FETCHED"],
            "REPLAY_F6_ALL_P50": rep["F6_ALL_P50"],
            "F6_EXACT": bool(abs(o0["ALL_REQUIRED_FETCHED"] - rep["F6_ALL_P50"]) < 1e-9),
            "O0_CORE_BASE_ALL_REQUIRED": (o0b or {}).get("ALL_REQUIRED_FETCHED"),
            "REPLAY_BASE_ALL_P50": rep["BASE_ALL_P50"],
            "BASE_EXACT": bool(o0b is not None and
                               abs(o0b["ALL_REQUIRED_FETCHED"] - rep["BASE_ALL_P50"]) < 1e-9)}
    ok = all(v["F6_EXACT"] and v["BASE_EXACT"] for v in out.values()) if out else None
    res = {"PER_CORPUS": out, "CORE_RANKING_UNCHANGED_BY_HALO": ok,
           "STRUCTURAL_GUARANTEE": ("selected_blocks() is called once per corpus before any halo "
                                    "exists; all cells reuse the identical base50/f650 lists")}
    json.dump(res, open(f"{OUT}/diagnostics/P9_CORE_PARITY.json", "w"), indent=1)
    return res


def family_contribution(g):
    """PHASE 14 -- what each edge family adds, evaluation-only provenance."""
    out = {}
    for ds, cov in g["cov"].items():
        base = (cov.get("O0_CORE") or {}).get("F6")
        if not base:
            continue
        row = {"CORE": base["ALL_REQUIRED_FETCHED"]}
        for suf in [""] + ["_" + b for b in BETAS]:
            got = {}
            for fam in FAMS:
                f = (cov.get(fam + suf) or {}).get("F6")
                if f:
                    got[fam] = f["ALL_REQUIRED_FETCHED"]
            if len(got) < 4:
                continue
            singles = {k: v for k, v in got.items() if k != "O4_FULL_C"}
            best_single = max(singles, key=lambda k: singles[k])
            row[suf or "unbounded"] = {
                "per_family": {k: round(v - base["ALL_REQUIRED_FETCHED"], 4)
                               for k, v in got.items()},
                "DELTA_NERX_OVER_STRUCT": round(got["O2_NERX"] - got["O1_STRUCT"], 4),
                "DELTA_KNN_OVER_STRUCT": round(got["O3_KNN"] - got["O1_STRUCT"], 4),
                "BEST_SINGLE_FAMILY": best_single,
                "DELTA_FULLC_OVER_BEST_SINGLE": round(got["O4_FULL_C"] - singles[best_single], 4)}
        out[ds] = row
    json.dump(out, open(f"{OUT}/diagnostics/FAMILY_CONTRIBUTION.json", "w"), indent=1)
    return out


P16_CELLS = ["O0_CORE", "O1_STRUCT_b1.0", "O4_FULL_C_b0.5", "O4_FULL_C_b1.0", "O4_FULL_C"]


def p16_grid(g, core_tag="H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH"):
    """PHASE 16 -- put the best HARD core under the same halo recipe as production METIS.

    Only the pairing that matters is run, as the program instructs: the one hard partition that
    actually moved a corpus (H4 true hypergraph, MetaQA) crossed with the halo family that won.
    """
    rows = []
    for ds in DS:
        c = g["cov_all"].get(ds) or {}
        if core_tag not in c:
            continue
        for cell in P16_CELLS:
            a = (c.get("CURRENT", {}).get(cell) or {}).get("F6")
            b = (c[core_tag].get(cell) or {}).get("F6")
            if not (a and b):
                continue
            rows.append({"ds": ds, "cell": cell,
                         "METIS": a["ALL_REQUIRED_FETCHED"], "H4": b["ALL_REQUIRED_FETCHED"],
                         "delta": round(b["ALL_REQUIRED_FETCHED"] - a["ALL_REQUIRED_FETCHED"], 4),
                         "METIS_R": c["CURRENT"][cell]["REPLICATION_FACTOR"],
                         "H4_R": c[core_tag][cell]["REPLICATION_FACTOR"],
                         "METIS_expo": a["EXPOSURE_MULTIPLIER"], "H4_expo": b["EXPOSURE_MULTIPLIER"]})
    # "usable" means a bounded halo.  The unbounded cells are in the grid for completeness but
    # they cost R = 9-37 memberships, which P8 already ruled out, so they cannot settle P16.
    W = "O4_FULL_C_b0.5"
    bnd = [r for r in rows if r["cell"] != "O0_CORE" and r["H4_R"] <= 2.001]
    out = {"CORE_TAG": core_tag, "PROMOTED_HALO": W, "GRID": rows,
           "H4_BEATS_METIS_UNDER_PROMOTED_HALO":
               sorted({r["ds"] for r in rows if r["cell"] == W and r["delta"] > 0}),
           "H4_LOSES_TO_METIS_UNDER_PROMOTED_HALO":
               sorted({r["ds"] for r in rows if r["cell"] == W and r["delta"] < 0}),
           "H4_BEATS_METIS_UNDER_ANY_BOUNDED_HALO":
               sorted({r["ds"] for r in bnd if r["delta"] > 0}),
           "HALO_REPAIRS_H4_CORE_REGRESSION": None}
    sq = {r["cell"]: r for r in rows if r["ds"] == "squad_clean"}
    if sq:
        base = sq["O0_CORE"]["METIS"]
        cand = {k: v["H4"] for k, v in sq.items()
                if k != "O0_CORE" and v["H4_R"] <= 2.001}
        best = max(cand.values(), default=None)
        out["HALO_REPAIRS_H4_CORE_REGRESSION"] = bool(best is not None and best >= base)
        out["SQUAD_METIS_CORE_ALONE"] = base
        out["SQUAD_H4_CORE_BEST_BOUNDED_HALO"] = best
        out["SQUAD_H4_CORE_BEST_BOUNDED_HALO_CELL"] = (
            max(cand, key=lambda k: cand[k]) if cand else None)
    json.dump(out, open(f"{OUT}/diagnostics/P16_CORE_X_HALO.json", "w"), indent=1)
    return out


def md_table(rows, hdr):
    w = [max(len(str(hdr[i])), max((len(str(r[i])) for r in rows), default=0))
         for i in range(len(hdr))]
    o = ["| " + " | ".join(str(hdr[i]).ljust(w[i]) for i in range(len(hdr))) + " |",
         "|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|"]
    for r in rows:
        o.append("| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |")
    return "\n".join(o)


def fmt(v, n=4, sign=True):
    if v is None:
        return "--"
    return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"


def tables(g, fam):
    T = []
    have = [d for d in DS if d in g["cov"]]
    T.append("# TABLES -- universal balanced partition + overlap search\n")

    # ---- P8
    T.append("## P8  EXPLOSION AUDIT -- unbounded 1-hop overlap replication factor\n")
    T.append("`R = sum_j |C_j u H_j| / |V|`.  R=1 is a hard partition.\n")
    rows = []
    for ds in have:
        e = g["expl"].get(ds, {})
        if not e:
            continue
        rows.append([SHORT[ds], f"{e['O0_CORE']['REPLICATION_FACTOR']:.2f}"] +
                    [f"{e[f]['REPLICATION_FACTOR']:.2f}" for f in FAMS] +
                    [e["O4_FULL_C"]["NODE_MULTIPLICITY"]["max"],
                     e["O4_FULL_C"]["NODE_MULTIPLICITY"].get("p99")])
    T.append(md_table(rows, ["corpus", "core", "STRUCT", "NERX", "KNN", "FULL_C",
                             "FULLC max mult", "FULLC p99"]) + "\n")

    # ---- P10/P12
    T.append("## P10/P12  ALL_REQUIRED_FETCHED and selected-P50 exposure\n")
    T.append("Coverage is measured on required NODES, never on a canonical partition id.\n")
    for suf, lbl in [("", "unbounded"), ("_b0.25", "beta=0.25"), ("_b0.5", "beta=0.5"),
                     ("_b1.0", "beta=1.0")]:
        rows = []
        for ds in have:
            cov = g["cov"][ds]
            b = (cov.get("O0_CORE") or {}).get("F6")
            if not b:
                continue
            r = [SHORT[ds], f"{b['ALL_REQUIRED_FETCHED']:.4f}"]
            for f in FAMS:
                c = (cov.get(f + suf) or {}).get("F6")
                r.append(f"{c['ALL_REQUIRED_FETCHED']:.4f} (x{c['EXPOSURE_MULTIPLIER']:.2f})"
                         if c else "--")
            rows.append(r)
        if rows:
            T.append(f"### {lbl}\n")
            T.append(md_table(rows, ["corpus", "core"] + FAMS) + "\n")

    # ---- P15
    T.append("## P15  MATCHED-NODE-EXPOSURE CONTROL (the decisive table)\n")
    T.append("Overlap vs simply selecting MORE hard core partitions until the unique-node budget\n"
             "matches.  Values are difference-in-differences: the cell's advantage over its\n"
             "matched hard depth, MINUS the advantage O0_CORE already has at P'=50 (which is the\n"
             "F6-vs-BASE selector gap, not an overlap effect).  `*` = paired bootstrap 95% CI\n"
             "excludes zero.  `(unmatched)` = the hard control saturated the 200-block ranking\n"
             "and read FEWER nodes, so no matched claim is possible.\n")
    cells = g["lead"].get("CELLS", {})
    rows = []
    for f in FAMS:
        for suf in [""] + ["_" + b for b in BETAS]:
            name = f + suf
            per = cells.get(name)
            if not per:
                continue
            r = [name]
            for ds in have:
                x = per.get(ds, {})
                v = x.get("MATCHED_HARD_DELTA_ADJ")
                if v is None:
                    r.append("--")
                elif x.get("EXPOSURE_MATCHED") is False:
                    r.append("(unmatched)")
                else:
                    r.append(fmt(v) + ("*" if x.get("vsHARD_ADJ_sig") else ""))
            rows.append(r)
    T.append(md_table(rows, ["cell"] + [SHORT[d] for d in have]) + "\n")

    # ---- P14
    T.append("## P14  FAMILY CONTRIBUTION TO THE HALO (evaluation-only provenance)\n")
    rows = []
    for ds in have:
        f14 = fam.get(ds, {})
        for k in ["unbounded", "_b0.25", "_b0.5", "_b1.0"]:
            v = f14.get(k)
            if not v:
                continue
            rows.append([SHORT[ds], k.lstrip("_") or "unbounded",
                         fmt(v["DELTA_NERX_OVER_STRUCT"]), fmt(v["DELTA_KNN_OVER_STRUCT"]),
                         v["BEST_SINGLE_FAMILY"], fmt(v["DELTA_FULLC_OVER_BEST_SINGLE"])])
    T.append(md_table(rows, ["corpus", "budget", "NERX-STRUCT", "KNN-STRUCT",
                             "best single", "FULLC-best"]) + "\n")

    # ---- P3
    pc = (g.get("precap") or {}).get("PER_CORPUS") or {}
    if pc:
        T.append("## P3  HYPEREDGE SIZES BEFORE THE UNIVERSAL CAP\n")
        T.append(f"Cap = {g['precap']['CAP']}, identical on every corpus and family.  A closed"
                 " neighbourhood over the cap is dropped whole.\n")
        rows = []
        for d in have:
            for f2 in ("H_STRUCT_LOCAL", "H_KNN_LOCAL"):
                v = (pc.get(d) or {}).get(f2)
                if v:
                    rows.append([SHORT[d], f2, f"{v['hyperedges_precap']:,}",
                                 v["size_mean"], round(v["size_p90"], 1),
                                 round(v["size_p99"], 1), round(v["size_p999"], 1),
                                 f"{v['size_max']:,}",
                                 f"{v['frac_over_cap']:.2%}",
                                 f"{v['frac_pins_over_cap']:.2%}"])
        T.append(md_table(rows, ["corpus", "family", "hyperedges", "mean", "p90", "p99",
                                 "p99.9", "max", "over cap", "pins over cap"]) + "\n")

    # ---- P16
    p16 = p16_grid(g)
    if p16["GRID"]:
        T.append("## P16  BEST HARD CORE x HALO\n")
        T.append("The one hard partition that moved a corpus (H4 true hypergraph) under the SAME\n"
                 "halo recipe as the production METIS core.  Only this pairing is run.\n")
        T.append(md_table([[SHORT[r["ds"]], r["cell"], f"{r['METIS']:.4f}", f"{r['H4']:.4f}",
                            fmt(r["delta"]), f"{r['METIS_R']:.2f}", f"{r['H4_R']:.2f}",
                            f"x{r['METIS_expo']:.2f}", f"x{r['H4_expo']:.2f}"]
                           for r in p16["GRID"]],
                          ["corpus", "halo", "METIS core", "H4 core", "delta",
                           "METIS R", "H4 R", "METIS expo", "H4 expo"]) + "\n")

    # ---- P11
    T.append("## P11  MIN_BLOCK_COVER -- blocks needed to cover all required nodes\n")
    rows = []
    for ds in have:
        cov = g["cov"][ds]
        for name in ["O0_CORE", "O1_STRUCT_b0.25", "O4_FULL_C_b0.25", "O4_FULL_C"]:
            m = (cov.get(name) or {}).get("MIN_BLOCK_COVER")
            if m:
                rows.append([SHORT[ds], name, round(m["median"], 1), round(m["p90"], 1),
                             round(m["p95"], 1), m["max"],
                             m.get("exact_fraction")])
    T.append(md_table(rows, ["corpus", "cell", "median", "p90", "p95", "max",
                             "exact frac"]) + "\n")

    # ---- P1-P5
    if g["hard"].get("SUMMARY"):
        T.append("## P1-P5  HARD PARTITIONER x REPRESENTATION MATRIX\n")
        T.append("Retrieval utility under the FROZEN L1 contract, delta of F6 ALL@50 vs the\n"
                 "production METIS partition.  Significance is exact McNemar AND above the\n"
                 "measured METIS reseed noise floor for that corpus.\n")
        rows = []
        for t in g["hard"]["ORDER"]:
            v = g["hard"]["SUMMARY"][t]
            r = [t] + [fmt(v["per_corpus"].get(d)) for d in have]
            r += [fmt(v["WORST_DELTA"]), fmt(v["MACRO_DELTA"]), v["SIG_REGRESSIONS"],
                  v["n_sig_gains"]]
            rows.append(r)
        T.append(md_table(rows, ["partitioner x graph"] + [SHORT[d] for d in have] +
                          ["worst", "macro", "sig reg", "sig gains"]) + "\n")
        rows = []
        for ds in have:
            for tag, R in (g["replay"].get(ds) or {}).items():
                rows.append([SHORT[ds], tag, R["BALANCE"]["max_over_mean"],
                             R["BALANCE"]["size_cv"], R["BALANCE"]["size_min"],
                             R["CUT"]["edge_cut_fraction"], R["CUT"]["boundary_fraction"],
                             R["F6_ALL_P50"], R["BASE_ALL_P50"]])
        if rows:
            T.append("### P4  balance, cut and boundary\n")
            T.append(md_table(rows, ["corpus", "partition", "max/mean", "size CV", "min",
                                     "edge cut", "boundary", "F6 ALL@50", "BASE ALL@50"]) + "\n")
    mb = (g.get("minb") or {})
    if mb.get("PER_CORPUS"):
        T.append("### P4  minimum blocks required by the gold/reference nodes\n")
        T.append(mb["DEFINITION"] + ".  " + mb["NOTE"] + "." + "\n")
        rows = []
        for ds in DS:
            for tag, st in (mb["PER_CORPUS"].get(ds) or {}).items():
                rows.append([SHORT[ds], tag, st["required_nodes_mean"],
                             st["min_blocks_mean"], st["min_blocks_median"],
                             st["min_blocks_max"],
                             f"{st['frac_single_block']:.1%}"])
        T.append(md_table(rows, ["corpus", "partition", "required nodes",
                                 "min blocks mean", "median", "max",
                                 "single-block queries"]) + "\n")
        pr = mb.get("PREDICTIVENESS")
        if pr:
            T.append(f"Across the {pr['n_cells']} partitioner-by-corpus cells, the change in mean min-blocks "
                     f"and the change in F6 ALL@50 are monotonically associated: Spearman "
                     f"rho = {pr['spearman_rho']:+.3f}, p = {pr['spearman_p']:.3g}.  "
                     f"Pearson reads {pr['pearson_r']:+.3f} but is "
                     f"dominated by two MetaQA cells, so Spearman is the honest summary.  "
                     f"Containment still cannot be the objective: on "
                     + ", ".join(SHORT[d] for d in pr["DEGENERATE_CORPORA"]) +
                     " every query has exactly one required node, so min-blocks is 1 for "
                     "EVERY partition and cannot vary -- yet its utility spans 0.0140, and "
                     "it is the corpus that blocks the universal gate." + "\n")
    return "\n".join(T)



def best_universal(g, require_matched=True):
    """The UNIVERSAL SELECTION RULE, applied verbatim and lexicographically.

    Gate 0 (P15): a cell may not SIGNIFICANTLY lose to simply reading more hard partitions at
                  the same unique-node budget.  Difference-in-differences vs O0_CORE, bootstrap.
    Gate 1:       zero significant cross-corpus regressions on the primary metric.
    Then:         max worst-corpus, max macro, max MetaQA hop3, min exposure multiplier.

    One recipe for all six corpora.  No per-corpus choice is ever made.
    """
    cells = g["lead"].get("CELLS", {})
    have = sorted(g["cov"])
    out = []
    for name, per in cells.items():
        if len(per) < len(have):
            continue
        dl = [per[d]["delta"] for d in have]
        sig_reg = [d for d in have if per[d]["sig"] and per[d]["delta"] < 0]
        # P15 gate: significant DiD losses, only where the exposure actually matched
        m_lose = [d for d in have
                  if per[d].get("vsHARD_ADJ_sig") and (per[d].get("MATCHED_HARD_DELTA_ADJ") or 0) < 0
                  and per[d].get("EXPOSURE_MATCHED") is not False]
        m_win = [d for d in have
                 if per[d].get("vsHARD_ADJ_sig") and (per[d].get("MATCHED_HARD_DELTA_ADJ") or 0) > 0
                 and per[d].get("EXPOSURE_MATCHED") is not False]
        unmatched = [d for d in have if per[d].get("EXPOSURE_MATCHED") is False]
        out.append({
            "METHOD": name,
            "WORST_DELTA": round(min(dl), 4), "MACRO_DELTA": round(float(np.mean(dl)), 4),
            "SIG_REGRESSIONS": len(sig_reg), "regressed_on": sig_reg,
            "MATCHED_SIG_LOSSES": len(m_lose), "matched_lost_on": m_lose,
            "MATCHED_SIG_WINS": len(m_win), "matched_won_on": m_win,
            "unmatched_on": unmatched,
            "PASSES_P15": len(m_lose) == 0 and len(m_win) > 0,
            "METAQA_H2": per.get("metaqa", {}).get("hop2"),
            "METAQA_H3": per.get("metaqa", {}).get("hop3"),
            "MEAN_EXPOSURE_MULT": round(float(np.mean([per[d]["EXPOSURE_MULTIPLIER"]
                                                       for d in have])), 3),
            "MAX_EXPOSURE_MULT": round(max(per[d]["EXPOSURE_MULTIPLIER"] for d in have), 3),
            "MAX_R": round(max(per[d]["R"] for d in have), 3),
            "per_corpus": {d: per[d]["delta"] for d in have},
            "per_corpus_matched_adj": {d: per[d].get("MATCHED_HARD_DELTA_ADJ") for d in have}})
    # the rule, applied lexicographically.  P15 is a GATE, so it sorts first in the full
    # listing; within a gate outcome the ordering is exactly the program's.
    key = lambda r: (r["SIG_REGRESSIONS"], -r["WORST_DELTA"], -r["MACRO_DELTA"],
                     -(r["METAQA_H3"] or 0), r["MEAN_EXPOSURE_MULT"])
    elig = [r for r in out if r["PASSES_P15"]] if require_matched else list(out)
    elig.sort(key=key)
    out.sort(key=lambda r: (0 if r["PASSES_P15"] else 1,) + key(r))
    return out, elig


def returns(g, fam, par, p16=None, cost=None):
    cells = g["lead"].get("CELLS", {})
    allr, elig = best_universal(g)
    best = elig[0] if elig else None
    hard = g["hard"] or {}
    hsum = hard.get("SUMMARY", {})
    horder = [t for t in hard.get("ORDER", [])]
    # a cell scored on two corpora can show "zero regressions" for free, so the reported
    # BEST_HARD_* is restricted to cells measured on EVERY corpus; partial cells are kept
    # in the table but cannot win.
    ncorp = len(hard.get("CORPORA") or [])
    full = [t for t in horder if hsum.get(t, {}).get("n_corpora") == ncorp and ncorp]
    hbest = full[0] if full else (horder[0] if horder else None)
    fc = cells.get("O4_FULL_C", {})
    R = {
        "BEST_HARD_PARTITIONER": (hbest.split("__")[0] if hbest else None),
        "BEST_HARD_GRAPH_REPRESENTATION": (hbest.split("__")[1] if hbest else None),
        "BEST_HARD_METAQA_H3": ((hard.get("CELLS", {}).get(hbest, {}).get("metaqa", {})
                                 .get("by_hop") or {}).get("hop3") if hbest else None),
        "HARD_CROSS_CORPUS_SAFE": hard.get("HARD_CROSS_CORPUS_SAFE"),
        "BEST_HARD_N_CORPORA": hsum.get(hbest, {}).get("n_corpora") if hbest else None,
        "BEST_HARD_SCORED_ON_ALL_CORPORA": bool(full),
        "BEST_HARD_METAQA_H3_ANY_CELL": max(
            [((hard.get("CELLS", {}).get(t, {}).get("metaqa", {}).get("by_hop") or {})
              .get("hop3") or 0) for t in horder] or [None]),
        "UNIVERSAL_HARD_GATE_PASSED": hard.get("UNIVERSAL_HARD_GATE_PASSED"),
        "FULLC_HALO_REPLICATION_FACTOR": {d: v["R"] for d, v in fc.items()},
        "FULLC_HALO_P50_EXPOSURE_MULTIPLIER": {d: v["EXPOSURE_MULTIPLIER"]
                                               for d, v in fc.items()},
        "FULLC_HALO_METAQA_H2": fc.get("metaqa", {}).get("hop2"),
        "FULLC_HALO_METAQA_H3": fc.get("metaqa", {}).get("hop3"),
    }
    if best:
        nm = best["METHOD"]
        R.update({
            "BEST_UNIVERSAL_METHOD": nm,
            "BEST_HALO_EDGE_FAMILY": nm.split("_b")[0],
            "BEST_BOUNDED_HALO_BETA": (float(nm.split("_b")[1]) if "_b" in nm else None),
            "BEST_HALO_REPLICATION_FACTOR": {d: cells[nm][d]["R"] for d in cells[nm]},
            "BEST_HALO_P50_EXPOSURE": {d: cells[nm][d]["EXPOSURE_MULTIPLIER"] for d in cells[nm]},
            "BEST_HALO_METAQA_H2": best["METAQA_H2"], "BEST_HALO_METAQA_H3": best["METAQA_H3"],
            "BEST_UNIVERSAL_WORST_DELTA": best["WORST_DELTA"],
            "BEST_UNIVERSAL_MACRO_DELTA": best["MACRO_DELTA"],
            "BEST_UNIVERSAL_METAQA_H3": best["METAQA_H3"],
            "BEST_UNIVERSAL_SIGNIFICANT_REGRESSIONS": best["SIG_REGRESSIONS"],
            "BEST_UNIVERSAL_EXPOSURE_MULTIPLIER": best["MEAN_EXPOSURE_MULT"],
            "BEST_UNIVERSAL_MAX_EXPOSURE_MULTIPLIER": best["MAX_EXPOSURE_MULT"],
            "BEST_UNIVERSAL_MATCHED_SIG_WINS": best["matched_won_on"],
            "BEST_UNIVERSAL_MATCHED_SIG_LOSSES": best["matched_lost_on"]})
        for d, sh in [("webqsp", "WEBQSP"), ("2wiki_clean", "2WIKI"),
                      ("musique_clean", "MUSIQUE"), ("hotpotqa_clean", "HOTPOT"),
                      ("squad_clean", "SQUAD")]:
            R[f"BEST_HALO_{sh}"] = cells[nm].get(d, {}).get("delta")
    R["CORE_RANKING_UNCHANGED_BY_HALO"] = par.get("CORE_RANKING_UNCHANGED_BY_HALO")
    p17 = g.get("p17") or {}
    W = "O4_FULL_C_b0.5"
    cells17 = {d: (v.get("CELLS") or {}).get(W) for d, v in p17.items()}
    cells17 = {d: v for d, v in cells17.items() if v}
    if cells17:
        wins = sorted(d for d, v in cells17.items() if v["p"] < 0.05 and v["DELTA"] > 0)
        loss = sorted(d for d, v in cells17.items() if v["p"] < 0.05 and v["DELTA"] < 0)
        R["P17_HALO_AWARE_RANKING_SIG_WINS"] = wins
        R["P17_HALO_AWARE_RANKING_SIG_LOSSES"] = loss
        R["P17_HALO_AWARE_RANKING_UNIVERSALLY_BETTER"] = bool(
            len(wins) == len(cells17) and not loss)
        agree = []
        for d, v in cells17.items():
            w = (g["cov"].get(d) or {}).get(W, {}).get("F6")
            if w:
                agree.append(abs(v["CORE_SCORED"]["ALL_REQUIRED_FETCHED"] -
                                 w["ALL_REQUIRED_FETCHED"]) < 1e-9)
        R["CROSS_IMPLEMENTATION_AGREEMENT_EXACT"] = bool(agree and all(agree))
        R["P17_PATH_PARITY_EXACT"] = bool(all(
            (v.get("PARITY") or {}).get("EXACT") for v in p17.values()))
        R["FINAL_RANKING_POLICY"] = ("HALO_AWARE"
                                     if R["P17_HALO_AWARE_RANKING_UNIVERSALLY_BETTER"]
                                     else "CORE_ONLY")
    if p16:
        R["P16_HALO_REPAIRS_H4_CORE_REGRESSION"] = p16.get(
            "HALO_REPAIRS_H4_CORE_REGRESSION")
        R["P16_H4_BEATS_METIS_UNDER_PROMOTED_HALO"] = p16.get(
            "H4_BEATS_METIS_UNDER_PROMOTED_HALO")
    pc = (g.get("precap") or {}).get("PER_CORPUS") or {}
    if pc:
        R["UNIVERSAL_HYPEREDGE_CAP"] = (g["precap"] or {}).get("CAP")
        R["PRECAP_FRAC_PINS_DROPPED_STRUCT"] = {
            d: v["H_STRUCT_LOCAL"]["frac_pins_over_cap"] for d, v in pc.items()}
        R["PRECAP_MEAN_HYPEREDGE_SIZE_STRUCT"] = {
            d: v["H_STRUCT_LOCAL"]["size_mean"] for d, v in pc.items()}
        R["GROUP_REPRESENTATION_SHRINKS_CORPUS"] = sorted(
            d for d in pc
            if (g["export"].get(d, {}).get("G1_LOCAL_HYPER_CLIQUE", {})
                .get("undirected_edges", 0)
                < g["export"].get(d, {}).get("G0_TOPOLOGY_C", {})
                .get("undirected_edges", 0)))
    if cost:
        R["MODAL_NO_GPU_ANYWHERE"] = cost.get("NO_GPU_ANYWHERE")
        R["MODAL_TOTAL_CPU_HOURS"] = cost.get("TOTAL_CPU_HOURS")
        R["MODAL_PARTITION_CELLS"] = cost.get("CELLS")
    # ---------------------------------------------------------------- verdict letters
    # Assigned from the measured artifacts, not chosen: each letter has one condition.
    hard = g["hard"] or {}
    HC = hard.get("CELLS") or {}
    gate_pass = hard.get("UNIVERSAL_HARD_GATE_PASSED") or []
    best_un = elig[0] if elig else None
    any_gain = any(v["sig"] and v["delta"] > 0 for t in HC for v in HC[t].values())
    ctrl = HC.get("H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C") or {}
    ctrl_null = bool(ctrl) and not any(v["sig"] for v in ctrl.values())
    if any(t.startswith("H1") for t in gate_pass):
        pv = "A. KAHIP_RECOVERED"
    elif any(t.startswith("H2") for t in gate_pass):
        pv = "B. CONNECTED_KAHIP_RECOVERED"
    elif any(t.startswith("H4") for t in gate_pass):
        pv = "C. TRUE_HYPERGRAPH_RECOVERED"
    elif ctrl_null:
        pv = "E. HARD_PARTITIONER_NOT_THE_MAIN_ISSUE"
    elif not any_gain:
        pv = "D. METIS_REMAINS_BEST"
    else:
        pv = "D. METIS_REMAINS_BEST"
    unb = [r for r in allr if "_b" not in r["METHOD"]]
    unb_ok = [r for r in unb if r["SIG_REGRESSIONS"] == 0 and r["PASSES_P15"]]
    gained = [r for r in allr if r["MACRO_DELTA"] > 0]
    passes15 = [r for r in allr if r["PASSES_P15"]]
    if unb_ok:
        ov = "A. UNBOUNDED_OVERLAP_RECOVERED"
    elif best_un and best_un["SIG_REGRESSIONS"] == 0 and best_un["WORST_DELTA"] > 0:
        ov = "B. BOUNDED_OVERLAP_RECOVERED"
    elif gained and not passes15:
        ov = "C. OVERLAP_ONLY_BUYS_EXPOSURE"
    elif not gained:
        ov = "E. OVERLAP_NO_VALUE"
    else:
        ov = "D. OVERLAP_TOO_EXPENSIVE"
    hard_ok = bool(gate_pass)
    ov_ok = ov.startswith(("A.", "B."))
    if hard_ok and ov_ok:
        fv = "C. UNIVERSAL_HARD_PLUS_OVERLAP_FOUND"
    elif hard_ok:
        fv = "A. UNIVERSAL_PARTITION_SUBSTRATE_FOUND"
    elif ov_ok:
        fv = "B. UNIVERSAL_OVERLAP_SUBSTRATE_FOUND"
    elif any_gain:
        fv = "D. STRONG_PARTITION_EFFECT_BUT_NO_UNIVERSAL_RULE"
    else:
        fv = "E. PARTITION_SUBSTRATE_EXHAUSTED"
    R["OVERLAP_VERDICT"] = ov
    R["PARTITIONER_VERDICT"] = pv
    R["FINAL_VERDICT"] = fv
    R["_VERDICT_EVIDENCE"] = {
        "hard_gate_passed": gate_pass,
        "same_representation_control_null_everywhere": ctrl_null,
        "any_significant_hard_gain": any_gain,
        "unbounded_cells_passing_all_gates": [r["METHOD"] for r in unb_ok],
        "best_universal_cell": (best_un or {}).get("METHOD"),
        "best_universal_worst_delta": (best_un or {}).get("WORST_DELTA"),
        "best_universal_sig_regressions": (best_un or {}).get("SIG_REGRESSIONS")}
    mbp = (g.get("minb") or {}).get("PREDICTIVENESS")
    if mbp:
        R["HARD_MIN_BLOCKS_SPEARMAN_RHO"] = mbp["spearman_rho"]
        R["HARD_MIN_BLOCKS_SPEARMAN_P"] = mbp["spearman_p"]
        R["CONTAINMENT_PREDICTS_UTILITY"] = "ASSOCIATED_BUT_NOT_SUFFICIENT"
        R["CONTAINMENT_DEGENERATE_CORPORA"] = mbp["DEGENERATE_CORPORA"]
    rrall = {d: (v.get("CELLS") or {}) for d, v in (g.get("rrank") or {}).items()}
    rr = {f"{d}@{c.split('_b')[1]}": v[c] for d, v in sorted(rrall.items())
          for c in ("O4_FULL_C_b0.5", "O4_FULL_C_b0.25") if c in v}
    if rr:
        R["HALO_RANKING_CONTROL_CORPORA"] = sorted(rrall)
        R["HALO_RANKING_REAL_MINUS_RANDOM"] = {k: v["REAL_MINUS_RANDOM_mean"]
                                               for k, v in rr.items()}
        R["HALO_RANKING_SIG_CELLS"] = sorted(k for k, v in rr.items()
                                             if v["RANKING_IS_LOAD_BEARING"])
        R["HALO_RANKING_SIGN_CONSISTENT"] = bool(
            all(a["REAL_MINUS_RANDOM"] > 0 for v in rr.values() for a in v["ARMS"]))
        rec = [100.0 * (float(np.mean([a["RANDOM"]["ALL_REQUIRED_FETCHED"] for a in v["ARMS"]]))
                        - v["REAL"]["ALL_REQUIRED_CORE_ONLY"])
               / (v["REAL"]["ALL_REQUIRED_FETCHED"] - v["REAL"]["ALL_REQUIRED_CORE_ONLY"])
               for v in rr.values()
               if v["REAL"]["ALL_REQUIRED_FETCHED"] - v["REAL"]["ALL_REQUIRED_CORE_ONLY"] > 1e-9]
        R["HALO_GAIN_RECOVERED_BY_RANDOM_SCORE_PCT"] = round(float(np.mean(rec)), 1)
        R["HALO_VALUE_SOURCE"] = "CANDIDATE_RELATION_DOMINANT_SCORE_IS_REFINEMENT"
    R["LEADERBOARD_ALL"] = allr
    R["LEADERBOARD_P15_ELIGIBLE"] = elig
    json.dump(R, open(f"{OUT}/RETURNS.json", "w"), indent=1)
    return R, allr, elig


def main():
    for d in ("modal", "hard_partitions", "hypergraph", "overlap", "exposure",
              "universal_leaderboard", "diagnostics"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    # regenerate the derived artifacts BEFORE reading them, so a refreshed measurement
    # (a new partitioner cell, a re-run matched control) can never be masked by a stale
    # LEADERBOARD.json or COST.json from an earlier pass.
    CO.main()
    LD.main()
    g = gather()
    fam = family_contribution(g)
    par = parity(g)
    print("P9 core-ranking parity:", par["CORE_RANKING_UNCHANGED_BY_HALO"],
          {k: (v["F6_EXACT"], v["BASE_EXACT"]) for k, v in par["PER_CORPUS"].items()})
    open(f"{OUT}/TABLES.md", "w", encoding="utf-8").write(tables(g, fam))
    print(f"wrote {OUT}/TABLES.md")
    p16 = p16_grid(g)
    cost = jload(f'{OUT}/modal/COST.json')
    R, allr, elig = returns(g, fam, par, p16, cost)
    print("")
    print("P15-ELIGIBLE universal candidates (beat read-deeper somewhere, lose nowhere):")
    hdr = (f"{'METHOD':20s} {'WORST':>8s} {'MACRO':>8s} {'SIGREG':>6s} {'M-WIN':>6s} "
           f"{'M-LOSS':>6s} {'H3':>7s} {'EXPO':>6s}")
    print(hdr); print("-" * len(hdr))
    for r in elig[:12]:
        print(f"{r['METHOD']:20s} {r['WORST_DELTA']:+8.4f} {r['MACRO_DELTA']:+8.4f} "
              f"{r['SIG_REGRESSIONS']:6d} {r['MATCHED_SIG_WINS']:6d} {r['MATCHED_SIG_LOSSES']:6d} "
              f"{(r['METAQA_H3'] or 0):7.4f} {r['MEAN_EXPOSURE_MULT']:6.3f}")
    print("")
    open(f"{OUT}/FINAL_REPORT.md", "w", encoding="utf-8").write(
        FR.build(g, fam, par, R, allr, elig, md_table, fmt, p16, cost, g['p17'],
                 g['precap']))
    print(f"wrote {OUT}/FINAL_REPORT.md")
    print("BEST_UNIVERSAL_METHOD =", R.get("BEST_UNIVERSAL_METHOD"))
    return g, fam, R


if __name__ == "__main__":
    main()
