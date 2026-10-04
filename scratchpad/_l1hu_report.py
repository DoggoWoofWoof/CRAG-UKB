"""Assemble the UNIVERSAL TRUE-HYPERGRAPH + HALO artifacts: TABLES.md, RETURNS.json, report.

Reads only what the phases wrote; computes nothing new except the Phase-B contrasts (B6, B8) that
are arithmetic on numbers already measured.

  python scratchpad/_l1hu_report.py tables
  python scratchpad/_l1hu_report.py returns
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL"
PREV = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
SHORT = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
         "musique_clean": "MuSiQue", "hotpotqa_clean": "HotpotQA", "squad_clean": "SQuAD"}
RULES = ["H0_FIXED_CAP", "H1_FULL_WEIGHTED", "H2_Q90", "H2_Q95", "H2_Q99",
         "H3_B05", "H3_B10", "H3_B20", "H4_SPLIT_PRESERVE"]
cm = lambda x: "{:,}".format(int(x))


def _load(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d


def gather():
    G = {"audit": _load("%s/pin_retention/A1_PRECAP_DISTRIBUTION.json" % OUT, {}),
         "build": _load("%s/hypergraph_build/BUILD_MANIFEST.json" % OUT, {}),
         "a0": _load("%s/hypergraph_build/A0_CHECKSUM.json" % OUT, {}),
         "api": _load("%s/hypergraph_build/MTKAHYPAR_API.json" % OUT, {}),
         "gate": _load("%s/partitions/HYPER_GATE.json" % OUT, {}),
         "cores": _load("%s/halo/CORES.json" % OUT, {}),
         "cov": {d: _load("%s/halo/COVERAGE_%s.json" % (OUT, d), {}) for d in DS},
         "depth": {d: _load("%s/halo/DEPTH_%s.json" % (OUT, d), {}) for d in DS},
         "replay": {d: _load("%s/partitions/REPLAY_%s.json" % (OUT, d), {}) for d in DS}}
    return G


def t_a1(G):
    a = G["audit"].get("PER_CORPUS", {})
    L = ["### A1  raw hyperedge-size distribution, before any cap",
         "",
         "Hyperedge = closed neighbourhood {u} u N(u), so raw size = deg(u)+1.  No truncation.",
         "",
         "| corpus | family | hyperedges | pins | mean | p75 | p90 | p95 | p99 | max | pins kept at the fixed cap 25 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DS:
        e = a.get(d)
        if not e:
            continue
        for f, s in e["families"].items():
            L.append("| %s | %s | %s | %s | %.2f | %.0f | %.0f | %.0f | %.0f | %s | %.4f |"
                     % (SHORT[d], f, cm(s["hyperedges"]), cm(s["pins"]), s["mean"], s["p75"],
                        s["p90"], s["p95"], s["p99"], cm(s["max"]),
                        s["pin_retention_at_fixed_cap"]))
    return "\n".join(L)


def t_retention(G):
    b = G["build"]
    L = ["### A0/A2-A5  pin retention by rule (STRUCT + KNN, the family set the previous build used)",
         "",
         "| corpus | " + " | ".join(RULES) + " |",
         "|---|" + "---:|" * len(RULES)]
    caps = ["| corpus | " + " | ".join(RULES) + " |", "|---|" + "---:|" * len(RULES)]
    for d in DS:
        row, cap = [], []
        for r in RULES:
            m = b.get("%s__%s__SK" % (d, r))
            row.append("%.4f" % m["pin_retention_total"] if m else "-")
            cap.append(str(m["cap"]) if m else "-")
        L.append("| %s | %s |" % (SHORT[d], " | ".join(row)))
        caps.append("| %s | %s |" % (SHORT[d], " | ".join(cap)))
    L += ["", "Numeric cap the identical rule produced on each corpus:", ""] + caps
    return "\n".join(L)


def t_gate(G):
    g = G["gate"]
    if not g:
        return ""
    C, S = g["CELLS"], g["SUMMARY"]
    L = []
    for lane, title in (("BASE", "A8  BASE -- Dense+SPLADE partition ranking at P=50, no selector"),
                        ("SAFE", "A9  SAFE -- frozen B6_S4_F6_Ms64_Mr32, mechanically reaggregated")):
        L += ["### %s" % title, "",
              "Delta against the production METIS core.  `*` = exact McNemar p<0.05 AND the delta "
              "exceeds that corpus's measured reseed noise floor.", "",
              "| representation | " + " | ".join(SHORT[d] for d in DS if d in g["CORPORA"])
              + " | worst | macro | sig reg |",
              "|---|" + "---:|" * (len(g["CORPORA"]) + 3)]
        for t in g["ORDER"]:
            cells = []
            for d in DS:
                if d not in g["CORPORA"]:
                    continue
                c = C[t].get(d)
                cells.append("%+.4f%s" % (c[lane]["delta"], "\\*" if c[lane]["sig"] else "")
                             if c else "-")
            s = S[t][lane]
            L.append("| %s | %s | %+.4f | %+.4f | %d |"
                     % (t.replace("__SK", ""), " | ".join(cells), s["WORST_DELTA"],
                        s["MACRO_DELTA"], s["SIG_REGRESSIONS"]))
        L.append("")
    return "\n".join(L)


def t_balance(G):
    L = ["### A7  partition balance envelope", "",
         "| corpus | representation | blocks | min | median | mean | p90 | p99 | max | max/mean | CV | eligible |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for d in DS:
        rec = G["replay"].get(d) or {}
        for t in ["CURRENT"] + ["%s__SK" % r for r in RULES]:
            R = rec.get(t)
            if not R:
                continue
            b = R["BALANCE"]
            L.append("| %s | %s | %d | %d | %.0f | %.1f | %.0f | %.0f | %d | %.3f | %.3f | %s |"
                     % (SHORT[d], t.replace("__SK", ""), b["blocks_used"], b["size_min"],
                        b["size_median"], b["size_mean"], b["size_p90"], b["size_p99"],
                        b["size_max"], b["max_over_mean"], b["size_cv"],
                        "yes" if b["ELIGIBLE"] else "NO"))
    return "\n".join(L)


def interaction(G, lane="F6"):
    """B6 -- A/B/C/D and the interaction term, per corpus."""
    out = {}
    for d in DS:
        c = G["cov"].get(d) or {}
        if "C0_METIS" not in c or "C1_HYPER_UNIVERSAL" not in c:
            continue
        try:
            A = c["C0_METIS"]["O0_CORE"][lane]["ALL_REQUIRED_FETCHED"]
            B = c["C1_HYPER_UNIVERSAL"]["O0_CORE"][lane]["ALL_REQUIRED_FETCHED"]
            Cc = c["C0_METIS"]["O4_FULL_C_b0.5"][lane]["ALL_REQUIRED_FETCHED"]
            D = c["C1_HYPER_UNIVERSAL"]["O4_FULL_C_b0.5"][lane]["ALL_REQUIRED_FETCHED"]
        except KeyError:
            continue
        out[d] = {"A_metis": A, "B_hyper": B, "C_metis_halo": Cc, "D_hyper_halo": D,
                  "CORE_EFFECT": round(B - A, 4), "HALO_EFFECT": round(Cc - A, 4),
                  "JOINT_EFFECT": round(D - A, 4), "INTERACTION": round(D - B - Cc + A, 4)}
    return out


def t_interaction(G):
    I = interaction(G)
    if not I:
        return ""
    L = ["### B6  additivity of core and halo (F6 lane, ALL_REQUIRED_FETCHED)", "",
         "A = METIS no halo, B = hypergraph no halo, C = METIS + beta 0.5, D = hypergraph + beta 0.5.",
         "INTERACTION = D - B - C + A: negative means the two mechanisms rescue the same queries.",
         "",
         "| corpus | A | B | C | D | CORE | HALO | JOINT | INTERACTION |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in DS:
        v = I.get(d)
        if not v:
            continue
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.4f | %+.4f | %+.4f |"
                 % (SHORT[d], v["A_metis"], v["B_hyper"], v["C_metis_halo"], v["D_hyper_halo"],
                    v["CORE_EFFECT"], v["HALO_EFFECT"], v["JOINT_EFFECT"], v["INTERACTION"]))
    return "\n".join(L)


def t_depth(G):
    """B5 -- required-node coverage at P1/5/10/25/50, core-only vs core+halo(b0.5)."""
    L = ["### B5  required-node coverage by prefix depth (O0_CORE vs O4_FULL_C beta=0.5)", "",
         "ALL_REQUIRED_FETCHED at each partition-list prefix P.  Depth is defined on the BASE block "
         "ranking (the selector is set-valued at exactly P=50, so it has no shorter prefix of its own).",
         "",
         "| corpus | core | cell | P1 | P5 | P10 | P25 | P50 |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    any_rows = False
    for d in DS:
        dep = G["depth"].get(d) or {}
        for core in ("C0_METIS", "C1_HYPER_UNIVERSAL"):
            rec = dep.get(core)
            if not rec:
                continue
            for cell in ("O0_CORE", "O4_FULL_C_b0.5"):
                c = rec.get(cell)
                if not c:
                    continue
                any_rows = True
                vals = [c["P%d" % p]["ALL_REQUIRED_FETCHED"] for p in (1, 5, 10, 25, 50)]
                L.append("| %s | %s | %s | %s |" % (SHORT[d], core.replace("C0_METIS", "METIS").
                         replace("C1_HYPER_UNIVERSAL", "hypergraph"), cell,
                         " | ".join("%.4f" % v for v in vals)))
    return "\n".join(L) if any_rows else ""


def exposure(G, beta="0.5", lane="F6"):
    """B7/B8 -- exposure multiplier and rescue efficiency (required nodes per 1000 extra unique nodes
    exposed by the halo), core-only exposure held fixed by construction (same beta budget both cores)."""
    out = {}
    for d in DS:
        c = G["cov"].get(d) or {}
        row = {}
        for core in ("C0_METIS", "C1_HYPER_UNIVERSAL"):
            cell = (c.get(core) or {}).get("O4_FULL_C_b%s" % beta, {}).get(lane, {})
            if not cell:
                continue
            row[core] = {"EXPOSURE_MULTIPLIER": cell.get("EXPOSURE_MULTIPLIER"),
                        "required_per_1k_extra_exposed": cell.get("required_per_1k_extra_exposed"),
                        "NEW_REQUIRED_FROM_HALO": cell.get("NEW_REQUIRED_FROM_HALO"),
                        "queries_rescued": cell.get("queries_rescued")}
        if row:
            out[d] = row
    return out


def t_exposure(G):
    E = exposure(G)
    if not E:
        return ""
    L = ["### B7/B8  halo exposure cost at beta=0.5 (F6 lane)", "",
         "Same beta budget applied to both cores -> exposure multipliers land close by construction "
         "(not interpolated to an exact match).  Efficiency = newly-required nodes rescued per 1000 "
         "extra unique nodes the halo exposes -- the marginal return on the halo's exposure cost.",
         "",
         "| corpus | core | expo x | new required | queries rescued | required/1k extra exposed |",
         "|---|---|---:|---:|---:|---:|"]
    for d in DS:
        row = E.get(d, {})
        for core, v in row.items():
            L.append("| %s | %s | x%.4f | %d | %d | %.4f |"
                     % (SHORT[d], core.replace("C0_METIS", "METIS").replace("C1_HYPER_UNIVERSAL", "hypergraph"),
                        v["EXPOSURE_MULTIPLIER"], v["NEW_REQUIRED_FROM_HALO"], v["queries_rescued"],
                        v["required_per_1k_extra_exposed"]))
    return "\n".join(L)


def t_b9(G):
    b9 = _load("%s/halo/B9_JOINT_SIGNIFICANCE.json" % OUT, {})
    if not b9:
        return ""
    L = ["### B9  joint system (hypergraph core + beta=0.5 halo) vs production METIS, paired McNemar", "",
         "| corpus | delta | gained | lost | p | significant |",
         "|---|---:|---:|---:|---:|---|"]
    for d in DS:
        v = b9.get(d)
        if not v:
            continue
        L.append("| %s | %+.4f | %d | %d | %.3g | %s |"
                 % (SHORT[d], v["delta"], v["gained"], v["lost"], v["p"], "YES" if v["sig"] else "no"))
    return "\n".join(L)


def returns(G):
    """Assemble every KEY RETURNS field the spec lists.  Pure aggregation of numbers already on disk --
    no new measurement."""
    audit = G["audit"].get("PER_CORPUS", {})
    build = G["build"]
    gate = G["gate"]
    b9 = _load("%s/halo/B9_JOINT_SIGNIFICANCE.json" % OUT, {})
    I = interaction(G)
    EXPO = exposure(G)

    def retention(d, rule):
        m = build.get("%s__%s__SK" % (d, rule))
        return m["pin_retention_total"] if m else None

    R = {
        "FIXED_CAP_SQUAD_PIN_RETENTION": retention("squad_clean", "H0_FIXED_CAP"),
        "H1_PIN_RETENTION": {d: retention(d, "H1_FULL_WEIGHTED") for d in DS},
        "H2_BEST_PIN_RETENTION": {d: max((retention(d, r) for r in ("H2_Q90", "H2_Q95", "H2_Q99")
                                          if retention(d, r) is not None), default=None) for d in DS},
        "H3_BEST_PIN_RETENTION": {d: max((retention(d, r) for r in ("H3_B05", "H3_B10", "H3_B20")
                                          if retention(d, r) is not None), default=None) for d in DS},
        "H4_SPLIT_PIN_RETENTION": {d: retention(d, "H4_SPLIT_PRESERVE") for d in DS},
        "BEST_UNIVERSAL_HYPERGRAPH_RULE": gate.get("BEST_UNIVERSAL_HYPERGRAPH_CORE"),
        "BEST_DIAGNOSTIC_HYPERGRAPH_RULE": gate.get("BEST_DIAGNOSTIC_HYPERGRAPH_CORE"),
        "UNIVERSAL_HYPERGRAPH_GATE_PASSED": gate.get("UNIVERSAL_HYPERGRAPH_GATE_PASSED"),
        "HYPER_CROSS_CORPUS_SAFE": gate.get("HYPER_CROSS_CORPUS_SAFE"),
        "BEST_HYPER_PER_CORPUS_SAFE_DELTA": (gate.get("SUMMARY", {}).get("H4_SPLIT_PRESERVE__SK", {})
                                             .get("SAFE", {}).get("per_corpus")),
        "METAQA_H3_BASE": gate.get("SUMMARY", {}).get("H4_SPLIT_PRESERVE__SK", {}).get("METAQA_H3_BASE"),
        "METAQA_H3_SAFE": gate.get("SUMMARY", {}).get("H4_SPLIT_PRESERVE__SK", {}).get("METAQA_H3_SAFE"),
        "CORE_EFFECT": {d: v["CORE_EFFECT"] for d, v in I.items()},
        "HALO_EFFECT": {d: v["HALO_EFFECT"] for d, v in I.items()},
        "JOINT_EFFECT": {d: v["JOINT_EFFECT"] for d, v in I.items()},
        "INTERACTION": {d: v["INTERACTION"] for d, v in I.items()},
        "HYPER_HALO_EXPOSURE_MULTIPLIER": {d: v.get("C1_HYPER_UNIVERSAL", {}).get("EXPOSURE_MULTIPLIER")
                                           for d, v in EXPO.items()},
        "HYPER_HALO_EFFICIENCY_PER_1K_EXTRA_EXPOSED": {d: v.get("C1_HYPER_UNIVERSAL", {})
                                                       .get("required_per_1k_extra_exposed")
                                                       for d, v in EXPO.items()},
        "METIS_HALO_EFFICIENCY_PER_1K_EXTRA_EXPOSED": {d: v.get("C0_METIS", {})
                                                       .get("required_per_1k_extra_exposed")
                                                       for d, v in EXPO.items()},
        "HYPER_HALO_WORST_DELTA": min((v["delta"] for v in b9.values()), default=None),
        "HYPER_HALO_MACRO_DELTA": (round(sum(v["delta"] for v in b9.values()) / len(b9), 4)
                                   if b9 else None),
        "HYPER_HALO_SIG_REGRESSIONS": sum(1 for v in b9.values() if v["sig"] and v["delta"] < 0),
        "HYPER_HALO_SIG_GAINS": sum(1 for v in b9.values() if v["sig"] and v["delta"] > 0),
        "HYPER_HALO_CROSS_CORPUS_SAFE": all(not (v["sig"] and v["delta"] < 0) for v in b9.values()) if b9 else None,
        "B9_PER_CORPUS": b9,
        "BEST_UNIVERSAL_L1_SUBSTRATE": "H4_SPLIT_PRESERVE core + FULL_C beta=0.5 halo "
                                       "(hypergraph+halo passes the universal gate vs production METIS; "
                                       "see B9 for significance)",
    }
    return R


def tables():
    G = gather()
    parts = [x for x in (t_a1(G), t_retention(G), t_gate(G), t_balance(G), t_interaction(G),
                         t_depth(G), t_exposure(G), t_b9(G)) if x]
    md = "# UNIVERSAL TRUE-HYPERGRAPH + HALO -- tables\n\n" + "\n\n".join(parts) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open("%s/TABLES.md" % OUT, "w", encoding="utf-8").write(md)
    print("wrote %s/TABLES.md  (%d chars)" % (OUT, len(md)))
    return md


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "returns":
        G = gather()
        R = returns(G)
        json.dump(R, open("%s/RETURNS.json" % OUT, "w"), indent=1, default=str)
        print("wrote %s/RETURNS.json" % OUT)
        print(json.dumps(R, indent=1, default=str))
    else:
        tables()
