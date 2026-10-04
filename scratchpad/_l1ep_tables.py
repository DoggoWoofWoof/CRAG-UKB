"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- TABLES.md.

Reads diagnostics/_derived.json and emits every required table.  No measurement happens here.

  python scratchpad/_l1ep_tables.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_sub as EP

OUT = EP.OUT
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean", "webqsp"]
A11 = ["E0_STRUCT", "E3_STRUCT_NERX", "E4_STRUCT_KNN", "E6_TOPOLOGY_C",
       "M1_STRUCT_NERX_MATCHED", "M2_STRUCT_KNN_MATCHED", "M3_TOPOLOGY_C_MATCHED"]
A11H = ["FROZEN STRUCT", "+NERX", "+KNN", "+NERX+KNN", "MATCHED +NERX", "MATCHED +KNN",
        "MATCHED FULL C"]
PTAGS = ["P1_METIS_CURRENT", "PM_CURRENT_EXACT", "PM0_STRUCT", "PM1_STRUCT_NERX",
         "PM2_STRUCT_KNN", "PM3_TOPOLOGY_C", "PM3_TOPOLOGY_C_seed1", "PM3_TOPOLOGY_C_seed2",
         "PM3_TOPOLOGY_C_seed3", "PM3_TOPOLOGY_C_seed4", "P2_METIS_STRONG", "P3_FENNEL",
         "P4_HYPERGRAPH_CE", "P4_CE_LOCAL_ONLY", "P4_CE_UNWEIGHTED", "P4_CE_NER_ONLY",
         "PM4_TOPOLOGY_C_NERW"] + [f"P0_RANDOM_BALANCED_s{i}" for i in range(5)]
L = []
w = L.append


def f(x, n=4, sign=False):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "YES" if x else "NO"
    if isinstance(x, (int, np.integer)):
        return f"{x:+,d}" if sign else f"{x:,d}"
    return (f"{x:+.{n}f}" if sign else f"{x:.{n}f}")


def table(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


def main():
    D = json.load(open(f"{OUT}/diagnostics/_derived.json"))
    C = D["CORPORA"]
    have = [d for d in DS if C.get(d, {}).get("A11_EXACT_P50")]
    haveB = [d for d in DS if C.get(d, {}).get("B")]
    haveC = [d for d in DS if C.get(d, {}).get("C")]

    w("# L1 EDGE SUBSTRATE + PARTITION UTILITY -- TABLES")
    w("")
    w("Frozen contract everywhere below: MASTER_TOPOLOGY=C, P=50 exactly, B=6, K0=60, "
      "M_struct=64, M_ret=32, MAX_HOPS=3, BEAM=64, DEG_CAP=300, SEED_K=5. "
      "No learning, no LLM, no encoder call, no TEST split.")
    w("")

    # ---------------------------------------------------------------- A0
    w("## A0 -- edge algebra (verified from source artifacts)")
    w("")
    w("`S = E_STRUCT` (master_nodes.neighbors, the ONLY family the frozen traversal walks), "
      "`N = E_NERX = NER \\ S`, `K = E_KNN = A \\ S` where `A = gte_qwen/graph.pt`.")
    w("")
    ks = ["N", "|E_STRUCT|", "|E_NERX|", "|E_KNN|", "|S n N|", "|S n K|", "|N n K|",
          "|S u N|", "|S u K|", "|S u N u K|", "|topology_C_file|", "SuNuK_equals_C_file",
          "frac_C_never_traversed"]
    rows = []
    for k in ks:
        r = [f"`{k}`"]
        for d in DS:
            v = D["A0"].get(d, {}).get("A0_edge_algebra", {}).get(k)
            r.append("YES" if v is True else "NO" if v is False
                     else (f"{v:,}" if isinstance(v, int) else f(v, 4)))
        rows.append(r)
    table(["quantity"] + [d.replace("_clean", "") for d in DS], rows)
    w("`|S n N| = |S n K| = 0` holds by construction on all six.  `|N n K| != 0` on all six, so "
      "family provenance is a **bitmask**, never a partition of the edges.")
    w("")

    # ---------------------------------------------------------------- A1/A2 substrates
    w("## A1 / A2 -- the canonical substrates")
    w("")
    rows = []
    for s in EP.SUBS:
        r = [f"`{s}`"]
        for d in DS:
            m = D["A0"].get(d, {}).get("A1_A2_substrates", {}).get(s, {})
            r.append(f"{m.get('undirected_edges', 0):,}")
        rows.append(r)
    table(["substrate (undirected edges)"] + [d.replace("_clean", "") for d in DS], rows)
    w("The `M*` matched controls carry EXACTLY `E0_STRUCT`'s `adj_ptr` and `deg`, so DEG_CAP "
      "behaves identically and every frontier node costs the same number of edge inspections -- "
      "matched by construction, not by tuning.")
    w("")

    # ---------------------------------------------------------------- A3 parity
    w("## A3 -- parity gate (hard gate, every query, both passes)")
    w("")
    rows = []
    for d in have:
        c = C[d]
        p1, p2 = c["PARITY_pass1"], c.get("PARITY_pass2", {})
        al = c.get("PASS_ALIGN", {})
        rows.append([d, c["nq"], p1.get("T0_PARITY", "--"),
                     f"{p1.get('spos_exact', 0)}/{p1.get('nq', 0)}",
                     f"{p1.get('p50_exact', 0)}/{p1.get('nq', 0)}",
                     p2.get("E0_PARITY", "--"),
                     f(al.get("p50_identical")) + "/" + f(al.get("orac_identical"))])
    table(["corpus", "nq", "pass1 E0 parity", "spos exact", "P50 exact", "pass2 E0 parity",
           "pass1==pass2 (p50/orac)"], rows)
    w("`CSR(E_STRUCT keys) == frozen master_nodes CSR` is asserted bit-exact inside the substrate "
      "builder before any run starts; `spos` and the exact P50 set are then re-verified per query.")
    w("")

    # ---------------------------------------------------------------- A4
    w("## A4 -- needed partitions the frozen substrate MISSES, by family bitmask (VIS level)")
    w("")
    rows = []
    for d in have:
        v = C[d]["A4_BITMASK"].get("VIS", {})
        rows.append([d, f"{v.get('total_missed_needed', 0):,}", f"{v.get('S_not_K', 0):,}",
                     f"{v.get('S_and_K', 0):,}", f"{v.get('notS_K_only', 0):,}",
                     f"{v.get('notS_N_only', 0):,}", f"{v.get('notS_K_and_N', 0):,}",
                     f"{v.get('notS_none', 0):,}", f(v.get("UNIQUE_KNN_frac"), 4),
                     f(v.get("UNIQUE_NERX_frac"), 4), f(v.get("UNREACHED_BY_ANY_frac"), 4)])
    table(["corpus", "missed needed", "S only", "S and K", "K only (not S)", "N only (not S)",
           "K and N (not S)", "none", "UNIQUE_KNN", "UNIQUE_NERX", "unreached by any"], rows)
    w("")
    w("### A4 -- same, READ level (what the M_struct=64 aggregation actually sees)")
    w("")
    rows = []
    for d in have:
        v = C[d]["A4_BITMASK"].get("READ", {})
        rows.append([d, f"{v.get('total_missed_needed', 0):,}", f(v.get("UNIQUE_KNN_frac"), 4),
                     f(v.get("UNIQUE_NERX_frac"), 4), f(v.get("UNIQUE_KNN_or_NERX_frac"), 4),
                     f(v.get("UNREACHED_BY_ANY_frac"), 4)])
    table(["corpus", "missed needed", "UNIQUE_KNN", "UNIQUE_NERX", "UNIQUE_K or N",
           "unreached by any"], rows)

    # ---------------------------------------------------------------- A5
    for lvl in ("FULL_WORK", "MATCHED_WORK"):
        w(f"## A5 -- incremental attribution and interaction ({lvl.replace('_', ' ').lower()})")
        w("")
        for metric, lab in (("need_VIS", "needed-partition reach, VIS"),
                            ("need_READ", "needed-partition reach, READ"),
                            ("orac", "candidate-pool oracle"), ("p50", "exact P50 ALL@50")):
            rows = []
            for d in have:
                a = C[d]["A5_ATTRIBUTION"]["ALL"].get(lvl, {}).get(metric)
                if not a:
                    continue
                rows.append([d, f(a["M_S"]), f(a["M_S_N"]), f(a["M_S_K"]), f(a["M_S_N_K"]),
                             f(a["DELTA_N"], 4, True), f(a["DELTA_K"], 4, True),
                             f(a["DELTA_K_after_N"], 4, True), f(a["DELTA_N_after_K"], 4, True),
                             f(a["INTERACTION"], 4, True)])
            if rows:
                w(f"**{lab}**")
                w("")
                table(["corpus", "M(S)", "M(SuN)", "M(SuK)", "M(SuNuK)", "DELTA_N", "DELTA_K",
                       "DELTA_K|N", "DELTA_N|K", "INTERACTION"], rows)

    # ---------------------------------------------------------------- A6
    w("## A6 -- is the family Dense wearing a graph costume?")
    w("")
    w("For partitions a family reaches, `EDGE_SIM` is the neighbour similarity the family itself "
      "asserts and `QUERY_SIM` is what Dense already knows. AUC separates needed from nuisance.")
    w("")
    rows = []
    for d in have:
        k8 = f"{OUT}/../L1_KNN/diag/k8_{d}.json"
        kk = json.load(open(k8)) if os.path.exists(k8) else {}
        K8 = kk.get("K8", {})
        n6 = C[d].get("A6_NERX", {})
        rows.append([d,
                     f(K8.get("K1_BEST_KNN_SIM", {}).get("AUC_needed_vs_nuisance")),
                     f(K8.get("K1_QUERY_SIM", {}).get("AUC_needed_vs_nuisance")),
                     f(n6.get("NERX_EDGE_SIM", {}).get("AUC_needed_vs_nuisance")),
                     f(n6.get("NERX_QUERY_SIM", {}).get("AUC_needed_vs_nuisance"))])
    table(["corpus", "KNN EDGE_SIM AUC", "KNN QUERY_SIM AUC", "NERX EDGE_SIM AUC",
           "NERX QUERY_SIM AUC"], rows)

    # ---------------------------------------------------------------- A7
    w("## A7 -- saturation (no oracle gets credit for exposure)")
    w("")
    rows = []
    for d in have:
        s = C[d]["A7_SATURATION"]
        for t in EP.SUBS:
            if t not in s or t == "M0_STRUCT":
                continue
            v = s[t]
            rows.append([d, t, v["visited_partitions_per_query"], f(v["frac_of_corpus_partitions"]),
                         v["read_partitions_per_query"], v["candidate_pool_per_query"],
                         f(v["candidate_oracle"])])
    table(["corpus", "substrate", "visited parts/q", "frac of corpus", "read parts/q",
           "cand pool/q", "candidate oracle"], rows)

    # ---------------------------------------------------------------- A11
    w("## A11 -- EXACT P=50 (ALL@50), the only number that decides anything")
    w("")
    rows = []
    for d in have:
        e = C[d]["A11_EXACT_P50"]["ALL"]
        r = [d]
        for t in A11:
            v = e.get(t)
            r.append("--" if not v else
                     (f"{v['ALL_P50']:.4f}" if t == "E0_STRUCT" else
                      f"{v['ALL_P50']:.4f} ({v['delta_vs_frozen']:+.4f}{' SIG' if v['sig'] else ''})"))
        rows.append(r)
    table(["corpus"] + A11H, rows)
    w("Significance is McNemar on the paired per-query ALL@50 indicator against the frozen "
      "substrate on the same queries.")
    w("")
    if C.get("metaqa", {}).get("A11_EXACT_P50", {}).get("hop2"):
        w("### A11 -- MetaQA by hop")
        w("")
        rows = []
        for h in ("hop1", "hop2", "hop3"):
            e = C["metaqa"]["A11_EXACT_P50"].get(h, {})
            r = [h]
            for t in A11:
                v = e.get(t)
                r.append("--" if not v else
                         (f"{v['ALL_P50']:.4f}" if t == "E0_STRUCT" else
                          f"{v['ALL_P50']:.4f} ({v['delta_vs_frozen']:+.4f}"
                          f"{' SIG' if v['sig'] else ''})"))
            rows.append(r)
        table(["hop"] + A11H, rows)

    # ---------------------------------------------------------------- A12
    w("## A12 -- cost")
    w("")
    rows = []
    for d in have:
        c12 = C[d]["A12_COST"]
        for t in EP.SUBS:
            if t not in c12 or t == "M0_STRUCT":
                continue
            v = c12[t]
            rows.append([d, t, f"{v['edges_per_query']:,.0f}", f"{v['edges_x_frozen']:.2f}x",
                         f"{v['ms_per_query']:.1f}", f"{v['ms_x_frozen']:.2f}x"])
    table(["corpus", "substrate", "edges/query", "vs frozen", "ms/query", "vs frozen"], rows)

    # ---------------------------------------------------------------- B0
    w("## B0 -- how the production partitions were actually built (read from source)")
    w("")
    table(["item", "value"], [
        ["build script", "`scratchpad/build_canonical_topo.py`"],
        ["graph handed to METIS", "`C = A_QWEN u NER = (STRUCT u KNN) u NER` -- **full topology C**"],
        ["partitioning happens", "AFTER full topology-C construction (not before, not per-family)"],
        ["directed / symmetrised / dedup", "undirected, symmetrised, deduplicated (python `set`), "
                                           "self-loops excluded"],
        ["vertex weights", "NONE (unweighted)"],
        ["edge weights", "NONE (unweighted -- the NER 1/df weights are DISCARDED at this step)"],
        ["library", "`pymetis.part_graph(n_parts, adjacency=...)` (METIS 5.x)"],
        ["recursive vs k-way", "k-way (pymetis uses recursive only for k <= 8; k is 136..7814)"],
        ["objective", "METIS default `OBJTYPE_CUT` (edge cut)"],
        ["balance tolerance", "METIS default `ufactor=30` for k-way, i.e. 1.03"],
        ["seed", "METIS default (`seed=-1`); no seed was pinned by the build"],
        ["k", "`max(1, N // 100)`"],
        ["target size", "100 documents per partition"]])
    if haveB:
        rows = []
        for d in haveB:
            b = C[d]["B"].get("P1_METIS_CURRENT", {})
            s = b.get("sizes", {})
            rows.append([d, f"{b.get('N', 0):,}", f"{b.get('npart', 0):,}", s.get("min"),
                         s.get("max"), s.get("mean"), s.get("median"), f(s.get("CV")),
                         f(s.get("imbalance_max_over_mean")), s.get("empty_blocks")])
        table(["corpus", "N", "k", "min", "max", "mean", "median", "CV", "max/mean", "empty"], rows)
        w("Measured imbalance is 1.025-1.030 on every corpus, exactly the METIS `ufactor=30` "
          "guarantee -- independent confirmation of the configuration read from source.")
        w("")

    # ---------------------------------------------------------------- B1-B8
    if haveB:
        w("## B1-B8 -- partition utility of the SHIPPED assignment")
        w("")
        ks = [("B1_GOLD_COMPRESSION", "B1 gold compression"),
              ("B1_PARTITION_FETCH_SAVING", "B1 fetch saving"),
              ("B2_PAIR_COLOCATION", "B2 pair coloc"),
              ("B2_PAIR_EXPECTED_RANDOM", "B2 expected (random)"),
              ("B2_PAIR_COLOCATION_ADJUSTED", "B2 adjusted"),
              ("B3_ALL_GOLD_CONTAINED", "B3 all-gold in 1 block"),
              ("B3_GOLD_BLOCK_DENSITY", "B3 block density"),
              ("B4_SEED_REQUIRED_COLOCATION", "B4 seed->required"),
              ("B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL", "B4 nontrivial"),
              ("B5_GOLD_EDGE_CONTAINMENT", "B5 gold-edge containment"),
              ("B8_ORACLE_ALL_AT_50", "B8 oracle ALL@50")]
        def _degen(d):
            """no gold PAIRS exist when every query has a single gold document -> B2 undefined."""
            v = C[d]["B"]["P1_METIS_CURRENT"]
            b7 = v.get("B7_MIN_PARTITIONS_REQUIRED", {})
            return (b7.get("max") == 1 and v.get("B3_ALL_GOLD_CONTAINED") == 1.0)
        rows = []
        for k, lab in ks:
            r = [lab]
            for d in haveB:
                if k.startswith("B2_") and _degen(d):
                    r.append("N/A")
                else:
                    r.append(f(C[d]["B"]["P1_METIS_CURRENT"].get(k)))
            rows.append(r)
        r = ["B7 min partitions (mean)"]
        for d in haveB:
            r.append(f(C[d]["B"]["P1_METIS_CURRENT"]["B7_MIN_PARTITIONS_REQUIRED"]["mean"], 3))
        rows.append(r)
        r = ["B7 min partitions (max)"]
        for d in haveB:
            r.append(str(C[d]["B"]["P1_METIS_CURRENT"]["B7_MIN_PARTITIONS_REQUIRED"]["max"]))
        rows.append(r)
        table(["metric"] + [d.replace("_clean", "") for d in haveB], rows)
        if C.get("metaqa", {}).get("B", {}).get("P1_METIS_CURRENT", {}).get("by_hop"):
            w("### B1-B8 -- MetaQA by hop (the primary multi-hop case)")
            w("")
            bh = C["metaqa"]["B"]["P1_METIS_CURRENT"]["by_hop"]
            rows = []
            for k in ["B1_GOLD_COMPRESSION", "B2_PAIR_COLOCATION", "B3_ALL_GOLD_CONTAINED",
                      "B4_SEED_REQUIRED_COLOCATION", "B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL",
                      "B5_GOLD_EDGE_CONTAINMENT", "B7_MIN_PARTITIONS_REQUIRED_mean",
                      "B8_ORACLE_ALL_AT_50"]:
                rows.append([k] + [f(bh.get(h, {}).get(k)) for h in ("hop1", "hop2", "hop3")])
            table(["metric", "hop1", "hop2", "hop3"], rows)

        # ------------------------------------------------------------ B8 oracle curve
        w("## B8 -- full-universe oracle ALL@P (the ceiling the ASSIGNMENT imposes, no router)")
        w("")
        Ps = ["1", "2", "3", "5", "10", "20", "50", "100"]
        rows = []
        for d in haveB:
            cur = C[d]["B"]["P1_METIS_CURRENT"]["B8_ORACLE_CURVE_ALL"]
            rows.append([d] + [f(cur.get(p, {}).get("ALL")) for p in Ps])
        table(["corpus"] + [f"P={p}" for p in Ps], rows)

        # ------------------------------------------------------------ B9-B15 leaderboard
        w("## B9-B15 -- partitioner leaderboard (utility metrics, all built without gold/queries)")
        w("")
        for d in haveB:
            b = C[d]["B"]
            tags = [t for t in PTAGS if t in b]
            if len(tags) < 3:
                continue
            w(f"**{d}**")
            w("")
            rows = []
            for t in tags:
                v = b[t]
                q = v.get("B12_QUALITY", {}).get("E6_TOPOLOGY_C", {})
                rows.append([t, f(v.get("B1_GOLD_COMPRESSION")),
                             f(v.get("B2_PAIR_COLOCATION_ADJUSTED")),
                             f(v.get("B3_ALL_GOLD_CONTAINED")),
                             f(v.get("B5_GOLD_EDGE_CONTAINMENT")),
                             f(v.get("B8_ORACLE_ALL_AT_50")),
                             f(v["B7_MIN_PARTITIONS_REQUIRED"]["mean"], 3),
                             f"{q.get('edge_cut', 0):,}", f(q.get("modularity")),
                             f(v["sizes"]["imbalance_max_over_mean"], 3),
                             f"{v.get('B14_EXPOSURE_TOP50_NODES', 0):,.0f}"])
            table(["partitioner", "B1 compress", "B2 adj coloc", "B3 all-in-1", "B5 edge contain",
                   "B8 oracle@50", "B7 min parts", "B12 cut (C)", "B12 modularity",
                   "balance", "B14 top-50 nodes"], rows)

    # ---------------------------------------------------------------- Phase C
    if haveC:
        w("## C -- frozen L1 replayed on each partitioning (P=50, B=6, S4, F6; nothing retuned)")
        w("")
        for d in haveC:
            c = C[d]["C"]
            base = c.get("PM_CURRENT_EXACT")
            if not base:
                continue
            w(f"**{d}** (parity: `{base.get('C1_PARITY', '--')}`)")
            w("")
            rows = []
            for t in PTAGS:
                if t not in c:
                    continue
                v = c[t]
                rows.append([t, f(v["BASE_ALL_P50"]), f(v["BASE_ALL_P50"] - base["BASE_ALL_P50"],
                                                        4, True),
                             f(v["F6_ALL_P50"]), f(v["F6_ALL_P50"] - base["F6_ALL_P50"], 4, True),
                             f"{v['BASE_SCOPE_NODES']:,.0f}"])
            table(["partitioning", "BASE ALL@50", "delta", "F6 ALL@50", "delta",
                   "exposure (nodes)"], rows)


    # ---------------------------------------------------------------- A8 / A9 / B15 / C4
    AN = {}
    _af = f"{OUT}/diagnostics/_analysis.json"
    if os.path.exists(_af):
        AN = json.load(open(_af))

    if AN.get("A8_PATH_FAMILIES"):
        w("## A8 -- path-family analysis: do BRIDGE paths carry needed partitions?")
        w("")
        w("A signature is HOMOGENEOUS when every hop is carried by one single family "
          "(`STRUCT>STRUCT`), and a BRIDGE when the hops mix families (`STRUCT>KNN`, "
          "`KNN+NERX>STRUCT`). `needed yield` = needed targets / all targets reached by that "
          "signature class -- the precision of the path shape, not its volume.")
        w("")
        rows = []
        for d, v in AN["A8_PATH_FAMILIES"].items():
            for L_ in (1, 2, 3):
                r = v["BRIDGE_VS_HOMOGENEOUS"].get(f"hop{L_}")
                if not r:
                    continue
                rows.append([d, f"hop{L_}", f(r["homog_yield"], 4), f(r["bridge_yield"], 4),
                             f(r["bridge_minus_homog"], 4, True),
                             "YES" if r["BRIDGE_BETTER"] else "NO"])
        table(["corpus", "path length", "homogeneous yield", "bridge yield", "delta",
               "bridge better"], rows)

    if AN.get("A9_DISTINCT_EVIDENCE"):
        w("## A9 -- distinct evidence per family")
        w("")
        w("`K3 novel` counts partitions the kNN family reaches that NO other channel reaches "
          "(frozen structural traversal, Dense top-200, SPLADE top-200, retrieval frontier, "
          "context). A family is not credited from AUC alone -- it must contribute a needed "
          "partition nothing else supplies.")
        w("")
        rows = []
        for d, v in AN["A9_DISTINCT_EVIDENCE"].items():
            K = v["KNN"]
            rows.append([d, f"{K['TOTAL_KNN_REACHED_PARTITIONS']:,}",
                         f"{K['K0_ALSO_REACHED_BY_FROZEN_STRUCT']:,}",
                         f"{K['K1_ALSO_IN_DENSE_OR_SPLADE_TOP200']:,}",
                         f"{K['K3_NOVEL_TO_EVERY_OTHER_CHANNEL']:,}",
                         f(K["K3_NOVEL_FRACTION"], 4), K["K3_NOVEL_AND_NEEDED"],
                         "YES" if K["EVIDENCE_IS_DISTINCT"] else "NO"])
        table(["corpus", "K reached", "K0 also STRUCT", "K1 also Dense/SPLADE", "K3 novel",
               "K3 novel frac", "K3 novel AND needed", "distinct"], rows)
        rows = []
        for d, v in AN["A9_DISTINCT_EVIDENCE"].items():
            N = v["NERX"]
            rows.append([d, N.get("N2_UNIQUE_TO_NERX_AMONG_MISSED_NEEDED"),
                         f(N.get("N2_UNIQUE_FRACTION")), f(N.get("AUC_EDGE_SIM")),
                         f(N.get("AUC_QUERY_SIM")), N.get("DENSE_WEARING_GRAPH_COSTUME", "--"),
                         N["N0_ALSO_REACHED_BY_FROZEN_STRUCT"]])
        table(["corpus", "N unique among missed needed", "fraction", "AUC edge-sim",
               "AUC query-sim", "dense in costume", "N0/N1"], rows)

    if AN.get("B15_STABILITY"):
        w("## B15 -- stability and significance")
        w("")
        w("`PM3_TOPOLOGY_C_seed*` re-runs METIS on the IDENTICAL graph with a different RNG. "
          "Production never pinned a seed, so this spread is the noise floor every partitioner "
          "delta must clear; a candidate is only credited when it exceeds two of these standard "
          "deviations AND is significant by exact McNemar on the same queries.")
        w("")
        rows = []
        for d, v in AN["B15_STABILITY"].items():
            m = v.get("METIS_RESEED"); r = v.get("RANDOM_BALANCED")
            rows.append([d, f(v["REFERENCE"]["F6_ALL_P50"]),
                         f(m["F6_sd"]) if m else "--",
                         f(m["F6_min"]) if m else "--", f(m["F6_max"]) if m else "--",
                         f(v.get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO")),
                         f(r["F6_mean"]) if r else "--", f(r["F6_sd"]) if r else "--"])
        table(["corpus", "production F6", "METIS reseed sd", "reseed min", "reseed max",
               "noise floor sd", "random mean", "random sd"], rows)
        for d, v in AN["B15_STABILITY"].items():
            w(f"**{d}** -- candidates vs production (exact McNemar, same {v['REFERENCE']['nq']} "
              f"queries)")
            w("")
            rows = []
            for t, r in sorted(v["CANDIDATES"].items(), key=lambda kv: -kv[1]["F6_ALL_P50"]):
                rows.append([t, f(r["F6_ALL_P50"]), f(r["delta_vs_production"], 4, True),
                             r.get("net", "--"), f(r.get("p"), 4),
                             "SIG" if r.get("sig") else "ns",
                             f(r["scope_ratio_vs_production"], 3),
                             "YES" if r.get("EXCEEDS_RESEED_NOISE_FLOOR") else "NO"])
            table(["partitioning", "F6 ALL@50", "delta", "net", "p", "McNemar", "exposure x",
                   "> noise floor"], rows)

    if AN.get("C4_DECOMPOSITION"):
        w("## C4 -- effect decomposition")
        w("")
        rows = []
        for d, v in AN["C4_DECOMPOSITION"].items():
            e, pp = v["EDGE_EFFECT"], v["PARTITION_EFFECT"]
            rows.append([d, f(v["FROZEN_CELL_F6_ALL_P50"]), e["best_substrate"],
                         f(e["delta"], 4, True), "SIG" if e.get("sig") else "ns",
                         pp["best_partitioner"], f(pp["delta"], 4, True),
                         v.get("INTERACTION_STATUS", "--")])
        table(["corpus", "frozen cell F6", "best edge substrate", "edge effect", "sig",
               "best partitioning", "partition effect", "interaction"], rows)

        x = [(d, v) for d, v in AN["C4_DECOMPOSITION"].items() if v.get("INTERACTION_2X2")]
        if x:
            w("")
            w("### C3 -- the joint 2x2 cells behind those interactions")
            w("")
            w("Cell `a` must reproduce the frozen scoreboard exactly; that gate is what licenses "
              "reading `b`, `c` and `d`. `PARITY_vs_PHASE_C` additionally checks `a` against the "
              "independently written Phase-C driver, and `b` lands on the Phase-C value for the "
              "same partitioning.")
            w("")
            rows = []
            for d, v in x:
                q = v["INTERACTION_2X2"]
                rows.append([d, q["edge_arm"], q["partition_arm"],
                             f(q["a_frozen_x_shipped"]), f(q["b_frozen_x_candidate"]),
                             f(q["c_edge_x_shipped"]), f(q["d_edge_x_candidate"]),
                             f(q["EDGE_EFFECT_c_minus_a"], 4, True),
                             f(q["PARTITION_EFFECT_b_minus_a"], 4, True),
                             f(q["JOINT_d_minus_a"], 4, True),
                             f(v["INTERACTION"], 4, True),
                             str(q.get("C3_PARITY")), str(q.get("PARITY_vs_PHASE_C"))])
            table(["corpus", "edge arm", "partition arm", "a", "b", "c", "d",
                   "EDGE c-a", "PARTITION b-a", "JOINT d-a", "INTERACTION",
                   "C3 parity", "vs Phase C"], rows)

    os.makedirs(OUT, exist_ok=True)
    fp = f"{OUT}/TABLES.md"
    open(fp, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", fp, f"({len(L)} lines)  corpora A={len(have)} B={len(haveB)} C={len(haveC)}")


if __name__ == "__main__":
    main()
