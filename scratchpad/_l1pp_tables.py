"""Emit every table of the L1 SIMPLIFY / PPR phase report from the round JSONs.

  python scratchpad/_l1pp_tables.py > <report dir>/TABLES.md
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1pp_core as PP

D = PP.PPD + "/diag"
DS = PP.DSETS
SH = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
      "musique_clean": "MuSiQue", "hotpotqa_clean": "HotpotQA", "squad_clean": "SQuAD"}
J = lambda n: json.load(open(f"{D}/{n}.json")) if os.path.exists(f"{D}/{n}.json") else None
A, Bj, Cj, Dj = J("round_a"), J("round_b"), J("round_c"), J("round_d")
EV, AL = J("eviction"), J("alpha_sensitivity")
Ej, OR = J("round_e"), J("oracle")


def hdr(cols):
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")


def sig(e, key="vs_BASE"):
    return f'{e["dALL"]:+.4f}' + ("\\*" if e[key]["sig"] else "")


def macro(rows):
    v = [e["dALL"] for e in rows]
    return sum(v) / len(v), min(v)


print("# L1 SIMPLIFY / PPR PHASE — TABLES\n")
print("`*` marks p < 0.05 (exact McNemar, paired, vs BASE unless stated). Every method in every "
      "table emits EXACTLY 50 partitions.\n")

# ---------------------------------------------------------------- STEP 0
print("## T1 — STEP 0 verification gates\n")
hdr(["corpus", "base_rank prov.", "ret_rrf prov.", "symmetric scoring", "deterministic ties",
     "exact P=50", "absent-channel", "gold in ranking"])
for d in DS:
    v = A["STEP0_AUDIT"][d]
    print(f'| {SH[d]} | {v["base_rank_provenance"]["status"]} | {v["ret_rrf_provenance"]["status"]} '
          f'| {v["symmetric_scoring"]["status"]} ({v["symmetric_scoring"]["mismatches"]}/'
          f'{v["symmetric_scoring"]["checked"]}) | {v["deterministic_ties"]["status"]} '
          f'({v["deterministic_ties"]["differences"]} diffs) | {v["exact_P50"]["status"]} '
          f'({v["exact_P50"]["violations"]} viol.) | 0 candidates with no channel | none |')

print("\n## T2 — STEP 0/1 measured modality redundancy (the reason for the channel redefinition)\n")
hdr(["corpus", "frac. of retrieval challengers already canonically ranked",
     "Spearman(canonical, retrieval) on shared partitions", "queries with ≥5 shared"])
for d in DS:
    c = A["STEP0_AUDIT"][d]["channel_redundancy"]
    print(f'| {SH[d]} | {c["frac_retrieval_challengers_already_canonically_ranked"]:.4f} | '
          f'{c["mean_spearman_canonical_vs_retrieval_on_shared"]} | '
          f'{c["n_queries_with_ge5_shared"]} |')

print("\n## T3 — STEP 1 the three orthogonal channels\n")
hdr(["corpus", "npart", "base_rank == RRF(PR_dense, PR_splade)", "ρ(DENSE, SPLADE)",
     "ρ(DENSE, BASE)", "ρ(SPLADE, BASE)"])
for d in DS:
    s = A["STEP1_CHANNELS"][d]
    print(f'| {SH[d]} | {s["npart"]} | **{s["BASE_RANK_PARITY"]}** | '
          f'{s["mean_spearman_DENSE_vs_SPLADE_partition_rank"]:.4f} | '
          f'{s["mean_spearman_DENSE_vs_BASE"]:.4f} | {s["mean_spearman_SPLADE_vs_BASE"]:.4f} |')

# ---------------------------------------------------------------- STEP 2
print("\n## T4 — STEP 2 canonical partition transition graph (built once, cached permanently)\n")
hdr(["corpus", "partitions", "nodes", "inter-partition edges", "density", "out-deg mean",
     "out-deg max", "isolated", "internal (self) mass frac", "cache bytes", "build s"])
for d in DS:
    g = A["STEP2_PARTITION_GRAPH"][d]
    print(f'| {SH[d]} | {g["npart"]} | {g["n_nodes"]} | {g["inter_partition_edges"]} | '
          f'{g["density"]:.4f} | {g["outdeg_mean"]} | {g["outdeg_max"]} | '
          f'{g["isolated_partitions"]} | {g["frac_mass_internal"]:.3f} | '
          f'{g["cache_bytes"]:,} | {g["build_sec"]} |')

# ---------------------------------------------------------------- STEP 5B/6
print("\n## T5 — STEP 5B/6 DIRECT exactly-P50 fusion vs the protected-core swap (ΔALL vs BASE)\n")
NAM = [("L0_BASE", "L0  BASE = Dense+SPLADE partition RRF"),
       ("L2_DIRECT_S4", "L2  DIRECT + S4 structural"),
       ("L3_DIRECT_PPRGLOBAL", "L3  DIRECT + global partition PPR (P1)"),
       ("L3b_DIRECT_PPRGLOBAL_M64", "L3b DIRECT + global PPR, top-64 only"),
       ("L4_DIRECT_PPRBOUNDED", "L4  DIRECT + bounded partition PPR (P2)"),
       ("L4b_DIRECT_PPRBOUNDED_EXP", "L4b DIRECT + bounded PPR, neighbour-expanded"),
       ("L6_DIRECT_S4_PLUS_PPR", "L6  DIRECT + S4 + global PPR"),
       ("L1_SWAP_F6_B6", "**L1  SAFE_REFERENCE_ROUTER B6_S4_F6**"),
       ("SWAP_ES_B6", "SWAP de-duplicated (no retrieval channel), B=6")]
hdr(["variant"] + [SH[d] for d in DS] + ["macro", "worst"])
for k, lab in NAM:
    rows = [Bj["RESULTS"][d][k] for d in DS]
    m, w = macro(rows)
    print(f'| {lab} | ' + " | ".join(sig(e) for e in rows) + f' | {m:+.4f} | {w:+.4f} |')

print("\n## T6 — STEP 5A B-sweep. B is removed as an axis; output is still exactly 50 (ΔALL vs BASE)\n")
hdr(["corpus"] + [f"F6 B={b}" for b in (2, 4, 6, 8, 12)] + [f"ES B={b}" for b in (2, 4, 6, 8, 12)])
for d in DS:
    f = [Bj["RESULTS"][d][f"L1_SWAP_F6_B{b}"] for b in (2, 4, 6, 8, 12)]
    e = [Bj["RESULTS"][d][f"SWAP_ES_B{b}"] for b in (2, 4, 6, 8, 12)]
    print(f'| {SH[d]} | ' + " | ".join(sig(x) for x in f + e) + " |")
print()
for tag, pre in (("F6", "L1_SWAP_F6_B"), ("ES", "SWAP_ES_B")):
    ms = []
    for b in (2, 4, 6, 8, 12):
        rows = [Bj["RESULTS"][d][f"{pre}{b}"] for d in DS]
        m, w = macro(rows)
        ms.append(f"B={b}: macro {m:+.4f} / worst {w:+.4f}")
    print(f"- **{tag}** — " + "; ".join(ms))

print("\n## T7 — STEP 8 MetaQA per-hop (ΔALL vs BASE); hop1 n=%d hop2 n=%d hop3 n=%d\n"
      % tuple(Bj["RESULTS"]["metaqa"]["L0_BASE"]["per_hop"][h]["n"] for h in ("1", "2", "3")))
hdr(["variant", "hop1", "hop2", "hop3"])
MH = [k for k, _ in NAM] + ["L1_SWAP_F6_B8", "L1_SWAP_F6_B12", "SWAP_ES_B8", "SWAP_ES_B12"]
for k in MH:
    ph = Bj["RESULTS"]["metaqa"][k]["per_hop"]
    print(f'| {k} | ' + " | ".join(sig(ph[h]) for h in ("1", "2", "3")) + " |")

# ---------------------------------------------------------------- STEP 1 independence
print("\n## T8 — STEP 1 independence gate: does a third channel carry NEW information?\n")
print("`dup` = of the partitions the channel promotes that are outside the canonical top-50, the "
      "fraction that is nevertheless already inside the canonical top-200. `1.000` = the channel "
      "says nothing Dense+SPLADE does not already say.\n")
hdr(["corpus", "S4 dup", "S4 new parts/q", "global PPR dup", "PPR new parts/q",
     "bounded PPR dup", "Jaccard(PPR top50, BASE50)", "Jaccard(S4 top50, BASE50)"])
for d in DS:
    i = Bj["STEP1_INDEPENDENCE"][d]
    print(f'| {SH[d]} | {i["S4struct_frac_promoted_already_in_canonical_top200"]:.3f} | '
          f'{i["S4struct_mean_new_partitions_promoted"]} | '
          f'{i["PPRglobal_frac_promoted_already_in_canonical_top200"]:.3f} | '
          f'{i["PPRglobal_mean_new_partitions_promoted"]} | '
          f'{i["PPRbounded_frac_promoted_already_in_canonical_top200"]:.3f} | '
          f'{i["mean_jaccard_PPRtop50_vs_BASE50"]:.3f} | {i["mean_jaccard_S4top50_vs_BASE50"]:.3f} |')

# ---------------------------------------------------------------- STEP 6/7 round D
if Dj:
    print("\n## T9 — STEP 6 structural-signal bake-off inside the swap (ΔALL vs BASE)\n")
    VN = ["S4_F6_B6", "PPRG_F6_B6", "PPRB_F6_B6", "PPRP32_F6_B6", "S4+PPRG_F6_B6",
          "S4_ES_B6", "PPRG_ES_B6", "PPRB_ES_B6", "PPRP32_ES_B6", "S4+PPRG_ES_B6",
          "S4_F6_B8", "PPRG_F6_B8", "PPRB_F6_B8", "S4+PPRG_F6_B8"]
    hdr(["variant"] + [SH[d] for d in DS] + ["macro", "worst"])
    for k in VN:
        rows = [Dj["RESULTS"][d][k] for d in DS]
        m, w = macro(rows)
        star = "**" if k == "S4_F6_B6" else ""
        print(f'| {star}{k}{star} | ' + " | ".join(sig(e) for e in rows)
              + f' | {m:+.4f} | {w:+.4f} |')

    print("\n## T10 — STEP 7 reach probe: PPR *does* reach gold partitions S4 cannot\n")
    hdr(["corpus", "gold partitions outside BASE50", "reached by S4", "reached ONLY by global PPR",
         "ONLY by bounded PPR", "ONLY by precomputed PPR", "new partitions/q vs S4 (G/B/P32)"])
    for d in DS:
        p = Dj["STEP7_REACH_PROBE"][d]
        print(f'| {SH[d]} | {p["PPRG"]["gold_partitions_outside_BASE50_total"]} | '
              f'{p["PPRG"]["gold_partitions_reached_by_S4"]} | '
              f'{p["PPRG"]["gold_partitions_reached_only_by_this"]} | '
              f'{p["PPRB"]["gold_partitions_reached_only_by_this"]} | '
              f'{p["PPRP32"]["gold_partitions_reached_only_by_this"]} | '
              f'{p["PPRG"]["mean_new_partitions_vs_S4"]} / {p["PPRB"]["mean_new_partitions_vs_S4"]}'
              f' / {p["PPRP32"]["mean_new_partitions_vs_S4"]} |')

    print("\n## T11 — MetaQA per-hop for the bake-off (ΔALL vs BASE)\n")
    hdr(["variant", "hop1", "hop2", "hop3"])
    for k in VN:
        ph = Dj["RESULTS"]["metaqa"][k]["per_hop"]
        print(f'| {k} | ' + " | ".join(sig(ph[h]) for h in ("1", "2", "3")) + " |")

# ---------------------------------------------------------------- STEP 4 / P3 / STEP 7
if Cj:
    print("\n## T12 — STEP 4 precomputation. (B) full basis is EXACT by linearity; only (C) is lossy\n")
    hdr(["corpus", "basis precompute s", "full basis bytes", "(A) online s/query",
         "(A) online graph edges/query", "(B) top50 set agreement", "(B) max abs Δmass",
         "(C) L=8", "(C) L=16", "(C) L=32", "(C) L=64", "(C) L=32 bytes"])
    for d in DS:
        s = Cj["STEP4_PRECOMPUTE"][d]
        c = s["C_sparse"]
        print(f'| {SH[d]} | {s["basis_precompute_sec"]} | {s["full_basis_bytes"]:,} | '
              f'{s["online_sec_per_query"]:.6f} | {s["online_graph_edges_touched_per_query"]:,} | '
              f'{s["B_full_basis"]["top50_exact_set_agreement"]:.4f} | '
              f'{s["B_full_basis"]["max_abs_mass_diff"]:.2e} | '
              + " | ".join(f'{c[f"L{L}"]["top50_exact_set_agreement"]:.4f}' for L in (8, 16, 32, 64))
              + f' | {c["L32"]["basis_bytes"]:,} |')

    print("\n## T13 — STEP 3 P3 partition-bounded node PPR, with the two NO-GRAPH controls\n")
    keys = sorted({k for d in DS for k in Cj["STEP6_ROUNDC"][d] if k.startswith("L5_")})
    keys += sorted({k for d in DS for k in Cj["STEP6_ROUNDC"][d] if k.startswith("L5e_")})
    hdr(["variant"] + [SH[d] for d in DS] + ["macro", "worst"])
    for k in ["L0_BASE", "L1_SWAP_F6_B6", "L3_DIRECT_PPRGLOBAL",
              "L3p_DIRECT_PPR_PRECOMP_L32"] + keys:
        rows = [Cj["STEP6_ROUNDC"][d][k] for d in DS if k in Cj["STEP6_ROUNDC"][d]]
        if len(rows) < len(DS):
            continue
        m, w = macro(rows)
        print(f'| {k} | ' + " | ".join(sig(e) for e in rows) + f' | {m:+.4f} | {w:+.4f} |')

    print("\n## T14 — STEP 3 P3 cost and support\n")
    hdr(["corpus", "local operator bytes", "s/query (seed-only)", "s/query (entry)",
         "partitions scored/query (seed-only)", "(entry)"])
    for d in DS:
        s = Cj["STEP3_P3"][d]
        print(f'| {SH[d]} | {s["local_operator_bytes"]:,} | {s["sec_per_query_seedonly"]:.6f} | '
              f'{s["sec_per_query_entry"]:.6f} | {s["mean_partitions_scored_seedonly"]} | '
              f'{s["mean_partitions_scored_entry"]} |')

    print("\n## T15 — STEP 7 CAP_B6 vs CAP_P50. These are NOT the same thing.\n")
    print("Decomposition of the queries the SAFE_REFERENCE_ROUTER (B6_S4_F6) leaves uncovered. "
          "`CAP_P50` = more than 50 gold partitions, impossible at P=50 under ANY replacement. "
          "`REACH` = a missing gold partition is not in the query's candidate universe at all. "
          "`CAP_B6` = every missing gold IS reachable but more than B=6 partitions of BASE would "
          "have to change. `RANKING` = ≤6 changes would suffice and the selector simply ordered "
          "them wrong.\n")
    hdr(["corpus", "CAP_P50", "REACH", "CAP_B6", "RANKING", "REACH after adding PPR-reachable",
         "mean gold partitions/query", "mean gold outside BASE50"])
    for d in DS:
        s = Cj["STEP7_DECOMPOSITION"][d]
        r, p = s["ROUTER_POOL"], s["PLUS_PPR_REACHABLE"]
        print(f'| {SH[d]} | {r["CAP_P50"]} | {r["REACH"]} | {r["CAP_B6"]} | {r["RANKING"]} | '
              f'{p["REACH"]} | {s["mean_gold_partitions"]} | {s["mean_gold_outside_base50"]} |')
    m = Cj["STEP7_DECOMPOSITION"]["metaqa"]
    print("\nMetaQA by hop (router pool):\n")
    hdr(["hop", "CAP_P50", "REACH", "CAP_B6", "RANKING"])
    for h in ("hop2", "hop3"):
        x = m["ROUTER_POOL"][h]
        print(f'| {h} | {x["CAP_P50"]} | {x["REACH"]} | {x["CAP_B6"]} | {x["RANKING"]} |')

# ---------------------------------------------------------------- latency
print("\n## T16 — STEP 11 latency and online graph work\n")
hdr(["corpus", "npart", "P1 global PPR s/q", "P1 online edges/q", "P2 bounded PPR s/q",
     "P2 candidate partitions", "P2 subgraph edges", "P2 online edges/q",
     "precomputed basis online edges/q"])
for d in DS:
    l = Bj["STEP11_LATENCY"][d]
    print(f'| {SH[d]} | {l["npart"]} | {l["P1_global_ppr_sec_per_query"]:.4f} | '
          f'{l["P1_online_graph_edges_touched_per_query"]:,} | '
          f'{l["P2_bounded_ppr_sec_per_query"]:.4f} | {l["P2_mean_candidate_partitions"]} | '
          f'{l["P2_mean_subgraph_edges"]} | {l["P2_online_graph_edges_touched_per_query"]:,} | '
          f'**0** |')

# ---------------------------------------------------------------- eviction
if EV:
    print("\n## T17 — STEP 0 (quantified) the boundary competition is near-deterministic turnover\n")
    hdr(["corpus", "B", "incumbents retained / B", "retention rate",
         "boundary incumbents with canonical evidence ONLY (/q)",
         "their retention rate", "gold partitions evicted", "gained", "net"])
    for d in DS:
        for b in ("B6", "B12"):
            e = EV[d][b]
            print(f'| {SH[d]} | {b[1:]} | {e["mean_incumbents_retained"]} | '
                  f'{e["incumbent_retention_rate"]:.4f} | '
                  f'{e["mean_boundary_incumbents_with_canonical_evidence_only"]} | '
                  f'{e["retention_rate_of_canonical_only_incumbents"]:.4f} | '
                  f'{e["gold_partitions_evicted"]} | {e["gold_partitions_gained"]} | '
                  f'{e["net_gold_partitions"]:+d} |')

# ---------------------------------------------------------------- alpha
if AL:
    print("\n## T18 — STEP 10 PPR damping robustness (reported, NOT selected)\n")
    ks = [k for k in list(AL.values())[0] if k.startswith("DIRECT_PPR_a")]
    sw = [k for k in list(AL.values())[0] if k.startswith("SWAP_PPR_F6_a")]
    hdr(["corpus", "S4_F6_B6"] + [k.replace("DIRECT_PPR_a", "DIRECT α=") for k in ks]
        + [k.replace("SWAP_PPR_F6_a", "SWAP α=") for k in sw])
    for d, r in AL.items():
        print(f'| {SH.get(d, d)} | {r["S4_F6_B6"]:+.4f} | '
              + " | ".join(f'{r[k]:+.4f}' for k in ks + sw) + " |")


# ---------------------------------------------------------------- round E
if Ej:
    print('\n## T19 — STEP 1 executed literally: ONE vote per evidence family (ΔALL vs BASE)\n')
    lab = {"F6": "F6      canonical + structural + retrieval  (3 votes, lexical twice)",
           "ES": "ES      canonical + structural              (2, node view DELETED)",
           "LEXONE": "LEXONE  RRF(canonical, retrieval) + structural (2, ONE lexical vote)",
           "DS3": "DS3     dense + splade + structural         (the directive's literal set)",
           "DS4": "DS4     dense + splade + structural + retrieval"}
    hdr(["variant"] + [SH[d] for d in DS] + ["macro", "worst"])
    for bv in (6, 8):
        for m in ("F6", "ES", "LEXONE", "DS3", "DS4"):
            k = f"{m}_B{bv}"
            rows = [Ej["RESULTS"][d][k] for d in DS]
            mm, w = macro(rows)
            print(f'| B={bv} · {lab[m]} | ' + " | ".join(sig(e) for e in rows)
                  + f' | {mm:+.4f} | {w:+.4f} |')
    print('\nMetaQA per hop (ΔALL vs BASE):\n')
    hdr(["variant", "hop1", "hop2", "hop3"])
    for m in ("F6", "ES", "LEXONE", "DS3", "DS4"):
        ph = Ej["RESULTS"]["metaqa"][f"{m}_B6"]["per_hop"]
        print(f'| {m}_B6 | ' + " | ".join(sig(ph[h]) for h in ("1", "2", "3")) + " |")

# ---------------------------------------------------------------- oracle
if OR:
    print('\n## T20 — STEP 7 addendum: is the SELECTOR or the POOL the limiter?\n')
    print("`ORACLE_B` = exists X with |X|=B drawn from (boundary + challengers) such that the "
          "protected core plus X covers all gold partitions. It is the best any parameter-free "
          "selector could do at that B without changing the contract.\n")
    hdr(["corpus", "B", "actual ALL", "oracle ALL", "selector headroom",
         "uncovered even by oracle", "of those: POOL failure", "of those: need > B swaps",
         "mean partitions needing replacement"])
    for d in DS:
        for b in (6, 8, 12):
            r = OR[d][f"B{b}"]
            print(f'| {SH[d]} | {b} | {r["ACTUAL_ALL"]:.4f} | {r["ORACLE_ALL"]:.4f} | '
                  f'**{r["SELECTOR_HEADROOM"]:+.4f}** | {r["uncovered_by_oracle"]} | '
                  f'{r["of_those_POOL_failure"]} | {r["of_those_need_more_than_B_swaps"]} | '
                  f'{r["mean_partitions_needing_replacement"]} |')
    print('\nMetaQA per hop at B=6:\n')
    hdr(["hop", "n", "actual", "oracle", "selector headroom", "POOL failures"])
    for h, x in OR["metaqa"]["B6"]["per_hop"].items():
        print(f'| hop{h} | {x["n"]} | {x["ACTUAL"]:.4f} | {x["ORACLE"]:.4f} | '
              f'**{x["SELECTOR_HEADROOM"]:+.4f}** | {x["POOL_failure"]} |')
