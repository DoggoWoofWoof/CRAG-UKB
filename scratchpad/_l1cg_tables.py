"""TABLES.md for the L1 CANDIDATE GENERATION phase."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1cg_core as CG
import _l1cg_run as RUN

D = f"{CG.CGD}/diag"
J = lambda n: json.load(open(f"{D}/{n}.json")) if os.path.exists(f"{D}/{n}.json") else {}
AUD, S4, RUNJ = J("audit"), J("step4_metaqa_hop3"), J("run")
S6, S11, BLK, DEP = J("step6"), J("step11"), J("blocker"), J("depth")
CEIL = json.load(open(f"{CG.ROOT}/L1_CEILING_AUDIT/ceiling_audit.json"))
DS = [d for d in CG.DSETS if d in AUD]
SHORT = {"A_DENSE_PARTITION": "A dense-part", "B_SPLADE_PARTITION": "B splade-part",
         "C_NODE_DENSE_CONT": "C node-dense", "D_NODE_SPLADE_CONT": "D node-splade",
         "E_S4_STRUCT": "E S4-struct", "F_PPR_REACH": "F ppr-reach",
         "G_GRAPH_NBR_RAW": "G nbr-raw", "G_GRAPH_NBR_NORM": "G nbr-norm",
         "H_STRUCT_FRONTIER": "H frontier", "I_CANON_CONT": "I canon-cont"}
L = []
w = L.append


def tbl(hdr, rows):
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        assert len(r) == len(hdr), f"ragged: {r}"
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


w("# L1 CANDIDATE GENERATION -- TABLES")
w("")
w("Selector frozen at R0 / B6_S4_F6_Ms64_Mr32 throughout. Every proposal family contributes "
  "candidate MEMBERSHIP only; no family's score enters the selector.")
w("")

# ---------------------------------------------------------------- T1
w("## T1 -- corpora and the pool-limited target set (STEP 1)")
w("")
w("`pool-limited` = the query is P50-feasible and at least one required gold partition lies "
  "outside the current candidate pool (boundary incumbents + S4/ret challengers).")
w("")
rows = []
for ds in DS:
    a = AUD[ds]; r = RUNJ.get(ds, {})
    k0 = "hop3" if ds == "metaqa" else "ALL"
    rows.append([ds, a["npart"], a["nq"], a["n_pool_limited"],
                 f"{a['n_pool_limited']/a['nq']:.3f}",
                 f"{r.get('SAFE',{}).get(k0,float('nan')):.4f}",
                 f"{r.get('CURRENT_POOL_ORACLE',{}).get(k0,float('nan')):.4f}", k0])
tbl(["corpus", "npart", "nq", "pool-limited q", "frac", "SAFE", "current-pool oracle", "slice"], rows)

# ---------------------------------------------------------------- T2
w("## T2 -- proposal-set size, and how much of the universe each family can see (STEP 1)")
w("")
w("`frac universe @256` = mean proposals actually available at M=256, divided by the number of "
  "non-base50 partitions. A family near 1.00 is close to enumerating the corpus and its recall is "
  "NOT comparable with a selective family's.")
w("")
for ds in ["metaqa", "webqsp"]:
    if ds not in AUD:
        continue
    w(f"**{ds}** (npart = {AUD[ds]['npart']}, non-base50 universe = {AUD[ds]['npart']-CG.P})")
    w("")
    rows = [[SHORT[f], AUD[ds]["PROPOSAL_SET_SIZE"][f]["mean_available"],
             f"{AUD[ds]['PROPOSAL_SET_SIZE'][f]['frac_of_universe_at_M256']:.3f}"]
            for f in CG.FAMS if f in AUD[ds]["PROPOSAL_SET_SIZE"]]
    tbl(["family", "mean proposals available", "frac universe @256"], rows)

# ---------------------------------------------------------------- T3 / T4
w("## T3 -- ALL_REQUIRED_PARTITIONS_PROPOSED@M, MetaQA by hop (STEP 2, primary metric)")
w("")
w("Fraction of pool-limited queries whose ENTIRE missing gold-partition set is proposed. "
  "A query is not solved because one partition of a chain was found.")
w("")
for k in ["hop1", "hop2", "hop3"]:
    a = AUD["metaqa"]["RECALL"]
    n = a["A_DENSE_PARTITION"]["8"].get(k, {}).get("n_target", 0)
    w(f"**MetaQA {k}** (pool-limited queries = {n})")
    w("")
    rows = [[SHORT[f]] + [f"{a[f][str(M)][k]['ALL_REQUIRED_PROPOSED']:.4f}"
                          if k in a[f][str(M)] else "-" for M in CG.MS]
            for f in CG.FAMS if f in a]
    tbl(["family"] + [f"M={M}" for M in CG.MS], rows)

w("## T4 -- ALL_REQUIRED_PARTITIONS_PROPOSED@M, WebQSP and the text controls (STEP 2 / STEP 10)")
w("")
for ds in DS:
    if ds == "metaqa":
        continue
    a = AUD[ds]["RECALL"]
    n = a["A_DENSE_PARTITION"]["8"]["ALL"]["n_target"]
    w(f"**{ds}** (pool-limited queries = {n})")
    w("")
    rows = [[SHORT[f]] + [f"{a[f][str(M)]['ALL']['ALL_REQUIRED_PROPOSED']:.4f}" for M in CG.MS]
            for f in CG.FAMS if f in a]
    tbl(["family"] + [f"M={M}" for M in CG.MS], rows)

# ---------------------------------------------------------------- T5
w("## T5 -- ANY vs ALL at M=256 (STEP 2, secondary)")
w("")
w("The gap between ANY and ALL is the multi-hop penalty: how often a family finds part of the "
  "chain but not all of it.")
w("")
rows = []
for ds in DS:
    k = "hop3" if ds == "metaqa" else "ALL"
    a = AUD[ds]["RECALL"]
    for f in ["B_SPLADE_PARTITION", "F_PPR_REACH", "I_CANON_CONT", "G_GRAPH_NBR_RAW"]:
        e = a[f]["256"][k]
        rows.append([ds, k, SHORT[f], f"{e['ANY_REQUIRED_PROPOSED']:.4f}",
                     f"{e['ALL_REQUIRED_PROPOSED']:.4f}", f"{e['micro_partition_recall']:.4f}"])
tbl(["corpus", "slice", "family", "ANY", "ALL", "micro partition recall"], rows)

# ---------------------------------------------------------------- T6
w("## T6 -- marginal complementarity at M=64 (STEP 3)")
w("")
w("Judged only on DISCOVERY. `unique` = recovered by this family and no other. `greedy` adds "
  "families in order of how many additional pool-limited queries become fully covered.")
w("")
for ds in DS:
    m = AUD[ds]["MARGINAL"]
    w(f"**{ds}** (pool-limited queries = {m['n_pool_limited']})")
    w("")
    rows = [[SHORT[f], v["missing_partitions_recovered"], v["unique_missing_partitions"],
             v["queries_fully_solved"], v["unique_queries_fully_solved"]]
            for f, v in m["per_family"].items()]
    tbl(["family", "missing partitions recovered", "unique partitions",
         "queries fully solved", "unique queries"], rows)
    rows = [[i + 1, SHORT[o["family"]], o["marginal_queries_solved"],
             o["cumulative_queries_solved"], f"{o['cumulative_frac_of_pool_limited']:.4f}"]
            for i, o in enumerate(m["greedy_order"])]
    tbl(["step", "family added", "marginal queries solved", "cumulative", "cumulative frac"], rows)

# ---------------------------------------------------------------- T7
w("## T7 -- overlap between proposal families, MetaQA at M=64 (STEP 3)")
w("")
w("Jaccard over the set of (query, recovered missing partition) pairs.")
w("")
jac = AUD["metaqa"]["MARGINAL"]["jaccard_overlap"]
fams = [f for f in CG.FAMS if f in jac]
tbl([""] + [SHORT[g].split()[0] for g in fams],
    [[SHORT[f]] + [f"{jac[f][g]:.2f}" for g in fams] for f in fams])

# ---------------------------------------------------------------- T8 / T9
if S4:
    w("## T8 -- anatomy of the MetaQA hop3 CURRENT_POOL_LIMIT failures (STEP 4)")
    w("")
    w(f"{S4['n_target_queries']} pool-limited hop3 queries, {S4['n_missing_partitions']} required "
      f"gold partitions they cannot see. `frac defined` = fraction of those partitions the channel "
      f"ranks at all (a rank of -1 means the channel never lists it).")
    w("")
    rows = []
    for k, v in S4["ATTRIBUTES"].items():
        if not v.get("n"):
            rows.append([k, 0, "-", "-", "-", "-", "-"]); continue
        rows.append([k, v["n"], f"{v['frac_defined']:.4f}", f"{v['p10']:.1f}",
                     f"{v['median']:.1f}", f"{v['p90']:.1f}", f"{v['max']:.0f}"])
    tbl(["attribute", "n defined", "frac defined", "p10", "median", "p90", "max"], rows)

    w("## T9 -- how those failures group, derived from T8 (STEP 4)")
    w("")
    rows = [[k, v["n"], f"{v['frac']:.4f}"] for k, v in S4["GROUPS_PARTITION_LEVEL"].items()]
    tbl(["group (partition level)", "n", "frac"], rows)
    rows = [[k, v["n"], f"{v['frac']:.4f}"] for k, v in S4["GROUPS_QUERY_LEVEL_worst_partition"].items()]
    tbl(["group (query level, worst partition)", "n", "frac"], rows)
    w("Graph distance from the nearest retrieval-seed partition, and depth in the precomputed "
      "top-32 neighbour tables (-1 = absent from the tables):")
    w("")
    tbl(["min graph distance", "n"], [[k, v] for k, v in S4["DIST_graph_dist_min"].items()])
    tbl(["precomputed table depth", "n"], [[k, v] for k, v in S4["DIST_precomputed_table_depth"].items()])
    rows = [[SHORT[f], f"{v:.4f}"] for f, v in S4["COVERAGE_by_family_at_256"].items()]
    tbl(["family", "frac of the 2596 missing partitions proposed at M=256"], rows)

# ---------------------------------------------------------------- T10
w("## T10 -- PPR as a PROPOSER only (STEP 5)")
w("")
w("PPR contributes candidate membership only; no PPR score reaches F6/R0. Both bookends are "
  "reported for the same pool, as STEP 8 requires.")
w("")
rows = []
for ds in DS:
    k = "hop3" if ds == "metaqa" else "ALL"
    r = RUNJ[ds]
    for M in RUN.MPPR:
        e = r["STEP5_PPR_PROPOSAL_ONLY"][str(M)][k]
        rows.append([ds, k, M, f"{r['SAFE'][k]:.4f}", f"{r['CURRENT_POOL_ORACLE'][k]:.4f}",
                     f"{e['POOL_ORACLE']:.4f}", f"{e['ACTUAL']:.4f}",
                     f"{e['delta_vs_SAFE']:+.4f}", f"+{e['gained']}/-{e['lost']}"])
tbl(["corpus", "slice", "M_ppr", "SAFE", "current-pool oracle", "PPR-pool oracle",
     "ACTUAL frozen", "delta", "gained/lost"], rows)

# ---------------------------------------------------------------- T11
if S6:
    w("## T11 -- online structural discovery vs the precomputed tables (STEP 6)")
    w("")
    w("Same target set as T8. `edges` = ONLINE_GRAPH_EDGES_TOUCHED per query. The precomputed "
      "tables are corpus-side, so K is a storage knob, never a query-time cost.")
    w("")
    rows = []
    for key, v in S6.items():
        for name, e in list(v["ONLINE"].items()):
            rows.append([key, v["npart"], v["mean_partition_degree"], "ONLINE " + name,
                         f"{e['missing_partition_recall']:.4f}", f"{e['mean_proposals']:.1f}",
                         f"{e['mean_frac_of_universe']:.3f}",
                         f"{e['ONLINE_GRAPH_EDGES_TOUCHED_mean']:.0f}", "-"])
        for name, e in v["PRECOMPUTED"].items():
            rows.append([key, v["npart"], v["mean_partition_degree"], "TABLE " + name,
                         f"{e['missing_partition_recall']:.4f}", f"{e['mean_proposals']:.1f}", "-",
                         "0", f"{e['table_bytes']/1e6:.2f}"])
    tbl(["corpus", "npart", "mean partition degree", "source", "missing-partition recall",
         "mean proposals", "frac universe", "edges touched", "table MB"], rows)

# ---------------------------------------------------------------- T12
w("## T12 -- union proposers, candidate-pool oracle at a single global budget (STEP 7)")
w("")
w("Round-robin interleave of the member families, deduped, cut at M_TOTAL. One budget for every "
  "corpus; no per-corpus tuning; no learned component. `gf` = touches no graph at query time; "
  "`pc` = fully served by corpus-side precomputed tables.")
w("")
for ds in DS:
    k = "hop3" if ds == "metaqa" else "ALL"
    r = RUNJ[ds]
    w(f"**{ds}** slice `{k}` -- SAFE {r['SAFE'][k]:.4f}, current-pool oracle "
      f"{r['CURRENT_POOL_ORACLE'][k]:.4f}")
    w("")
    rows = [[n, "yes" if u["graph_free"] else "no", "yes" if u["fully_precomputed"] else "no"]
            + [f"{u[str(M)][k]['POOL_ORACLE']:.4f}" for M in RUN.MTOT]
            + [f"{u['256'][k]['ACTUAL']:.4f}"]
            for n, u in r["STEP7_UNION"].items()]
    tbl(["union", "gf", "pc"] + [f"oracle M={M}" for M in RUN.MTOT], rows) if False else tbl(
        ["union", "gf", "pc"] + [f"oracle M={M}" for M in RUN.MTOT] + ["ACTUAL M=256"], rows)

# ---------------------------------------------------------------- T13
w("## T13 -- MANDATORY MetaQA hop table (STEP 9)")
w("")
CH = CEIL["metaqa"]["REFERENCE_INDICATORS"]
r = RUNJ["metaqa"]
best_single = max(CG.FAMS, key=lambda f: r["SINGLE_FAMILY"][f]["hop3"]["POOL_ORACLE"])
best_union = max(r["STEP7_UNION"], key=lambda n: r["STEP7_UNION"][n]["256"]["hop3"]["POOL_ORACLE"])
best_ppr = max(RUN.MPPR, key=lambda M: r["STEP5_PPR_PROPOSAL_ONLY"][str(M)]["hop3"]["POOL_ORACLE"])
REC = "U_PC5"
rows = [
    ["SAFE current (frozen selector, current pool)"] + [f"{r['SAFE'][k]:.4f}" for k in
                                                        ["hop1", "hop2", "hop3", "ALL"]],
    ["current-pool oracle"] + [f"{r['CURRENT_POOL_ORACLE'][k]:.4f}" for k in
                               ["hop1", "hop2", "hop3", "ALL"]],
    [f"PPR-proposal pool oracle (M_ppr={best_ppr})"] +
    [f"{r['STEP5_PPR_PROPOSAL_ONLY'][str(best_ppr)][k]['POOL_ORACLE']:.4f}"
     for k in ["hop1", "hop2", "hop3", "ALL"]],
    [f"best single proposer pool oracle ({SHORT[best_single]}, M=128)"] +
    [f"{r['SINGLE_FAMILY'][best_single][k]['POOL_ORACLE']:.4f}" for k in
     ["hop1", "hop2", "hop3", "ALL"]],
    [f"best union proposer pool oracle ({best_union}, M=256)"] +
    [f"{r['STEP7_UNION'][best_union]['256'][k]['POOL_ORACLE']:.4f}" for k in
     ["hop1", "hop2", "hop3", "ALL"]],
    [f"best union + FROZEN selector ({best_union}, M=256)"] +
    [f"{r['STEP7_UNION'][best_union]['256'][k]['ACTUAL']:.4f}" for k in
     ["hop1", "hop2", "hop3", "ALL"]],
    ["unlimited-pool oracle at B=6"] + [f"{CH['ORACLE_UNLIMITED_POOL_B6']['per_hop'][h]:.4f}"
                                        for h in "123"] +
    [f"{CH['ORACLE_UNLIMITED_POOL_B6']['aggregate']:.4f}"],
    ["full-universe P50 oracle"] + [f"{CH['ORACLE_ABSTRACTION_ONLY_P50']['per_hop'][h]:.4f}"
                                    for h in "123"] +
    [f"{CH['ORACLE_ABSTRACTION_ONLY_P50']['aggregate']:.4f}"]]
if REC != best_union:
    rows.insert(5, [f"recommended union pool oracle ({REC}, M=256)"] +
                [f"{r['STEP7_UNION'][REC]['256'][k]['POOL_ORACLE']:.4f}"
                 for k in ["hop1", "hop2", "hop3", "ALL"]])
tbl(["configuration", "hop1", "hop2", "hop3", "ALL"], rows)
w(f"The pool-oracle-maximising union and the union recommended under the STEP 12 criteria are the "
  f"same object here ({REC}), so no separate row is needed." if REC == best_union else "")
w("")
w("")
h3 = float(r["STEP7_UNION"][REC]["256"]["hop3"]["POOL_ORACLE"])
cur = float(r["CURRENT_POOL_ORACLE"]["hop3"]); ceil6 = CH["ORACLE_UNLIMITED_POOL_B6"]["per_hop"]["3"]
w(f"Pool headroom closed on hop3 by {REC} at M=256: "
  f"({h3:.4f} - {cur:.4f}) / ({ceil6:.4f} - {cur:.4f}) = {(h3-cur)/(ceil6-cur):.3f}.")
w("")

# ---------------------------------------------------------------- T14
w("## T14 -- the recommended generator on every corpus (STEP 10)")
w("")
w(f"`{REC}` = round-robin over " + ", ".join(SHORT[f] for f in RUN.UNIONS[REC]) +
  ". Identical on every corpus.")
w("")
rows = []
for ds in DS:
    k = "hop3" if ds == "metaqa" else "ALL"
    rr = RUNJ[ds]; u = rr["STEP7_UNION"][REC]
    rows.append([ds, k, f"{rr['SAFE'][k]:.4f}", f"{rr['CURRENT_POOL_ORACLE'][k]:.4f}"]
                + [f"{u[str(M)][k]['POOL_ORACLE']:.4f}" for M in RUN.MTOT]
                + [f"{u['256'][k]['ACTUAL']:.4f}", f"{u['256']['mean_proposals']:.0f}"])
tbl(["corpus", "slice", "SAFE", "current-pool oracle"] + [f"oracle M={M}" for M in RUN.MTOT]
    + ["ACTUAL M=256", "mean proposals"], rows)

# ---------------------------------------------------------------- T15
if S11:
    w("## T15 -- B = {6, 8, 12} re-run against the EXPANDED pool (STEP 11)")
    w("")
    w("The ceiling audit measured B as non-binding against the OLD pool. Re-measured here once the "
      "generator has moved the pool, as the directive requires.")
    w("")
    rows = []
    for ds in DS:
        if ds not in S11:
            continue
        k = "hop3" if ds == "metaqa" else "ALL"
        for Bv, v in S11[ds]["B"].items():
            e = v[k]
            rows.append([ds, k, Bv, f"{e['SAFE']:.4f}", f"{e['CURRENT_POOL_ORACLE']:.4f}",
                         f"{e['EXPANDED_POOL_ORACLE']:.4f}", f"{e['EXPANDED_ACTUAL']:.4f}"])
    tbl(["corpus", "slice", "B", "SAFE", "current-pool oracle", "expanded-pool oracle",
         "ACTUAL frozen"], rows)

# ---------------------------------------------------------------- T16
if BLK:
    w("## T16 -- why the frozen selector converts none of it (verification)")
    w("")
    w("Union of A, B, F, I, G at M=256. A `truly new` candidate is one outside base50 and outside "
      "the frozen challenger set, so it carries no structural and no retrieval rank.")
    w("")
    rows = [[ds, v["nq"], v["queries_offered_a_truly_new_candidate"],
             v["truly_new_candidates_offered_total"], v["truly_new_candidates_ever_SELECTED"],
             v["selections_bit_identical_to_SAFE"],
             f"{v['margin_min_over_queries']:.2e}" if v["margin_min_over_queries"] is not None else "-"]
            for ds, v in BLK.items()]
    tbl(["corpus", "nq", "queries offered a new candidate", "new candidates offered",
         "new candidates SELECTED", "selections identical to SAFE", "min score margin"], rows)
    b = list(BLK.values())[0]
    w(f"Bound: any candidate outside base50 scores at most 1/(K0+P) = "
      f"{b['theoretical_bound_new_max']:.8f}; every boundary incumbent scores at least "
      f"1/(K0+P-1) = {b['theoretical_bound_incumbent_min']:.8f}. There are exactly B incumbents for "
      f"B slots, so the new candidate is strictly dominated for every B.")
    w("")

# ---------------------------------------------------------------- T17
if DEP:
    w("## T17 -- generator aggregation depth, the one CONVERTIBLE knob")
    w("")
    w("M_struct / M_ret set how deep the generator aggregates. Raising them extends the spos / rpos "
      "rank lists themselves, so new candidates arrive WITH channel evidence. The scoring rule is "
      "untouched and ranks of already-listed partitions are unchanged. Whether this counts as "
      "inside the frozen contract is the user's call; it is reported separately for that reason.")
    w("")
    rows = []
    for ds, v in DEP.items():
        k = "hop3" if ds == "metaqa" else "ALL"
        for cfg, e in v.items():
            rows.append([ds, k, cfg, f"{e['mean_pool']:.1f}", f"{e[k]['POOL_ORACLE']:.4f}",
                         f"{e[k]['ACTUAL']:.4f}", f"{e[k]['delta_vs_Ms64_Mr32']:+.4f}",
                         f"+{e[k]['gained']}/-{e[k]['lost']}", f"{e[k]['p']:.3g}",
                         "yes" if e[k]["sig"] else "no"])
    tbl(["corpus", "slice", "config", "mean pool", "pool oracle", "ACTUAL", "delta vs Ms64/Mr32",
         "gained/lost", "p", "sig"], rows)

out = f"{CG.CGD}/TABLES.md"
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", out, len(L), "lines")
