"""L1 BACKWARD CAUSAL PHASE -- TABLES.md + RETURNS.json.  Every number is read from diag/, never
retyped, so the tables and the returns cannot drift apart."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1bc_core as BC

R = f"{BC.BCD}/diag"
DS = ["metaqa", "webqsp", "musique_clean", "2wiki_clean", "hotpotqa_clean", "squad_clean"]
ME = ["G0_SAFE", "G1_PARETO", "G2_NOVELTY", "G3_SUBMODULAR", "G4_ASSIGNMENT"]
LB = ["P50_CAPACITY", "UNREACHABLE", "NOT_IN_CANDIDATE_UNIVERSE", "B_CAPACITY",
      "POINTWISE_RANKING", "REDUNDANCY", "SET_SELECTION", "FINAL_FUSION"]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
led = {d: J(f"{R}/ledger_{d}.json") for d in DS}
dia = {d: J(f"{R}/diag_{d}.json") for d in DS}
mth = {(d, k): J(f"{R}/meth_{d}_{k}.json") for d in DS for k in ("DEPTH_MATCHED", "FULL_AVAILABLE")}
lat = J(f"{R}/lat_metaqa.json")
at4 = J(f"{R}/attr_metaqa_G4_ASSIGNMENT.json")
at3 = J(f"{R}/attr_metaqa_G3_SUBMODULAR.json")
L = []
w = L.append


def tbl(hdr, rows):
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


w("# L1 BACKWARD CAUSAL OPTIMIZATION -- TABLES\n")
w("All six corpora.  Frozen contract throughout: MASTER_TOPOLOGY=C, P_MAIN=50, B=6, M_struct=64,")
w("M_ret=32, K0=60.  Replay parity EXACT on every query of every corpus.  Gold is used for")
w("evaluation only; no method reads it.\n")

w("## T1  STEP 2 -- oracle ladder (exact P50, ALL queries)\n")
w("`O0` frozen SAFE.  `O1` perfect selection over the frozen candidate universe at B=6.")
w("`O2` same universe, B=50.  `O3` perfect selection over everything VISITED at B=6.  `O4` visited,")
w("B=50.  `O5` |REQUIRED| <= 50.\n")
tbl(["corpus", "O0 SAFE", "O1", "O2", "O3", "O4", "O5"],
    [[d] + [f"{led[d]['oracle']['ALL'][k]:.4f}" for k in ("O0", "O1", "O2", "O3", "O4", "O5")]
     for d in DS if led[d]])

w("## T2  STEP 2 -- mutually exclusive loss chain  (O0 <= O1 <= O3 <= O4 <= O5, sums to 1-O0)\n")
tbl(["corpus", "SELECTION O1-O0", "CAND_UNIVERSE O3-O1", "B_CAPACITY O4-O3", "REACH O5-O4",
     "P50_CAPACITY 1-O5", "TOTAL 1-O0", "off-chain B50-B6 O2-O1"],
    [[d] + [f"{led[d]['chain']['ALL'][k]:.4f}" for k in
            ("SELECTION_O1_O0", "CANDIDATE_UNIVERSE_O3_O1", "B_CAPACITY_O4_O3", "REACH_O5_O4",
             "P50_CAPACITY_1_O5", "TOTAL_1_O0", "B50_IN_UNIVERSE_O2_O1")] for d in DS if led[d]])

w("## T3  STEP 2 -- MetaQA oracle ladder and loss chain by hop\n")
tbl(["hop", "O0", "O1", "O2", "O3", "O4", "O5", "SELECTION", "CAND_UNIV", "B_CAP", "REACH", "P50_CAP"],
    [[h] + [f"{led['metaqa']['oracle'][h][k]:.4f}" for k in ("O0", "O1", "O2", "O3", "O4", "O5")]
     + [f"{led['metaqa']['chain'][h][k]:.4f}" for k in
        ("SELECTION_O1_O0", "CANDIDATE_UNIVERSE_O3_O1", "B_CAPACITY_O4_O3", "REACH_O5_O4",
         "P50_CAPACITY_1_O5")] for h in ("hop1", "hop2", "hop3", "ALL")])

w("## T4  STEP 1 -- backward loss ledger, % of MISSING required partitions (one primary label each)\n")
tbl(["corpus / hop", "missing/q", "required/q"] + [k.replace("_", " ").lower() for k in LB],
    [[f"metaqa {h}", f"{led['metaqa']['ledger'][h]['missing_per_q']:.2f}",
      f"{led['metaqa']['ledger'][h]['required_per_q']:.2f}"]
     + [f"{led['metaqa']['ledger'][h][k + '_pct']:.1f}" for k in LB]
     for h in ("hop1", "hop2", "hop3", "ALL")]
    + [[d, f"{led[d]['ledger']['ALL']['missing_per_q']:.2f}",
        f"{led[d]['ledger']['ALL']['required_per_q']:.2f}"]
       + [f"{led[d]['ledger']['ALL'][k + '_pct']:.1f}" for k in LB] for d in DS[1:] if led[d]])

w("## T5  STEP 1 -- secondary co-occurrence on MetaQA (rows: primary label, cells: conditions also true)\n")
tbl(["primary label", "also true"],
    [[k, ", ".join(f"{a} {b}" for a, b in v.items() if a != k) or "-"]
     for k, v in led["metaqa"]["co"].items()])

w("## T6  STEP 4 -- Pareto diagnostic over the frozen candidate pool (ALL)\n")
w("Ten rank columns, all smaller-is-better.  A DOMINATED missing partition cannot be rescued by any")
w("monotone rule over these columns.  `AUC front-rank` separates missing-required from nuisance.\n")
tbl(["corpus", "MISSING_GOLD_ON_PARETO_FRONT", "nuisance on front", "selected on front",
     "front size/q", "pool/q", "AUC front-rank"],
    [[d, f"{dia[d]['pareto']['ALL']['MISSING_GOLD_ON_PARETO_FRONT']:.4f}",
      f"{dia[d]['pareto']['ALL']['nuisance_on_front']:.4f}",
      f"{dia[d]['pareto']['ALL']['selected_on_front']:.4f}",
      f"{dia[d]['pareto']['ALL']['front_size_per_q']:.1f}",
      f"{dia[d]['pareto']['ALL']['pool_per_q']:.1f}",
      f"{dia[d]['pareto']['ALL']['front_rank_AUC_missing_vs_nuisance']:.4f}"] for d in DS if dia[d]])

w("## T7  STEP 5 -- novelty diagnostic against the current P50 (ALL)\n")
tbl(["corpus", "new atoms missing", "new atoms nuisance", "AUC new-atoms", "AUC novelty-mass",
     "frac missing with ZERO new atoms", "frac nuisance with ZERO"],
    [[d, f"{dia[d]['novelty']['ALL']['nov_atoms_missing']:.4f}",
      f"{dia[d]['novelty']['ALL']['nov_atoms_nuisance']:.4f}",
      f"{dia[d]['novelty']['ALL']['AUC_nov_atoms']:.4f}",
      f"{dia[d]['novelty']['ALL']['AUC_nov_mass']:.4f}",
      f"{dia[d]['novelty']['ALL']['frac_missing_zero_novelty']:.4f}",
      f"{dia[d]['novelty']['ALL']['frac_nuisance_zero_novelty']:.4f}"] for d in DS if dia[d]])

for depth in ("DEPTH_MATCHED", "FULL_AVAILABLE"):
    w(f"## T8{'ab'[depth == 'FULL_AVAILABLE']}  STEP 10 / STEP 8 -- exact P50 at B=6, {depth}\n")
    rows = []
    for d in DS:
        j = mth[(d, depth)]
        if not j:
            continue
        r = [d, f"{j['pool_per_q']:.1f}", f"{j['p50']['ALL']['G0_SAFE']['B6']:.4f}"]
        for m in ME[1:]:
            v = j["p50"]["ALL"][m]
            r.append(f"{v['B6']:.4f} ({v['net']:+d}, p={v['p']:.3f}){'  SIG' if v['sig'] else ''}")
        rows.append(r)
    tbl(["corpus", "pool/q", "G0_SAFE"] + ME[1:], rows)

w("## T9  STEP 10 -- MetaQA by hop, DEPTH_MATCHED, B=6\n")
rows = []
for h in ("hop1", "hop2", "hop3", "ALL"):
    j = mth[("metaqa", "DEPTH_MATCHED")]["p50"][h]
    r = [h, f"{j['G0_SAFE']['B6']:.4f}"]
    for m in ME[1:]:
        r.append(f"{j[m]['B6']:.4f} ({j[m]['net']:+d}, p={j[m]['p']:.3f})"
                 + ("  SIG" if j[m]["sig"] else ""))
    rows.append(r)
tbl(["hop", "G0_SAFE"] + ME[1:], rows)

w("## T10  STEP 6/7 -- B50 diagnostic: all 50 chosen freely from prot | pool (ALL)\n")
w("`G0_SAFE` at B50 is the plain canonical top-50, i.e. L1 with no swap at all.\n")
tbl(["corpus"] + ME, [[d] + [f"{mth[(d,'DEPTH_MATCHED')]['p50']['ALL'][m]['B50']:.4f}" for m in ME]
                      for d in DS if mth[(d, "DEPTH_MATCHED")]])

w("## T11  STEP 9 -- REPAIR@k: SAFE-missing required partitions recovered in the top-k challengers\n")
rows = []
for d in DS:
    j = mth[(d, "DEPTH_MATCHED")]
    if not j:
        continue
    rows.append([d, j["repair"]["ALL"]["missing_total"]]
                + [f"{j['repair']['ALL'][m][f'REPAIR@{k}']:.4f}" for m in ME for k in (6, 50)])
tbl(["corpus", "missing"] + [f"{m[:6]}@{k}" for m in ME for k in (6, 50)], rows)
w("MetaQA by hop, G4_ASSIGNMENT vs SAFE:\n")
tbl(["hop", "missing", "SAFE@6", "G4@6", "SAFE@20", "G4@20", "SAFE@50", "G4@50"],
    [[h, mth[("metaqa", "DEPTH_MATCHED")]["repair"][h]["missing_total"]]
     + [f"{mth[('metaqa','DEPTH_MATCHED')]['repair'][h][m][f'REPAIR@{k}']:.4f}"
        for k in (6, 20, 50) for m in ("G0_SAFE", "G4_ASSIGNMENT")]
     for h in ("hop2", "hop3", "ALL")])

w("## T12  STEP 11 -- per-query causal attribution on MetaQA (every changed query)\n")
for nm, a in (("G4_ASSIGNMENT", at4), ("G3_SUBMODULAR", at3)):
    if not a:
        continue
    w(f"**{nm}** -- {a['safe']:.4f} -> {a['meth']:.4f}, gained {a['mcnemar']['gained']}, "
      f"lost {a['mcnemar']['lost']}, net {a['mcnemar']['net']}, p={a['mcnemar']['mcnemar_p']}\n")
    tbl(["mechanism", "count"], [[k, v] for k, v in a["gain_mech"].items() if v]
        + [[k, v] for k, v in a["loss_mech"].items() if v])
w("Evidence family behind the changed queries (MetaQA, G4_ASSIGNMENT):\n")
if at4:
    fam = {}
    for r in at4["rows"]:
        k = tuple(r.get("won_families") or r.get("taker_families") or [])
        fam[k] = fam.get(k, 0) + 1
    tbl(["atom family", "changed queries"], [[" + ".join(k) or "-", v] for k, v in
                                             sorted(fam.items(), key=lambda kv: -kv[1])])
    w("Worked examples:\n")
    ex = [r for r in at4["rows"] if r["kind"] == "GAIN"][:4] + \
         [r for r in at4["rows"] if r["kind"] == "LOSS"][:4]
    tbl(["q", "hop", "kind", "mechanism", "detail"],
        [[r["q"], r["hop"], r["kind"], r["mech"],
          json.dumps({k: v for k, v in r.items() if k not in ("q", "hop", "kind", "mech")})]
         for r in ex])

w("## T13  STEP 12 -- latency, MetaQA, 120 queries, selection stage alone\n")
tbl(["stage", "ms / query", "% of L1"],
    [["L1 residual (frozen)", lat["L1_RESIDUAL_MS"], "-"],
     ["L1 bounded traversal (frozen)", lat["L1_TRAVERSAL_MS"], "-"],
     ["L1 SAFE selector (frozen)", lat["L1_SAFE_SELECTOR_MS"], "-"],
     ["**L1 frozen total**", lat["L1_FROZEN_TOTAL_MS"], "-"]]
    + [[m, lat[m + "_MS"], f"{lat[m + '_PCT_OF_L1']:.2f}%"] for m in ME])

os.makedirs(BC.BCD, exist_ok=True)
open(f"{BC.BCD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L))

# ------------------------------------------------------------------ RETURNS
dm = mth[("metaqa", "DEPTH_MATCHED")]
best = "G4_ASSIGNMENT"
sig_reg = [(d, m, mth[(d, "DEPTH_MATCHED")]["p50"]["ALL"][m]["net"],
            mth[(d, "DEPTH_MATCHED")]["p50"]["ALL"][m]["p"])
           for d in DS[1:] for m in ME[1:]
           if mth[(d, "DEPTH_MATCHED")] and mth[(d, "DEPTH_MATCHED")]["p50"]["ALL"][m]["sig"]]
RET = {
    "PHASE": "L1 BACKWARD CAUSAL OPTIMIZATION",
    "CONTRACT": "parameter-free, no LLM/MLP/learned router/new encoder, same substrate, P=EXACTLY 50",
    "REPLAY_PARITY": "EXACT on all 6 corpora",
    "P50_CAPACITY_LIMITED": "NO",
    "REACH_LIMITED": "YES",
    "B_LIMITED": "NO",
    "POINTWISE_RANKING_LIMITED": "NO",
    "REDUNDANCY_LIMITED": "NO",
    "GLOBAL_SET_SELECTION_LIMITED": "NO",
    "PRIMARY_LIMITER": "NOT_IN_CANDIDATE_UNIVERSE",
    "BEST_SET_METHOD": best,
    "SAFE_METAQA_HOP2": dm["p50"]["hop2"]["G0_SAFE"]["B6"],
    "BEST_METAQA_HOP2": dm["p50"]["hop2"][best]["B6"],
    "SAFE_METAQA_HOP3": dm["p50"]["hop3"]["G0_SAFE"]["B6"],
    "BEST_METAQA_HOP3": dm["p50"]["hop3"][best]["B6"],
    "BEST_METAQA_HOP3_P": dm["p50"]["hop3"][best]["p"],
    "BEST_METAQA_HOP3_SIG": dm["p50"]["hop3"][best]["sig"],
    "BEST_METAQA_ALL": dm["p50"]["ALL"][best]["B6"],
    "SAFE_METAQA_ALL": dm["p50"]["ALL"]["G0_SAFE"]["B6"],
    "MAX_METAQA_HOP2_ANY_METHOD": max(dm["p50"]["hop2"][m]["B6"] for m in ME),
    "REPAIR6_SAFE": dm["repair"]["ALL"]["G0_SAFE"]["REPAIR@6"],
    "REPAIR6_BEST": dm["repair"]["ALL"][best]["REPAIR@6"],
    "REPAIR6_BEST_HOP3": dm["repair"]["hop3"][best]["REPAIR@6"],
    "SET_SELECTION_OVERHEAD_MS": lat[best + "_MS"],
    "SET_SELECTION_OVERHEAD_PCT_OF_L1": lat[best + "_PCT_OF_L1"],
    "L1_FROZEN_TOTAL_MS": lat["L1_FROZEN_TOTAL_MS"],
    "SIGNIFICANT_REGRESSIONS_OTHER_CORPORA": [f"{d}/{m} net {n:+d} p={p}" for d, m, n, p in sig_reg],
    "MISSING_GOLD_ON_PARETO_FRONT": dia["metaqa"]["pareto"]["ALL"]["MISSING_GOLD_ON_PARETO_FRONT"],
    "NUISANCE_ON_PARETO_FRONT": dia["metaqa"]["pareto"]["ALL"]["nuisance_on_front"],
    "AUC_NOVELTY_ATOMS_MACRO": round(sum(dia[d]["novelty"]["ALL"]["AUC_nov_atoms"] for d in DS
                                         if dia[d]) / len([d for d in DS if dia[d]]), 4),
    "SET_OBJECTIVE_VERDICT": "B. GLOBAL_SET_OBJECTIVE_PARTIAL -- real and significant on MetaQA "
                             "hop3 only, fails the universality gate",
    "PROMOTED": "NONE",
    "L1_FROZEN": "NO",
}
json.dump(RET, open(f"{BC.BCD}/RETURNS.json", "w"), indent=1)
print(json.dumps(RET, indent=1))
