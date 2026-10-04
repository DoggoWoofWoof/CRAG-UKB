"""L1 CANDIDATE ADMISSION PHASE -- TABLES.md + RETURNS.json, read straight from diag/."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1bc_core as BC
import _l1ca_admit as AD

CAD = AD.CAD
DS = ["metaqa", "webqsp", "musique_clean", "2wiki_clean", "hotpotqa_clean", "squad_clean"]
POOLS, PROD, SELS = AD.POOLS, AD.PROD, AD.SELS
CELLS = [f"{k}+{s}" for k in ["SAFE_POOL", "A2_HYBRID_ADMIT", "A3_INTERLEAVE", "A4_SUBSUME"] for s in SELS]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
adm = {d: J(f"{CAD}/diag/admit_{d}.json") for d in DS}
aud = {d: J(f"{CAD}/diag/audit_{d}.json") for d in DS}
st0 = J(f"{BC.BCD}/diag/step0_ALL.json")
L = []
w = L.append


def tbl(hdr, rows):
    w("| " + " | ".join(str(x) for x in hdr) + " |")
    w("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


w("# L1 CANDIDATE ADMISSION -- TABLES\n")
w("Frozen contract unchanged: P = exactly 50, B = 6, protected core 44, same substrate, no new")
w("traversal.  Every production pool holds exactly the frozen per-query candidate count K(q).")
w("`F6_PARITY_ON_SAFE_POOL` gates the harness: the re-expressed frozen selector must reproduce")
w("`KB.f6_select` on the SAFE pool for every query.\n")

w("## T0  STEP 0 -- the two B questions, and the B ladder at fixed P = 50\n")
if st0:
    tbl(["corpus / hop", "O0", "O1", "O2", "O3", "O4", "O5", "B now (O2-O1)", "B after (O4-O3)"],
        [[f"metaqa/{h}"] + [f"{st0['metaqa']['ladder'][h][k]:.4f}" for k in
                            ("O0", "O1", "O2", "O3", "O4", "O5")]
         + [f"{st0['metaqa']['B'][h]['B_CURRENT_HEADROOM_O2_O1']:.4f}",
            f"{st0['metaqa']['B'][h]['B_AFTER_FULL_VISITED_HEADROOM_O4_O3']:.4f}"]
         for h in ("hop1", "hop2", "hop3", "ALL")]
        + [[d] + [f"{st0[d]['ladder']['ALL'][k]:.4f}" for k in ("O0", "O1", "O2", "O3", "O4", "O5")]
           + [f"{st0[d]['B']['ALL']['B_CURRENT_HEADROOM_O2_O1']:.4f}",
              f"{st0[d]['B']['ALL']['B_AFTER_FULL_VISITED_HEADROOM_O4_O3']:.4f}"]
           for d in DS[1:] if st0.get(d)])
    bl = sorted(int(k) for k in st0["metaqa"]["B"]["ALL"]["SAFE_universe_B_ladder"])
    w("B ladder: protect `base_rank[:50-b]`, choose the remaining b perfectly from the universe.\n")
    tbl(["universe / hop"] + [f"b={b}" for b in bl],
        [[f"SAFE universe, metaqa/{h}"] + [f"{st0['metaqa']['B'][h]['SAFE_universe_B_ladder'][str(b)]:.4f}"
                                           for b in bl] for h in ("hop2", "hop3", "ALL")]
        + [[f"FULL_VISITED, metaqa/{h}"] + [f"{st0['metaqa']['B'][h]['FULL_VISITED_B_ladder'][str(b)]:.4f}"
                                            for b in bl] for h in ("hop2", "hop3", "ALL")])

w("## T1  STEP 1/2 -- what the missing required partitions look like at the admission stage\n")
tbl(["corpus / hop", "missing", "visited", "has SRC atom", "already admitted"],
    [[f"metaqa/{h}", adm["metaqa"]["admission"][h]["missing_total"],
      f"{adm['metaqa']['admission'][h]['missing_visited']:.4f}",
      f"{adm['metaqa']['admission'][h]['missing_with_SRC_atom']:.4f}",
      f"{adm['metaqa']['admission'][h]['missing_already_admitted']:.4f}"]
     for h in ("hop2", "hop3", "ALL")]
    + [[d, adm[d]["admission"]["ALL"]["missing_total"],
        f"{adm[d]['admission']['ALL']['missing_visited']:.4f}",
        f"{adm[d]['admission']['ALL']['missing_with_SRC_atom']:.4f}",
        f"{adm[d]['admission']['ALL']['missing_already_admitted']:.4f}"]
       for d in DS[1:] if adm[d]])

w("## T2  STEP 2/3/4 -- pool composition (all production pools are exactly K(q))\n")
tbl(["corpus", "K(q)", "A1 filled", "A2 retained SAFE", "A2 new vs SAFE", "UNION (diagnostic)"],
    [[d, f"{adm[d]['pool_sizes']['K_per_q']:.1f}", f"{adm[d]['pool_sizes']['A1_filled_per_q']:.1f}",
      f"{adm[d]['pool_sizes']['A2_retained_SAFE_per_q']:.1f}",
      f"{adm[d]['pool_sizes']['A2_new_vs_SAFE_per_q']:.1f}",
      f"{adm[d]['pool_sizes']['UNION_per_q']:.1f}"] for d in DS if adm[d]])

w("## T3  STEP 1 -- MISSING_REQUIRED_ADMISSION_RECALL by pool\n")
tbl(["corpus / hop"] + POOLS,
    [[f"metaqa/{h}"] + [f"{adm['metaqa']['admission'][h][k + '_MISSING_RECALL']:.4f}" for k in POOLS]
     for h in ("hop2", "hop3", "ALL")]
    + [[d] + [f"{adm[d]['admission']['ALL'][k + '_MISSING_RECALL']:.4f}" for k in POOLS]
       for d in DS[1:] if adm[d]])

w("## T4  STEP 5 -- candidate-pool oracle, perfect selection at B = 6, exact P50\n")
tbl(["corpus / hop"] + POOLS,
    [[f"metaqa/{h}"] + [f"{adm['metaqa']['pool_oracle'][h][k]:.4f}" for k in POOLS]
     for h in ("hop1", "hop2", "hop3", "ALL")]
    + [[d] + [f"{adm[d]['pool_oracle']['ALL'][k]:.4f}" for k in POOLS] for d in DS[1:] if adm[d]])

w("## T5  STEP 6 -- the 2x2 causal matrix, exact P50 (net / p vs SAFE_POOL+F6)\n")
for scope, rows in (("metaqa by hop", [("metaqa", h) for h in ("hop1", "hop2", "hop3", "ALL")]),
                    ("all corpora, ALL", [(d, "ALL") for d in DS])):
    w(f"**{scope}**\n")
    tbl(["row"] + CELLS,
        [[f"{d}/{h}" if scope.startswith("all") else h]
         + [(lambda v: f"{v['acc']:.4f} ({v['net']:+d}, p={v['p']:.3f})"
             + ("  SIG" if v["sig"] else ""))(adm[d]["p50_2x2"][h][cell]) for cell in CELLS]
         for d, h in rows if adm[d]])

w("## T6  STEP 7 -- does hybrid admission preserve what plain G4 lost?\n")
tbl(["corpus", "partitions plain G4 lost", "of which no SRC atom", "preserved by A2+G4",
     "no-SRC ones preserved"],
    [[d, aud[d]["preserve"]["g4_lost_partitions"], aud[d]["preserve"]["of_which_no_src"],
      f"{aud[d]['preserve']['frac_preserved_by_A2_G4']:.4f}",
      f"{aud[d]['preserve']['frac_no_src_preserved_by_A2_G4']:.4f}"] for d in DS if aud[d]])

w("## T7  STEP 7 -- changed-query audit, A2_HYBRID+G4 against each control\n")
for d in DS:
    if not aud[d]:
        continue
    w(f"**{d}**\n")
    tbl(["comparison", "gains", "losses", "gain from a NEWLY admitted partition",
         "losses with SRC", "losses without SRC", "lost but still in pool",
         "lost partition evidence (dense/splade/struct/retcont)"],
        [[k, v["n_gain"], v["n_loss"], v["gain_newly_admitted"], v["loss_has_SRC"],
          v["loss_no_SRC"], v["loss_still_in_pool"],
          "/".join(str(v["loss_evidence"][x]) for x in ("dense", "splade", "struct", "retcont"))]
         for k, v in aud[d]["audit"].items()])

w("## T8  STEP 8 -- query-local applicability, inference-safe statistics (no gate, no threshold)\n")
key = "A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6"
for d in DS:
    if not aud[d] or key not in aud[d]["applicability"]:
        continue
    a = aud[d]["applicability"][key]
    w(f"**{d}** -- gains {a['n_gain']}, losses {a['n_loss']}, unchanged {a['n_same']}\n")
    tbl(["statistic", "gain queries", "loss queries", "unchanged", "AUC gain vs loss"],
        [[s, a[s]["gain"], a[s]["loss"], a[s]["same"], a[s]["AUC_gain_vs_loss"]] for s in AD.__dict__
         and ["src_atoms", "src_targets", "src_conc", "src_safe_overlap", "src_dense_agree",
              "src_splade_agree", "src_targets_per_source", "frac_miss_with_src"]])

w("## T9  STEP 10 -- admission latency (ms/query)\n")
tbl(["corpus", "frozen admission", "A1 SRC assignment", "A2 hybrid", "F6 selector", "G4 selector"],
    [[d] + [adm[d]["latency_ms"][k] for k in ("frozen_admit", "a1", "a2", "F6", "G4")]
     for d in DS if adm[d]])
lat_l1 = J(f"{BC.BCD}/diag/lat_metaqa.json")
if lat_l1 and adm["metaqa"]:
    base = lat_l1["L1_FROZEN_TOTAL_MS"]
    ov = adm["metaqa"]["latency_ms"]["a1"] + adm["metaqa"]["latency_ms"]["a2"]
    w(f"MetaQA frozen L1 total **{base} ms/q**; admission overhead (A1+A2) **{ov:.4f} ms/q "
      f"= {100*ov/(base+ov):.2f}%** of L1.\n")

os.makedirs(CAD, exist_ok=True)
open(f"{CAD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L))

m = adm["metaqa"]
sig_reg = [f"{d}/{cell} net {adm[d]['p50_2x2']['ALL'][cell]['net']:+d} "
           f"p={adm[d]['p50_2x2']['ALL'][cell]['p']}"
           for d in DS[1:] for cell in ("A2_HYBRID_ADMIT+F6", "A2_HYBRID_ADMIT+G4")
           if adm[d] and adm[d]["p50_2x2"]["ALL"][cell]["sig"]
           and adm[d]["p50_2x2"]["ALL"][cell]["net"] < 0]
RET = {
    "PHASE": "L1 CANDIDATE ADMISSION CAUSAL",
    "CONTRACT": "no LLM/MLP/learning/new encoder/new traversal; P=50, B=6, production pool = K(q)",
    "F6_PARITY_ON_SAFE_POOL": {d: adm[d]["F6_PARITY_ON_SAFE_POOL"] for d in DS if adm[d]},
    "B_CURRENT_HEADROOM": {h: st0["metaqa"]["B"][h]["B_CURRENT_HEADROOM_O2_O1"]
                           for h in ("hop2", "hop3", "ALL")} if st0 else None,
    "B_AFTER_ADMISSION_HEADROOM": {h: st0["metaqa"]["B"][h]["B_AFTER_FULL_VISITED_HEADROOM_O4_O3"]
                                   for h in ("hop2", "hop3", "ALL")} if st0 else None,
    "SAFE_POOL_MISSING_RECALL": m["admission"]["ALL"]["SAFE_POOL_MISSING_RECALL"],
    "SRC_POOL_MISSING_RECALL": m["admission"]["ALL"]["A1_SRC_ASSIGN_MISSING_RECALL"],
    "HYBRID_POOL_MISSING_RECALL": m["admission"]["ALL"]["A2_HYBRID_ADMIT_MISSING_RECALL"],
    "UNION_POOL_MISSING_RECALL": m["admission"]["ALL"]["SAFE_U_SRC_MISSING_RECALL"],
    "SAFE_POOL_MISSING_RECALL_HOP3": m["admission"]["hop3"]["SAFE_POOL_MISSING_RECALL"],
    "HYBRID_POOL_MISSING_RECALL_HOP3": m["admission"]["hop3"]["A2_HYBRID_ADMIT_MISSING_RECALL"],
    "SAFE_POOL_ORACLE_HOP3": m["pool_oracle"]["hop3"]["SAFE_POOL"],
    "SRC_POOL_ORACLE_HOP3": m["pool_oracle"]["hop3"]["A1_SRC_ASSIGN"],
    "HYBRID_POOL_ORACLE_HOP3": m["pool_oracle"]["hop3"]["A2_HYBRID_ADMIT"],
    "UNION_POOL_ORACLE_HOP3": m["pool_oracle"]["hop3"]["SAFE_U_SRC"],
    "SAFE_F6_HOP3": m["p50_2x2"]["hop3"]["SAFE_POOL+F6"]["acc"],
    "SAFE_G4_HOP3": m["p50_2x2"]["hop3"]["SAFE_POOL+G4"]["acc"],
    "F6_HYBRID_HOP3": m["p50_2x2"]["hop3"]["A2_HYBRID_ADMIT+F6"]["acc"],
    "G4_HYBRID_HOP3": m["p50_2x2"]["hop3"]["A2_HYBRID_ADMIT+G4"]["acc"],
    "G4_HYBRID_HOP3_NET_P": [m["p50_2x2"]["hop3"]["A2_HYBRID_ADMIT+G4"]["net"],
                             m["p50_2x2"]["hop3"]["A2_HYBRID_ADMIT+G4"]["p"]],
    "G4_HYBRID_ALL": m["p50_2x2"]["ALL"]["A2_HYBRID_ADMIT+G4"]["acc"],
    "SIGNIFICANT_REGRESSIONS_OTHER_CORPORA": sig_reg,
    "CROSS_CORPUS_SAFE": "YES" if not sig_reg else "NO",
    "ADMISSION_OVERHEAD_MS": round(adm["metaqa"]["latency_ms"]["a1"]
                                   + adm["metaqa"]["latency_ms"]["a2"], 4),
}
json.dump(RET, open(f"{CAD}/RETURNS.json", "w"), indent=1)
print(json.dumps(RET, indent=1))
