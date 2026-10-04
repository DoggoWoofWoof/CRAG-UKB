"""Assemble the six-dataset COMBINED-residual table from results/GENERALIZATION/_g2_comb_*.json."""
import json, os

DSETS = [("metaqa", "MetaQA", "KB"), ("webqsp", "WebQSP", "KB"),
         ("2wiki_clean", "2Wiki", "text"), ("musique_clean", "MuSiQue", "text"),
         ("hotpotqa_clean", "HotpotQA", "text"), ("squad_clean", "SQuAD", "text")]
D = {}
for k, lab, fam in DSETS:
    p = f"results/GENERALIZATION/_g2_comb_{k}.json"
    D[k] = json.load(open(p)) if os.path.exists(p) else None

L = []
w = L.append


def cell(d, arm, f="dALL"):
    if d is None or arm not in d["RESULTS"]:
        return "—"
    v = d["RESULTS"][arm]
    if f == "dALL":
        return f"{v['ALL']:.4f} ({v['dALL_vs_BASE']:+.4f})"
    if f == "d":
        return f"{v['dALL_vs_BASE']:+.4f}"
    if f == "net":
        return f"{v['queries_newly_ALL_covered']:+d}"
    return f"{v[f]}"


w("# G2 FINAL L1 EXPERIMENT — COMBINED PARAMETER-FREE RESIDUAL (six corpora)\n")
w("`BASE P50 + retrieval residual + structural residual`, all additive (BASE never evicted).\n")

w("\n## 0. Substrate + protocol\n")
w("| corpus | family | dev n | sample | docs | parts | BASE scope | scope % corpus | golds/q | parity |")
w("|---|---|--:|---|--:|--:|--:|--:|--:|:--|")
for k, lab, fam in DSETS:
    d = D[k]
    if d is None:
        w(f"| {lab} | {fam} | — | *NOT RUN* | | | | | | |"); continue
    sc = d["BASE_SCOPE_NODES"]
    w(f"| {lab} | {fam} | {d['n_dev_queries']} | {d['sample_rule'].split(' (')[0]} | {d['n_docs']:,} | "
      f"{d['npart']:,} | {sc:.0f} | {100.0*sc/d['n_docs']:.1f}% | "
      f"{d['RESULTS']['BASE'].get('n', '')and''}{d.get('_g','')} | {d['BASE_PARITY']['status']} |")

w("\n## 1. THE SIX-DATASET TABLE (ALL@scope, Δ vs BASE)\n")
w("| corpus | BASE | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED 32+32 | uniq added | scope growth |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for k, lab, fam in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | | | | |"); continue
    C = d["RESULTS"]["COMBINED_32_32"]
    w(f"| **{lab}** ({fam}) | {d['BASE_ALL_P50']:.4f} | {cell(d,'RET32')} | {cell(d,'STRUCT32')} | "
      f"{cell(d,'RET64')} | {cell(d,'STRUCT64')} | **{cell(d,'COMBINED_32_32')}** | "
      f"{C['unique_added_nodes_mean']:.1f} | {C['scope_growth_pct']:.2f}% |")

w("\n## 2. Queries newly ALL-covered / gold occurrences recovered\n")
w("| corpus | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED | RANDOM matched |")
w("|---|--:|--:|--:|--:|--:|--:|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | | |"); continue
    r = d["RESULTS"]
    def q(a):
        v = r[a]
        return f"{v['queries_newly_ALL_covered']} / {v['newly_recovered_gold_occurrences']}"
    w(f"| **{lab}** | {q('RET32')} | {q('STRUCT32')} | {q('RET64')} | {q('STRUCT64')} | "
      f"**{q('COMBINED_32_32')}** | {q('RANDOM_matchedCOMBINED')} |")

w("\n## 3. Overlap + marginal complementarity\n")
w("| corpus | \\|RET32∩STRUCT32\\| | overlap frac | ret-only | struct-only | unique | "
  "STRUCT after RET | RET after STRUCT | additivity gap |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | | | | |"); continue
    o, m = d["OVERLAP"], d["MARGINAL"]
    sa = m["paired_COMBINED_vs_RET32"]; ra = m["paired_COMBINED_vs_STRUCT32"]
    w(f"| **{lab}** | {o['mean_RET32_INTERSECT_STRUCT32']:.2f} | {o['overlap_fraction_of_32']:.1%} | "
      f"{o['retrieval_only_mean']:.1f} | {o['structural_only_mean']:.1f} | {o['unique_added_mean']:.1f} | "
      f"{m['STRUCT_MARGINAL_AFTER_RET']:+.4f} (+{sa['net']}, p={sa['mcnemar_p']}) | "
      f"{m['RET_MARGINAL_AFTER_STRUCT']:+.4f} (+{ra['net']}, p={ra['mcnemar_p']}) | "
      f"{m['additivity_gap']:+.4f} |")

w("\n## 4. COMBINED vs the single-family 64-node budgets (matched total budget)\n")
w("| corpus | COMBINED − RET64 | paired | COMBINED − STRUCT64 | paired |")
w("|---|--:|:--|--:|:--|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | |"); continue
    m = d["MARGINAL"]; a = m["paired_COMBINED_vs_RET64"]; b = m["paired_COMBINED_vs_STRUCT64"]
    w(f"| **{lab}** | {m['COMBINED_minus_RET64']:+.4f} | net {a['net']:+}, p={a['mcnemar_p']}"
      f"{' **sig**' if a['sig_p<0.05'] else ''} | {m['COMBINED_minus_STRUCT64']:+.4f} | "
      f"net {b['net']:+}, p={b['mcnemar_p']}{' **sig**' if b['sig_p<0.05'] else ''} |")

w("\n## 5. ANY@scope + availability + monotonicity check\n")
w("| corpus | BASE ANY | COMBINED ANY | struct avail/q | ret avail/q (rrf200) | q with <64 ret | worsened (all arms) |")
w("|---|--:|--:|--:|--:|--:|--:|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | | |"); continue
    ws = sum(v.get("queries_worsened", 0) for v in d["RESULTS"].values())
    w(f"| **{lab}** | {d['BASE_ANY_P50']:.4f} | {d['RESULTS']['COMBINED_32_32']['ANY']:.4f} | "
      f"{d['structural_available_per_query']:.1f} | {d['retrieval_available_per_query']['rrf200']:.1f} | "
      f"{d['queries_with_fewer_than_64_retrieval_candidates']['rrf200']} | {ws} |")

w("\n## 6. Cost\n")
w("| corpus | BASE s | RET build s | STRUCT traversal s | ×BASE | edges traversed | "
  "COMBINED increment over STRUCT32 |")
w("|---|--:|--:|--:|--:|--:|--:|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | | |"); continue
    c = d["COST"]
    w(f"| **{lab}** | {c['BASE_sec']} | {c['RET_build_sec']} | {c['STRUCT_traversal_sec']} | "
      f"{c['STRUCT_x_BASE']}× | {c['STRUCT_edges_traversed']:,} | "
      f"+{c['COMBINED_incremental_sec_above_STRUCT32']}s ({c['COMBINED_incremental_pct_above_STRUCT32']}%) |")

d = D["metaqa"]
if d and "per_hop" in d["RESULTS"]["BASE"]:
    w("\n## 7. MetaQA per-hop ALL\n")
    w("| arm | hop1 | hop2 | hop3 |")
    w("|---|--:|--:|--:|")
    for a in ["BASE", "RET32", "STRUCT32", "RET64", "STRUCT64", "COMBINED_32_32"]:
        ph = d["RESULTS"][a]["per_hop"]
        w(f"| {a} | " + " | ".join(f"{ph[h]['ALL']:.4f}" for h in ("1", "2", "3")) + " |")

w("\n## 8. Retrieval-channel sensitivity (is rrf200 the right canonical channel?)\n")
w("| corpus | RET32 rrf200 | RET32 dense200 | RET32 splade200 | COMBINED rrf200 | COMBINED dense200 |")
w("|---|--:|--:|--:|--:|--:|")
for k, lab, _ in DSETS:
    d = D[k]
    if d is None:
        w(f"| **{lab}** | *NOT RUN* | | | | |"); continue
    w(f"| **{lab}** | {cell(d,'RET32_rrf200','d')} | {cell(d,'RET32_dense200','d')} | "
      f"{cell(d,'RET32_splade200','d')} | {cell(d,'COMBINED_32_32','d')} | "
      f"{cell(d,'COMBINED_32_32_dense200','d')} |")

out = "results/GENERALIZATION/_g2_comb_tables.md"
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote " + out)
