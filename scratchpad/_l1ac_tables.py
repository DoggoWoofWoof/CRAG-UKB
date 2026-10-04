"""Render TABLES.md for the L1 ARCHITECTURAL CEILING AUDIT."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1pp_core as PP

AUD = f"{PP.ROOT}/L1_CEILING_AUDIT"
d = json.load(open(f"{AUD}/ceiling_audit.json"))
SHORT = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
         "musique_clean": "MuSiQue", "hotpotqa_clean": "HotpotQA", "squad_clean": "SQuAD"}
DS = list(d)
TEXT = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
L = []
w = L.append
blocks = [("MetaQA hop1", d["metaqa"]["PER_HOP"]["1"], 401),
          ("MetaQA hop2", d["metaqa"]["PER_HOP"]["2"], 401),
          ("MetaQA hop3", d["metaqa"]["PER_HOP"]["3"], 401),
          ("MetaQA aggregate", d["metaqa"]["AGGREGATE"], 401)] + \
         [(SHORT[ds], d[ds]["AGGREGATE"], d[ds]["npart"]) for ds in DS if ds != "metaqa"]

w("# L1 ARCHITECTURAL CEILING AUDIT — TABLES\n")
w("Gold is used for oracle evaluation only; nothing here is an inference method. "
  "`base_rank` replay parity vs the cached canonical ranking is **EXACT on all six corpora**.\n")

w("\n## T1 — Full-universe oracle ALL by partition budget (= the representational ceiling)\n")
w("An oracle free to pick any P partitions from the complete canonical universe covers a query "
  "exactly when its gold evidence fits in P partitions, so this column *is* "
  "`GOLD_PARTITION_COUNT <= P`. ANY is budget-independent at P>=1.\n")
w("| block | npart | P=10 | P=25 | P=50 | P=75 | P=100 | P=150 | ANY |")
w("|---|---|---|---|---|---|---|---|---|")
for tag, e, npd in blocks:
    o = e["FULL_UNIVERSE_ORACLE_ALL"]
    w(f"| {tag} | {npd} | " + " | ".join(f"{o[str(p)]:.4f}" for p in (10, 25, 50, 75, 100, 150))
      + f" | {e['ANY']:.4f} |")

w("\n## T2 — Required gold-partition count: how many partitions must be selected at all\n")
w("| block | mean | median | p75 | p90 | p95 | p99 | max | =1 | =2 | =3 | =4 | >=5 | >=10 | >25 | >50 |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag, e, _ in blocks:
    c = e["GOLD_PARTITION_COUNT"]
    w(f"| {tag} | {c['mean']:.2f} | {c['median']:.0f} | {c['p75']:.0f} | {c['p90']:.0f} | "
      f"{c['p95']:.0f} | {c['p99']:.0f} | {c['max']} | {c['frac_1']:.3f} | {c['frac_2']:.3f} | "
      f"{c['frac_3']:.3f} | {c['frac_4']:.3f} | {c['frac_5plus']:.3f} | {c['frac_10plus']:.3f} | "
      f"{c['frac_gt25']:.3f} | {c['frac_gt50']:.3f} |")

w("\n## T3 — Canonical Dense+SPLADE ranking gap\n")
w("`worst required-gold rank` is the variable that governs ALL coverage: a top-P canonical cut "
  "covers the query iff worst < P. Ranks are 0-indexed over the full universe.\n")
w("| block | npart | best med | best p90 | **worst med** | worst p75 | worst p90 | worst p95 | "
  "worst p99 | worst max |")
w("|---|---|---|---|---|---|---|---|---|---|")
for tag, e, npd in blocks:
    b, W = e["BEST_REQUIRED_GOLD_RANK"], e["WORST_REQUIRED_GOLD_RANK"]
    w(f"| {tag} | {npd} | {b['median']:.0f} | {b['p90']:.0f} | **{W['median']:.0f}** | "
      f"{W['p75']:.0f} | {W['p90']:.0f} | {W['p95']:.0f} | {W['p99']:.0f} | {W['max']} |")

w("\n## T4 — Coverage from simply taking the canonical top-P (no router at all)\n")
w("| block | P=25 | P=50 | P=75 | P=100 | P=150 | full-universe oracle P=50 | ranking gap at P=50 |")
w("|---|---|---|---|---|---|---|---|")
for tag, e, _ in blocks:
    c = e["CANONICAL_TOPP_ALL"]; o = e["FULL_UNIVERSE_ORACLE_ALL"]["50"]
    w(f"| {tag} | " + " | ".join(f"{c[str(p)]:.4f}" for p in (25, 50, 75, 100, 150))
      + f" | {o:.4f} | **{o - c['50']:+.4f}** |")

w("\n## T5 — MANDATORY MetaQA ceiling comparison\n")
K = [("BASE P50", "BASE_P50"), ("SAFE F6", "SAFE_F6"), ("R1 LIFT", "R1_LIFT"),
     ("current-pool oracle B6", "CURRENT_POOL_ORACLE_B6"),
     ("PPR-expanded-pool oracle B6", "PPR_EXPANDED_POOL_ORACLE_B6")]
w("| method / ceiling | hop1 | hop2 | hop3 | aggregate |")
w("|---|---|---|---|---|")
ri = d["metaqa"]["REFERENCE_INDICATORS"]
for tag, k in K:
    v = ri[k]
    w(f"| {tag} | " + " | ".join(f"{v['per_hop'][h]:.4f}" for h in "123")
      + f" | {v['aggregate']:.4f} |")
for p in (25, 50, 75, 100, 150):
    cells = [d["metaqa"]["PER_HOP"][h]["FULL_UNIVERSE_ORACLE_ALL"][str(p)] for h in "123"]
    ag = d["metaqa"]["AGGREGATE"]["FULL_UNIVERSE_ORACLE_ALL"][str(p)]
    bold = "**" if p == 50 else ""
    w(f"| {bold}full-universe oracle P{p}{bold} | "
      + " | ".join(f"{bold}{c:.4f}{bold}" for c in cells) + f" | {bold}{ag:.4f}{bold} |")

w("\n## T6 — Constraint ladder: which frozen constraint actually binds\n")
w("Each row keeps the partition abstraction and switches ONE constraint off. Gold-evaluated.\n")
LAD = [("actual router (SAFE F6)", "SAFE_F6"),
       ("+ perfect selector, current pool, B=6", "CURRENT_POOL_ORACLE_B6"),
       ("+ perfect selector, current pool, B unlimited", "ORACLE_CURRENT_POOL_UNLIMITED_B"),
       ("+ perfect selector, unlimited pool, B=6", "ORACLE_UNLIMITED_POOL_B6"),
       ("+ abstraction only (gold fits in 50 partitions)", "ORACLE_ABSTRACTION_ONLY_P50")]
w("| ladder step | MetaQA hop2 | MetaQA hop3 | MetaQA agg | " +
  " | ".join(SHORT[ds] for ds in DS if ds != "metaqa") + " |")
w("|---|---|---|---|" + "---|" * (len(DS) - 1))
for tag, k in LAD:
    v = ri[k]
    row = f"| {tag} | {v['per_hop']['2']:.4f} | {v['per_hop']['3']:.4f} | {v['aggregate']:.4f} | "
    row += " | ".join(f"{d[ds]['REFERENCE_INDICATORS'][k]['aggregate']:.4f}"
                      for ds in DS if ds != "metaqa")
    w(row + " |")

w("\n## T7 — WebQSP control (full-universe oracle)\n")
e = d["webqsp"]["AGGREGATE"]
w("| budget | ALL | ANY |")
w("|---|---|---|")
for p in (25, 50, 75, 100, 150):
    w(f"| P={p} | {e['FULL_UNIVERSE_ORACLE_ALL'][str(p)]:.4f} | {e['ANY']:.4f} |")
c = e["GOLD_PARTITION_COUNT"]
w(f"\nRequired gold-partition count: mean {c['mean']:.2f}, median {c['median']:.0f}, "
  f"p90 {c['p90']:.0f}, p99 {c['p99']:.0f}, max {c['max']}; "
  f"{c['frac_1']:.1%} need 1, {c['frac_5plus']:.1%} need >=5, {c['frac_gt50']:.1%} need >50. "
  f"npart = {d['webqsp']['npart']}.\n")

w("\n## T8 — Text controls\n")
w("| corpus | npart | BASE P50 | SAFE router | full-universe oracle P50 | full-universe oracle P100 | ranking gap |")
w("|---|---|---|---|---|---|---|")
for ds in TEXT:
    r = d[ds]["REFERENCE_INDICATORS"]; o = d[ds]["AGGREGATE"]["FULL_UNIVERSE_ORACLE_ALL"]
    w(f"| {SHORT[ds]} | {d[ds]['npart']} | {r['BASE_P50']['aggregate']:.4f} | "
      f"{r['SAFE_F6']['aggregate']:.4f} | {o['50']:.4f} | {o['100']:.4f} | "
      f"**{o['50'] - r['SAFE_F6']['aggregate']:+.4f}** |")

w("\n## T9 — MetaQA hop3 architectural decomposition\n")
h3 = d["metaqa"]["HOP3_DECOMPOSITION"]
w(f"All {h3['n_uncovered_hop3']} hop3 queries the SAFE router fails, assigned to exactly one "
  f"category. Precedence: {h3['precedence']}\n")
w("| category | definition | count | % of uncovered hop3 |")
w("|---|---|---|---|")
DEF = {"A_REPRESENTATION_LIMIT": "not coverable at any budget (gold unmapped / no gold)",
       "B_P50_CAPACITY_LIMIT": "more than 50 gold partitions — needs a larger P",
       "D_CURRENT_POOL_LIMIT": "a required partition is outside boundary ∪ challengers",
       "C_RANKING_LIMIT": "all required in pool, but >B=6 of them sit outside the protected core",
       "E_SELECTOR_LIMIT": "all required in pool and fits B=6 — the selector chose wrong"}
for k in ["A_REPRESENTATION_LIMIT", "B_P50_CAPACITY_LIMIT", "D_CURRENT_POOL_LIMIT",
          "C_RANKING_LIMIT", "E_SELECTOR_LIMIT"]:
    w(f"| **{k}** | {DEF[k]} | {h3['counts'][k]} | {h3['pct'][k]:.2f}% |")
w(f"\nOverlap disclosure: of the {h3['D_total']} category-D queries, "
  f"**{h3['D_queries_that_ALSO_need_more_than_B_swaps']} ({h3['D_frac_also_B_bound']:.1%})** would "
  f"*also* fail on the B=6 swap cap even with an unlimited candidate pool. Widening the pool alone "
  f"does not fix them.\n")

w("\n## T10 — R1 ruling support: gain against each available headroom (MetaQA)\n")
w("| level | SAFE F6 | R1 | R1 gain | full-universe P50 oracle | R1 gain / P50-oracle headroom | "
  "current-pool B6 oracle | R1 gain / selector headroom |")
w("|---|---|---|---|---|---|---|---|")
for tag, key in (("hop2", "2"), ("hop3", "3"), ("aggregate", None)):
    f6 = ri["SAFE_F6"]["per_hop"][key] if key else ri["SAFE_F6"]["aggregate"]
    r1 = ri["R1_LIFT"]["per_hop"][key] if key else ri["R1_LIFT"]["aggregate"]
    oc = ri["CURRENT_POOL_ORACLE_B6"]["per_hop"][key] if key else \
        ri["CURRENT_POOL_ORACLE_B6"]["aggregate"]
    src = d["metaqa"]["PER_HOP"][key] if key else d["metaqa"]["AGGREGATE"]
    o50 = src["FULL_UNIVERSE_ORACLE_ALL"]["50"]
    w(f"| {tag} | {f6:.4f} | {r1:.4f} | {r1 - f6:+.4f} | {o50:.4f} | "
      f"**{(r1 - f6) / (o50 - f6):.1%}** | {oc:.4f} | {(r1 - f6) / (oc - f6):.1%} |")

open(f"{AUD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote TABLES.md", len(L), "lines")
