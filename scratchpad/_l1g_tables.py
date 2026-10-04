"""Markdown tables for results/L1_GEOM/REPORT.md from the lane's JSON records (no computation, no rounding beyond the records)."""
import json
import os

import _l1g_core as G

CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
LAB = {"metaqa": "metaqa MtK", "metaqa_phg": "metaqa PHG", "squad": "squad MtK", "squad_phg": "squad PHG", "musique": "musique PHG"}


def load(kind, c):
    p = os.path.join(G.OUT, "%s_A_%s.json" % (kind, c))
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


L = {c: load("ladder", c) for c in CACHES}
Dd = {c: load("dir", c) for c in CACHES}
H = {c: load("headroom", c) for c in CACHES}
have = [c for c in CACHES if L[c] is not None]


def cell(rec, tag):
    if rec is None or tag not in rec["arms"]:
        return "—"
    a = rec["arms"][tag]
    if a["gained"] == 0 and a["lost"] == 0:
        return "%.4f" % a["ALL"]
    star = "**" if a["p"] < 0.05 else ""
    return "%s%.4f%s (+%d/−%d p=%.0e)" % (star, a["ALL"], star, a["gained"], a["lost"], a["p"])


ROWS = [
    ("P0/P1", "BASE = Dense(q)+SPLADE(q), frozen RRF, P50", "L", "P BASE (frozen RRF)"),
    ("F1a", "evidence-masked RRF", "L", "F F1a evidence-masked RRF"),
    ("F1b", "RRF × MNZ", "L", "F F1b RRF x MNZ"),
    ("F1c", "round-robin interleave", "L", "F F1c interleave (round robin)"),
    ("F1d", "agreement blend (set overlap A_q)", "L", "F F1d agreement blend (set overlap)"),
    ("F1e", "agreement blend (reciprocal-rank A_q)", "L", "F F1e agreement blend (reciprocal-rank agreement)"),
    ("G1", "+ Dense(μ) channel, frozen RRF", "L", "G G1 centroid mu | + BASE | F0 frozen RRF"),
    ("G2", "+ Dense(2q−μ) channel, frozen RRF", "L", "G G2 extrapolated 2q - mu | + BASE | F0 frozen RRF"),
    ("G3", "+ Dense(q−μ) channel, frozen RRF", "L", "G G3 residual direction q - mu | + BASE | F0 frozen RRF"),
    ("G3e", "+ Dense(q−μ), agreement blend F1e", "L", "G G3 residual direction q - mu | + BASE | F1e agreement blend (reciprocal-rank agreement)"),
    ("G4", "+ Dense(q̂ = q + P(q−μ)) channel, frozen RRF", "L", "G G4 projected q_hat = q + P(q - mu) | + BASE | F0 frozen RRF"),
    ("G5", "+ shifted seeds e_i + P(q−μ), frozen RRF", "L", "G G5 shifted seeds e_i + P(q - mu) | + BASE | F0 frozen RRF"),
    ("G12", "+ Dense(μ) + Dense(2q−μ), frozen RRF", "L", "G12 mu + (2q - mu) | + BASE | F0 frozen RRF"),
    ("G12e", "+ Dense(μ) + Dense(2q−μ), agreement blend F1e", "L", "G12 mu + (2q - mu) | + BASE | F1e agreement blend (reciprocal-rank agreement)"),
    ("G124", "+ μ + 2q−μ + q̂, frozen RRF", "L", "G124 mu + (2q - mu) + q_hat | + BASE | F0 frozen RRF"),
    ("D1b", "directional per-seed node RRF → table (alone)", "D", "D1b per-seed node RRF -> table | alone"),
    ("D1d", "directional per-seed block max → RRF (alone)", "D", "D1d per-seed block max -> RRF over seeds | alone"),
    ("D2a", "BASE + directional mean-cos nodes → table", "D", "D2a BASE + D1a (frozen RRF, 3 ch)"),
    ("D2b", "BASE + directional per-seed node RRF → table", "D", "D2b BASE + D1b (frozen RRF, 3 ch)"),
    ("D2c", "BASE + directional block max over actual nodes", "D", "D2c BASE + D1c (frozen RRF, 3 ch)"),
    ("D2d", "BASE + directional per-seed block max → RRF", "D", "D2d BASE + D1d (frozen RRF, 3 ch)"),
]
print("| arm | mechanism | " + " | ".join(LAB[c] for c in have) + " |")
print("|---|---|" + "---|" * len(have))
for short, desc, kind, tag in ROWS:
    recs = L if kind == "L" else Dd
    print("| %s | %s | %s |" % (short, desc, " | ".join(cell(recs[c], tag) for c in have)))

print()
print("| DEV_A | " + " | ".join(LAB[c] for c in have) + " |")
print("|---|" + "---|" * len(have))
for k, lab in (("feasible_P50", "feasible (≤ P50 gold blocks)"), ("reach_any_evidence", "every gold block reached by a served hit"), ("union_of_channel_top50", "union of the two channels' top-50")):
    print("| %s | %s |" % (lab, " | ".join("%.4f" % L[c]["diag"]["ceilings"][k] for c in have)))
for k, lab in (("BASE_ALL", "BASE ALL"), ("fail", "fail"), ("fail_infeasible", "  infeasible"), ("fail_unreached", "  UNREACHED (needs new evidence)"), ("fail_reached_weak", "  reached but outside both top-50s"),
               ("fail_fusion_fixable", "  fusion-fixable (inside the union of top-50s)"), ("gold_blocks_per_query", "gold blocks / query"), ("unreached_gold_blocks_per_failed_query", "unreached gold blocks / failed query")):
    print("| %s | %s |" % (lab, " | ".join(("%.4f" % H[c][k]) if H.get(c) else "—" for c in have)))
print("| BASE by A_q quartile (low→high) | %s |" % " | ".join(str(L[c]["diag"]["agreement"]["BASE_ALL_by_A_set_quartile"]) for c in have))
print("| A_q mean | %s |" % " | ".join("%.3f" % L[c]["diag"]["agreement"]["A_set_mean"] for c in have))

print()
oc = [c for c in have if Dd.get(c)]
print("| gold node in top-100 (DEV_A) | " + " | ".join(LAB[c] for c in oc) + " |")
print("|---|" + "---|" * len(oc))
for k, lab in (("q_with_gold_node_in_top100_dense_q", "Dense(q)"), ("q_with_gold_node_in_top100_directional_mean", "directional mean over seeds"), ("q_with_gold_node_in_top100_directional_rrf", "directional per-seed RRF"),
               ("q_with_gold_node_in_top100_directional_seed0_only", "directional seed-1 only"), ("q_with_gold_node_in_top100_directional_max_over_seeds", "directional max over seeds"), ("q_with_gold_node_in_top100_either", "Dense(q) or directional mean"),
               ("gold_best_rank_under_mean_cos_median", "median best gold rank under directional mean (of N)")):
    print("| %s | %s |" % (lab, " | ".join(str(Dd[c]["diag"]["orientation"][k]) for c in oc)))
for c in oc:
    bh = Dd[c]["diag"]["orientation"].get("by_hop")
    if bh:
        print("| by hop (%s) | %s |" % (LAB[c], "; ".join("%s: dense %.3f dir_rrf %.3f dir_mean %.3f" % (h, v["dense_q"], v["dir_rrf"], v["dir_mean"]) for h, v in bh.items())))
print()
print("| new evidence (DEV_A) | " + " | ".join(LAB[c] for c in have) + " |")
print("|---|" + "---|" * len(have))
for kind, key, lab in (("L", "G2 extrapolated 2q - mu", "2q−μ: q gaining an unreached gold block / reach after"), ("L", "G3 residual direction q - mu", "q−μ: same"), ("L", "G4 projected q_hat = q + P(q - mu)", "q̂: same"),
                       ("D", "D1b per-seed node RRF", "directional per-seed RRF: same")):
    recs = L if kind == "L" else Dd
    vals = []
    for c in have:
        ne = (recs[c] or {}).get("diag", {}).get("new_evidence", {}).get(key)
        vals.append("%.3f / %.3f" % (ne["q_gaining_unreached_gold_block"], ne["reach_after"]) if ne else "—")
    print("| %s | %s |" % (lab, " | ".join(vals)))
pe = os.path.join(G.OUT, "partition_effect_A.json")
if os.path.exists(pe):
    P = json.load(open(pe, encoding="utf-8"))
    print()
    print("| paired DEV_A (same queries) | Mt-KaHyPar | PHG | gained/lost by PHG | p |")
    print("|---|---|---|---|---|")
    for ds, e in P.items():
        for arm, v in e["arms"].items():
            print("| %s — %s | %.4f | %.4f | +%d/−%d | %.1e |" % (ds, arm, v["MTKAHYPAR"], v["PHG"], v["gained_by_PHG"], v["lost_by_PHG"], v["p"]))
        for tag in ("MTKAHYPAR", "PHG"):
            g = e[tag]
            print("| %s %s geometry | blocks %d, sizes [%d, %d], gold blocks/q %.3f, feasible %.4f, directed cut-edge fraction %.4f | | | |" % (ds, tag, g["blocks"], g["size_min"], g["size_max"], g["gold_blocks_per_query_DEV_A"], g["feasible_P50_DEV_A"], g["directed_struct_edges_cut_fraction"]))

# ---- directional controls (DEV_A) and combinations
ctl = {c: load("dircontrol", c) for c in CACHES}
cc = [c for c in CACHES if ctl.get(c)]
if cc:
    print()
    print("| directional controls (DEV_A, third channel + BASE via frozen RRF / alone) | " + " | ".join(LAB[c] for c in cc) + " |")
    print("|---|" + "---|" * len(cc))
    print("| BASE | %s |" % " | ".join("%.4f" % ctl[c]["arms"]["D0 BASE (frozen RRF)"]["ALL"] for c in cc))
    for k, lab in (("true", "true direction q − e_s (= D1d)"), ("reversed", "reversed direction e_s − q"), ("random", "random unit direction"),
                   ("qmax", "exhaustive dense block max (no seed, no direction)"), ("pooled", "true direction, pool = dense∪SPLADE top-1000"), ("size", "block size")):
        row = []
        for c in cc:
            a2 = ctl[c]["arms"]["C2 %-8s | + BASE (frozen RRF, 3 ch)" % k]
            a1 = ctl[c]["arms"]["C1 %-8s | alone" % k]
            row.append("%s / alone %.4f" % (cell({"arms": {"x": a2}}, "x"), a1["ALL"]))
        print("| %s | %s |" % (lab, " | ".join(row)))
    print("| block sizes min / median / max | %s |" % " | ".join("%d / %d / %d" % (ctl[c]["block_sizes"]["min"], ctl[c]["block_sizes"]["median"], ctl[c]["block_sizes"]["max"]) for c in cc))
cb = {c: load("combo", c) for c in CACHES}
cc = [c for c in CACHES if cb.get(c)]
if cc:
    print()
    print("| combinations (DEV_A) | " + " | ".join(LAB[c] for c in cc) + " |")
    print("|---|" + "---|" * len(cc))
    for tag in cb[cc[0]]["arms"]:
        print("| %s | %s |" % (tag, " | ".join(cell(cb[c], tag) for c in cc)))
    for c in cc:
        for k, v in cb[c].get("pairwise_DEV_A", {}).items():
            print("| paired (%s) %s | +%d/−%d p=%.1e |" % (LAB[c], k, v["gained"], v["lost"], v["p"]))
