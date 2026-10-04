"""Emit the L1 ARCHITECTURE DECISION tables from _g2_resid_{ds}.json + _g2_priv_metaqa.json."""
import json, os, sys
import numpy as np

P = "results/GENERALIZATION"
DS = [("metaqa", "MetaQA"), ("2wiki_clean", "2wiki"), ("musique_clean", "MuSiQue"), ("squad_clean", "SQuAD")]
M_B = [8, 16, 32, 64]
PS_B = [1, 2, 4, 8]
J = {}
for d, _n in DS:
    f = f"{P}/_g2_resid_{d}.json"
    if os.path.exists(f):
        J[d] = json.load(open(f))
OUT = []
w = OUT.append


def cell(r, key="ALL"):
    if r is None:
        return "—"
    s = f"{r[key]:.4f}"
    p = r.get("paired_vs_BASE_ALL", {})
    if p.get("sig_p<0.05"):
        s = f"**{s}**"
    return s


def dcell(r):
    if r is None:
        return "—"
    d = r["dALL_vs_BASE"]; p = r.get("paired_vs_BASE_ALL", {})
    net = p.get("net")
    s = f"{d:+.4f}"
    if net is not None:
        s += f" ({net:+d})"
    if p.get("sig_p<0.05"):
        s = "**" + s + "**"
    return s


# ------------------------------------------------------------------ 1. BASE
w("## 1. BASE coverage (STEP 0) — authoritative canonical L1, parity-gated\n")
w("| corpus | n dev q | docs | partitions | BASE_ANY_P50 | BASE_ALL_P50 | BASE_SCOPE_NODES | "
  "scope % of corpus | BASE_PARTITION_PARITY |")
w("|---|--:|--:|--:|--:|--:|--:|--:|:--|")
for d, n in DS:
    if d not in J:
        w(f"| {n} | — | — | — | — | — | — | — | NOT RUN |"); continue
    j = J[d]
    w(f"| **{n}** | {j['n_dev_queries']} | {j['n_docs']:,} | {j['npart']} | {j['BASE_ANY_P50']:.4f} | "
      f"{j['BASE_ALL_P50']:.4f} | {j['BASE_SCOPE_NODES']:.0f} | "
      f"{100*j['BASE_SCOPE_NODES']/j['n_docs']:.1f} % | `{j['BASE_PARTITION_PARITY']['status']}` |")
if "metaqa" in J:
    ph = J["metaqa"]["RESULTS"]["BASE"]["per_hop"]
    w(f"\nMetaQA per-hop BASE ALL: hop1 {ph['1']['ALL']:.4f} (n={ph['1']['n']}) · "
      f"hop2 {ph['2']['ALL']:.4f} (n={ph['2']['n']}) · hop3 {ph['3']['ALL']:.4f} (n={ph['3']['n']})")

# ------------------------------------------------------------------ 2. NODE arms
w("\n## 2/3/5. NODE arms — matched added-node budget M (STEPS 3B, 4, 6)\n")
w("Every arm adds **exactly M nodes** on top of the same BASE P50 union; BASE is never evicted, so "
  "ΔALL ≥ 0 by construction. Bold = McNemar p<0.05 vs BASE. `(+n)` = net queries newly ALL-covered.\n")
ROWS = [("RANDOM_NODE_ADD", "RANDOM_NODE (5 seeds)"),
        ("RETRIEVAL_NODE_ADD_dense", "RETRIEVAL_NODE dense (exact)"),
        ("RETRIEVAL_NODE_ADD_splade200", "RETRIEVAL_NODE splade200"),
        ("RETRIEVAL_NODE_ADD_rrf200", "RETRIEVAL_NODE rrf200"),
        ("STRUCT_NODE_ADD_RRF", "**STRUCT_NODE** RRF seeds"),
        ("STRUCT_NODE_ADD_DENSE", "STRUCT_NODE dense seeds"),
        ("STRUCT_NODE_ADD_SPLADE", "STRUCT_NODE splade seeds"),
        ("STRUCT_NODE_ADD_RRFdeep", "STRUCT_NODE RRF, beam 256")]
for d, n in DS:
    if d not in J:
        continue
    R = J[d]["RESULTS"]
    w(f"\n### {n} — BASE ALL {J[d]['BASE_ALL_P50']:.4f}\n")
    w("| arm | " + " | ".join(f"M={m}" for m in M_B) + " |")
    w("|---|" + "--:|" * len(M_B))
    for k, lab in ROWS:
        w(f"| {lab} | " + " | ".join(dcell(R.get(f"{k}_M{m}")) for m in M_B) + " |")

# ------------------------------------------------------------------ 6. PARTITION arms
w("\n## 6. PARTITION arms — additive out-of-P50 partitions (STEP 3C)\n")
w("`STRUCT_PART` maps each out-of-BASE structural node to its canonical C partition, converts node rank "
  "j to the fixed RRF contribution 1/(K0+j), sums per partition and takes the top-Ps. BASE P50 is never "
  "evicted. Added-node counts are ~Ps × mean partition size, so the node-matched controls below are the "
  "load-bearing comparison.\n")
PROWS = [("RANDOM_PART_ADD", "RANDOM_PART (5 seeds)"),
         ("RETRIEVAL_PART_ADD", "RETRIEVAL_PART (fused rank 50..50+Ps)"),
         ("STRUCT_PART_ADD_RRF", "**STRUCT_PART** RRF seeds"),
         ("STRUCT_PART_ADD_RRFdeep", "STRUCT_PART RRF, beam 256"),
         ("RANDOM_NODE_matchedPART", "· node-matched RANDOM"),
         ("RETRIEVAL_NODE_dense_matchedPART", "· node-matched RETRIEVAL dense"),
         ("STRUCT_NODE_matchedPART", "· node-matched STRUCT_NODE")]
for d, n in DS:
    if d not in J:
        continue
    R = J[d]["RESULTS"]
    w(f"\n### {n} — BASE ALL {J[d]['BASE_ALL_P50']:.4f}, mean partition {J[d]['mean_partition_size']:.0f} nodes\n")
    w("| arm | " + " | ".join(f"Ps={p}" for p in PS_B) + " |")
    w("|---|" + "--:|" * len(PS_B))
    for k, lab in PROWS:
        w(f"| {lab} | " + " | ".join(dcell(R.get(f"{k}_Ps{p}")) for p in PS_B) + " |")
    w("| *added nodes* | " + " | ".join(
        f"{R[f'STRUCT_PART_ADD_RRF_Ps{p}']['added_nodes_mean']:.0f}" for p in PS_B) + " |")
    w("| *scope growth* | " + " | ".join(
        f"{R[f'STRUCT_PART_ADD_RRF_Ps{p}']['scope_growth_pct']:.1f} %" for p in PS_B) + " |")

# ------------------------------------------------------------------ 7. STEP 7
w("\n## 7. STEP 7 — structure-specific gain at matched scope\n")
w("| corpus | budget | ΔALL STRUCT | ΔALL RANDOM | ΔALL RETRIEVAL | over RANDOM | over RETRIEVAL |")
w("|---|---|--:|--:|--:|--:|--:|")
for d, n in DS:
    if d not in J:
        continue
    S = J[d]["STEP7_structure_specific_gain"]
    for m in M_B:
        k = f"NODE_M{m}"
        if k not in S:
            continue
        v = S[k]
        w(f"| {n} | NODE M={m} | {v['dALL_STRUCT']:+.4f} | {v['dALL_RANDOM']:+.4f} | "
          f"{v['dALL_RETRIEVAL_best']:+.4f} ({v['best_retrieval_channel']}) | "
          f"{v['STRUCT_NODE_GAIN_OVER_RANDOM']:+.4f} | {v['STRUCT_NODE_GAIN_OVER_RETRIEVAL']:+.4f} |")
    for p in PS_B:
        k = f"PART_Ps{p}"
        if k not in S:
            continue
        v = S[k]
        w(f"| {n} | PART Ps={p} | {v['dALL_STRUCT']:+.4f} | {v['dALL_RANDOM']:+.4f} | "
          f"{v['dALL_RETRIEVAL']:+.4f} | {v['STRUCT_PART_GAIN_OVER_RANDOM']:+.4f} | "
          f"{v['STRUCT_PART_GAIN_OVER_RETRIEVAL']:+.4f} |")

# ------------------------------------------------------------------ 8. MetaQA hop table
if "metaqa" in J:
    R = J["metaqa"]["RESULTS"]; PREP = J["metaqa"]["PREPARTITION_REFERENCE"]
    w("\n## 8. The central MetaQA hop table (STEP 6)\n")
    w("| arm | added nodes | hop1 ALL | hop2 ALL | hop3 ALL |")
    w("|---|--:|--:|--:|--:|")

    def hrow(lab, r, add=None):
        ph = r["per_hop"]
        a = r.get("added_nodes_mean", add)
        w(f"| {lab} | {a if a is not None else '—'} | " +
          " | ".join(f"{ph[h]['ALL']:.4f}" for h in ("1", "2", "3")) + " |")
    hrow("BASE", R["BASE"])
    _c = [k for k in PREP if not k.startswith("_") and not k.startswith("A1b")
          and not k.endswith("_M0")]
    bestp = max(_c, key=lambda k: PREP[k]["ALL@P50"]) if _c else None
    if bestp and PREP[bestp].get("per_hop"):
        ph = PREP[bestp]["per_hop"]
        w(f"| PREPARTITION ({bestp}) | 0 (substitution) | " +
          " | ".join(f"{ph[h]['ALL']:.4f}" for h in ("1", "2", "3")) + " |")
    for k, lab in (("RANDOM_NODE_ADD_M32", "RANDOM_NODE +32"),
                   ("RETRIEVAL_NODE_ADD_dense_M32", "RETRIEVAL_NODE +32 (dense)"),
                   ("STRUCT_NODE_ADD_RRF_M32", "**STRUCT_NODE +32**"),
                   ("RANDOM_PART_ADD_Ps4", "RANDOM_PART +4"),
                   ("RETRIEVAL_PART_ADD_Ps4", "RETRIEVAL_PART +4"),
                   ("STRUCT_PART_ADD_RRF_Ps4", "**STRUCT_PART +4**")):
        if k in R:
            hrow(lab, R[k])
    pf = f"{P}/_g2_priv_metaqa.json"
    if os.path.exists(pf):
        PJ = json.load(open(pf))
        D2 = PJ["PART2_controlled_decomposition"]
        for k in ("PRIVILEGED_bracket|KB_privileged|UNBOUNDED", "PRIVILEGED_bracket|KB_privileged|M32"):
            if k in D2:
                r = D2[k]; ph = r["per_hop_ALL"]
                w(f"| *PRIVILEGED_OLD ({k.split('|')[-1]})* | {r['added_nodes_mean']:.0f} | " +
                  " | ".join(f"{ph[h]:.4f}" for h in ("1", "2", "3")) + " |")

# ------------------------------------------------------------------ 9. cross-dataset
w("\n## 9. Cross-dataset table at one global operating point (STEP 10)\n")
w("| corpus | BASE ALL | PREPARTITION best | STRUCT_NODE +32 | STRUCT_PART +4 | "
  "RETRIEVAL_NODE +32 | RANDOM_NODE +32 |")
w("|---|--:|--:|--:|--:|--:|--:|")
for d, n in DS:
    if d not in J:
        continue
    j = J[d]; R = j["RESULTS"]; PREP = j["PREPARTITION_REFERENCE"]
    cands = [k for k in PREP if not k.startswith("_") and not k.startswith("A1b")
             and not k.endswith("_M0")]
    bp = max(cands, key=lambda k: PREP[k]["ALL@P50"]) if cands else None
    pv = f"{PREP[bp]['ALL@P50']:.4f} ({PREP[bp]['net']:+d}, {bp})" if bp else "—"
    w(f"| **{n}** | {j['BASE_ALL_P50']:.4f} | {pv} | {dcell(R.get('STRUCT_NODE_ADD_RRF_M32'))} | "
      f"{dcell(R.get('STRUCT_PART_ADD_RRF_Ps4'))} | {dcell(R.get('RETRIEVAL_NODE_ADD_dense_M32'))} | "
      f"{dcell(R.get('RANDOM_NODE_ADD_M32'))} |")

# ------------------------------------------------------------------ 10. cost
w("\n## 10. Cost table (STEP 9)\n")
w("| corpus | method | edges traversed | structural nodes evaluated | wall s | × BASE | "
  "residual available/q |")
w("|---|---|--:|--:|--:|--:|--:|")
for d, n in DS:
    if d not in J:
        continue
    C = J[d]["COST"]; b = C["BASE"]["sec"]
    w(f"| **{n}** | BASE router | 0 | 0 | {b:.1f} | 1.0 | — |")
    for k, v in C.items():
        if k == "BASE":
            continue
        w(f"| | {k} | {v['edges_traversed']:,} | {v['structural_nodes_evaluated']:,} | {v['sec']:.1f} | "
          f"{v['sec']/max(b,1e-9):.1f}× | {v['residual_available_per_query']:.1f} |")

# ------------------------------------------------------------------ 11. privileged decomposition
pf = f"{P}/_g2_priv_metaqa.json"
if os.path.exists(pf):
    PJ = json.load(open(pf))
    w("\n## 11. Exact decomposition of the old MetaQA gain (STEP 5)\n")
    r1 = PJ["PART1_exact_reproduction"]
    w(f"**Part 1 — exact reproduction of the published run.** `MATCHES_PUBLISHED = "
      f"{r1['MATCHES_PUBLISHED']}`\n")
    w("| hop | mine P50→UNION | published P50→UNION | added nodes (mine / published) |")
    w("|---|--:|--:|--:|")
    for h in (1, 2, 3):
        a = r1[f"hop{h}"]; b = r1["PUBLISHED"][f"hop{h}"]
        w(f"| hop{h} | {a['P50_ALL']:.4f} → {a['UNION_ALL']:.4f} | {b['P50_ALL']:.4f} → "
          f"{b['UNION_ALL']:.4f} | {a['added_mean']:.0f} / {b['added_mean']:.0f} |")
    D2 = PJ["PART2_controlled_decomposition"]; meta = PJ["_meta"]
    w(f"\n**Part 2 — controlled decomposition** (same traversal implementation, main dev sample "
      f"n={meta['n_dev_queries']}, BASE ALL {meta['BASE_ALL']:.4f}). Only the seed source and the "
      f"adjacency vary.\n")
    w("| seeds | adjacency | " + " | ".join(f"M={m}" for m in M_B) +
      " | UNBOUNDED | added (unb.) | hop2 | hop3 |")
    w("|---|---|" + "--:|" * (len(M_B) + 4))
    ph0 = D2["BASE"]["per_hop_ALL"]
    w(f"| — | BASE | " + " | ".join(f"{D2['BASE']['ALL']:.4f}" for _ in M_B) +
      f" | {D2['BASE']['ALL']:.4f} | 0 | {ph0['2']:.4f} | {ph0['3']:.4f} |")
    for k in [x for x in D2 if x.endswith("UNBOUNDED")]:
        pre = k.rsplit("|", 1)[0]; sd, aj = pre.split("|")
        u = D2[k]; ph = u["per_hop_ALL"]
        w(f"| {sd} | {aj} | " + " | ".join(f"{D2[f'{pre}|M{m}']['ALL']:.4f}" for m in M_B) +
          f" | {u['ALL']:.4f} | {u['added_nodes_mean']:.0f} | {ph['2']:.4f} | {ph['3']:.4f} |")

txt = "\n".join(OUT)
open(f"{P}/_g2_resid_tables.md", "w", encoding="utf-8", newline="\n").write(txt)
print(txt)
