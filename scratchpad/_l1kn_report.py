"""L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- table assembly.

Reads scratchpad/_l1kn/run_{ds}.json (STEPS 2-7, 9, 10) and
results/.../L1_KNN/diag/k8_{ds}.json (STEPS 8, 11) and writes TABLES.md + diag/_derived.json.

  python scratchpad/_l1kn_report.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1kn_sub as SUB

KND, CACHE, DS, SUBS = SUB.KND, SUB.CACHE, SUB.DS, SUB.SUBS
SHORT = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
         "musique_clean": "MuSiQue", "hotpotqa_clean": "Hotpot", "squad_clean": "SQuAD"}
L = []
w = L.append


def have(ds):
    return os.path.exists(f"{CACHE}/run_{ds}.json")


def load(ds):
    R = json.load(open(f"{CACHE}/run_{ds}.json"))
    kp = f"{KND}/diag/k8_{ds}.json"
    R["_k8"] = json.load(open(kp)) if os.path.exists(kp) else None
    return R


def masks(ds, R):
    n = R["nq"]
    if ds == "metaqa":
        h = np.asarray(R["hq"])
        return [("hop1", h == 1), ("hop2", h == 2), ("hop3", h == 3), ("ALL", np.ones(n, bool))]
    return [("ALL", np.ones(n, bool))]


def main():
    D = {d: load(d) for d in DS if have(d)}
    sub = json.load(open(f"{KND}/diag/step1_substrates.json"))
    der = {}
    w("# L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- TABLES\n")
    w(f"Corpora with a completed replay: **{', '.join(SHORT[d] for d in D)}**"
      f"{'' if len(D) == 6 else ' (INCOMPLETE)'}.\n")

    # ------------------------------------------------------------------ STEP 1
    w("\n## STEP 1 -- EDGE SUBSTRATES\n")
    w("### 1a. The three precomputed families, and how much of topology C the traversal never sees\n")
    w("| corpus | N | STRUCT (traversed) | KNN | NERX | KNN&NERX | edges of C never traversed |")
    w("|---|--:|--:|--:|--:|--:|--:|")
    for d in DS:
        if d not in sub:
            continue
        s = sub[d]
        w(f"| {SHORT[d]} | {s['N']:,} | {s['STRUCT_undirected']:,} | {s['KNN_undirected']:,} | "
          f"{s['NERX_undirected']:,} | {s.get('KNN_and_NERX_overlap', 0):,} | "
          f"**{s['frac_topologyC_edges_never_traversed']:.1%}** |")
    der["FRAC_TOPOLOGY_C_NEVER_TRAVERSED"] = {
        d: sub[d]["frac_topologyC_edges_never_traversed"] for d in sub}

    w("\n### 1b. The traversal substrates\n")
    w("| corpus | substrate | undirected edges | mean deg | max deg | isolated | over DEG_CAP |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for d in DS:
        if d not in sub:
            continue
        for t in SUBS:
            m = sub[d]["subs"][t]
            extra = (f" (kNN slots {m['frac_slots_knn']:.1%})" if "frac_slots_knn" in m else "")
            w(f"| {SHORT[d]} | `{t}`{extra} | {m['undirected_edges']:,} | {m['mean_deg']:.2f} | "
              f"{m['max_deg']:,} | {m['isolated_nodes']:,} | {m['nodes_over_DEG_CAP']:,} |")

    # ------------------------------------------------------------------ STEP 2
    w("\n## STEP 2 -- PARITY\n")
    w("`T0_FROZEN` must reproduce frozen L1 exactly: the structural rank map `spos` identical on "
      "every query, and the exact-P50 outcome identical on every query.\n")
    w("| corpus | nq | `spos` exact | P50 exact | T0_PARITY | seeds/q | edges/q | nodes/q | "
      "partitions/q | hops |")
    w("|---|--:|--:|--:|---|--:|--:|--:|--:|--:|")
    par = {}
    for d, R in D.items():
        p, a = R["PAR"], R["A"]["T0_FROZEN"]
        hp = p["hops_reached"]
        w(f"| {SHORT[d]} | {R['nq']} | {p['spos_exact']}/{p['nq']} | {p['p50_exact']}/{p['nq']} | "
          f"**{p['T0_PARITY']}** | {p['seeds_per_q']:.2f} | {np.mean(a['edges']):,.0f} | "
          f"{np.mean(a['vis_nodes']):,.0f} | {np.mean(a['vis_parts']):,.1f} | "
          f"{int(np.argmax(hp))} |")
        par[d] = p["T0_PARITY"]
    der["T0_PARITY"] = par

    # ------------------------------------------------------------------ STEP 3
    w("\n## STEP 3 -- UNIQUE REACH\n")
    w("Every gold partition MISSED by the frozen SAFE output, classified by which substrate "
      "reaches it. `VISITED` = anything the bounded search scored; `READ` = what the M_struct=64 "
      "read actually hands the S4 aggregation.  `KNN_UNIQUE` is the primary number.\n")
    for lev in ("VIS", "READ"):
        w(f"\n**{ 'VISITED' if lev=='VIS' else 'READ (M_struct=64)'}**\n")
        w("| corpus / slice | missed needed | STRUCT only | **KNN only** | both | neither | "
          "NERX only | **KNN_UNIQUE_NEEDED_RECALL** |")
        w("|---|--:|--:|--:|--:|--:|--:|--:|")
        for d, R in D.items():
            rows = [("ALL", R["CLS"])] + ([(k, v) for k, v in sorted(R["CLSH"].items())]
                                          if d == "metaqa" else [])
            for nm, C in rows:
                c = C[lev]
                tot = c["STRUCT_ONLY"] + c["KNN_ONLY"] + c["BOTH"] + c["NEITHER"]
                r = c["KNN_ONLY"] / max(tot, 1)
                lbl = SHORT[d] + ("" if nm == "ALL" else f" {nm}")
                w(f"| {lbl} | {tot:,} | {c['STRUCT_ONLY']:,} | **{c['KNN_ONLY']:,}** | "
                  f"{c['BOTH']:,} | {c['NEITHER']:,} | {c['NERX_ONLY']:,} | **{r:.4f}** |")
                if nm == "ALL":
                    der.setdefault(f"KNN_UNIQUE_NEEDED_RECALL_{lev}", {})[d] = round(r, 4)
                else:
                    der.setdefault(f"KNN_UNIQUE_NEEDED_RECALL_{lev}_metaqa_hops", {})[nm] = \
                        round(r, 4)

    w("\n### 3c. What turning the family ON actually buys (needed-partition recall vs `T0`)\n")
    w("`T1` runs at a FRACTION of `T0`'s edge budget (the kNN graph is sparse), so its standalone "
      "reach understates the family. `T2`/`T5` are the operational question: does adding the family "
      "to the frozen traversal move needed recall?\n")
    w("| corpus / slice | level | T0 | T1 kNN-only | T2 +kNN | T3 matched | T4 NERX-only | "
      "T5 topology C | **T2 - T0** |")
    w("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for d, R in D.items():
        nt = np.asarray(R["need_tot"])
        for nm, m in masks(d, R):
            for lev in ("VIS", "READ"):
                vals = {t: np.asarray(R["A"][t]["need_" + lev])[m].sum() / max(nt[m].sum(), 1)
                        for t in SUBS}
                lbl = SHORT[d] + ("" if nm == "ALL" else f" {nm}")
                w(f"| {lbl} | {lev} | " + " | ".join(f"{vals[t]:.4f}" for t in SUBS)
                  + f" | **{vals['T2_FULL_UNION'] - vals['T0_FROZEN']:+.4f}** |")
                key = d if nm == "ALL" else f"{d}/{nm}"
                der.setdefault(f"NEEDED_RECALL_{lev}", {})[key] = {
                    t: round(float(vals[t]), 4) for t in SUBS}

    # ------------------------------------------------------------------ STEP 4
    w("\n## STEP 4 -- DENSE REDUNDANCY\n")
    w("Every partition the kNN-only traversal reaches, tested against the channels the frozen "
      "system already has. `novel` = in none of dense / SPLADE / canonical-200 / retrieval "
      "continuation / structural-T0.\n")
    w("| corpus | kNN partitions | in Dense | in SPLADE | in canon-200 | in RF | in struct T0 | "
      "**novel** | needed | needed&novel | P(needed \\| novel) | P(needed \\| dup) |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for d, R in D.items():
        r = R["RED"]; n = max(r["knn_parts"], 1)
        nv = max(r["novel"], 1); dp = max(r["knn_parts"] - r["novel"], 1)
        w(f"| {SHORT[d]} | {r['knn_parts']:,} | {r['in_dense']/n:.1%} | {r['in_splade']/n:.1%} | "
          f"{r['in_canon200']/n:.1%} | {r['in_RF']/n:.1%} | {r['in_struct_T0']/n:.1%} | "
          f"**{r['novel']/n:.1%}** | {r['needed']:,} | {r['novel_needed']:,} | "
          f"{r['novel_needed']/nv:.4f} | {r['dup_needed']/dp:.4f} |")
        der.setdefault("KNN_PARTITION_NOVELTY", {})[d] = round(r["novel"] / n, 4)
        der.setdefault("KNN_DENSE_REDUNDANCY", {})[d] = round(r["in_dense"] / n, 4)
        der.setdefault("KNN_NEEDED_ENRICHMENT_NOVEL_VS_DUP", {})[d] = [
            round(r["novel_needed"] / nv, 5), round(r["dup_needed"] / dp, 5)]

    # ------------------------------------------------------------------ STEP 5
    w("\n## STEP 5 -- EDGE-FAMILY SATURATION\n")
    w("| corpus | substrate | nodes visited/q | partitions visited/q | % of corpus partitions | "
      "admitted partitions/q | read partitions/q |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for d, R in D.items():
        for t in SUBS:
            a = R["A"][t]
            fr = np.mean(a["vis_parts"]) / R["npart"]
            w(f"| {SHORT[d]} | `{t}` | {np.mean(a['vis_nodes']):,.0f} | "
              f"{np.mean(a['vis_parts']):,.1f} | **{fr:.1%}** | {np.mean(a['add_parts']):.1f} | "
              f"{np.mean(a['read_parts']):.1f} |")
            der.setdefault("PARTITIONS_PER_QUERY", {}).setdefault(d, {})[t] = \
                round(float(np.mean(a["vis_parts"])), 2)
            der.setdefault("SATURATION_FRAC", {}).setdefault(d, {})[t] = round(float(fr), 4)

    # ------------------------------------------------------------------ STEP 6 + 9
    w("\n## STEPS 6 + 9 -- MATCHED WORK AND CANDIDATE ORACLES\n")
    w("`T3` spends the SAME per-node edge budget as `T0` (same `adj_ptr`, same `deg`, same "
      "DEG_CAP), reallocated between families. Needed-partition recall is per gold partition; the "
      "candidate oracle asks whether ANY B=6 subset of that substrate's F6 pool could cover the "
      "query.\n")
    w("| corpus / slice | substrate | edges/q | needed recall VIS | needed recall READ | "
      "candidate oracle | partitions/q | exact P50 |")
    w("|---|---|--:|--:|--:|--:|--:|--:|")
    for d, R in D.items():
        nt = np.asarray(R["need_tot"])
        for nm, m in masks(d, R):
            for t in SUBS:
                a = R["A"][t]
                nv = np.asarray(a["need_VIS"])[m].sum() / max(nt[m].sum(), 1)
                nr = np.asarray(a["need_READ"])[m].sum() / max(nt[m].sum(), 1)
                lbl = SHORT[d] + ("" if nm == "ALL" else f" {nm}")
                w(f"| {lbl} | `{t}` | {np.mean(np.asarray(a['edges'])[m]):,.0f} | {nv:.4f} | "
                  f"{nr:.4f} | {np.mean(np.asarray(a['orac'])[m]):.4f} | "
                  f"{np.mean(np.asarray(a['vis_parts'])[m]):,.1f} | "
                  f"{np.mean(np.asarray(a['p50'])[m]):.4f} |")
                key = d if nm == "ALL" else f"{d}/{nm}"
                der.setdefault("ORACLE", {}).setdefault(key, {})[t] = \
                    round(float(np.mean(np.asarray(a["orac"])[m])), 4)
                der.setdefault("SUBSTITUTION_P50", {}).setdefault(key, {})[t] = \
                    round(float(np.mean(np.asarray(a["p50"])[m])), 4)

    # ------------------------------------------------------------------ STEP 7
    w("\n## STEP 7 -- PATH PROVENANCE\n")
    w("Family sequence of every path `T2_FULL_UNION` actually explored (the `parent_tid` chain of "
      "each scored edge). `needed-target rate` = fraction of that sequence's edges whose target "
      "node lies in a gold partition.\n")
    w("A raw comparison across sequences is confounded by LENGTH -- a hop-1 edge lands on a gold "
      "partition far more often than a hop-3 edge whatever family it belongs to -- so every "
      "sequence is compared only against sequences of the SAME length.\n")
    for d, R in D.items():
        P7 = R["PATH"]
        tot = sum(v["paths"] for v in P7.values())
        w(f"\n**{SHORT[d]}** ({tot:,} scored edges)\n")
        w("| length | family sequence | paths | share | needed-target rate | distinct needed "
          "partitions |")
        w("|--:|---|--:|--:|--:|--:|")
        best, bv = None, -1
        for ln in (1, 2, 3):
            rows = sorted([(k, v) for k, v in P7.items() if k.count(">") == ln - 1],
                          key=lambda kv: -kv[1]["paths"])
            if not rows:
                continue
            for k, v in rows[:7]:
                rt = v["needed_targets"] / max(v["targets"], 1)
                w(f"| {ln} | `{k}` | {v['paths']:,} | {v['paths']/max(tot,1):.2%} | {rt:.5f} | "
                  f"{v['needed_parts']:,} |")
            for nmz, sel in (("**STRUCT-pure**", lambda k: "KNN" not in k and "NERX" not in k),
                             ("**contains KNN**", lambda k: "KNN" in k)):
                g = [v for k, v in rows if sel(k)]
                pth = sum(v["paths"] for v in g)
                tg = sum(v["targets"] for v in g)
                nd = sum(v["needed_targets"] for v in g)
                rt = nd / max(tg, 1)
                w(f"| {ln} | {nmz} (all) | {pth:,} | {pth/max(tot,1):.2%} | **{rt:.5f}** | |")
                if nmz.startswith("**contains") and pth:
                    der.setdefault("PATH_LEN_MATCHED", {}).setdefault(d, {})[f"len{ln}_knn"] = \
                        round(rt, 5)
                elif pth:
                    der.setdefault("PATH_LEN_MATCHED", {}).setdefault(d, {})[f"len{ln}_struct"] = \
                        round(rt, 5)
            for k, v in rows:
                rt = v["needed_targets"] / max(v["targets"], 1)
                if v["paths"] >= 0.01 * tot and rt > bv:
                    bv, best = rt, k
        der.setdefault("BEST_PATH_FAMILY_SEQUENCE", {})[d] = best

    # ------------------------------------------------------------------ STEP 8 + 11
    if any(R["_k8"] for R in D.values()):
        w("\n## STEP 8 -- KNN AS ITS OWN EVIDENCE FAMILY\n")
        w("| corpus | queries with a kNN edge | kNN partitions/q | needed reached | "
          "missed-needed reached |")
        w("|---|--:|--:|--:|--:|")
        for d, R in D.items():
            k = R["_k8"]
            if not k:
                continue
            z = k["K0"]
            w(f"| {SHORT[d]} | {z['queries']:,} | {z['parts_reached']/max(z['queries'],1):,.1f} | "
              f"{z['needed_reached']}/{z['needed_total']} "
              f"({z['needed_reached']/max(z['needed_total'],1):.1%}) | "
              f"{z['missed_reached']}/{z['missed_total']} "
              f"({z['missed_reached']/max(z['missed_total'],1):.1%}) |")
        w("\n| corpus | signal | mean needed | mean nuisance | AUC needed vs nuisance | "
          "AUC missed vs nuisance |")
        w("|---|---|--:|--:|--:|--:|")
        for d, R in D.items():
            k = R["_k8"]
            if not k:
                continue
            for s, v in k["K8"].items():
                w(f"| {SHORT[d]} | `{s}` | {v['mean_needed']} | {v['mean_nuisance']} | "
                  f"**{v['AUC_needed_vs_nuisance']}** | {v['AUC_missed_vs_nuisance']} |")
                der.setdefault("K8_AUC", {}).setdefault(d, {})[s] = v["AUC_needed_vs_nuisance"]

        w("\n## STEP 11 -- COST\n")
        w("Uniform instrumentation (`want_edges=True` for every substrate, so the timings are "
          "comparable). `edges/q` is the adjacency volume inspected; `scored/q` the edges that "
          "survived DEG_CAP and entered the matvec.\n")
        w("| corpus | substrate | edges/q | scored/q | scope/q | ms/q | x T0 edges | x T0 ms |")
        w("|---|---|--:|--:|--:|--:|--:|--:|")
        for d, R in D.items():
            k = R["_k8"]
            if not k:
                continue
            b = k["COST"]["T0_FROZEN"]
            for t in SUBS:
                c = k["COST"][t]
                w(f"| {SHORT[d]} | `{t}` | {c['edges_per_q']:,.0f} | {c['scored_per_q']:,.0f} | "
                  f"{c['scope_per_q']:,.0f} | {c['ms_per_q']:.2f} | "
                  f"{c['edges_per_q']/max(b['edges_per_q'],1e-9):.2f} | "
                  f"{c['ms_per_q']/max(b['ms_per_q'],1e-9):.2f} |")
                der.setdefault("COST", {}).setdefault(d, {})[t] = c

    # ------------------------------------------------------------------ STEP 10
    w("\n## STEP 10 -- EXACT P50\n")
    w("The frozen channels are untouched; the kNN structural ranking is added as a FOURTH "
      "symmetric RRF channel. `K_CHAN_VOTE` lets it re-score the frozen candidate list only; "
      "`K_CHAN_FULL` also lets it contribute challengers. `*` = McNemar significant.\n")
    w("| corpus / slice | FROZEN | K_CHAN_VOTE | K_CHAN_FULL | T2 substitution | T3 substitution |")
    w("|---|--:|--:|--:|--:|--:|")
    for d, R in D.items():
        base = np.asarray(R["base"])
        for nm, m in masks(d, R):
            cells = []
            for v in ("K_CHAN_VOTE", "K_CHAN_FULL"):
                cur = np.asarray(R["TEN"][v])
                st = PP.mcnemar(cur[m], base[m])
                cells.append(f"{cur[m].mean():.4f} ({st['net']:+d}{'*' if st['sig'] else ''})")
                key = d if nm == "ALL" else f"{d}/{nm}"
                der.setdefault("STEP10", {}).setdefault(key, {})[v] = {
                    "acc": round(float(cur[m].mean()), 4), "net": st["net"],
                    "gained": st["gained"], "lost": st["lost"],
                    "p": round(st["mcnemar_p"], 4), "sig": bool(st["sig"])}
            for t in ("T2_FULL_UNION", "T3_MATCHED_HYBRID"):
                cur = np.asarray(R["A"][t]["p50"]).astype(np.int8)
                st = PP.mcnemar(cur[m], base[m])
                cells.append(f"{cur[m].mean():.4f} ({st['net']:+d}{'*' if st['sig'] else ''})")
                key = d if nm == "ALL" else f"{d}/{nm}"
                der.setdefault("STEP10_SUBSTITUTION", {}).setdefault(key, {})[t] = {
                    "acc": round(float(cur[m].mean()), 4), "net": st["net"],
                    "p": round(st["mcnemar_p"], 4), "sig": bool(st["sig"])}
            lbl = SHORT[d] + ("" if nm == "ALL" else f" {nm}")
            w(f"| {lbl} | {base[m].mean():.4f} | " + " | ".join(cells) + " |")

    os.makedirs(f"{KND}/diag", exist_ok=True)
    json.dump(der, open(f"{KND}/diag/_derived.json", "w"), indent=1)
    open(f"{KND}/TABLES.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {KND}/TABLES.md ({len(L)} lines) and diag/_derived.json")


if __name__ == "__main__":
    main()
