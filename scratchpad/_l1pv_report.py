"""CORE-EXIT PROVENANCE AUDIT -- aggregation and the STEP-2 universality verdict.

Reads results/.../L1_PROVENANCE/diag/{step1_families.json, pv_<ds>.json} and writes TABLES.md.
Pure assembly: no model, no fitting, no threshold search.

  python scratchpad/_l1pv_report.py
"""
import os, sys, json
import numpy as np

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PROVENANCE"
D = f"{KTD}/diag"
ORD = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
NICE = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
        "musique_clean": "MuSiQue", "hotpotqa_clean": "Hotpot", "squad_clean": "SQuAD"}
SC = ["SRC_ALL", "SRC_NER", "SRC_STRUCT_ONLY", "ALL_NER", "ALL_STRUCT_ONLY"]
BUCK = ["ONLY_NER", "ONLY_STRUCT", "MULTI_FAMILY", "NO_CORE_EXIT_EVIDENCE"]
PL = "A1_SRC"
# the seven evaluation rows the directive names
ROWS = [("MetaQA hop2", "metaqa", 2), ("MetaQA hop3", "metaqa", 3), ("WebQSP", "webqsp", None),
        ("2Wiki", "2wiki_clean", None), ("MuSiQue", "musique_clean", None),
        ("Hotpot", "hotpotqa_clean", None), ("SQuAD", "squad_clean", None)]


def f(x, n=4):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def cell(o, key, hop):
    if hop is None:
        return o["STEP2_swap"].get(f"{PL}/{key}")
    return o["STEP2_swap_by_hop"].get(f"{PL}/{key}/hop{hop}")


def main():
    FAMS = json.load(open(f"{D}/step1_families.json"))
    P = {d: json.load(open(f"{D}/pv_{d}.json")) for d in ORD if os.path.exists(f"{D}/pv_{d}.json")}
    have = [d for d in ORD if d in P]
    L = []
    A = L.append
    A("# L1 CORE-EXIT PROVENANCE AUDIT -- TABLES\n")
    A(f"Corpora with a completed audit pass: **{len(have)}/6** "
      f"({', '.join(NICE[d] for d in have)}).\n")

    # ---------------- STEP 1 ----------------------------------------------------------------
    A("\n## STEP 1 -- EDGE FAMILY LEDGER\n")
    A("### 1a. Corpus-level: which families exist in the FROZEN traversal adjacency\n")
    A("The traversal adjacency is `master_nodes_{ds}.json` `neighbors`, i.e. **STRUCT**. "
      "`A = gte_qwen/graph.pt` is `STRUCT u KNN`; `KNN = A \\ STRUCT`; `NER = ner_edges_w_df25.pkl`. "
      "Undirected canonical keys, the existing `ac_edge_recon` convention.\n")
    A("| corpus | nodes | traversal edges | also NER | frac NER | also KNN | not in A | "
      "STRUCT provenance (canonical manifest) |")
    A("|---|--:|--:|--:|--:|--:|--:|---|")
    for d in ORD:
        s = FAMS.get(d)
        if not s:
            continue
        cm = s.get("canonical_struct_manifest") or {}
        prov = f"`{cm.get('edge_family')}`"
        if cm.get("edge_subtype"):
            prov += f" / `{cm['edge_subtype']}`"
        if cm.get("n_relations"):
            prov += f" / {cm['n_relations']} rel"
        A(f"| {NICE[d]} | {s['N']:,} | {s['traversal_undirected_edges']:,} | "
          f"{s['traversal_edges_also_NER']:,} | **{s['frac_traversal_also_NER']*100:.2f}%** | "
          f"{s['traversal_edges_also_KNN']} | {s.get('traversal_not_in_A', 0):,} | {prov} |")
    A("\n**KNN is structurally absent from L1 traversal on all six corpora (0 edges everywhere).** "
      "The frozen beam expands `nd.neighbors` only, so the semantic/Qwen-kNN family can never "
      "appear in a core-exit transition. The only live provenance axis is "
      "`STRUCT_ONLY` vs `STRUCT_AND_NER`.\n")

    A("\n### 1b. Per-query core-exit TRANSITION counts, by family\n")
    A("| corpus | queries | parity | edges/q | core-exit/q | core-exit NER/q | "
      "core-exit STRUCT_ONLY/q | frac NER (mean) | frac NER (median) |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for d in have:
        o = P[d]
        t = o["STEP1_transitions_per_query"]
        A(f"| {NICE[d]} | {o['n_queries']} | {o['parity']} | {t['all_edges']['mean']:,.0f} | "
          f"{t['core_exit']['mean']:,.0f} | {t['core_exit_NER']['mean']:,.0f} | "
          f"{t['core_exit_STRUCT_ONLY']['mean']:,.0f} | "
          f"**{t['frac_core_exit_NER']['mean']:.4f}** | "
          f"{t['frac_core_exit_NER']['median']:.4f} |")

    A("\n### 1c. Core-exit transitions by hop, and their NER share\n")
    A("| corpus | " + " | ".join(f"hop{h}" for h in (1, 2, 3)) + " | " +
      " | ".join(f"NER frac hop{h}" for h in (1, 2, 3)) + " |")
    A("|---|" + "--:|" * 6)
    for d in have:
        o = P[d]
        h1 = o["STEP1_core_exit_by_hop"]
        h2 = o["STEP1_core_exit_NER_frac_by_hop"]
        A(f"| {NICE[d]} | " + " | ".join(f"{h1.get(str(h), 0):,}" for h in (1, 2, 3)) + " | " +
          " | ".join(f(h2.get(str(h)), 4) for h in (1, 2, 3)) + " |")
    if "metaqa" in P and P["metaqa"].get("STEP1_by_query_hop"):
        A("\nMetaQA per-query, split by question hop:\n")
        A("| question hop | core-exit/q | core-exit NER/q |")
        A("|---|--:|--:|")
        for h, v in P["metaqa"]["STEP1_by_query_hop"].items():
            A(f"| {h} | {v['core_exit']['mean']:,.0f} | {v['core_exit_NER']['mean']:,.0f} |")

    # ---------------- STEP 2 ----------------------------------------------------------------
    A("\n## STEP 2 -- GOOD/BAD SWAP BY EDGE FAMILY\n")
    A(f"Pool `{PL}`. GOOD/BAD definitions verbatim from the prior phase "
      "(`_l1kt_partb.run`): at identical K(q) the SRC pool evicts SAFE candidates and admits new "
      "ones; a labelled pair has exactly one required-and-missing side. Every family is scored on "
      "the **same** pair population -- the pools are family-independent, only the score changes. "
      "`AUC > 0.5` means the family separates GOOD from BAD in the right direction.\n")
    for key in SC:
        A(f"\n### {key}\n")
        A("| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | "
          "AUC (both have evidence) |")
        A("|---|--:|--:|--:|--:|--:|--:|--:|")
        for nm, d, hop in ROWS:
            if d not in P:
                continue
            c = cell(P[d], key, hop)
            if c is None:
                continue
            A(f"| {nm} | {c['n_good']} | {c['n_bad']} | {f(c['mean_good'])} | {f(c['mean_bad'])} | "
              f"**{f(c['AUC'])}** | {f(c['sign_acc'])} | {f(c.get('AUC_both_evid'))} |")

    A("\n### IS ANY EDGE FAMILY SIGN-CONSISTENT ACROSS CORPORA?\n")
    A("A family is sign-consistent if its AUC lands on the same side of 0.5 on every eligible row "
      "(a row is eligible when it has at least one GOOD and one BAD pair, so the AUC is defined).\n")
    A("| family | eligible rows | AUC > 0.5 | AUC < 0.5 | min AUC | max AUC | SIGN-CONSISTENT? |")
    A("|---|--:|--:|--:|--:|--:|---|")
    verd = {}
    for key in SC:
        vals = []
        for nm, d, hop in ROWS:
            if d not in P:
                continue
            c = cell(P[d], key, hop)
            if c and c["AUC"] is not None:
                vals.append((nm, c["AUC"]))
        hi = [x for x in vals if x[1] > 0.5]
        lo = [x for x in vals if x[1] < 0.5]
        ok = len(vals) > 1 and (len(lo) == 0 or len(hi) == 0)
        verd[key] = {"n_eligible": len(vals), "n_above": len(hi), "n_below": len(lo),
                     "min": min((x[1] for x in vals), default=None),
                     "max": max((x[1] for x in vals), default=None),
                     "sign_consistent": bool(ok), "rows": dict(vals)}
        A(f"| {key} | {len(vals)} | {len(hi)} | {len(lo)} | "
          f"{f(verd[key]['min'])} | {f(verd[key]['max'])} | "
          f"{'**YES**' if ok else 'NO'} |")

    # ---------------- STEP 3 ----------------------------------------------------------------
    A("\n## STEP 3 -- EXCLUSIVE SUPPORT\n")
    A("Each admitted candidate is bucketed by which families supply its core-exit evidence. "
      "`gain/loss` is `good_swaps / bad_swaps` -- above 1 means candidates in that bucket are more "
      "often the required-and-missing side than the collateral side.\n")
    A("| corpus | " + " | ".join(f"{b}<br>good / bad / ratio" for b in BUCK) + " |")
    A("|---|" + "---|" * len(BUCK))
    for d in have:
        o = P[d]["STEP3_exclusive_support"]
        cs = []
        for b in BUCK:
            v = o.get(f"{PL}/{b}", {})
            r = v.get("gain_loss_ratio")
            cs.append(f"{v.get('good_swaps', 0)} / {v.get('bad_swaps', 0)} / "
                      f"{'--' if r is None else (f'{r:.3f}' if np.isfinite(r) else 'inf')}")
        A(f"| {NICE[d]} | " + " | ".join(cs) + " |")

    # ---------------- STEP 4 ----------------------------------------------------------------
    A("\n## STEP 4 -- SOURCE QUALITY (diagnostic only)\n")
    A("`p_required` = P(admitted candidate is required-and-missing | bin), attributed to its best "
      "supporting edge in that family. Bin edges are fixed by the frozen contract "
      "(`prot` = canonical ranks 0..43 in four equal blocks, `bnd` = 44..49), not searched. "
      "A universal lever would need the same monotone direction on every corpus.\n")
    for axis, ttl in (("core_source", "source in protected core (1) vs not (0)"),
                      ("hop", "hop of the supporting edge"),
                      ("src_cpos_quartile", "source partition canonical rank"),
                      ("src_conf_quartile", "source retrieval confidence quartile (T1_SOURCE)")):
        A(f"\n### {axis} -- {ttl}\n")
        bins = sorted({b for d in have for fam in ("NER", "STRUCT_ONLY")
                       for b in P[d]["STEP4_source_quality"].get(fam, {}).get(axis, {})})
        A("| corpus | family | " + " | ".join(f"{b}<br>p_req (n)" for b in bins) + " |")
        A("|---|---|" + "---|" * len(bins))
        for d in have:
            for fam in ("NER", "STRUCT_ONLY"):
                r = P[d]["STEP4_source_quality"].get(fam, {}).get(axis, {})
                if not r:
                    continue
                cs = [(f"{r[b]['p_required']:.4f} ({r[b]['n']})" if b in r else "--") for b in bins]
                A(f"| {NICE[d]} | {fam} | " + " | ".join(cs) + " |")

    # ---------------- STEPS 5-7 --------------------------------------------------------------
    W = {d: json.load(open(f"{D}/pvswap_{d}.json")) for d in ORD
         if os.path.exists(f"{D}/pvswap_{d}.json")}
    if W:
        A("\n## STEPS 5-7 -- FAMILY-RESTRICTED ASSIGNMENT AT EXACT P50\n")
        A("Fixed-K replacement, `K = K0`, `B = 6`, `P = 50`; the ordering key is the "
          "family-restricted provenance score and nothing else. `SRC_ALL` is the parity anchor: it "
          "is `V6_EXIT_SRC_SIM` from the prior phase and must reproduce it exactly.\n")
        A("\n### Candidate coverage -- what fraction of the K(q) universe the family can score "
          "at all\n")
        A("| corpus | " + " | ".join(SC) + " |")
        A("|---|" + "--:|" * len(SC))
        for d in W:
            c = W[d]["candidate_evidence_coverage"]
            A(f"| {NICE[d]} | " + " | ".join(f(c.get(k)) for k in SC) + " |")
        for sel in ("F6", "G4"):
            A(f"\n### Exact P50 coverage, selector `{sel}` (net = gained - lost vs FROZEN)\n")
            A("| corpus | slice | FROZEN | " + " | ".join(SC) + " |")
            A("|---|---|--:|" + "--:|" * len(SC))
            for d in W:
                for nm, dd in W[d]["p50"].items():
                    cs = []
                    for k in SC:
                        v = dd.get(f"{k}/{sel}")
                        cs.append("--" if v is None else
                                  f"{v['acc']:.4f} {v['net']:+d}"
                                  f"{' **SIG**' if v['sig'] else ''}")
                    A(f"| {NICE[d]} | {nm} | {dd['FROZEN']:.4f} | " + " | ".join(cs) + " |")

    os.makedirs(KTD, exist_ok=True)
    open(f"{KTD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"step2_universality": verd, "corpora": have, "pvswap_corpora": sorted(W)},
              open(f"{D}/_report_derived.json", "w"), indent=1)
    print(f"wrote {KTD}/TABLES.md  ({len(L)} lines)")
    for k, v in verd.items():
        print(f"  {k:18s} eligible={v['n_eligible']} above={v['n_above']} below={v['n_below']} "
              f"consistent={v['sign_consistent']}")


if __name__ == "__main__":
    main()
