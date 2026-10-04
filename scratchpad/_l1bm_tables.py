"""Emit TABLES.md straight from the diag JSONs.  Every cell is read, never retyped.

  python scratchpad/_l1bm_tables.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1bm_core as BM

D = f"{BM.BMD}/diag"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]


def J(name):
    p = f"{D}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def tbl(hdr, rows):
    """a markdown table with a guaranteed-rectangular body."""
    n = len(hdr)
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * n) + "|"]
    for r in rows:
        r = [("" if x is None else str(x)) for x in r]
        assert len(r) == n, f"ragged row {len(r)} vs {n}: {r}"
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def f4(x):
    return "" if x is None else f"{x:.4f}"


def main():
    L = ["# L1 BEAM RECOVERY -- tables", "",
         "All values are read directly out of `diag/*.json`.  `SAFE` is the frozen incumbent",
         "(N0_STATIC beam 64 -> S4 -> F6 -> exact P50).", ""]

    # ---------------------------------------------------------------- T1 parity
    L += ["## T1  STEP 1 -- baseline parity gate", ""]
    rows = []
    for ds in DSETS:
        j = J(f"step1_{ds}")
        if not j:
            continue
        rows.append([ds, j["nq"], f"{j['all_field_parity']}/{j['nq']}",
                     f"{j['s4_ranking_parity']}/{j['nq']}",
                     f4(j["SAFE"]["ALL"]), f4(j["REPLAY"]["ALL"]), j["BASELINE_PARITY"]])
    L += [tbl(["dataset", "queries", "all 4 frozen fields", "S4 ranking",
               "SAFE ALL", "replay ALL", "verdict"], rows), ""]

    # ---------------------------------------------------------------- T2 where the prune binds
    j1 = J("step1_metaqa")
    if j1:
        st = j1["stats"]
        L += ["## T2  Where the beam actually prunes (MetaQA, measured on the exact replay)", "",
              tbl(["path position", "frontier in", "distinct fresh candidates", "kept (beam)",
                   "prune ratio"],
                  [[i + 1, st["frontier"][i], st["cands"][i], st["kept"][i],
                    f"{st['cands'][i] / max(1e-9, st['kept'][i]):.1f}x"] for i in range(3)]), "",
              "The directive's premise put position 2 at ~2,560 candidates.  That figure came from an",
              "estimate in the previous phase report (64 frontier x ~40 edges), not from a measurement.",
              "Measured, position 1 does not prune at all, position 2 prunes ~7x, and position 3 is the",
              "*harder* prune at ~12x.  Mean legal out-degree is ~17, not ~40.", ""]

    # ---------------------------------------------------------------- T3 STEP 3 tie mass
    j3 = J("step3_metaqa")
    if j3:
        L += ["## T3  STEP 3 -- tie mass available to a lexicographic rule (MetaQA, position 2)", "",
              tbl(["quantity", "value"],
                  [["position-2 candidates scored", f"{j3['position2_candidates']:,}"],
                   ["no legal fresh child (future = sentinel)",
                    f"{j3['no_legal_child_sentinel']:,} ({j3['sentinel_frac']:.2%})"],
                   ["candidates tied in the FUTURE key",
                    f"{j3['tied_in_future_key']:,} ({j3['tied_future_frac']:.2%})"],
                   ["candidates tied in the CURRENT key",
                    f"{j3['tied_in_current_key']:,} ({j3['tied_current_frac']:.2%})"],
                   ["sentinel candidates entering beam 64",
                    f"{j3['sentinel_entering_beam64']:,} "
                    f"({j3['sentinel_per_query_in_beam']:.2f}/query)"]]), ""]
    j8 = J("step8_metaqa")
    if j8 and j8.get("STEP3"):
        L += [tbl(["lexicographic rule", "identical node list to its primary key", "fraction"],
                  [[k.split("_vs_")[0], f"{v['identical_node_lists']}/{v['of']}",
                    f"{v['identical_frac']:.4f}"] for k, v in j8["STEP3"].items()]), ""]

    # ---------------------------------------------------------------- T4 STEP 7 binding hop
    j7 = J("step7_metaqa")
    if j7:
        for blk in ["ALL", "hop2", "hop3"]:
            rows = []
            for p in j7["policies"]:
                a = j7["STEP7"][p].get(blk)
                if not a:
                    continue
                rows.append([p, a["n_cand_p2"],
                             f"{a['gold_cand_surv_p2']}/{a['gold_cand_p2']}",
                             f"{a['gold_cand_survival']:.4f}",
                             f"{a['gold_parent_surv_p2']}/{a['gold_parent_p2']}",
                             f"{a['gold_parent_survival']:.4f}",
                             f4(a["gold_cand_pct"]), f4(a["gold_parent_pct"])])
            L += [f"## T4{'' if blk == 'ALL' else '.' + blk}  STEP 7 -- position-2 diagnostic "
                  f"(MetaQA {blk})", "",
                  tbl(["policy", "cands @p2", "gold cand survived", "gold cand survival",
                       "gold parent survived", "gold parent survival",
                       "gold cand pctrank", "gold parent pctrank"], rows), ""]

    # ---------------------------------------------------------------- T5 STEP 4 three cuts
    j4 = J("step4_metaqa")
    if j4:
        for blk in ["ALL", "hop2", "hop3"]:
            rows = []
            for p, v in j4["STEP4"].items():
                a = v.get(blk)
                if not a:
                    continue
                rows.append([p, v["unique_nodes_evaluated_per_q"], v["graph_edges_inspected_per_q"],
                             f"{a['frac_discovered']:.4f}", f"{a['frac_added']:.4f}",
                             f"{a['frac_read']:.4f}", a["n_gold"], v["latency_ms_per_q"]])
            L += [f"## T5{'' if blk == 'ALL' else '.' + blk}  STEP 4 -- the three cuts after "
                  f"discovery (MetaQA {blk})", "",
                  tbl(["policy", "nodes visited/q", "edges/q", "gold DISCOVERED",
                       "gold in added(256)", f"gold READ by S4({j4['M_struct']})", "gold total",
                       "ms/q"], rows), ""]

    # ---------------------------------------------------------------- T6 STEP 8 exact P50
    if j8:
        rows = []
        for p, v in j8["STEP8"].items():
            d = v["P50"]
            rows.append([p, f4(d["ALL"]), f4(d.get("hop1")), f4(d.get("hop2")), f4(d.get("hop3")),
                         f"{v['vs_SAFE_gained']}/{v['vs_SAFE_lost']}", f"{v['vs_SAFE_net']:+d}",
                         f"{v['vs_SAFE_mcnemar_p']:.3g}", "YES" if v["vs_SAFE_sig"] else "no"])
        L += ["## T6  STEP 8 -- EXACT P50 end-to-end (MetaQA)", "",
              f"SAFE = ALL {j8['SAFE']['ALL']:.4f} / hop1 {j8['SAFE'].get('hop1', 0):.4f} / "
              f"hop2 {j8['SAFE'].get('hop2', 0):.4f} / hop3 {j8['SAFE'].get('hop3', 0):.4f}", "",
              tbl(["policy", "ALL", "hop1", "hop2", "hop3", "gained/lost", "net", "McNemar p",
                   "sig"], rows), ""]

        # ------------------------------------------------------------ T7 STEP 9 work vs scope
        rows = []
        for p, v in j8["STEP8"].items():
            i, o = v["INTERNAL"], v["OUTPUT"]
            rows.append([p, f"{i['graph_edges_per_q']:,.0f}", f"{i['lookahead_edges_per_q']:,.0f}",
                         f"{i['total_edges_per_q']:,.0f}", f"{i['distinct_nodes_evaluated_per_q']:,.0f}",
                         f"{i['beam_scope_nodes_per_q']:,.0f}", f"{i['latency_ms_per_q']:.1f}",
                         o["partitions"], f"{o['scope_nodes_per_q']:,.0f}"])
        L += ["## T7  STEP 9 -- internal routing work vs final downstream scope (MetaQA)", "",
              tbl(["policy", "graph edges/q", "lookahead edges/q", "total edges/q",
                   "distinct nodes evaluated/q", "beam scope nodes/q", "ms/q",
                   "FINAL partitions", "FINAL scope nodes/q"], rows), "",
              "Internal work is what the router touches while deciding.  Final scope is what L2 sees.",
              "The output contract is unchanged at exactly 50 partitions for every row.", ""]

    # ---------------------------------------------------------------- T9 the transition table
    jf = J("funnel_metaqa")
    if jf:
        SHORT = {"REACHABLE": "reachable", "ALIVE_p1": "alive p1", "ALIVE_p2": "alive p2",
                 "ALIVE_p3": "alive p3", "NODE_added256": "in added(256)",
                 "NODE_read_S4": "read by S4(64)", "PART_in_S4": "partition in S4",
                 "PART_in_P50": "partition in P50"}
        # the node funnel is nested and monotone; the partition rows are NOT nested inside it
        # (a partition can also be reached through a different gold node), so they are separated
        cut = jf["rows"].index("NODE_read_S4") + 1
        NODE_R, PART_R = jf["rows"][:cut], jf["rows"][cut:]
        for blk in ["ALL", "hop2", "hop3"]:
            rows, prows = [], []
            for p, F in jf["FUNNEL"].items():
                a = F.get(blk)
                if not a:
                    continue
                rows.append([p] + [f"{a[r]['node_frac']:.4f}" for r in NODE_R])
                prows.append([p] + [f"{a[r]['node_frac']:.4f}" for r in PART_R]
                             + [f"{a['PART_in_P50']['query_frac']:.4f}"])
            if not rows:
                continue
            n = jf["FUNNEL"][list(jf["FUNNEL"])[0]][blk]
            L += [f"## T9{'' if blk == 'ALL' else '.' + blk}  Transition table -- where a REQUIRED "
                  f"gold node is lost (MetaQA {blk})", "",
                  f"Fraction of the {n['gold_nodes']:,} required gold nodes over "
                  f"{n['queries']:,} queries still in play at each stage of the structural search.",
                  "This funnel is nested, so it is monotone by construction.", "",
                  tbl(["policy"] + [SHORT[r] for r in NODE_R], rows), "",
                  "The two partition rows are NOT nested inside that funnel -- a gold partition can",
                  "also be reached through a different gold node, or through the canonical and",
                  "retrieval channels, which is why they sit above the node rows.  The last column is",
                  "the query-level all-gold metric, i.e. exactly the reported EXACT-P50 number.", "",
                  tbl(["policy"] + [SHORT[r] for r in PART_R] + ["EXACT P50 (queries)"], prows), ""]
        # where the loss sits, and whether relaxing a cut removes it or merely moves it downstream
        SHOW = [p for p in ("M0_BASELINE", "B1_PARENT_DIVERSE", "L1_MAX_FUTURE", "D0_DELAYED_PRUNE")
                if p in jf["FUNNEL"]]
        for blk in ["ALL", "hop2"]:
            rows = []
            for i in range(len(NODE_R) - 1):
                r = [f"{SHORT[NODE_R[i]]} -> {SHORT[NODE_R[i+1]]}"]
                for p in SHOW:
                    f = jf["FUNNEL"][p][blk]
                    tot = f[NODE_R[0]]["node_frac"] - f[NODE_R[-1]]["node_frac"]
                    d = f[NODE_R[i]]["node_frac"] - f[NODE_R[i + 1]]["node_frac"]
                    r += [f"{d:.4f} ({d / max(1e-9, tot):.0%})"]
                rows.append(r)
            rows.append(["TOTAL structural loss"] +
                        [f"{jf['FUNNEL'][p][blk][NODE_R[0]]['node_frac'] - jf['FUNNEL'][p][blk][NODE_R[-1]]['node_frac']:.4f}"
                         for p in SHOW])
            L += [f"### Conservation of loss -- where each cut spends it (MetaQA {blk})", "",
                  tbl(["transition"] + SHOW, rows), ""]
        L += ["`D0_DELAYED_PRUNE` is the oracle-free control: it removes the position-2 and",
              "position-3 prunes entirely, so its `alive p2` and `alive p3` rows are pinned to the",
              "reachable set by construction.  It does not remove the loss -- it relocates all of it",
              "onto the static-score-ordered `added(256)` cut, where the same score now has to",
              "discriminate a pool an order of magnitude larger, and does so worse.", ""]

    # ---------------------------------------------------------------- T10 clean latency
    jl = J("latency_metaqa")
    if jl:
        L += ["## T10  Clean latency benchmark (MetaQA)", "",
              f"{jl['n_queries']} queries spread across the hop blocks, 30-query warm-up, median of",
              f"{jl['repeats']} repeats, nothing else on the CPU.  The ms/q column inside the sweep",
              "JSONs is contaminated (jobs shared the CPU and finished partway through) and is not",
              "reported anywhere.", "",
              tbl(["policy", "ms/query", "x SAFE"],
                  [[p, f"{v:.2f}", f"{jl['multiplier_vs_SAFE'][p]:.2f}x"]
                   for p, v in jl["ms_per_query"].items()]), ""]

    # ---------------------------------------------------------------- T8 all six
    rows = []
    for ds in DSETS:
        j = J(f"step8_{ds}")
        if not j:
            continue
        for p, v in j["STEP8"].items():
            if p == "M0_BASELINE":
                continue
            rows.append([ds, p, f4(j["SAFE"]["ALL"]), f4(v["P50"]["ALL"]),
                         f"{v['P50']['ALL'] - j['SAFE']['ALL']:+.4f}",
                         f"{v['vs_SAFE_net']:+d}", f"{v['vs_SAFE_mcnemar_p']:.3g}",
                         "YES" if v["vs_SAFE_sig"] else "no"])
    if rows:
        L += ["## T8  STEP 8 -- promotion check on all six corpora (EXACT P50)", "",
              tbl(["dataset", "policy", "SAFE ALL", "policy ALL", "delta", "net", "McNemar p",
                   "sig"], rows), ""]

    out = "\n".join(L) + "\n"
    open(f"{BM.BMD}/TABLES.md", "w", encoding="utf-8").write(out)
    print(f"wrote {BM.BMD}/TABLES.md  ({len(L)} lines)")


if __name__ == "__main__":
    main()
