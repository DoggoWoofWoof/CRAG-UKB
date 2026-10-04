"""Derive the eight required return flags from the diag JSONs, mechanically.

The verdicts are computed from the measurements rather than written by hand, so the report cannot
drift from the data.  The decision order is the one set for this phase, not candidate survival:

    1. exact-P50 hop3 gain      2. exact-P50 hop2 gain      3. p2/p3 gold survival mechanism
    4. latency multiplier       5. cross-corpus safety

So BEST_BEAM_METHOD is resolved FIRST, from end-to-end exact P50 with the no-significant-regression
rule on all six corpora, and the survival numbers are then reported FOR THAT METHOD.  The policy that
maximises position-2 survival is reported separately, as a mechanism note together with what it did
downstream -- quoting its survival gain as the headline would be dishonest when it loses end-to-end.

  python scratchpad/_l1bm_returns.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1bm_core as BM

D = f"{BM.BMD}/diag"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
MATERIAL = 0.02        # predeclared: "materially" = >= 2 points absolute, not +0.002
FROZEN = "M0_BASELINE"


def J(n):
    p = f"{D}/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    j7, j8, j4, jf = J("step7_metaqa"), J("step8_metaqa"), J("step4_metaqa"), J("funnel_metaqa")
    jl = J("latency_metaqa")
    R = {}

    # ---- 1/2. end-to-end first: which policies are even promotable on all six corpora ----------
    prom = {}
    for p in (j8["STEP8"] if j8 else {}):
        if p == FROZEN:
            continue
        rows, bad, net = {}, [], 0
        for ds in DSETS:
            jd = J(f"step8_{ds}")
            if not jd or p not in jd["STEP8"]:
                continue
            v = jd["STEP8"][p]
            rows[ds] = {"delta": round(v["P50"]["ALL"] - jd["SAFE"]["ALL"], 4),
                        "net": v["vs_SAFE_net"], "p": v["vs_SAFE_mcnemar_p"],
                        "sig": v["vs_SAFE_sig"]}
            net += v["vs_SAFE_net"]
            if v["vs_SAFE_sig"] and v["vs_SAFE_net"] < 0:
                bad.append(ds)
        m = j8["STEP8"][p]["P50"]
        prom[p] = {"per_dataset": rows, "sig_regressions": bad, "net_total": net,
                   "corpora_tested": len(rows),
                   "metaqa_hop2_delta": round(m.get("hop2", 0) - j8["SAFE"].get("hop2", 0), 4),
                   "metaqa_hop3_delta": round(m.get("hop3", 0) - j8["SAFE"].get("hop3", 0), 4),
                   "NO_SIG_REGRESSION": (len(rows) == len(DSETS) and not bad)}
    R["_promotion"] = prom
    # promotion needs BOTH safety and a material MetaQA multi-hop gain -- "do not celebrate +0.002"
    win = [p for p, v in prom.items() if v["NO_SIG_REGRESSION"] and v["net_total"] > 0
           and max(v["metaqa_hop2_delta"], v["metaqa_hop3_delta"]) >= MATERIAL]
    R["BEST_BEAM_METHOD"] = (max(win, key=lambda p: prom[p]["metaqa_hop3_delta"]) if win else
                             "B0_GLOBAL (the frozen static-score beam) -- nothing cleared promotion")
    best = win[0] if win else FROZEN
    R["_best_policy_key"] = best

    if j8:
        m = j8["STEP8"][best]["P50"]
        R["EXACT_P50_METAQA_HOP2"] = round(m["hop2"], 4)
        R["EXACT_P50_METAQA_HOP3"] = round(m["hop3"], 4)
        R["_exact_p50_deltas"] = {
            "hop2": round(m["hop2"] - j8["SAFE"]["hop2"], 4),
            "hop3": round(m["hop3"] - j8["SAFE"]["hop3"], 4),
            "ALL": round(j8["STEP8"][best]["P50"]["ALL"] - j8["SAFE"]["ALL"], 4),
            "SAFE_hop2": j8["SAFE"]["hop2"], "SAFE_hop3": j8["SAFE"]["hop3"]}
        # the best hop3 ANY policy reached, so the flag cannot hide a win that exists somewhere
        R["_best_hop3_any_policy"] = max(
            ((round(v["P50"]["hop3"], 4), p) for p, v in j8["STEP8"].items()), default=None)

    # ---- 3. the survival mechanism, for the winner AND for the survival-maximiser ---------------
    if j7:
        b = j7["STEP7"][FROZEN]["ALL"]
        par = {p: round(j7["STEP7"][p]["ALL"]["gold_parent_survival"]
                        - b["gold_parent_survival"], 4) for p in j7["policies"]}
        cnd = {p: round(j7["STEP7"][p]["ALL"]["gold_cand_survival"]
                        - b["gold_cand_survival"], 4) for p in j7["policies"]}
        R["_p2_parent_gain"], R["_p2_cand_gain"] = par, cnd
        mx = max((p for p in par if p != "D0_DELAYED_PRUNE"), key=par.get)
        R["_p2_survival_maximiser"] = mx
        R["POSITION2_GOLD_SURVIVAL_GAIN"] = (
            f"{par[best]:+.4f} for the promoted method; gold-parent survival stays "
            f"{j7['STEP7'][best]['ALL']['gold_parent_survival']:.4f}. The largest gain any bounded "
            f"policy achieved is {par[mx]:+.4f} ({b['gold_parent_survival']:.4f} -> "
            f"{j7['STEP7'][mx]['ALL']['gold_parent_survival']:.4f}, {mx}), but it pays for that by "
            f"losing gold CANDIDATES ({cnd[mx]:+.4f}, {b['gold_cand_survival']:.4f} -> "
            f"{j7['STEP7'][mx]['ALL']['gold_cand_survival']:.4f}) and it does not convert -- see "
            f"HOP3_GOLD_NODE_RECOVERY_GAIN.")

    # ---- gold-node recovery, for the winner and for the survival-maximiser ---------------------
    if j4:
        b3 = j4["STEP4"][FROZEN]["hop3"]
        mx = R.get("_p2_survival_maximiser", FROZEN)

        def rec(p):
            a = j4["STEP4"][p]["hop3"]
            return (f"{a['gold_READ_by_S4'] - b3['gold_READ_by_S4']:+d} gold nodes read by S4 "
                    f"({b3['gold_READ_by_S4']} -> {a['gold_READ_by_S4']} of {b3['n_gold']}, "
                    f"{a['frac_read'] - b3['frac_read']:+.4f})")
        R["HOP3_GOLD_NODE_RECOVERY_GAIN"] = (
            f"{rec(best)} for the promoted method. The position-2 survival maximiser {mx} gives "
            f"{rec(mx)}; the oracle-free delayed-prune control D0_DELAYED_PRUNE, which removes the "
            f"position-2 and position-3 prunes entirely, gives {rec('D0_DELAYED_PRUNE')}."
            if "D0_DELAYED_PRUNE" in j4["STEP4"] else rec(best))

        d, bb = j4["STEP4"]["D0_DELAYED_PRUNE"]["ALL"], j4["STEP4"][FROZEN]["ALL"]
        R["_delayed_prune_control"] = {
            "frac_discovered": [bb["frac_discovered"], d["frac_discovered"]],
            "frac_added": [bb["frac_added"], d["frac_added"]],
            "frac_read": [bb["frac_read"], d["frac_read"]]}
        # "premature" only if delaying the prune puts gold IN FRONT OF S4, not merely in scope
        R["PREMATURE_PRUNING_CONFIRMED"] = (
            "YES" if d["frac_read"] - bb["frac_read"] >= MATERIAL else
            ("DISCOVERY_ONLY -- delaying the prune raises gold DISCOVERY "
             f"{bb['frac_discovered']:.4f} -> {d['frac_discovered']:.4f} but LOWERS what S4 reads "
             f"{bb['frac_read']:.4f} -> {d['frac_read']:.4f} and lowers exact-P50 ALL "
             f"{j8['SAFE']['ALL']:.4f} -> {j8['STEP8']['D0_DELAYED_PRUNE']['P50']['ALL']:.4f}; "
             "the answer to the phase question is therefore NO")
            if d["frac_discovered"] - bb["frac_discovered"] >= MATERIAL else "NO")

    # ---- the funnel: where any benefit disappears ----------------------------------------------
    if jf:
        f0 = jf["FUNNEL"][FROZEN]["ALL"]
        rows = jf["rows"]
        R["_funnel_ALL_frozen"] = {r: f0[r]["node_frac"] for r in rows}
        # stage-to-stage losses among the STRUCTURAL rows (the last row is not nested in them)
        st = rows[:rows.index("NODE_read_S4") + 1]
        R["_funnel_stage_loss"] = {f"{st[i]} -> {st[i+1]}":
                                   round(f0[st[i]]["node_frac"] - f0[st[i + 1]]["node_frac"], 4)
                                   for i in range(len(st) - 1)}
        R["_funnel_biggest_drop"] = max(R["_funnel_stage_loss"].items(), key=lambda kv: kv[1])

    # ---- 4. work and latency -------------------------------------------------------------------
    if j8:
        i = j8["STEP8"][best]["INTERNAL"]; o = j8["STEP8"][best]["OUTPUT"]
        lk = j8["STEP8"].get("L1_MAX_FUTURE", {}).get("INTERNAL", {})
        R["ONLINE_EDGES"] = (
            f"{i['total_edges_per_q']:,.0f} graph-edge inspections/query "
            f"({i['graph_edges_per_q']:,.0f} beam + {i['lookahead_edges_per_q']:,.0f} lookahead), "
            f"{i['distinct_nodes_evaluated_per_q']:,.0f} distinct nodes scored, emitting EXACTLY "
            f"{o['partitions']} partitions = {o['scope_nodes_per_q']:,.0f} nodes downstream "
            f"(unchanged).")
        if lk:
            R["ONLINE_EDGES"] += (
                f" The one-step lookahead costs {lk['total_edges_per_q']:,.0f} edges/query "
                f"({lk['total_edges_per_q'] / max(1e-9, i['total_edges_per_q']):.1f}x) for the same "
                f"50-partition output -- a latency cost, not a compression cheat.")
    if jl:
        base = jl["ms_per_query"][FROZEN]
        R["_latency_ms_per_q"] = jl["ms_per_query"]
        R["_latency_multiplier"] = jl["multiplier_vs_SAFE"]
        R["LATENCY"] = (f"{base:.1f} ms/query structural routing for the promoted method "
                        f"(clean benchmark: {jl['n_queries']} queries, median of {jl['repeats']} "
                        f"repeats, nothing else on the CPU). Rejected alternatives: "
                        + ", ".join(f"{p} {jl['multiplier_vs_SAFE'][p]:.2f}x"
                                    for p in ("L1_MAX_FUTURE", "L2_TOP2_FUTURE", "D0_DELAYED_PRUNE",
                                              "B1_PARENT_DIVERSE") if p in jl["ms_per_query"]) + ".")
    else:
        R["LATENCY"] = "PENDING clean benchmark (the sweep ms/q is CPU-contaminated, not reportable)"

    json.dump(R, open(f"{BM.BMD}/RETURNS.json", "w"), indent=1)
    for k, v in R.items():
        if not k.startswith("_"):
            print(f"{k} = {v}\n")
    return R


if __name__ == "__main__":
    main()
