"""B1.9 STEP 2/4/5/6 — corpus-level reserve utility, monotonicity, and a leave-one-CORPUS-out policy.

STEP 2  U_C(R) for every development corpus that has a B1.x substrate (TRAIN labels; supervision only).
STEP 4  do label-free corpus descriptors track the reserve operating point? rank correlation + LOCO sign stability.
STEP 5  ONLY a 1-descriptor monotone map (no MLP, no attention, no dataset router).
STEP 6  leave one ENTIRE corpus out: descriptor SELECTION happens inside the fold, on training corpora only,
        so picking a descriptor after seeing the full table cannot leak. R_D is frozen before any query is scored.

BASELINES: GLOBAL_FIXED_R / SOURCE_SELECTED_FIXED_R / LABEL_FREE_CORPUS_CONFIGURED_R / TARGET_ORACLE_R(invalid).
"""
import os, sys, json, math
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from _b14_twochannel import load
from _b17_marginal import utilities, R_VALS

OUT = "results/GENERALIZATION/_g2_b19c_corpus_policy.json"
DESC = json.load(open("results/GENERALIZATION/_g2_b19_descriptors.json"))
FEATS = ["mean_degree", "neighbor_jaccard_redundancy", "StructSemOverlap@10", "StructSemOverlap@50",
         "StructSemOverlap@200", "edge_semantic_lift", "hop2_over_hop1", "edge_density",
         "degree_cv", "giant_component_frac"]
ALL = ["metaqa", "2wiki_clean", "squad_clean", "musique_clean", "hotpotqa_clean", "webqsp"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    avail = [ds for ds in ALL if os.path.exists(f"scratchpad/_b12/{ds}.npz")]
    print("corpora with a B1.x substrate:", avail)

    # ---------------- STEP 2: U_C(R)
    U, URs = {}, {}
    for ds in avail:
        d = load(ds); UR, D = utilities(d); nq = len(UR)
        net = np.array([UR[:, i].sum() for i in range(4)], float)
        bi = int(np.argmax(net))
        near = int(np.argmax(net >= net.max() - 0.05 * nq))
        URs[ds] = UR
        U[ds] = {"n_queries": nq,
                 "NET_by_R": {str(R_VALS[i]): float(net[i]) for i in range(4)},
                 "NET_per_query_by_R": {str(R_VALS[i]): float(net[i] / nq) for i in range(4)},
                 "best_constant_R": R_VALS[bi], "smallest_near_optimal_R": R_VALS[near],
                 "NET_per_query_at_best": float(net[bi] / nq),
                 "oracle_NET": float(UR.max(1).sum()),
                 "oracle_NET_per_query": float(UR.max(1).sum() / nq),
                 "curve_shape": ("monotone_up" if np.all(np.diff(net) >= 0)
                                 else ("peak_then_down" if net[bi] > net[-1] else "mixed"))}
    for ds in avail:
        u = U[ds]; p = u["NET_per_query_by_R"]
        print(f"  {ds:15s} nq {u['n_queries']:5d}  NET/q@R "
              f"{p['0']:.4f}/{p['4']:.4f}/{p['8']:.4f}/{p['16']:.4f}  "
              f"bestR {u['best_constant_R']:2d}  {u['curve_shape']}")

    y = np.array([U[ds]["NET_per_query_at_best"] for ds in avail])
    bestR = np.array([U[ds]["best_constant_R"] for ds in avail], float)

    # ---------------- STEP 4: monotonicity + LOCO sign stability
    step4 = {}
    for f in FEATS:
        x = np.array([DESC[ds][f] for ds in avail], float)
        rho = spearman(x, y); rhoR = spearman(x, bestR)
        loco = []
        for i in range(len(avail)):
            m = np.arange(len(avail)) != i
            r = spearman(x[m], y[m])
            loco.append(None if r is None else round(r, 4))
        sgn = [int(np.sign(v)) for v in loco if v is not None and abs(v) > 0.15]
        step4[f] = {"spearman_vs_NETperq": None if rho is None else round(rho, 4),
                    "spearman_vs_bestR": None if rhoR is None else round(rhoR, 4),
                    "LOCO_spearman": loco,
                    "sign_stable_LOCO": bool(len(sgn) == len(avail) and len(set(sgn)) == 1),
                    "values": {ds: DESC[ds][f] for ds in avail}}
    n_perfect = sum(1 for f in FEATS if step4[f]["spearman_vs_NETperq"] is not None
                    and abs(step4[f]["spearman_vs_NETperq"]) > 0.999)
    step4["_MULTIPLE_COMPARISONS"] = {
        "n_descriptors_tested": len(FEATS), "n_corpora": len(avail),
        "P_perfect_rank_by_chance_per_descriptor": round(2.0 / math.factorial(len(avail)), 6),
        "expected_perfect_by_chance": round(len(FEATS) * 2.0 / math.factorial(len(avail)), 4),
        "n_descriptors_achieving_perfect_rank": n_perfect,
        "NOTE": "descriptors were inspected before this table was built; the honest test is the LOCO "
                "protocol below, where descriptor selection happens inside each fold."}

    # ---------------- STEP 5/6: leave-one-CORPUS-out configured R (selection INSIDE the fold)
    rows = []
    ri = {r: k for k, r in enumerate(R_VALS)}
    for tgt in avail:
        src = [d for d in avail if d != tgt]
        Rs = np.array([U[d]["best_constant_R"] for d in src], float)
        scored = []
        for f in FEATS:
            xs = np.array([DESC[d][f] for d in src], float)
            r = spearman(xs, Rs)
            if r is not None:
                scored.append((abs(r), f, r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        _, fsel, rsel = scored[0]
        xs = np.array([DESC[d][fsel] for d in src], float)
        xt = float(DESC[tgt][fsel])
        j = int(np.argmin(np.abs(xs - xt)))          # monotone 1-NN in descriptor space; zero fitted capacity
        R_pred = int(Rs[j])
        UR = URs[tgt]; nq = len(UR)
        glob = np.array([sum(U[d]["NET_by_R"][str(R)] / U[d]["n_queries"] for d in avail) for R in R_VALS])
        R_glob = R_VALS[int(np.argmax(glob))]
        srcsum = np.array([sum(U[d]["NET_by_R"][str(R)] / U[d]["n_queries"] for d in src) for R in R_VALS])
        R_srcfix = R_VALS[int(np.argmax(srcsum))]
        rows.append({"target": tgt, "selected_descriptor": fsel,
                     "source_spearman_desc_vs_bestR": round(rsel, 4),
                     "nearest_source_corpus": src[j],
                     "R_configured": R_pred, "R_target_oracle_constant": U[tgt]["best_constant_R"],
                     "R_global_fixed": R_glob, "R_source_selected_fixed": R_srcfix,
                     "NET_configured": round(float(UR[:, ri[R_pred]].sum()), 1),
                     "NET_global_fixed": round(float(UR[:, ri[R_glob]].sum()), 1),
                     "NET_source_fixed": round(float(UR[:, ri[R_srcfix]].sum()), 1),
                     "NET_target_oracle_constant": round(float(UR[:, ri[U[tgt]["best_constant_R"]]].sum()), 1),
                     "oracle_per_query_upper": round(U[tgt]["oracle_NET_per_query"], 5),
                     "n_queries": nq, "R_correct": bool(R_pred == U[tgt]["best_constant_R"])})

    tot = {k: round(sum(r[k] for r in rows), 1) for k in
           ["NET_configured", "NET_global_fixed", "NET_source_fixed", "NET_target_oracle_constant"]}
    verdict = {"LOCO_rows": rows, "TOTALS": tot,
               "R_exact_match_rate": round(float(np.mean([r["R_correct"] for r in rows])), 4),
               "configured_beats_source_fixed": bool(tot["NET_configured"] > tot["NET_source_fixed"]),
               "configured_beats_global_fixed": bool(tot["NET_configured"] > tot["NET_global_fixed"]),
               "frac_of_constant_oracle_recovered":
                   round(tot["NET_configured"] / max(tot["NET_target_oracle_constant"], 1e-9), 4)}

    res = {"CONTRACT": {"encoder_passes": 0, "uses_dataset_id": False, "uses_target_labels_for_R": False,
                        "uses_target_query_stats": False, "uses_TEST": False,
                        "descriptors": "label-free index-time corpus/graph statistics only"},
           "STEP2_U_C": U, "STEP4_MONOTONICITY": step4, "STEP56_LOCO_POLICY": verdict}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print(f"\n=== STEP 4: descriptor vs corpus reserve utility (n_corpora={len(avail)}) ===")
    print(f"  {'descriptor':30s} {'rho(NET/q)':>11s} {'rho(bestR)':>11s} {'LOCOsignstable':>16s}")
    for f in FEATS:
        s = step4[f]
        print(f"  {f:30s} {str(s['spearman_vs_NETperq']):>11s} {str(s['spearman_vs_bestR']):>11s} "
              f"{str(s['sign_stable_LOCO']):>16s}")
    mc = step4["_MULTIPLE_COMPARISONS"]
    print(f"  [chance] P(perfect|1 desc)={mc['P_perfect_rank_by_chance_per_descriptor']}  "
          f"expected among {mc['n_descriptors_tested']} = {mc['expected_perfect_by_chance']}  "
          f"observed = {mc['n_descriptors_achieving_perfect_rank']}")

    print("\n=== STEP 6: leave-one-CORPUS-out configured reserve ===")
    print(f"  {'target':15s} {'desc (selected in-fold)':28s} {'Rcfg':>5s} {'Rorc':>5s} {'Rsrc':>5s} "
          f"{'NETcfg':>8s} {'NETsrc':>8s} {'NETorc':>8s}")
    for r in rows:
        print(f"  {r['target']:15s} {r['selected_descriptor']:28s} {r['R_configured']:5d} "
              f"{r['R_target_oracle_constant']:5d} {r['R_source_selected_fixed']:5d} "
              f"{r['NET_configured']:8.0f} {r['NET_source_fixed']:8.0f} {r['NET_target_oracle_constant']:8.0f}")
    print(f"  TOTAL configured {tot['NET_configured']} | source-fixed {tot['NET_source_fixed']} | "
          f"global-fixed {tot['NET_global_fixed']} | oracle-constant {tot['NET_target_oracle_constant']}")
    print(f"  R exact-match {verdict['R_exact_match_rate']:.1%} | recovers "
          f"{verdict['frac_of_constant_oracle_recovered']:.1%} of the constant-oracle")
    print("\nwrote " + OUT)


if __name__ == "__main__":
    main()
