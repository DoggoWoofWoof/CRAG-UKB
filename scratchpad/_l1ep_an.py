"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- the analyses that need no new traversal.

A8  path-family analysis         which edge-family SIGNATURES actually carry needed partitions,
                                 and whether MIXED (bridge) signatures beat HOMOGENEOUS ones.
A9  distinct evidence            K0/K1/K2/K3 for the kNN family (fully instrumented by the audit's
                                 redundancy pass) and the NERX family at the granularity that WAS
                                 instrumented -- un-measured cells are labelled, never estimated.
B15 stability / significance     the METIS reseed noise floor (same graph, same algorithm, different
                                 RNG) is the yardstick every partitioner delta is judged against;
                                 exact McNemar per candidate from the stored per-query indicators.
C4  effect decomposition         EDGE_EFFECT / PARTITION_EFFECT / INTERACTION.

  python scratchpad/_l1ep_an.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1ep_sub as EP

OUT = EP.OUT
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean", "webqsp"]


# ---------------------------------------------------------------- A8 path families
def _hops(sig):
    return sig.split(">")


def _homog(sig):
    """a signature is HOMOGENEOUS when every hop is carried by the same single family."""
    sets = [frozenset(h.split("+")) for h in _hops(sig)]
    return len(set(sets)) == 1 and len(sets[0]) == 1


def a8(PATH):
    if not PATH:
        return {}
    rows = {}
    for sig, v in PATH.items():
        tg, nt = int(v["targets"]), int(v["needed_targets"])
        rows[sig] = {"len": len(_hops(sig)), "homogeneous": _homog(sig),
                     "paths": int(v["paths"]), "targets": tg, "needed_targets": nt,
                     "needed_yield": round(nt / max(tg, 1), 5),
                     "parts": int(v["parts"]), "needed_parts": int(v["needed_parts"])}
    agg = {}
    for L in (1, 2, 3):
        for hom in (True, False):
            sel = [r for r in rows.values() if r["len"] == L and r["homogeneous"] == hom]
            if not sel:
                continue
            tg = sum(r["targets"] for r in sel)
            nt = sum(r["needed_targets"] for r in sel)
            agg[f"hop{L}_{'HOMOGENEOUS' if hom else 'BRIDGE'}"] = {
                "n_signatures": len(sel), "targets": tg, "needed_targets": nt,
                "needed_yield": round(nt / max(tg, 1), 5)}
    verdict = {}
    for L in (1, 2, 3):
        h = agg.get(f"hop{L}_HOMOGENEOUS")
        b = agg.get(f"hop{L}_BRIDGE")
        if h and b:
            verdict[f"hop{L}"] = {"homog_yield": h["needed_yield"],
                                  "bridge_yield": b["needed_yield"],
                                  "bridge_minus_homog": round(b["needed_yield"]
                                                              - h["needed_yield"], 5),
                                  "BRIDGE_BETTER": bool(b["needed_yield"] > h["needed_yield"])}
    top = sorted(rows.items(), key=lambda kv: -kv[1]["needed_targets"])[:12]
    return {"BY_SIGNATURE": rows, "BY_LENGTH_AND_HOMOGENEITY": agg,
            "BRIDGE_VS_HOMOGENEOUS": verdict,
            "TOP_SIGNATURES_BY_NEEDED": [k for k, _ in top]}


# ---------------------------------------------------------------- A9 distinct evidence
def a9(RED, BIT, A6):
    if not RED:
        return {}
    tot = int(RED.get("knn_parts", 0))
    novel = int(RED.get("novel", 0))
    K = {"K0_ALSO_REACHED_BY_FROZEN_STRUCT": int(RED.get("in_struct_T0", 0)),
         "K1_ALSO_IN_DENSE_OR_SPLADE_TOP200": int(RED.get("in_canon200", 0)),
         "K2_ALSO_IN_RETRIEVAL_FRONTIER_OR_CONTEXT": max(int(RED.get("in_RF", 0)),
                                                         int(RED.get("in_CONT", 0))),
         "K3_NOVEL_TO_EVERY_OTHER_CHANNEL": novel,
         "TOTAL_KNN_REACHED_PARTITIONS": tot,
         "K3_NOVEL_FRACTION": round(novel / max(tot, 1), 5),
         "K3_NOVEL_AND_NEEDED": int(RED.get("novel_needed", 0)),
         "NEEDED_REACHED_BY_KNN": int(RED.get("needed", 0)),
         "NEEDED_ALSO_REACHED_ELSEWHERE": int(RED.get("dup_needed", 0))}
    K["EVIDENCE_IS_DISTINCT"] = bool(K["K3_NOVEL_AND_NEEDED"] > 0)
    N = {"N0_ALSO_REACHED_BY_FROZEN_STRUCT": "NOT_MEASURED",
         "N1_ALSO_IN_DENSE_OR_SPLADE_TOP200": "NOT_MEASURED",
         "N2_UNIQUE_TO_NERX_AMONG_MISSED_NEEDED": None,
         "_note": "pass 2 instrumented NERX at the needed-partition bitmask level and at the "
                  "edge/query-similarity level, not at the channel-redundancy level; the "
                  "un-instrumented cells are reported NOT_MEASURED rather than estimated."}
    b = (BIT or {}).get("VIS")
    if b:
        N["N2_UNIQUE_TO_NERX_AMONG_MISSED_NEEDED"] = b["notS_N_only"]
        N["N2_UNIQUE_FRACTION"] = b["UNIQUE_NERX_frac"]
    if A6:
        e = A6.get("NERX_EDGE_SIM", {}).get("AUC_needed_vs_nuisance")
        q = A6.get("NERX_QUERY_SIM", {}).get("AUC_needed_vs_nuisance")
        N["AUC_EDGE_SIM"] = e
        N["AUC_QUERY_SIM"] = q
        if e is not None and q is not None:
            N["DENSE_WEARING_GRAPH_COSTUME"] = "YES" if q > e else "NO"
    return {"KNN": K, "NERX": N}


# ---------------------------------------------------------------- B15 stability
def _ind(C, tag, key="_ind_F6"):
    v = C.get(tag, {}).get(key)
    return np.asarray(v, np.int8) if v is not None else None


def b15(ds):
    fp = f"{OUT}/INTERACTION/C_{ds}.json"
    if not os.path.exists(fp):
        return {}
    C = json.load(open(fp))
    base = C.get("PM_CURRENT_EXACT")
    if base is None:
        return {}
    bi = _ind(C, "PM_CURRENT_EXACT")
    out = {"REFERENCE": {"tag": "PM_CURRENT_EXACT", "F6_ALL_P50": base["F6_ALL_P50"],
                         "BASE_ALL_P50": base["BASE_ALL_P50"], "nq": base["nq"]}}
    for fam, pref in (("RANDOM_BALANCED", "P0_RANDOM_BALANCED_s"),
                      ("METIS_RESEED", "PM3_TOPOLOGY_C_seed")):
        tags = sorted(t for t in C if t.startswith(pref))
        if not tags:
            continue
        f6 = np.array([C[t]["F6_ALL_P50"] for t in tags], float)
        bs = np.array([C[t]["BASE_ALL_P50"] for t in tags], float)
        out[fam] = {"n_seeds": len(tags), "tags": tags,
                    "F6_mean": round(float(f6.mean()), 4),
                    "F6_sd": round(float(f6.std(ddof=1)), 4) if len(f6) > 1 else 0.0,
                    "F6_min": round(float(f6.min()), 4), "F6_max": round(float(f6.max()), 4),
                    "BASE_mean": round(float(bs.mean()), 4),
                    "F6_delta_vs_production": round(float(f6.mean() - base["F6_ALL_P50"]), 4)}
    nf = None
    if "METIS_RESEED" in out:
        allv = [C[t]["F6_ALL_P50"] for t in out["METIS_RESEED"]["tags"]] + [base["F6_ALL_P50"]]
        nf = round(float(np.std(allv, ddof=1)), 4)
        out["NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO"] = nf
        out["NOISE_FLOOR_F6_RANGE"] = round(float(max(allv) - min(allv)), 4)
    cand = {}
    for t, v in C.items():
        if t == "PM_CURRENT_EXACT" or t.startswith("P0_RANDOM") or "_seed" in t:
            continue
        xi = _ind(C, t)
        rec = {"F6_ALL_P50": v["F6_ALL_P50"],
               "delta_vs_production": round(v["F6_ALL_P50"] - base["F6_ALL_P50"], 4),
               "BASE_ALL_P50": v["BASE_ALL_P50"],
               "delta_BASE": round(v["BASE_ALL_P50"] - base["BASE_ALL_P50"], 4),
               "scope_nodes": v["BASE_SCOPE_NODES"],
               "scope_ratio_vs_production": round(v["BASE_SCOPE_NODES"]
                                                  / max(base["BASE_SCOPE_NODES"], 1e-9), 4)}
        if xi is not None and bi is not None and len(xi) == len(bi):
            mc = RT.mcnemar(xi, bi)
            rec.update({"net": int(mc.get("net", 0)), "gained": mc.get("gained"),
                        "lost": mc.get("lost"), "p": mc.get("mcnemar_p"),
                        "sig": bool(mc.get("sig", False))})
        if nf is not None:
            rec["EXCEEDS_RESEED_NOISE_FLOOR"] = bool(abs(rec["delta_vs_production"]) > 2 * nf)
        cand[t] = rec
    out["CANDIDATES"] = cand
    return out


# ---------------------------------------------------------------- C4 decomposition
def c4(ds, D):
    """EDGE_EFFECT      best edge substrate at the FROZEN partitioning     (Phase A, exact P50)
       PARTITION_EFFECT best partitioning at the FROZEN edge substrate     (Phase C, exact P50)
       INTERACTION      needs the joint cell; computed only when Phase A licensed an edge arm."""
    rec = D["CORPORA"].get(ds, {})
    A = rec.get("A11_EXACT_P50", {}).get("ALL", {})
    C = rec.get("C", {})
    if not A or not C or "PM_CURRENT_EXACT" not in C:
        return {}
    frozen_edge = A.get("E0_STRUCT", {}).get("ALL_P50")
    prod = C["PM_CURRENT_EXACT"]["F6_ALL_P50"]
    best_e = max(((t, v) for t, v in A.items() if t != "E0_STRUCT"),
                 key=lambda kv: kv[1]["ALL_P50"], default=(None, None))
    cands = {t: v for t, v in C.items()
             if t != "PM_CURRENT_EXACT" and not t.startswith("P0_RANDOM") and "_seed" not in t}
    best_p = max(cands.items(), key=lambda kv: kv[1]["F6_ALL_P50"], default=(None, None))
    return {"FROZEN_CELL_F6_ALL_P50": prod,
            "PHASE_A_ANCHOR_ALL_P50": frozen_edge,
            "EDGE_EFFECT": {"best_substrate": best_e[0],
                            "delta": (round(best_e[1]["ALL_P50"] - frozen_edge, 4)
                                      if best_e[1] else None),
                            "sig": (best_e[1]["sig"] if best_e[1] else None)},
            "PARTITION_EFFECT": {"best_partitioner": best_p[0],
                                 "delta": (round(best_p[1]["F6_ALL_P50"] - prod, 4)
                                           if best_p[1] else None)},
            **_inter(ds, prod, A, C)}


def _inter(ds, prod, A, C):
    """the 2x2 joint cell, run by _l1ep_x.py once Phase A licensed an edge arm.

           a = E0 x shipped   b = E0 x candidate   c = E6 x shipped   d = E6 x candidate
           INTERACTION = d - b - c + a
    """
    fp = f"{OUT}/INTERACTION/X_{ds}.json"
    if not os.path.exists(fp):
        return {"INTERACTION": None, "INTERACTION_STATUS": "NOT_RUN"}
    X = json.load(open(fp))
    base = X.get("PM_CURRENT_EXACT") or {}
    cand = next((v for k, v in X.items() if k != "PM_CURRENT_EXACT"), None)
    if not base or not cand:
        return {"INTERACTION": None, "INTERACTION_STATUS": "INCOMPLETE"}
    g = lambda rec, t: (rec.get("CELLS", {}).get(t) or {}).get("F6_ALL_P50")
    EDGE = "E6_TOPOLOGY_C"
    a_, b_, c_, d_ = g(base, "E0_STRUCT"), g(cand, "E0_STRUCT"), g(base, EDGE), g(cand, EDGE)
    if None in (a_, b_, c_, d_):
        return {"INTERACTION": None, "INTERACTION_STATUS": "INCOMPLETE"}
    return {"INTERACTION": round(d_ - b_ - c_ + a_, 4),
            "INTERACTION_STATUS": "MEASURED",
            "INTERACTION_2X2": {
                "edge_arm": EDGE, "partition_arm": cand.get("partition_tag"),
                "a_frozen_x_shipped": a_, "b_frozen_x_candidate": b_,
                "c_edge_x_shipped": c_, "d_edge_x_candidate": d_,
                "EDGE_EFFECT_c_minus_a": round(c_ - a_, 4),
                "PARTITION_EFFECT_b_minus_a": round(b_ - a_, 4),
                "JOINT_d_minus_a": round(d_ - a_, 4),
                "C3_PARITY": base.get("C3_PARITY"),
                "PARITY_vs_PHASE_C": ("EXACT" if abs(a_ - prod) < 1e-9
                                      else f"MISMATCH phaseC={prod} x={a_}")}}


def main():
    D = json.load(open(f"{OUT}/diagnostics/_derived.json"))
    A = {"A8_PATH_FAMILIES": {}, "A9_DISTINCT_EVIDENCE": {}, "B15_STABILITY": {},
         "C4_DECOMPOSITION": {}}
    for ds in DS:
        rec = D["CORPORA"].get(ds, {})
        if rec.get("STEP7_PATH"):
            A["A8_PATH_FAMILIES"][ds] = a8(rec["STEP7_PATH"])
        if rec.get("STEP4_KNN_REDUNDANCY"):
            A["A9_DISTINCT_EVIDENCE"][ds] = a9(rec["STEP4_KNN_REDUNDANCY"],
                                               rec.get("A4_BITMASK", {}), rec.get("A6_NERX", {}))
        s = b15(ds)
        if s:
            A["B15_STABILITY"][ds] = s
        c = c4(ds, D)
        if c:
            A["C4_DECOMPOSITION"][ds] = c
    fp = f"{OUT}/diagnostics/_analysis.json"
    json.dump(A, open(fp, "w"), indent=1)
    print("wrote", fp)
    return A


if __name__ == "__main__":
    A = main()
    for ds, v in A["B15_STABILITY"].items():
        nf = v.get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO")
        print(f"\n{ds}  reference F6 {v['REFERENCE']['F6_ALL_P50']:.4f}  "
              f"noise_floor_sd={nf if nf is not None else 'n/a'}")
        for t, r in sorted(v["CANDIDATES"].items(), key=lambda kv: -kv[1]["F6_ALL_P50"]):
            print(f"   {t:24s} {r['F6_ALL_P50']:.4f}  d={r['delta_vs_production']:+.4f}  "
                  f"net={r.get('net', 0):+5d}  {'SIG' if r.get('sig') else '   '}  "
                  f"scope_x{r['scope_ratio_vs_production']:.3f}"
                  + ("  >NOISE" if r.get("EXCEEDS_RESEED_NOISE_FLOOR") else ""))
