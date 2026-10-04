"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- derive every named return.

Merges the two traversal passes (the kNN audit's six substrates under their canonical names, plus
the second pass that adds E3 / M1 / M3) and the Phase-B/C artifacts into ONE derived object.
Nothing is measured here; this file only selects, aligns and labels.

  python scratchpad/_l1ep_derive.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_sub as EP
import _l1ps_router as RT

OUT = EP.OUT
ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean", "webqsp"]
BACK = {v: k for k, v in EP.ALIAS.items() if k != "M0_STRUCT"}      # T-name -> canonical
A11 = ["E0_STRUCT", "E3_STRUCT_NERX", "E4_STRUCT_KNN", "E6_TOPOLOGY_C",
       "M1_STRUCT_NERX_MATCHED", "M2_STRUCT_KNN_MATCHED", "M3_TOPOLOGY_C_MATCHED"]


def load(ds):
    """canonical per-substrate per-query arrays from both passes, aligned on query order."""
    p1 = f"scratchpad/_l1kn/run_{ds}.json"
    p2 = f"scratchpad/_l1ep/a2_{ds}.json"
    if not os.path.exists(p1):
        return None
    r1 = json.load(open(p1))
    A = {}
    for t, v in r1["A"].items():
        A[BACK.get(t, t)] = {k: np.asarray(x, np.float64) for k, x in v.items()}
    A["M0_STRUCT"] = A["E0_STRUCT"]
    R = {"ds": ds, "nq": r1["nq"], "npart": r1["npart"], "stride": r1.get("stride", 1),
         "hq": np.asarray(r1["hq"], np.int32), "need_tot": np.asarray(r1["need_tot"], np.float64),
         "need_missed": np.asarray(r1["need_missed"], np.float64),
         "base": np.asarray(r1["base"], np.int8), "CLS": r1["CLS"], "CLSH": r1.get("CLSH", {}),
         "RED": r1["RED"], "PATH": r1.get("PATH", {}), "TEN": r1.get("TEN", {}),
         "PAR": r1["PAR"], "A": A, "pass2": False}
    if os.path.exists(p2):
        r2 = json.load(open(p2))
        if r2["nq"] == r1["nq"] and r2.get("stride", 1) == r1.get("stride", 1):
            for t, v in r2["subs"].items():
                if t == "E0_STRUCT":                      # pass-2 anchor: must match pass 1
                    a2 = np.asarray(v["p50"], np.float64)
                    R["PASS_ALIGN"] = {"p50_identical": bool(np.array_equal(a2, A["E0_STRUCT"]["p50"])),
                                       "orac_identical": bool(np.array_equal(
                                           np.asarray(v["orac"], np.float64), A["E0_STRUCT"]["orac"]))}
                    continue
                A[t] = {k: np.asarray(x, np.float64) for k, x in v.items()}
            R["A6_NERX"] = r2.get("A6_NERX", {})
            R["PAR2"] = r2.get("PARITY", {})
            R["pass2"] = True
    return R


def masks(R):
    m = {"ALL": np.ones(R["nq"], bool)}
    if R["ds"] == "metaqa":
        for h in (1, 2, 3):
            m[f"hop{h}"] = R["hq"] == h
    return m


def a4_bitmask(R):
    """A4: needed partitions the frozen substrate MISSES, cross-classified by which families reach
    them.  The non-STRUCT half is resolved exactly from the recorded marginals:
        |K n N n ~S| = |K n ~S| + |N n ~S| - |(K u N) n ~S|
    which is inclusion-exclusion, not an approximation.  The STRUCT half is only split by K, since
    the audit pass did not record the NERX marginal inside it."""
    out = {}
    for L, C in R["CLS"].items():
        s_k, s_nk = C["BOTH"], C["STRUCT_ONLY"]
        ns_k, ns_nk = C["KNN_ONLY"], C["NEITHER"]
        n_ns = C["NERX_ONLY"]
        kn_ns = C["KNN_OR_NERX_NOT_STRUCT"]
        both_ns = ns_k + n_ns - kn_ns                       # K and N and not S
        k_only = ns_k - both_ns                             # K only
        n_only = n_ns - both_ns                             # N only
        none_ = ns_nk - n_only                              # reached by nothing
        tot = s_k + s_nk + ns_k + ns_nk
        out[L] = {"total_missed_needed": int(tot),
                  "S_and_K": int(s_k), "S_not_K": int(s_nk),
                  "notS_K_only": int(k_only), "notS_N_only": int(n_only),
                  "notS_K_and_N": int(both_ns), "notS_none": int(none_),
                  "UNIQUE_KNN_frac": round(k_only / max(tot, 1), 5),
                  "UNIQUE_NERX_frac": round(n_only / max(tot, 1), 5),
                  "UNIQUE_KNN_or_NERX_frac": round(kn_ns / max(tot, 1), 5),
                  "UNREACHED_BY_ANY_frac": round(none_ / max(tot, 1), 5)}
    return out


def a5(R, mask=None):
    """A5: incremental attribution and the interaction term, at BOTH work levels."""
    m = np.ones(R["nq"], bool) if mask is None else mask
    A = R["A"]
    nt = float(R["need_tot"][m].sum())

    def M(t, metric):
        if t not in A:
            return None
        a = A[t]
        if metric in ("need_VIS", "need_READ", "need_ADD"):
            return float(a[metric][m].sum() / max(nt, 1))
        return float(a[metric][m].mean())

    out = {}
    for lvl, (S, SN, SK, SNK) in (
            ("FULL_WORK", ("E0_STRUCT", "E3_STRUCT_NERX", "E4_STRUCT_KNN", "E6_TOPOLOGY_C")),
            ("MATCHED_WORK", ("M0_STRUCT", "M1_STRUCT_NERX_MATCHED", "M2_STRUCT_KNN_MATCHED",
                              "M3_TOPOLOGY_C_MATCHED"))):
        d = {}
        for metric in ("need_VIS", "need_READ", "orac", "p50"):
            v = {k: M(t, metric) for k, t in (("S", S), ("SN", SN), ("SK", SK), ("SNK", SNK))}
            if any(x is None for x in v.values()):
                continue
            d[metric] = {
                "M_S": round(v["S"], 5), "M_S_N": round(v["SN"], 5), "M_S_K": round(v["SK"], 5),
                "M_S_N_K": round(v["SNK"], 5),
                "DELTA_N": round(v["SN"] - v["S"], 5), "DELTA_K": round(v["SK"] - v["S"], 5),
                "DELTA_K_after_N": round(v["SNK"] - v["SN"], 5),
                "DELTA_N_after_K": round(v["SNK"] - v["SK"], 5),
                "INTERACTION": round(v["SNK"] - v["SN"] - v["SK"] + v["S"], 5)}
        out[lvl] = d
    return out


def a11(R, mask=None):
    """A11: exact-P50 ALL@50 per substrate, absolute + delta + McNemar against the frozen one."""
    m = np.ones(R["nq"], bool) if mask is None else mask
    A = R["A"]
    b = A["E0_STRUCT"]["p50"][m].astype(np.int8)
    out = {}
    for t in A11:
        if t not in A:
            continue
        x = A[t]["p50"][m].astype(np.int8)
        mc = RT.mcnemar(x, b)
        out[t] = {"ALL_P50": round(float(x.mean()), 4),
                  "delta_vs_frozen": round(float(x.mean() - b.mean()), 4),
                  "net": int(mc.get("net", 0)), "gained": mc.get("gained"),
                  "lost": mc.get("lost"), "p": mc.get("mcnemar_p"),
                  "sig": bool(mc.get("sig", False)),
                  "n": int(m.sum())}
    return out


def a7(R):
    """A7: saturation -- exposure created by each substrate, so no oracle gets credit for it."""
    A = R["A"]; np_ = R["npart"]
    return {t: {"visited_partitions_per_query": round(float(A[t]["vis_parts"].mean()), 1),
                "frac_of_corpus_partitions": round(float(A[t]["vis_parts"].mean() / np_), 4),
                "read_partitions_per_query": round(float(A[t]["read_parts"].mean()), 1),
                "candidate_pool_per_query": round(float(A[t]["npool"].mean()), 1),
                "candidate_oracle": round(float(A[t]["orac"].mean()), 4)}
            for t in A}


def a12(R):
    A = R["A"]; b = A["E0_STRUCT"]
    return {t: {"edges_per_query": round(float(A[t]["edges"].mean()), 1),
                "edges_x_frozen": round(float(A[t]["edges"].mean() / max(b["edges"].mean(), 1e-9)), 3),
                "ms_per_query": round(float(A[t]["ms"].mean()), 2),
                "ms_x_frozen": round(float(A[t]["ms"].mean() / max(b["ms"].mean(), 1e-9)), 3)}
            for t in A}


def phase_b(ds):
    f = f"{OUT}/PARTITION_UTILITY/B_{ds}.json"
    return json.load(open(f)) if os.path.exists(f) else {}


def phase_c(ds):
    f = f"{OUT}/INTERACTION/C_{ds}.json"
    return json.load(open(f)) if os.path.exists(f) else {}


def main():
    D = {"CORPORA": {}, "A0": json.load(open(f"{OUT}/EDGE_SUBSTRATE/A0_A1_A2_substrates.json"))}
    for ds in DS:
        R = load(ds)
        rec = {}
        if R is not None:
            mk = masks(R)
            rec.update({
                "nq": R["nq"], "npart": R["npart"], "PARITY_pass1": R["PAR"],
                "PARITY_pass2": R.get("PAR2", {}), "PASS_ALIGN": R.get("PASS_ALIGN", {}),
                "pass2_present": R["pass2"],
                "A4_BITMASK": a4_bitmask(R),
                "A5_ATTRIBUTION": {k: a5(R, v) for k, v in mk.items()},
                "A6_NERX": R.get("A6_NERX", {}),
                "A7_SATURATION": a7(R),
                "A11_EXACT_P50": {k: a11(R, v) for k, v in mk.items()},
                "A12_COST": a12(R),
                "STEP4_KNN_REDUNDANCY": R["RED"], "STEP7_PATH": R["PATH"]})
        b, c = phase_b(ds), phase_c(ds)
        if b:
            rec["B"] = b
        if c:
            rec["C"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in c.items()}
        D["CORPORA"][ds] = rec
    # the artifact tree the directive asked for: one edge-substrate file per corpus, so
    # EDGE_SUBSTRATE/ is self-contained rather than pointing back at scratch.
    EK = ["nq", "npart", "PARITY_pass1", "PARITY_pass2", "PASS_ALIGN", "pass2_present",
          "A4_BITMASK", "A5_ATTRIBUTION", "A6_NERX", "A7_SATURATION", "A11_EXACT_P50",
          "A12_COST", "STEP4_KNN_REDUNDANCY", "STEP7_PATH"]
    os.makedirs(f"{OUT}/EDGE_SUBSTRATE", exist_ok=True)
    for ds, rec in D["CORPORA"].items():
        e = {k: rec[k] for k in EK if k in rec}
        if e:
            json.dump({"ds": ds, **e}, open(f"{OUT}/EDGE_SUBSTRATE/A_{ds}.json", "w"), indent=1)

    os.makedirs(f"{OUT}/diagnostics", exist_ok=True)
    fp = f"{OUT}/diagnostics/_derived.json"
    json.dump(D, open(fp, "w"), indent=1)
    print("wrote", fp)
    return D


if __name__ == "__main__":
    D = main()
    for ds, r in D["CORPORA"].items():
        if "A11_EXACT_P50" in r:
            print(f"\n{ds}  nq={r['nq']}  parity={r['PARITY_pass1'].get('T0_PARITY')}")
            for t, v in r["A11_EXACT_P50"]["ALL"].items():
                print(f"   {t:26s} {v['ALL_P50']:.4f}  d={v['delta_vs_frozen']:+.4f}  "
                      f"net={v['net']:+4d}  {'SIG' if v['sig'] else '   '}")
