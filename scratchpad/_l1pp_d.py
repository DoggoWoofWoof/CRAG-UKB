"""ROUND D -- the structural-signal bake-off inside the architecture that actually works.

ROUND B showed DIRECT exactly-P50 fusion is harmful on MetaQA: a third channel with a co-equal
FULL-CORPUS vote evicts canonical gold-bearing partitions everywhere (churn 9-19).  The protected
core is what keeps a structural channel honest.  So the remaining question -- which ONE orthogonal
structural signal belongs in L1 -- has to be asked inside the protected-core competition, with the
duplicated retrieval channel removed (ES) and, as a control, kept (F6).

Structural signals compared, all parameter-free and all emitting exactly 50 partitions:
    S4        frozen multi-signal structural RRF over the directional expansion
    PPRG      P1 global partition-graph PPR
    PPRB      P2 bounded (query-local) partition-graph PPR
    PPRP32    P1 reconstructed from the top-32 sparse precomputed diffusion basis
    S4+PPRG   both, one vote each

  python scratchpad/_l1pp_d.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1pp_b as B
import _l1pp_c as CC
import _l1kb_core as KB
import _l1kb_router as JR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P, M = PP.K0, PP.P, B.M_STRUCT


def swap(ctxs, chans, use_ret, Bv):
    """protected core + symmetric boundary competition; `chans` is a list of per-query
    {partition: rank} structural channels, each contributing 1/(K0+rank) exactly once."""
    nq = len(ctxs)
    out = np.empty((nq, P), np.int32)
    for qi, c in enumerate(ctxs):
        pool = []
        for ch in chans:
            pool += [p for p in ch[qi] if p not in c["prot_set"]]
        if use_ret:
            pool += [p for p in c["rpos"] if p not in c["prot_set"]]
        cands = c["bnd"] + [p for p in dict.fromkeys(pool) if p not in c["bnd"]]
        sc = []
        for p in cands:
            s = 1.0 / (K0 + c["cpos"][p]) if p in c["cpos"] else 0.0
            for ch in chans:
                if p in ch[qi]:
                    s += 1.0 / (K0 + ch[qi][p])
            if use_ret and p in c["rpos"]:
                s += 1.0 / (K0 + c["rpos"][p])
            sc.append((-s, c["cpos"].get(p, 10 ** 6), p))
        sc.sort()
        fs = c["prot"] + [p for _, _, p in sc[:Bv]]
        assert len(set(fs)) == P
        out[qi] = fs
    return out


def mass_channel(mass, tie, npart, nq, topM=M):
    """continuous partition mass -> {partition: rank} for the top-M positive-mass partitions."""
    _, order = B.contrib_mass(mass, tie, npart)
    ch = []
    for qi in range(nq):
        d = {}
        for r, p in enumerate(order[qi, :topM]):
            if mass[qi, p] > 0:
                d[int(p)] = r
        ch.append(d)
    return ch


def list_channel(lists, topM=M):
    return [{int(p): r for r, p in enumerate(l[:topM])} for l in lists]


def run_ds(ds, OUT):
    z, meta = PP.load(ds)
    nq = meta["n_dev_queries"]; goldp = PP.goldparts(z, meta); psz = z["part_sizes"]
    hops = z["hops"]
    topo = TA.load_topology(ds, log=lambda *a: None)
    ch = PP.channels(ds, z, meta, topo)
    npart, tie, hard = ch["npart"], ch["base_rank_replay"], ch["hard"]
    G = PP.partition_graph(ds, topo, log=lambda *a: None)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    sfull = C["sfull"][(M, "S4")]
    S = PP.seed_personalization(z, hard, npart, nq)
    Pmat = PP.transition(G, npart)

    mz = np.load(f"{PP.PPD}/ppr/mass_{ds}.npz")
    m_glob = mz["mass_global"].astype(np.float64)
    m_bnd = mz["mass_bounded"].astype(np.float64)
    _, trunc, nnz, t_basis = CC.basis_apply(S, Pmat, npart, Ls=(32,))
    m_pre = trunc[32]

    CH = {"S4": list_channel(sfull),
          "PPRG": mass_channel(m_glob, tie, npart, nq),
          "PPRB": mass_channel(m_bnd, tie, npart, nq),
          "PPRP32": mass_channel(m_pre, tie, npart, nq)}

    # ---- does any PPR variant REACH partitions S4 cannot? (STEP 7 reach probe) ----
    b50 = [set(int(x) for x in z["base_rank"][qi, :P]) for qi in range(nq)]
    reach = {}
    for k in ("PPRG", "PPRB", "PPRP32"):
        newp = sum(len(set(CH[k][qi]) - set(CH["S4"][qi]) - b50[qi]) for qi in range(nq)) / nq
        gnew = sum(len((goldp[qi] - b50[qi]) & (set(CH[k][qi]) - set(CH["S4"][qi])))
                   for qi in range(nq))
        gs4 = sum(len((goldp[qi] - b50[qi]) & set(CH["S4"][qi])) for qi in range(nq))
        gmiss = sum(len(goldp[qi] - b50[qi]) for qi in range(nq))
        reach[k] = {"mean_new_partitions_vs_S4": round(newp, 2),
                    "gold_partitions_reached_only_by_this": int(gnew),
                    "gold_partitions_reached_by_S4": int(gs4),
                    "gold_partitions_outside_BASE50_total": int(gmiss)}
    OUT.setdefault("STEP7_REACH_PROBE", {})[ds] = reach

    V = {}
    for Bv in (6, 8):
        ctxs = KB.contexts(z, meta, C, Bv)
        V[f"BASE"] = np.array([c["prot"] + c["bnd"] for c in ctxs], np.int32) if Bv == 6 else V["BASE"]
        for name, chans in (("S4", [CH["S4"]]), ("PPRG", [CH["PPRG"]]), ("PPRB", [CH["PPRB"]]),
                            ("PPRP32", [CH["PPRP32"]]),
                            ("S4+PPRG", [CH["S4"], CH["PPRG"]])):
            for tag, ur in (("ES", False), ("F6", True)):
                V[f"{name}_{tag}_B{Bv}"] = swap(ctxs, chans, ur, Bv)

    base = PP.score_top50(V["BASE"], goldp, psz)
    f6 = PP.score_top50(V["S4_F6_B6"], goldp, psz)
    res = {}
    for name, top in V.items():
        s = PP.score_top50(top, goldp, psz)
        e = {"ALL": round(s["ALL"], 4), "dALL": round(s["ALL"] - base["ALL"], 4),
             "dALL_vs_F6": round(s["ALL"] - f6["ALL"], 4),
             "vs_BASE": PP.mcnemar(s["ind"], base["ind"]),
             "vs_F6": PP.mcnemar(s["ind"], f6["ind"]),
             "scope_mean": round(s["scope_mean"], 1)}
        if ds == "metaqa":
            e["per_hop"] = {str(h): {
                "n": int((hops == h).sum()),
                "ALL": round(float(s["ind"][hops == h].mean()), 4),
                "dALL": round(float(s["ind"][hops == h].mean() - base["ind"][hops == h].mean()), 4),
                "dALL_vs_F6": round(float(s["ind"][hops == h].mean() - f6["ind"][hops == h].mean()), 4),
                "vs_BASE": PP.mcnemar(s["ind"][hops == h], base["ind"][hops == h])}
                for h in (1, 2, 3) if (hops == h).sum()}
        res[name] = e
        log("%-15s %-16s ALL %.4f  dBASE %+.4f (p %.4f)  dF6 %+.4f (p %.4f)" % (
            ds, name, e["ALL"], e["dALL"], e["vs_BASE"]["mcnemar_p"], e["dALL_vs_F6"],
            e["vs_F6"]["mcnemar_p"]))
    OUT.setdefault("RESULTS", {})[ds] = res


def main():
    dsets = sys.argv[1:] or PP.DSETS
    fp = f"{PP.PPD}/diag/round_d.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote round_d.json")


if __name__ == "__main__":
    main()
