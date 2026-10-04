"""CONTROL -- is the boundary-mass RANKING doing work, or would any 1-hop neighbours do?

P15 already shows the halo beats reading more hard partitions at the same node budget.  This
asks the sharper question about the halo itself: hold the candidate set (the same 1-hop FULL_C
pairs) and the same per-block quota beta*|C_j| fixed, and replace the parameter-free boundary
mass s(v,C_j) = sum_{u in C_j} w(u,v)/deg(u) with a DETERMINISTIC RANDOM score.

If the random arm matches, the halo's value is "1-hop neighbours, any of them" and the scoring
rule is decoration.  If the real arm wins, the ranking is load-bearing.

Core selection is the frozen F6 one in both arms (Phase 9 rule: the halo never votes), so the
only thing that differs is WHICH boundary nodes each block pays out.

  python scratchpad/_l1ov_randrank.py <ds> [cell ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as EV
import _l1ov_p17 as P17
import _l1ep_pu as PU

OUT, log = OV.OUT, OV.log
CELLS = [("O4_FULL_C", 0.5), ("O4_FULL_C", 0.25)]
SEEDS = (0, 1, 2)


def run(ds, cells=CELLS, log=log):
    hard, npart = PU.load_assignment(ds, "CURRENT")
    hard = np.asarray(hard, np.int64)
    N = len(hard)
    g, gptr, rows, hops = PU.gold_rows(ds)
    core_sizes = np.bincount(hard, minlength=npart).astype(np.int64)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = EV.selected_blocks(ds, hard, npart, log)
    nq = meta["n_dev_queries"]

    def cov(pairs):
        nptr, nidx = OV.to_node_csr(pairs, npart, N)
        bptr, bidx = OV.to_block_csr(pairs, npart, N)
        return P17.coverage(f650, hard, g, gptr, nq, nptr, nidx, bptr, bidx, npart, core_sizes)

    R = {"ds": ds, "CELLS": {}}
    for fam, beta in cells:
        name = f"{fam}_b{beta}"
        t0 = time.time()
        pairs = OV.halo_pairs(ds, hard, "CURRENT", fam, N, log)
        bm_k, mass = OV.boundary_mass(ds, hard, "CURRENT", fam, N, npart, log)
        real = OV.bounded_pairs(bm_k, mass, hard, npart, N, beta)
        A = cov(real)
        arms = []
        for s in SEEDS:
            rng = np.random.default_rng(1000 + s)
            rnd = OV.bounded_pairs(bm_k, rng.random(len(mass)), hard, npart, N, beta)
            assert len(rnd) == len(real), (len(rnd), len(real))   # identical node budget
            B = cov(rnd)
            gained, lost, p = P17.mcnemar(B["_ind"], A["_ind"])
            arms.append({"seed": s,
                         "RANDOM": {k: v for k, v in B.items() if k != "_ind"},
                         "REAL_MINUS_RANDOM": round(A["ALL_REQUIRED_FETCHED"]
                                                    - B["ALL_REQUIRED_FETCHED"], 4),
                         "real_gained": gained, "real_lost": lost, "p": float(f"{p:.3g}"),
                         "sig": bool(p < 0.05),
                         "jaccard_vs_real": round(float(
                             len(np.intersect1d(real, rnd)) / max(len(np.union1d(real, rnd)), 1)),
                             4)})
        d = [a["REAL_MINUS_RANDOM"] for a in arms]
        R["CELLS"][name] = {
            "REAL": {k: v for k, v in A.items() if k != "_ind"},
            "HALO_PAIRS": int(len(real)),
            "ARMS": arms,
            "REAL_MINUS_RANDOM_mean": round(float(np.mean(d)), 4),
            "REAL_MINUS_RANDOM_min": round(float(np.min(d)), 4),
            "RANKING_IS_LOAD_BEARING": bool(all(a["sig"] and a["REAL_MINUS_RANDOM"] > 0
                                                for a in arms)),
            "seconds": round(time.time() - t0, 1)}
        x = R["CELLS"][name]
        log(f"  {ds:16s} {name:16s} real {A['ALL_REQUIRED_FETCHED']:.4f}  random "
            f"{[a['RANDOM']['ALL_REQUIRED_FETCHED'] for a in arms]}  "
            f"real-random mean {x['REAL_MINUS_RANDOM_mean']:+.4f}  "
            f"load-bearing {x['RANKING_IS_LOAD_BEARING']}  "
            f"(jaccard {[a['jaccard_vs_real'] for a in arms]})")
    os.makedirs(f"{OUT}/diagnostics", exist_ok=True)
    fp = f"{OUT}/diagnostics/HALO_RANKING_CONTROL.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    rec[ds] = R
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return R


if __name__ == "__main__":
    ds = sys.argv[1]
    sel = [(c.split("_b")[0], float(c.split("_b")[1])) for c in sys.argv[2:]] or CELLS
    run(ds, sel)
