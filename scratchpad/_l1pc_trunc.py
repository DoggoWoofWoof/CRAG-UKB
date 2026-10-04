"""FULL_VISITED PARTITION CALIBRATION AUDIT -- STEP 7c, the depth-matched control.

The ledger (T11) shows that every query the FULL_VISITED orderings lose is lost by DISPLACING a
needed partition the frozen baseline's own six slots already held.  That indicts the SIZE of the
candidate list F6 receives, not the ORDER within it -- opening the universe grows the challenger
list from ~40 to ~260 entries, and F6 fuses structural rank with canonical and retrieval rank, so a
nuisance partition with a mediocre structural rank but a good canonical rank can outscore a needed
one.  Those two effects are confounded in STEP 7.

This control separates them.  For each query the calibrated FULL_VISITED ordering is truncated to
EXACTLY the number of partitions the frozen M64 ordering produced for that same query, then fed to
the unchanged F6.  Same P = 50, same B = 6, same selector, same per-query candidate-list length as
the frozen baseline -- the only difference left is WHICH partitions occupy those slots, which is
precisely the calibration question.

K is not a tuned constant and not a threshold grid: it is read off the frozen ordering per query.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1sr_eval as EV
import _l1pc_core as PC

DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
NORMS = ["N0_RAW", "N1_DEGREE_NORMALIZED", "N2_SIZE_NORMALIZED", "N3_EXPOSURE_NORMALIZED"]
DIRECT = ["D1_BEST_NODE_SCORE", "D2_SEED_TIMES_SDIR", "D3_S4_TWO_CHANNEL"]


def run(ds, log=print):
    fp = f"{PC.PCD}/data/table_{ds}.npz"
    if not os.path.exists(fp):
        log(f"[{ds}] no cached table -- run _l1pc_run.py {ds} first")
        return None
    S = EV.substrate(ds)
    T = dict(np.load(fp, allow_pickle=False))
    nq = int(T["nq"][0])
    npart = int(T["npart"][0])
    hard = S["hard"].astype(np.int64)
    G = PC.partition_geometry(ds, hard, npart, log=log)
    hops = np.asarray(S["z"]["hops"])[:nq] if "hops" in S["z"] else None
    BL = [("ALL", np.ones(nq, bool))]
    if hops is not None and len(hops) and int(min(hops)) >= 0:
        for h in sorted(set(int(x) for x in hops)):
            BL.append((f"hop{h}", hops == h))

    fold = np.arange(nq) % 2
    vn = [np.zeros(npart), np.zeros(npart)]
    nqf = [int((fold == 0).sum()), int((fold == 1).sum())]
    for qi in range(nq):
        r = PC.rows(T, qi)
        vn[fold[qi]][r["part"]] += r["nnodes"]
    EXP_F = [vn[0] / max(nqf[0], 1), vn[1] / max(nqf[1], 1)]

    def full_order(qi, norm):
        r = PC.rows(T, qi)
        p = r["part"]
        s = r["nnodes"].astype(np.float64)
        if norm == "N1_DEGREE_NORMALIZED":
            s = s / np.maximum(G["pdeg"][p], 1.0)
        elif norm == "N2_SIZE_NORMALIZED":
            s = s / np.maximum(G["psize"][p], 1.0)
        elif norm == "N3_EXPOSURE_NORMALIZED":
            s = s / np.maximum(EXP_F[1 - (qi % 2)][p], 1e-6)
        return PC.s4_order(p, r["first_jj"], s, r["min_hop"], r["max_sdir"])

    def direct_order(qi, rule):
        r = PC.rows(T, qi)
        p = r["part"].astype(np.int64)
        if rule == "D1_BEST_NODE_SCORE":
            key = np.lexsort((p, r["first_jj"], -r["max_sdir"]))
        elif rule == "D2_SEED_TIMES_SDIR":
            key = np.lexsort((p, r["first_jj"], -(r["nseeds"] * r["max_sdir"])))
        elif rule == "D3_S4_TWO_CHANNEL":
            m = len(p)
            rk = np.zeros(m)
            for k_, sg in ((r["first_jj"].astype(np.float64), +1.0),
                           (r["max_sdir"].astype(np.float64), -1.0)):
                o = np.lexsort((p, sg * k_))
                rr = np.empty(m, np.int64)
                rr[o] = np.arange(m)
                rk += 1.0 / (PC.K0 + rr)
            key = np.lexsort((p, -rk))
        else:
            raise ValueError(rule)
        return p[key]

    frozen = [EV.sf_from_cache(S["z"], qi, hard, "P0_S4") for qi in range(S["nq"])]
    K = [len(frozen[qi]) for qi in range(nq)]
    i_safe, _ = EV.evaluate(S, frozen, "A_F6", (EV.P,))
    i_safe = i_safe[EV.P][:nq]
    OUT = {"ds": ds, "nq": nq,
           "frozen_ordering_len": {"mean": round(float(np.mean(K)), 1),
                                   "median": int(np.median(K)), "max": int(max(K))},
           "full_visited_ordering_len":
               {"mean": round(float(np.mean([int(T["qptr"][i + 1] - T["qptr"][i])
                                             for i in range(nq)])), 1)},
           "SAFE": {b: round(float(i_safe[mk].mean()), 4) for b, mk in BL},
           "TRUNCATED": {}}
    log(f"[{ds}] frozen ordering {OUT['frozen_ordering_len']['mean']} partitions/q vs "
        f"FULL_VISITED {OUT['full_visited_ordering_len']['mean']}  "
        f"SAFE " + " ".join(f"{b} {OUT['SAFE'][b]:.4f}" for b, _ in BL))
    for name, mk in ([(n, lambda qi, n=n: full_order(qi, n)) for n in NORMS] +
                     [(n, lambda qi, n=n: direct_order(qi, n)) for n in DIRECT]):
        SF = [np.asarray(mk(qi))[:K[qi]] for qi in range(nq)] + frozen[nq:]
        ind, churn = EV.evaluate(S, SF, "A_F6", (EV.P,))
        ind = ind[EV.P][:nq]
        row = {b: round(float(ind[mk2].mean()), 4) for b, mk2 in BL}
        row["churn"] = round(float(churn[:nq].mean()), 3)
        row["vs_SAFE"] = PP.mcnemar(ind, i_safe)
        for b, mk2 in BL[1:]:
            row[f"vs_SAFE_{b}"] = PP.mcnemar(ind[mk2], i_safe[mk2])
        OUT["TRUNCATED"][name] = row
        m = row["vs_SAFE"]
        log(f"[{ds}] TRUNC/{name:26s} " + "  ".join(f"{b} {row[b]:.4f}" for b, _ in BL) +
            f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")
    # dose-response: the SAME N0_RAW ordering, cut at increasing multiples of the frozen length.
    # A diagnostic curve, not a setting to promote -- nothing here is selected on the result.
    OUT["DOSE_N0_RAW"] = {}
    for mult in (1, 2, 4, None):
        SF = [np.asarray(full_order(qi, "N0_RAW"))[:(None if mult is None else mult * K[qi])]
              for qi in range(nq)] + frozen[nq:]
        ind, _ = EV.evaluate(S, SF, "A_F6", (EV.P,))
        ind = ind[EV.P][:nq]
        row = {b: round(float(ind[mk2].mean()), 4) for b, mk2 in BL}
        row["mean_ordering_len"] = round(float(np.mean(
            [len(SF[qi]) for qi in range(nq)])), 1)
        row["vs_SAFE"] = PP.mcnemar(ind, i_safe)
        OUT["DOSE_N0_RAW"]["FULL" if mult is None else f"{mult}xK"] = row
        m = row["vs_SAFE"]
        tag = "FULL" if mult is None else f"{mult}xK"
        log(f"[{ds}] DOSE N0_RAW {tag:>5s}  len {row['mean_ordering_len']:6.1f}  " +
            "  ".join(f"{b} {row[b]:.4f}" for b, _ in BL) +
            f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")

    json.dump(OUT, open(f"{PC.PCD}/diag/trunc_{ds}.json", "w"), indent=1)
    log(f"[{ds}] wrote diag/trunc_{ds}.json")
    return OUT


if __name__ == "__main__":
    for d in (sys.argv[1:] or DS):
        run(d)
