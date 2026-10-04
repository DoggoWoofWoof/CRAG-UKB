"""FULL_VISITED PARTITION CALIBRATION AUDIT -- STEP 7 ledger: WHY the net is negative.

Exact-P50 is a set test, so a swap can lose a query in two ways that a recall number cannot tell
apart.  This script opens the frozen F6 selection itself and books both sides:

    ADMISSION  a needed partition (gold, outside the protected 44) enters X
    EVICTION   a gold partition that was ALREADY an incumbent at boundary ranks 44..49 is
               displaced out of X by a challenger

Nothing is re-searched and nothing is re-ranked: the cached FULL_VISITED table and the frozen F6
are the same objects the STEP 7/8 numbers came from.  Run after _l1pc_run.py has cached the table.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1pc_core as PC

DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
NORMS = ["N0_RAW", "N1_DEGREE_NORMALIZED", "N2_SIZE_NORMALIZED", "N3_EXPOSURE_NORMALIZED"]
DIRECT = ["D1_BEST_NODE_SCORE", "D2_SEED_TIMES_SDIR", "D3_S4_TWO_CHANNEL"]
B = EV.B                       # the swap budget, unchanged (STEP 6: do not touch B)


def picks_of(S, SFs, qi):
    sc, _ = EV.order_for(S["ctxs"][qi], SFs[qi], "A_F6")
    return [p for _, _, p in sc][:B]


def ledger(S, SFs, nq, BL):
    """per-block admission / eviction ledger for one structural ordering."""
    out = {b: {"queries": 0, "covered": 0, "needed": 0, "needed_admitted": 0,
               "gold_at_boundary": 0, "gold_boundary_evicted": 0,
               "novel_gold_admitted": 0, "slots_used_on_gold": 0,
               "slots_used_on_nuisance": 0} for b, _ in BL}
    ind = np.zeros(nq, np.int8)
    for qi in range(nq):
        c = S["ctxs"][qi]
        gp = S["goldp"][qi]
        prot = c["prot_set"]
        bnd = set(c["base50"]) - prot
        X = set(picks_of(S, SFs, qi))
        need = gp - prot
        gb = gp & bnd
        cov = int(need <= X)
        ind[qi] = cov
        for b, mk in BL:
            if not mk[qi]:
                continue
            o = out[b]
            o["queries"] += 1
            o["covered"] += cov
            o["needed"] += len(need)
            o["needed_admitted"] += len(need & X)
            o["gold_at_boundary"] += len(gb)
            o["gold_boundary_evicted"] += len(gb - X)
            o["novel_gold_admitted"] += len((gp & X) - bnd)
            o["slots_used_on_gold"] += len(gp & X)
            o["slots_used_on_nuisance"] += len(X) - len(gp & X)
    return out, ind


def flips(ind, i_safe, S, SFs, SF_safe, nq, BL):
    """attribute every query that CHANGED against the frozen baseline to eviction or admission."""
    out = {b: {"lost": 0, "lost_by_dropping_a_needed_the_baseline_had": 0,
               "lost_of_which_a_boundary_incumbent": 0, "lost_other": 0,
               "gained": 0, "gained_with_a_novel_admission": 0} for b, _ in BL}
    for qi in range(nq):
        if ind[qi] == i_safe[qi]:
            continue
        c = S["ctxs"][qi]
        gp = S["goldp"][qi]
        prot = c["prot_set"]
        bnd = set(c["base50"]) - prot
        Xn = set(picks_of(S, SFs, qi))
        Xs = set(picks_of(S, SF_safe, qi))
        gb = gp & bnd
        for b, mk in BL:
            if not mk[qi]:
                continue
            o = out[b]
            if ind[qi] < i_safe[qi]:
                o["lost"] += 1
                # a NEEDED partition the baseline's 6 slots did hold and this ordering does not:
                # the enlarged universe displaced it with a challenger that only exists there.
                drop = (gp - prot) & Xs - Xn
                o["lost_by_dropping_a_needed_the_baseline_had"] += int(bool(drop))
                o["lost_of_which_a_boundary_incumbent"] += int(bool(drop & gb))
                o["lost_other"] += int(not drop)
            else:
                o["gained"] += 1
                o["gained_with_a_novel_admission"] += int(bool((gp & Xn) - Xs - bnd))
    return out


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
    OUT = {"ds": ds, "nq": nq, "B": B, "PROTECTED": EV.P - B}
    lsafe, i_safe = ledger(S, frozen, nq, BL)
    OUT["FROZEN_M64/S4 (SAFE)"] = {"ledger": lsafe}
    for name, mk in ([(n, lambda qi, n=n: full_order(qi, n)) for n in NORMS] +
                     [(n, lambda qi, n=n: direct_order(qi, n)) for n in DIRECT]):
        SF = [mk(qi) for qi in range(nq)] + frozen[nq:]
        lg, ind = ledger(S, SF, nq, BL)
        fl = flips(ind, i_safe, S, SF, frozen, nq, BL)
        OUT[f"FULL_VISITED/{name}"] = {"ledger": lg, "flips": fl}
        b = "hop3" if "hop3" in dict(BL) else "ALL"
        g, f_ = lg[b], fl[b]
        log(f"[{ds}] {name:26s} {b}  admitted {g['novel_gold_admitted']:5d}  "
            f"evicted {g['gold_boundary_evicted']:5d}  "
            f"(baseline evicted {lsafe[b]['gold_boundary_evicted']:5d})  "
            f"lost {f_['lost']:3d} ({f_['lost_by_dropping_a_needed_the_baseline_had']:3d} by "
            f"dropping a needed the baseline held, {f_['lost_of_which_a_boundary_incumbent']:3d} "
            f"of those an incumbent)  gained {f_['gained']:3d}")
    json.dump(OUT, open(f"{PC.PCD}/diag/ledger_{ds}.json", "w"), indent=1)
    log(f"[{ds}] wrote diag/ledger_{ds}.json")
    return OUT


if __name__ == "__main__":
    for d in (sys.argv[1:] or DS):
        run(d)
