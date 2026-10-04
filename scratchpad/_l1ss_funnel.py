"""STEP 10 -- where the read-stage gain is lost, partition by partition.

The oracle read puts 100 percent of the discovered gold partitions into the S4 ranking and still moves
exact-P50 by less than one point.  This attributes that gap.  For every query the swap has to supply

    need = goldp - prot(44)        (the gold partitions the protected core does NOT already hold)

and it has exactly B = 6 slots.  Every query lands in exactly one bucket:

    FREE      need is empty                       -- the protected core already covers it
    CAPACITY  len(need) > B                        -- unreachable at B = 6 whatever the ranking is
    NOCAND    some needed partition is not a candidate at all (not an incumbent, not in SF, not in RF)
    LOST      every needed partition is a candidate, but the F6 order did not put them all in the 6
    WON       covered

NOCAND is the only bucket a better structural READ can move.  LOST and CAPACITY are downstream of S4
and are out of this phase by construction.  Nothing here changes any mechanism; it is accounting.

  python scratchpad/_l1ss_funnel.py ds [R0_STATIC,RX_ORACLE_READ,...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1ss_core as SS
import _l1ss_sep as SEP
import _l1ss_rank as RK

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
BUCKETS = ["FREE", "CAPACITY", "NOCAND", "LOST", "WON"]


def sf_for(D, qi, order_name):
    a, b = int(D["qptr"][qi]), int(D["qptr"][qi + 1])
    n = b - a
    F = {k: D[k][a:b] for k in SS.FEATS}
    F["_GOLD"] = D["gold"][a:b].astype(np.float64)
    o = RK.order_for(order_name, F, n) if n else np.empty(0, np.int64)
    sn = np.full(SS.BM.SMAX, -1, np.int32); sh = np.zeros(SS.BM.SMAX, np.int8)
    sd = np.zeros(SS.BM.SMAX, np.float32); sc = np.zeros(SS.BM.SMAX, np.int32)
    k = min(n, SS.BM.SMAX)
    if k:
        idx = o[:k]
        sn[:k] = D["node"][a:b][idx]
        sh[:k] = F["HOP"][idx].astype(np.int8)
        sd[:k] = F["STATIC_SDIR"][idx].astype(np.float32)
        sc[:k] = F["PATH_SUPPORT"][idx].astype(np.int32)
    return sn, sh, sd, sc, o, a, b


def run(ds="metaqa", ranks=("R0_STATIC", "R1_EQUAL_RRF", "R2_HOP_CONDITIONED", "RX_ORACLE_READ"), tag=None):
    D = SEP.load(tag or ds)
    S = EV.substrate(ds); nq = S["nq"]; hard = S["hard"]; hops = S["hops"]
    goldp, ctxs = S["goldp"], S["ctxs"]
    gr, _ = DG.gold_rows_of(ds, S["z"])
    Gs = [set(int(x) for x in g) for g in gr]
    seed_gold = np.zeros(nq, np.int64)
    for qi in range(nq):
        sd = set(int(s) for s in S["z"]["seeds"][qi] if s >= 0)
        seed_gold[qi] = len(sd & Gs[qi])
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    if hs and min(hs) < 0:
        hs = []
    OUT = {"ds": tag or ds, "substrate": ds, "nq": nq, "METHODS": {}}

    for rk in ranks:
        buck = {b: np.zeros(nq, np.int8) for b in BUCKETS}
        need_n = np.zeros(nq, np.int64)
        # per-needed-partition evidence, over the partitions the swap actually has to supply
        ev = {"need": 0, "in_SF": 0, "in_RF": 0, "incumbent": 0, "no_evidence": 0, "picked": 0}
        sfrank = []
        gold_read = np.zeros(nq, np.int64)
        for qi in range(nq):
            c = ctxs[qi]
            sn, sh, sdd, sc, o, a, b = sf_for(D, qi, rk)
            SF = EV.sf_from_cache({"s_node": sn[None], "s_hop": sh[None], "s_sdir": sdd[None],
                                   "s_cnt": sc[None]}, 0, hard, "P0_S4")
            g = D["gold"][a:b]
            gold_read[qi] = int(g[o[:min(b - a, EV.M_STRUCT)]].sum()) if b > a else 0
            spos = {p: r for r, p in enumerate(SF)}
            b50 = c["base50"]; RF = c["_RF"]; bnd = list(c["bnd"])
            chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
            cands = set(bnd) | set(chal)
            sc_ord, _ = EV.order_for(c, SF, "A_F6")
            picks = set(p for _, _, p in sc_ord[:EV.B])
            need = set(goldp[qi]) - c["prot_set"]
            need_n[qi] = len(need)
            for p in need:
                ev["need"] += 1
                i_sf, i_rf, i_bd = p in spos, p in RF, p in bnd
                ev["in_SF"] += int(i_sf); ev["in_RF"] += int(i_rf); ev["incumbent"] += int(i_bd)
                ev["no_evidence"] += int(not (i_sf or i_rf or i_bd))
                ev["picked"] += int(p in picks)
                if i_sf:
                    sfrank.append(spos[p])
            if not need:
                buck["FREE"][qi] = 1
            elif len(need) > EV.B:
                buck["CAPACITY"][qi] = 1
            elif not need <= cands:
                buck["NOCAND"][qi] = 1
            elif not need <= picks:
                buck["LOST"][qi] = 1
            else:
                buck["WON"][qi] = 1
        cov = (buck["FREE"] | buck["WON"]).astype(float)
        row = {"exact_P50_check": round(float(cov.mean()), 4),
               "gold_nodes_total": int(D["ngold"].sum()),
               "gold_nodes_seed": int(seed_gold.sum()),
               "gold_nodes_discovered": int(D["gold"].sum()),
               "gold_nodes_read64": int(gold_read.sum()),
               "read_recall@64": round(float(gold_read.sum() / max(1, D["ngold"].sum())), 4),
               "read_efficiency": round(float(gold_read.sum() /
                                        max(1, D["gold"].sum() + seed_gold.sum())), 4),
               "needed_partitions": ev["need"],
               "needed_with_structural_evidence": ev["in_SF"],
               "needed_with_retrieval_evidence": ev["in_RF"],
               "needed_incumbent": ev["incumbent"],
               "needed_with_NO_evidence": ev["no_evidence"],
               "needed_picked": ev["picked"],
               "needed_picked_frac": round(ev["picked"] / max(1, ev["need"]), 4),
               "median_SF_rank_of_needed": (int(np.median(sfrank)) if sfrank else None),
               "buckets": {bk: int(buck[bk].sum()) for bk in BUCKETS},
               "buckets_frac": {bk: round(float(buck[bk].mean()), 4) for bk in BUCKETS}}
        for h in hs:
            m = np.asarray(hops) == h
            row[f"hop{h}"] = {"exact_P50": round(float(cov[m].mean()), 4),
                              "buckets": {bk: int(buck[bk][m].sum()) for bk in BUCKETS},
                              "mean_need": round(float(need_n[m].mean()), 2)}
        OUT["METHODS"][rk] = row
        log(f"  {rk:20s} P50 {row['exact_P50_check']:.4f}  read@64 {row['read_recall@64']:.4f} "
            f" eff {row['read_efficiency']:.4f}  buckets " +
            " ".join(f"{bk}={row['buckets'][bk]}" for bk in BUCKETS))
        log(f"      needed parts {ev['need']}  SF {ev['in_SF']}  RF {ev['in_RF']}  "
            f"incumbent {ev['incumbent']}  NO-evidence {ev['no_evidence']}  picked {ev['picked']}")
    fp = f"{SS.SSD}/diag/funnel_{tag or ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
        sys.argv[2].split(",") if len(sys.argv) > 2 else
        ("R0_STATIC", "R1_EQUAL_RRF", "R2_HOP_CONDITIONED", "RX_ORACLE_READ"),
        tag=sys.argv[3] if len(sys.argv) > 3 else None)
