"""STEPS 6-10 -- three parameter-free node rankings, then the unchanged S4 -> F6 -> EXACT P50.

Where the ranking is applied, and why the beam is left alone
-----------------------------------------------------------
The beam only ever compares candidates that are at the SAME hop, because a hop expands one frontier
at a time.  So the cross-hop miscalibration that the separability audit found cannot affect the beam:
it affects exactly the two stages that ARE cross-hop, the `added` ordering and the `M_struct = 64`
read.  Those two are the same list, so applying one ranking to `added` automatically applies it to the
read -- STEP 7 is satisfied by construction, with one key, not two.  The winning ranking is then re-run
with the same key inside the beam as an explicit STEP 7 verification.

`s_hop`, `s_sdir` and `s_cnt` are the frozen per-node values (vmeta hop, max static s_dir, arrival
count).  A ranking permutes which nodes S4 sees and in what order; it never changes what S4 reads off
a node.  S4, F6, B = 6, M_struct = 64, M_ret = 32 and P = 50 are untouched.

  python scratchpad/_l1ss_rank.py ds [R0_STATIC,R1_EQUAL_RRF,...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1ss_core as SS
import _l1ss_sep as SEP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0 = 60                       # the frozen RRF constant, reused -- not a new parameter
RANKINGS = ["R0_STATIC", "R1_EQUAL_RRF", "R2_HOP_CONDITIONED"]
# R1: one channel per distinct MECHANISM -- geometry, multi-path support, retrieval provenance.
R1_CHANNELS = ["CS_ADMIT", "PATH_SUPPORT", "SEED_RANK"]
# R2: the mechanistically appropriate signal for each structural position.  A hop-1 node is adjacent
# to a seed, so the only thing that distinguishes it is WHICH seed; a deeper node is reached by
# several routes, so multi-path agreement is the evidence.
R2_BY_HOP = {1: "SEED_RANK", 2: "PATH_SUPPORT", 3: "PATH_SUPPORT"}


def _rank_in(vals, sign, ties):
    """0-based rank inside one query, ties broken by frozen `added` position."""
    return np.argsort(np.lexsort((ties, (-sign) * vals))).astype(np.float64)


def order_for(rank, F, n):
    """the permutation of one query candidate list under `rank`.  `F` maps feature name -> array."""
    ties = np.arange(n, dtype=np.int64)                 # frozen `added` position
    if rank == "R0_STATIC":
        return ties                                     # the list is already in frozen order
    if rank == "R1_EQUAL_RRF":
        s = np.zeros(n)
        for ch in R1_CHANNELS:
            s += 1.0 / (K0 + _rank_in(F[ch], SS.SIGN[ch], ties))
        return np.lexsort((ties, -s))
    if rank == "R2_HOP_CONDITIONED":
        hop = F["HOP"].astype(int)
        key = np.full(n, 1e18)
        for h in sorted(set(hop.tolist())):
            m = hop == h
            sg = R2_BY_HOP.get(h, "CS_ADMIT")
            key[m] = _rank_in(F[sg][m], SS.SIGN[sg], ties[m])
        return np.lexsort((ties, hop, key))             # round-robin: rank 0 of every stratum first
    if rank == "RS_BEST_SINGLE":
        # STEP 6, the directive first branch: "if one signal clearly dominates, use it directly".
        # RR_PARENT_SUPPORT is the STEP 2/3 winner (R@64 0.2061 vs the frozen key 0.1334).
        # One column, no weights, no thresholds, no combination.
        return np.lexsort((ties, -SS.SIGN["RR_PARENT_SUPPORT"] * F["RR_PARENT_SUPPORT"]))
    if rank == "RX_ORACLE_READ":
        # NOT a candidate ranking.  A gold-using upper bound that answers one question: if the read
        # stage were solved perfectly, how much exact-P50 would that be worth?  If the answer is
        # "almost none", no inference-safe signal can convert either, and the bottleneck is not here.
        return np.lexsort((ties, -F["_GOLD"]))
    raise ValueError(rank)


def run(ds, ranks, tag=None):
    D = SEP.load(tag or ds)
    S = EV.substrate(ds); nq = S["nq"]; hard = S["hard"]; hops = S["hops"]
    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    i_safe, _ = EV.evaluate(S, SAFE, "A_F6", (EV.P,))
    OUT = {"ds": tag or ds, "substrate": ds, "nq": nq, "SAFE": EV.by_hop(i_safe[EV.P], hops), "STEP9": {}, "STEP10": {}}
    log(f"SAFE {OUT['SAFE']}")
    qptr = D["qptr"]
    gold_total = float(D["ngold"].sum())

    for rk in ranks:
        SF, read_hit, read_hit_h = [], 0, {}
        ps4 = np.zeros(nq, np.int64); pg = np.zeros(nq, np.int64)
        for qi in range(nq):
            a, b = int(qptr[qi]), int(qptr[qi + 1])
            n = b - a
            F = {k: D[k][a:b] for k in SS.FEATS}
            F["_GOLD"] = D["gold"][a:b].astype(np.float64)
            o = order_for(rk, F, n) if n else np.empty(0, np.int64)
            sn = np.full(SS.BM.SMAX, -1, np.int32); sh = np.zeros(SS.BM.SMAX, np.int8)
            sd = np.zeros(SS.BM.SMAX, np.float32); sc = np.zeros(SS.BM.SMAX, np.int32)
            k = min(n, SS.BM.SMAX)
            if k:
                idx = o[:k]
                sn[:k] = D["node"][a:b][idx]
                sh[:k] = F["HOP"][idx].astype(np.int8)
                sd[:k] = F["STATIC_SDIR"][idx].astype(np.float32)
                sc[:k] = F["PATH_SUPPORT"][idx].astype(np.int32)
            SF.append(EV.sf_from_cache({"s_node": sn[None], "s_hop": sh[None], "s_sdir": sd[None],
                                        "s_cnt": sc[None]}, 0, hard, "P0_S4"))
            g = D["gold"][a:b]
            h = int(g[o[:min(n, EV.M_STRUCT)]].sum()) if n else 0
            read_hit += h
            hq = int(hops[qi]) if hops is not None else -1
            read_hit_h[hq] = read_hit_h.get(hq, 0) + h
            gp = set(int(x) for x in D["part"][a:b][g.astype(bool)])
            sfs = set(int(p) for p in SF[-1])
            ps4[qi] = len(gp & sfs); pg[qi] = len(gp)
        ind, churn = EV.evaluate(S, SF, "A_F6", (EV.P,))
        m = PP.mcnemar(ind[EV.P], i_safe[EV.P])
        blocks = {}
        hs = sorted(set(int(x) for x in hops)) if hops is not None else []
        if hs and min(hs) < 0:
            hs = []                       # only metaqa carries real hop annotations
        if hs:
            for h in hs:
                mk = np.asarray(hops) == h
                blocks[f"hop{h}"] = PP.mcnemar(ind[EV.P][mk], i_safe[EV.P][mk])
        OUT["STEP9"][rk] = {"P50": EV.by_hop(ind[EV.P], hops), "churn": round(float(churn.mean()), 3),
                            **{f"vs_SAFE_{k}": v for k, v in m.items()}, "vs_SAFE_by_hop": blocks}
        eff = {}
        for h in hs:
            mk = np.asarray(hops) == h
            gt = float(D["ngold"][mk].sum())
            eff[f"hop{h}"] = {"gold_read@64": read_hit_h.get(h, 0),
                              "read_recall@64": round(read_hit_h.get(h, 0) / max(1, gt), 4)}
        OUT["STEP10"][rk] = {"gold_read@64": read_hit,
                             "read_recall@64": round(read_hit / gold_total, 4),
                             "gold_parts_in_S4": int(ps4.sum()), "gold_parts_total": int(pg.sum()),
                             "gold_part_S4_frac": round(float(ps4.sum() / max(1, pg.sum())), 4),
                             **eff}
        for h in hs:
            mk = np.asarray(hops) == h
            OUT["STEP10"][rk][f"hop{h}"]["gold_part_S4_frac"] = round(
                float(ps4[mk].sum() / max(1, pg[mk].sum())), 4)
        v = OUT["STEP9"][rk]
        log(f"  {rk:22s} ALL {v['P50']['ALL']:.4f}  " + "  ".join(
            f"h{h} {v['P50'][f'hop{h}']:.4f}" for h in hs) +
            f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}"
            f"   read@64 {OUT['STEP10'][rk]['read_recall@64']:.4f}"
            f"   goldpart->S4 {OUT['STEP10'][rk]['gold_part_S4_frac']:.4f}")
    fp = f"{SS.SSD}/diag/rank_{tag or ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
        sys.argv[2].split(",") if len(sys.argv) > 2 else RANKINGS,
        tag=sys.argv[3] if len(sys.argv) > 3 else None)
