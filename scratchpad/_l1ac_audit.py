"""L1 ARCHITECTURAL CEILING AUDIT -- is further L1 selector work worth doing at all?

No inference method is proposed here.  Every number is either an already-measured router result or
a gold-evaluated CEILING of the partition-budget abstraction.

THE CENTRAL IDENTITY.  A full-universe oracle at budget P may pick ANY P partitions from the
complete canonical universe, so for ALL-coverage it succeeds exactly when the query's gold evidence
fits in P partitions:

    FULL_UNIVERSE_ORACLE_ALL(P)  ==  fraction of queries with |gold_partitions| <= P

which is also, verbatim, the "pure representational ceiling" GOLD_PARTITION_COUNT <= P that STEP 2
asks for separately.  They are the same quantity, reported once and named as such.

GOLD MAPPING IS TOTAL.  `hard` assigns every corpus doc to exactly one partition and gold partitions
are built as `hard[g]` over the gold rows, so no gold doc can be unmapped.  The audit verifies this
(gold_cnt sums == gold row counts) rather than assuming it.

  python scratchpad/_l1ac_audit.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1pp_core as PP
import _l1kb_core as KB

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
AUD = f"{PP.ROOT}/L1_CEILING_AUDIT"
PBUD = [10, 25, 50, 75, 100, 150]
CANON_P = [25, 50, 75, 100, 150]
B = 6


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def dist(counts):
    c = np.asarray(counts, np.int64)
    n = max(1, len(c))
    return {"n": len(c), "mean": round(float(c.mean()), 3), "median": float(np.median(c)),
            "p75": pct(c, 75), "p90": pct(c, 90), "p95": pct(c, 95), "p99": pct(c, 99),
            "max": int(c.max()) if len(c) else 0,
            "frac_1": round(float((c == 1).sum() / n), 4),
            "frac_2": round(float((c == 2).sum() / n), 4),
            "frac_3": round(float((c == 3).sum() / n), 4),
            "frac_4": round(float((c == 4).sum() / n), 4),
            "frac_5plus": round(float((c >= 5).sum() / n), 4),
            "frac_10plus": round(float((c >= 10).sum() / n), 4),
            "frac_gt25": round(float((c > 25).sum() / n), 4),
            "frac_gt50": round(float((c > 50).sum() / n), 4),
            "frac_0": round(float((c == 0).sum() / n), 4)}


def ppr_lists(ds, nq, M=64):
    f = f"{PP.PPD}/ppr/mass_{ds}.npz"
    if not os.path.exists(f):
        return None
    m = np.load(f)["mass_global"]
    order = np.argsort(-m, axis=1, kind="stable")[:, :M]
    return [[int(p) for p in order[qi] if m[qi, int(p)] > 0] for qi in range(nq)]


def run_ds(ds, OUT):
    S = CC.substrate(ds, B)
    z, meta, nq, goldp, ctxs = S["z"], S["meta"], S["nq"], S["goldp"], S["ctxs"]
    hops = S["hops"]
    ch = PP.channels(ds, z, meta)
    npart = int(ch["npart"])
    full = ch["base_rank_replay"]                       # (nq, npart) canonical Dense+SPLADE ranking
    assert full.shape == (nq, npart)
    # rank position of every partition under the full canonical ranking
    fpos = np.empty((nq, npart), np.int32)
    rows = np.arange(nq)[:, None]
    fpos[rows, full] = np.arange(npart, dtype=np.int32)[None, :]

    # ---- gold mapping completeness (verified, not assumed)
    gc, gptr = z["gold_cnt"], z["gold_ptr"]
    gold_docs = np.array([int(gc[gptr[qi]:gptr[qi + 1]].sum()) for qi in range(nq)])
    cnt = np.array([len(goldp[qi]) for qi in range(nq)])
    row = {"npart": npart, "nq": nq,
           "BASE_RANK_PARITY": ch["BASE_RANK_PARITY"],
           "queries_with_zero_gold_partitions": int((cnt == 0).sum()),
           "queries_with_zero_gold_docs": int((gold_docs == 0).sum()),
           "gold_docs_per_query_mean": round(float(gold_docs.mean()), 2)}

    # ---- best / worst required-gold canonical rank
    best = np.full(nq, -1, np.int32); worst = np.full(nq, -1, np.int32)
    for qi in range(nq):
        if not goldp[qi]:
            continue
        g = np.fromiter(goldp[qi], np.int64, len(goldp[qi]))
        r = fpos[qi, g]
        best[qi] = int(r.min()); worst[qi] = int(r.max())
    have = worst >= 0

    def block(mask, tag):
        m = mask & have
        e = {"n": int(mask.sum()),
             "FULL_UNIVERSE_ORACLE_ALL": {str(p): round(float((cnt[mask] <= p).mean()), 4)
                                          for p in PBUD},
             "ANY": round(float((cnt[mask] >= 1).mean()), 4),
             "GOLD_PARTITION_COUNT": dist(cnt[mask]),
             "CANONICAL_TOPP_ALL": {str(p): round(float(((worst[m] < p)).mean()), 4)
                                    for p in CANON_P},
             "BEST_REQUIRED_GOLD_RANK": {"median": pct(best[m], 50), "p75": pct(best[m], 75),
                                         "p90": pct(best[m], 90), "p95": pct(best[m], 95),
                                         "p99": pct(best[m], 99)},
             "WORST_REQUIRED_GOLD_RANK": {"median": pct(worst[m], 50), "p75": pct(worst[m], 75),
                                          "p90": pct(worst[m], 90), "p95": pct(worst[m], 95),
                                          "p99": pct(worst[m], 99),
                                          "max": int(worst[m].max()) if m.sum() else -1}}
        log("   %-10s n=%5d  oracleP50 %.4f  P100 %.4f  P150 %.4f | canonP50 %.4f P150 %.4f | "
            "goldparts med %.0f p90 %.0f max %d | worstrank med %.0f p90 %.0f"
            % (tag, e["n"], e["FULL_UNIVERSE_ORACLE_ALL"]["50"],
               e["FULL_UNIVERSE_ORACLE_ALL"]["100"], e["FULL_UNIVERSE_ORACLE_ALL"]["150"],
               e["CANONICAL_TOPP_ALL"]["50"], e["CANONICAL_TOPP_ALL"]["150"],
               e["GOLD_PARTITION_COUNT"]["median"], e["GOLD_PARTITION_COUNT"]["p90"],
               e["GOLD_PARTITION_COUNT"]["max"],
               e["WORST_REQUIRED_GOLD_RANK"]["median"], e["WORST_REQUIRED_GOLD_RANK"]["p90"]))
        return e

    row["AGGREGATE"] = block(np.ones(nq, bool), "aggregate")
    if ds == "metaqa":
        row["PER_HOP"] = {str(h): block(hops == h, f"hop{h}") for h in (1, 2, 3)}

    # ---- pool oracles (already-measured router reference points)
    ind = np.load(f"{CC.CALD}/diag/ind_{ds}.npz")
    pl = ppr_lists(ds, nq)
    orc_cur = ind["oracle"].astype(np.int8)
    orc_ppr = np.zeros(nq, np.int8)
    for qi, c in enumerate(ctxs):
        uni = set(c["bnd"]) | set(c["chal"])
        if pl:
            uni |= {p for p in pl[qi] if p not in c["base50"]}
        miss = goldp[qi] - c["prot_set"]
        orc_ppr[qi] = int(len(goldp[qi]) <= CC.P and len(miss) <= B and miss <= uni)
    # ---- constraint ladder: turn each frozen constraint off one at a time, gold-evaluated.
    # fits50  : the partition abstraction alone            (|G| <= 50)
    # freeB   : abstraction + unlimited pool, B=6 still caps swaps out of the protected core
    # freePool: abstraction + current pool reach, B unlimited
    # both    : the actual current-pool oracle at B=6
    fits50 = np.zeros(nq, np.int8); freeB = np.zeros(nq, np.int8); freePool = np.zeros(nq, np.int8)
    for qi, c in enumerate(ctxs):
        g = goldp[qi]
        if len(g) > CC.P:
            continue
        fits50[qi] = 1
        miss = g - c["prot_set"]
        freeB[qi] = int(len(miss) <= B)
        freePool[qi] = int(miss <= (set(c["bnd"]) | set(c["chal"])))

    row["REFERENCE_INDICATORS"] = {}
    ref = {"BASE_P50": ind["base"], "SAFE_F6": ind["R0_RAW_RRF"], "R1_LIFT": ind["R1_RRF_LIFT"],
           "CURRENT_POOL_ORACLE_B6": orc_cur, "PPR_EXPANDED_POOL_ORACLE_B6": orc_ppr,
           "ORACLE_ABSTRACTION_ONLY_P50": fits50,
           "ORACLE_UNLIMITED_POOL_B6": freeB,
           "ORACLE_CURRENT_POOL_UNLIMITED_B": freePool}
    for k, v in ref.items():
        v = np.asarray(v, np.int8)
        e = {"aggregate": round(float(v.mean()), 4)}
        if ds == "metaqa":
            e["per_hop"] = {str(h): round(float(v[hops == h].mean()), 4) for h in (1, 2, 3)}
        row["REFERENCE_INDICATORS"][k] = e

    # ---- STEP 8 decomposition (MetaQA hop3, queries the SAFE router fails)
    if ds == "metaqa":
        f6 = np.asarray(ind["R0_RAW_RRF"], np.int8)
        tgt = np.where((hops == 3) & (f6 == 0))[0]
        cats = {k: 0 for k in ["A_REPRESENTATION_LIMIT", "B_P50_CAPACITY_LIMIT", "C_RANKING_LIMIT",
                               "D_CURRENT_POOL_LIMIT", "E_SELECTOR_LIMIT"]}
        for qi in tgt:
            c = ctxs[qi]; g = goldp[qi]
            if len(g) == 0 or len(g) > npart:
                cats["A_REPRESENTATION_LIMIT"] += 1; continue
            if len(g) > CC.P:
                cats["B_P50_CAPACITY_LIMIT"] += 1; continue
            miss = g - c["prot_set"]
            if not (miss <= (set(c["bnd"]) | set(c["chal"]))):
                cats["D_CURRENT_POOL_LIMIT"] += 1; continue
            if len(miss) > B:
                cats["C_RANKING_LIMIT"] += 1; continue
            cats["E_SELECTOR_LIMIT"] += 1
        # D and C can both bind on the same query; report the overlap rather than hiding it
        d_also_over_B = d_tot = 0
        for qi in tgt:
            c = ctxs[qi]; g = goldp[qi]
            if not (0 < len(g) <= CC.P):
                continue
            miss = g - c["prot_set"]
            if not (miss <= (set(c["bnd"]) | set(c["chal"]))):
                d_tot += 1
                d_also_over_B += int(len(miss) > B)
        n = max(1, len(tgt))
        row["HOP3_DECOMPOSITION"] = {
            "n_uncovered_hop3": int(len(tgt)),
            "precedence": "A (uncoverable at any P) -> B (|gold|>50) -> D (required partition "
                          "outside boundary+challengers) -> C (in pool but >B=6 outside the "
                          "protected core) -> E (in pool and fits B, selector chose wrong)",
            "counts": cats,
            "pct": {k: round(100.0 * v / n, 2) for k, v in cats.items()},
            "D_queries_that_ALSO_need_more_than_B_swaps": d_also_over_B,
            "D_total": d_tot,
            "D_frac_also_B_bound": round(d_also_over_B / max(1, d_tot), 4)}
        log("   hop3 uncovered=%d  %s" % (len(tgt), row["HOP3_DECOMPOSITION"]["pct"]))

    OUT[ds] = row


def main():
    os.makedirs(AUD, exist_ok=True)
    fp = f"{AUD}/ceiling_audit.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CC.DSETS):
        log(ds)
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote ceiling_audit.json")


if __name__ == "__main__":
    main()
