"""FULL_VISITED PARTITION CALIBRATION AUDIT -- STEPS 1-8 for one corpus.

    python scratchpad/_l1pc_run.py <dataset>

Contract held fixed everywhere: FINAL P = 50, B = 6, the frozen bounded search, the frozen F6
boundary competition, no learned parameter, no threshold grid, no dataset branch, no TEST split.
The ONLY thing any variant changes is the ORDER of the structural partition ranking handed to F6.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import rankdata, spearmanr
import _l1pp_core as PP
import _l1sr_eval as EV
import _l1pc_core as PC

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
RECALL_AT = [6, 12, 20, 50]

# STEP 2 -- every signal is an EXISTING recorded quantity.  sign +1 = larger is better.
SIGNALS = [
    ("S4_RANK_FULL_VISITED", "s4rank", -1),
    ("BEST_STRUCT_NODE_SCORE", "max_sdir", +1),
    ("STRUCT_NODE_COUNT", "nnodes", +1),
    ("DISTINCT_SEED_COUNT", "nseeds", +1),
    ("DISTINCT_PARENT_COUNT", "path_support", +1),
    ("RANK_WEIGHTED_PARENT_SUPPORT", "sum_rr", +1),
    ("MIN_HOP", "min_hop", -1),
    ("BEST_PATH_SCORE", "max_psum", +1),
    ("BEST_PATH_MIN_SCORE", "max_pmin", +1),
    ("BEST_PARENT_SCORE", "max_psc", +1),
    ("FIRST_ARRIVAL_INDEX", "first_jj", -1),
    ("RRF_MASS", "rrf_sum", +1),
    ("ADMITTED_NODE_COUNT", "n_adm", +1),
    ("CANONICAL_RANK", "cpos", -1),
    ("DENSE_RANK", "r_dense", -1),
    ("SPLADE_RANK", "r_splade", -1),
    ("RETRIEVAL_RRF_RANK", "r_rrf", -1),
]
# STEP 5 -- coherence family, all derived from columns already present
COHERENCE = ["COH_SEEDS_PER_NODE", "COH_ARRIVALS_PER_NODE", "COH_BEST_NODE_RRF_SHARE",
             "COH_ADMITTED_FRACTION", "COH_SEED_TIMES_SDIR"]
# STEP 4 -- the normalisation family.  One division each, no exponent, no weight, no threshold.
NORMS = ["N0_RAW", "N1_DEGREE_NORMALIZED", "N2_SIZE_NORMALIZED", "N3_EXPOSURE_NORMALIZED"]


def blocks(S, nq):
    hops = np.asarray(S["z"]["hops"])[:nq] if "hops" in S["z"] else None
    out = [("ALL", np.ones(nq, bool))]
    if hops is not None and len(hops) and int(min(hops)) >= 0:
        for h in sorted(set(int(x) for x in hops)):
            out.append((f"hop{h}", hops == h))
    return out


def auc_recall(sig, pos, k_list):
    """per-query AUC + micro recall@k of the POSITIVE class under one signed signal.

    `sig` is already oriented so that LARGER is better.  Ties get average ranks, so a constant
    signal scores exactly 0.5 and cannot be mistaken for separation."""
    n = len(sig)
    npos = int(pos.sum())
    if npos == 0 or npos == n:
        return None, {k: (0, 0) for k in k_list}, None
    r = rankdata(sig)                                   # ascending, ties averaged
    auc = (r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * (n - npos))
    pct = float(np.mean((r[pos] - 1) / max(n - 1, 1)))  # percentile rank of the positives
    order = np.lexsort((np.arange(n), -sig))
    hit = {}
    for k in k_list:
        hit[k] = (int(pos[order[:k]].sum()), npos)
    return float(auc), hit, pct


def run(ds, log=log, nq_max=None):
    S, T = PC.build(ds, log=log, nq_max=nq_max)
    nq = int(T["nq"][0]); npart = int(T["npart"][0])
    hard = S["hard"].astype(np.int64)
    G = PC.partition_geometry(ds, hard, npart, log=log)
    goldp = S["goldp"]
    BL = blocks(S, nq)
    OUT = {"ds": ds, "nq": nq, "npart": npart,
           "PARITY": f"{int(T['parity'][0])}/{nq}",
           "WORK": dict(zip(["edges_per_q", "cands_per_q", "visited_per_q", "admitted_per_q",
                             "ms_per_q"], [round(float(x), 2) for x in T["work"]]))}

    # ---------------- exposure prior (label-free, query-independent), plus a 2-fold control ------
    vis_cnt = np.zeros(npart, np.float64)
    vis_nodes = np.zeros(npart, np.float64)
    fold = np.arange(nq) % 2
    vis_nodes_f = [np.zeros(npart), np.zeros(npart)]
    nq_f = [int((fold == 0).sum()), int((fold == 1).sum())]
    for qi in range(nq):
        r = PC.rows(T, qi)
        vis_cnt[r["part"]] += 1.0
        vis_nodes[r["part"]] += r["nnodes"]
        vis_nodes_f[fold[qi]][r["part"]] += r["nnodes"]
    P_VIS = vis_cnt / nq
    EXP = vis_nodes / nq                                  # expected visited nodes per query
    EXP_F = [vis_nodes_f[0] / max(nq_f[0], 1), vis_nodes_f[1] / max(nq_f[1], 1)]
    OUT["EXPOSURE"] = {"universe_saturation": round(float(P_VIS.mean()), 4),
                       "mean_partitions_visited_per_query":
                           round(float(sum(int(T["qptr"][i + 1] - T["qptr"][i])
                                           for i in range(nq)) / nq), 1),
                       "frac_partitions_ever_visited": round(float((P_VIS > 0).mean()), 4),
                       "frac_partitions_visited_in_over_half_of_queries":
                           round(float((P_VIS > 0.5).mean()), 4),
                       "exposure_prior_2fold_spearman":
                           round(float(spearmanr(EXP_F[0], EXP_F[1]).statistic), 4),
                       "queries_with_empty_visited_universe":
                           int(sum(1 for i in range(nq)
                                   if int(T["qptr"][i + 1]) == int(T["qptr"][i])))}

    # ---------------- STEP 1: the exact challenger universe --------------------------------------
    CU = []                                               # per query: (part, needed mask, rows)
    n_cand = n_need = 0
    need_out_total = need_in_univ = 0
    for qi in range(nq):
        r = PC.rows(T, qi)
        m = r["in_b50"] == 0
        p = r["part"][m]
        gp = goldp[qi]
        b50 = S["ctxs"][qi]["base50"]
        need = np.array([int(int(x) in gp) for x in p], bool)
        CU.append((p, need, {c: r[c][m] for c in PC.COLS}))
        n_cand += len(p); n_need += int(need.sum())
        need_out_total += len(gp - b50)
        need_in_univ += int(need.sum())
    OUT["STEP1"] = {
        "candidates_per_query": round(n_cand / nq, 1),
        "needed_per_query": round(n_need / nq, 3),
        "needed_prevalence": round(n_need / max(n_cand, 1), 6),
        "gold_partitions_outside_canonical_top50": need_out_total,
        "of_which_in_FULL_VISITED_universe": need_in_univ,
        "needed_reachability": round(need_in_univ / max(need_out_total, 1), 4)}
    for nm, mk in BL[1:]:
        c = sum(len(CU[qi][0]) for qi in range(nq) if mk[qi])
        n = sum(int(CU[qi][1].sum()) for qi in range(nq) if mk[qi])
        OUT["STEP1"][nm] = {"candidates_per_query": round(c / max(int(mk.sum()), 1), 1),
                            "needed_per_query": round(n / max(int(mk.sum()), 1), 3),
                            "needed_prevalence": round(n / max(c, 1), 6)}
    log(f"STEP1 candidates/q {OUT['STEP1']['candidates_per_query']}  "
        f"needed/q {OUT['STEP1']['needed_per_query']}  "
        f"prevalence {OUT['STEP1']['needed_prevalence']:.5f}  "
        f"saturation {OUT['EXPOSURE']['universe_saturation']:.4f}")

    # ---------------- signal matrix (raw + coherence + normalised), per query --------------------
    def sigmat(qi):
        p, need, r = CU[qi]
        s4 = PC.s4_order(r["part"], r["first_jj"], r["nnodes"], r["min_hop"], r["max_sdir"])
        pos = {int(x): i for i, x in enumerate(s4)}
        d = {c: r[c].astype(np.float64) for c in PC.COLS}
        d["s4rank"] = np.array([pos[int(x)] for x in p], np.float64)
        nn = np.maximum(d["nnodes"], 1.0)
        d["COH_SEEDS_PER_NODE"] = d["nseeds"] / nn
        d["COH_ARRIVALS_PER_NODE"] = d["path_support"] / nn
        d["COH_BEST_NODE_RRF_SHARE"] = (1.0 / (PC.K0 + d["first_jj"])) / np.maximum(d["rrf_sum"], 1e-12)
        d["COH_ADMITTED_FRACTION"] = d["n_adm"] / nn
        d["COH_SEED_TIMES_SDIR"] = d["nseeds"] * d["max_sdir"]
        d["psize"] = G["psize"][p]; d["pdeg"] = G["pdeg"][p]; d["pbdeg"] = G["pbdeg"][p]
        d["padj"] = G["padj"][p]; d["pnodedeg"] = G["pnodedeg"][p]
        d["pexp"] = EXP[p]; d["pvis"] = P_VIS[p]
        d["pexp_heldout"] = EXP_F[1 - (qi % 2)][p]
        return p, need, d

    SM = [sigmat(qi) for qi in range(nq)]

    # ---------------- STEP 2: needed-vs-nuisance separability ------------------------------------
    def audit(names, tag):
        res = {}
        for nm, col, sgn in names:
            per = {b: {"auc": [], "pct": [], "hit": {k: 0 for k in RECALL_AT}, "tot": 0}
                   for b, _ in BL}
            for qi in range(nq):
                p, need, d = SM[qi]
                if len(p) == 0:
                    continue
                a, hit, pct = auc_recall(sgn * d[col], need, RECALL_AT)
                for b, mk in BL:
                    if not mk[qi]:
                        continue
                    e = per[b]
                    if a is not None:
                        e["auc"].append(a); e["pct"].append(pct)
                    for k in RECALL_AT:
                        e["hit"][k] += hit[k][0]
                    e["tot"] += int(need.sum())
            res[nm] = {b: {"AUC": round(float(np.mean(per[b]["auc"])), 4) if per[b]["auc"] else None,
                           "needed_percentile_rank":
                               round(float(np.mean(per[b]["pct"])), 4) if per[b]["pct"] else None,
                           "queries_scored": len(per[b]["auc"]),
                           "needed_total": per[b]["tot"],
                           **{f"NEEDED_RECALL@{k}":
                              round(per[b]["hit"][k] / max(per[b]["tot"], 1), 4) for k in RECALL_AT}}
                       for b, _ in BL}
        OUT[tag] = res
        blk = "hop3" if "hop3" in dict(BL) else "ALL"
        for nm, _, _ in names:
            r = res[nm][blk]
            log(f"{tag} {nm:30s} {blk} AUC {str(r['AUC']):>7s}  "
                f"R@6 {r['NEEDED_RECALL@6']:.4f}  R@12 {r['NEEDED_RECALL@12']:.4f}  "
                f"R@20 {r['NEEDED_RECALL@20']:.4f}  R@50 {r['NEEDED_RECALL@50']:.4f}")
        return res

    audit(SIGNALS, "STEP2")
    audit([(c, c, +1) for c in COHERENCE], "STEP5")

    # ---------------- STEP 3: degree / exposure calibration --------------------------------------
    STAT = ["psize", "pdeg", "pbdeg", "padj", "pnodedeg", "pexp", "pvis"]
    nd_mean = {s: [[], []] for s in STAT}
    for qi in range(nq):
        p, need, d = SM[qi]
        if len(p) == 0:
            continue
        for s in STAT:
            if need.any():
                nd_mean[s][0].append(float(d[s][need].mean()))
            if (~need).any():
                nd_mean[s][1].append(float(d[s][~need].mean()))
    def _m(v):
        return float(np.mean(v)) if len(v) else float("nan")
    OUT["STEP3"] = {"needed_vs_nuisance_static": {
        s: {"needed_mean": round(_m(nd_mean[s][0]), 4),
            "nuisance_mean": round(_m(nd_mean[s][1]), 4),
            "ratio": round(_m(nd_mean[s][0]) / max(_m(nd_mean[s][1]), 1e-12), 4),
            "queries_with_a_needed_partition": len(nd_mean[s][0])}
        for s in STAT}}

    # corpus-level: does a static quantity predict how the raw ranking treats a partition?
    mean_pct = np.full(npart, np.nan)
    acc = np.zeros(npart); cnt = np.zeros(npart)
    for qi in range(nq):
        p, need, d = SM[qi]
        if len(p) < 2:
            continue
        pr = (rankdata(-d["s4rank"]) - 1) / (len(p) - 1)   # 1.0 = best structural rank
        acc[p] += pr; cnt[p] += 1
    m = cnt > 0
    mean_pct[m] = acc[m] / cnt[m]
    SARR = {**{k: G[k] for k in ("psize", "pdeg", "pbdeg", "padj", "pnodedeg")},
            "pexp": EXP, "pvis": P_VIS}
    OUT["STEP3"]["corpus_spearman_vs_raw_S4_percentile"] = {
        s: round(float(spearmanr(SARR[s][m], mean_pct[m]).statistic), 4) for s in STAT}
    OUT["STEP3"]["corpus_spearman_vs_P_visited"] = {
        s: round(float(spearmanr(SARR[s], P_VIS).statistic), 4) for s in STAT}
    log("STEP3 needed/nuisance static ratios  " +
        "  ".join(f"{s}={OUT['STEP3']['needed_vs_nuisance_static'][s]['ratio']:.3f}"
                  for s in STAT))
    log("STEP3 spearman(static, raw S4 percentile)  " +
        "  ".join(f"{s}={OUT['STEP3']['corpus_spearman_vs_raw_S4_percentile'][s]:+.3f}"
                  for s in STAT))

    # ---------------- STEP 4: the normalisation family -------------------------------------------
    def support_of(d, p, norm):
        s = d["nnodes"]
        if norm == "N0_RAW":
            return s
        if norm == "N1_DEGREE_NORMALIZED":
            return s / np.maximum(G["pdeg"][p], 1.0)
        if norm == "N2_SIZE_NORMALIZED":
            return s / np.maximum(G["psize"][p], 1.0)
        if norm == "N3_EXPOSURE_NORMALIZED":
            return s / np.maximum(d["pexp_heldout"], 1e-6)
        raise ValueError(norm)

    def full_order(qi, norm):
        """the structural ranking over the WHOLE visited universe (not just challengers), which is
        what F6 consumes.  Only the support column changes."""
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

    N4 = []
    for norm in NORMS:
        names = [(norm, "_norm", -1)]
        per = {b: {"auc": [], "pct": [], "hit": {k: 0 for k in RECALL_AT}, "tot": 0} for b, _ in BL}
        for qi in range(nq):
            p, need, d = SM[qi]
            if len(p) == 0:
                continue
            sup = support_of(d, p.astype(np.int64), norm)
            o = PC.s4_order(p.astype(np.int64), d["first_jj"], sup, d["min_hop"], d["max_sdir"])
            pos = {int(x): i for i, x in enumerate(o)}
            rk = np.array([pos[int(x)] for x in p], np.float64)
            a, hit, pct = auc_recall(-rk, need, RECALL_AT)
            for b, mk in BL:
                if not mk[qi]:
                    continue
                e = per[b]
                if a is not None:
                    e["auc"].append(a); e["pct"].append(pct)
                for k in RECALL_AT:
                    e["hit"][k] += hit[k][0]
                e["tot"] += int(need.sum())
        N4.append((norm, {b: {"AUC": round(float(np.mean(per[b]["auc"])), 4) if per[b]["auc"] else None,
                              "needed_percentile_rank":
                                  round(float(np.mean(per[b]["pct"])), 4) if per[b]["pct"] else None,
                              "needed_total": per[b]["tot"],
                              **{f"NEEDED_RECALL@{k}":
                                 round(per[b]["hit"][k] / max(per[b]["tot"], 1), 4)
                                 for k in RECALL_AT}} for b, _ in BL}))
    OUT["STEP4"] = dict(N4)
    blk = "hop3" if "hop3" in dict(BL) else "ALL"
    for norm, r in N4:
        log(f"STEP4 {norm:26s} {blk} AUC {r[blk]['AUC']}  R@6 {r[blk]['NEEDED_RECALL@6']:.4f}  "
            f"R@12 {r[blk]['NEEDED_RECALL@12']:.4f}  R@50 {r[blk]['NEEDED_RECALL@50']:.4f}")

    # ---------------- the attainable ceiling on NEEDED_RECALL@k --------------------------------
    ceil = {b: {k: [0, 0] for k in RECALL_AT} for b, _ in BL}
    for qi in range(nq):
        n = int(CU[qi][1].sum())
        for b, mk in BL:
            if not mk[qi]:
                continue
            for k in RECALL_AT:
                ceil[b][k][0] += min(k, n); ceil[b][k][1] += n
    OUT["NEEDED_RECALL_CEILING"] = {b: {f"NEEDED_RECALL@{k}": round(ceil[b][k][0] /
                                                                    max(ceil[b][k][1], 1), 4)
                                        for k in RECALL_AT} for b, _ in BL}

    # ---------------- STEP 3b: does FULL_VISITED *cause* exposure to dominate? -------------------
    # the identical measurement on the FROZEN M64 universe.  If the correlation is already there at
    # M64, FULL_VISITED inherits it rather than creating it.
    frozen_sf = [EV.sf_from_cache(S["z"], qi, hard, "P0_S4") for qi in range(S["nq"])]
    acc2 = np.zeros(npart); cnt2 = np.zeros(npart)
    for qi in range(nq):
        o = np.asarray(frozen_sf[qi], np.int64)
        if len(o) < 2:
            continue
        pr = 1.0 - np.arange(len(o)) / (len(o) - 1)
        acc2[o] += pr; cnt2[o] += 1
    m2 = cnt2 > 0
    mp2 = np.full(npart, np.nan); mp2[m2] = acc2[m2] / cnt2[m2]
    OUT["STEP3"]["M64_corpus_spearman_vs_S4_percentile"] = {
        s: round(float(spearmanr(SARR[s][m2], mp2[m2]).statistic), 4) for s in STAT}
    OUT["STEP3"]["EXPOSURE_DOMINANCE_CAUSED_BY_FULL_VISITED"] = {
        s: round(OUT["STEP3"]["corpus_spearman_vs_raw_S4_percentile"][s]
                 - OUT["STEP3"]["M64_corpus_spearman_vs_S4_percentile"][s], 4) for s in STAT}
    log("STEP3b spearman on the FROZEN M64 universe        " +
        "  ".join(f"{s}={OUT['STEP3']['M64_corpus_spearman_vs_S4_percentile'][s]:+.3f}"
                  for s in STAT))

    # ---------------- STEP 5 direct: rank the universe by the best single signals ----------------
    # These orderings are chosen AFTER seeing STEP 2, so they are diagnostics; only survival on all
    # six corpora under the promotion gate would make one of them a candidate.
    def direct_order(qi, rule, restrict=None):
        r = PC.rows(T, qi) if restrict is None else restrict
        p = r["part"].astype(np.int64)
        if rule == "D1_BEST_NODE_SCORE":
            key = np.lexsort((p, r["first_jj"], -r["max_sdir"]))
        elif rule == "D2_SEED_TIMES_SDIR":
            key = np.lexsort((p, r["first_jj"], -(r["nseeds"] * r["max_sdir"])))
        elif rule == "D3_S4_TWO_CHANNEL":
            m = len(p); rk = np.zeros(m)
            for k_, sg in ((r["first_jj"].astype(np.float64), +1.0),
                           (r["max_sdir"].astype(np.float64), -1.0)):
                o = np.lexsort((p, sg * k_)); rr = np.empty(m, np.int64); rr[o] = np.arange(m)
                rk += 1.0 / (PC.K0 + rr)
            key = np.lexsort((p, -rk))
        else:
            raise ValueError(rule)
        return p[key]

    DIRECT = ["D1_BEST_NODE_SCORE", "D2_SEED_TIMES_SDIR", "D3_S4_TWO_CHANNEL"]
    dres = {}
    for rule in DIRECT:
        per = {b: {"auc": [], "hit": {k: 0 for k in RECALL_AT}, "tot": 0} for b, _ in BL}
        for qi in range(nq):
            p, need, d = SM[qi]
            if len(p) == 0:
                continue
            o = direct_order(qi, rule, restrict=CU[qi][2])
            pos = {int(x): i for i, x in enumerate(o)}
            rk = np.array([pos[int(x)] for x in p], np.float64)
            a, hit, pct = auc_recall(-rk, need, RECALL_AT)
            for b, mk in BL:
                if not mk[qi]:
                    continue
                if a is not None:
                    per[b]["auc"].append(a)
                for k in RECALL_AT:
                    per[b]["hit"][k] += hit[k][0]
                per[b]["tot"] += int(need.sum())
        dres[rule] = {b: {"AUC": round(float(np.mean(per[b]["auc"])), 4) if per[b]["auc"] else None,
                          **{f"NEEDED_RECALL@{k}": round(per[b]["hit"][k] / max(per[b]["tot"], 1), 4)
                             for k in RECALL_AT}} for b, _ in BL}
    OUT["STEP5_DIRECT"] = dres
    blk = "hop3" if "hop3" in dict(BL) else "ALL"
    for rule in DIRECT:
        r = dres[rule][blk]
        log(f"STEP5D {rule:24s} {blk} AUC {r['AUC']}  R@6 {r['NEEDED_RECALL@6']:.4f}  "
            f"R@12 {r['NEEDED_RECALL@12']:.4f}  R@50 {r['NEEDED_RECALL@50']:.4f}")

    # ---------------- STEP 7/8: same semantics through the frozen F6, exact P50 ------------------
    frozen = [EV.sf_from_cache(S["z"], qi, hard, "P0_S4") for qi in range(S["nq"])]
    i_safe, ch_safe = EV.evaluate(S, frozen, "A_F6", (EV.P,))
    i_safe = i_safe[EV.P][:nq]
    OUT["SAFE"] = {b: round(float(i_safe[mk].mean()), 4) for b, mk in BL}
    OUT["STEP78"] = {}
    best_extra = {}
    for norm in NORMS:
        SF = [full_order(qi, norm) for qi in range(nq)]
        ind, churn = EV.evaluate(S, SF + frozen[nq:], "A_F6", (EV.P,))
        ind = ind[EV.P][:nq]
        row = {b: round(float(ind[mk].mean()), 4) for b, mk in BL}
        row["churn"] = round(float(churn[:nq].mean()), 3)
        row["vs_SAFE"] = PP.mcnemar(ind, i_safe)
        for b, mk in BL[1:]:
            row[f"vs_SAFE_{b}"] = PP.mcnemar(ind[mk], i_safe[mk])
        OUT["STEP78"][f"FULL_VISITED/{norm}"] = row
        best_extra[norm] = ind
        log(f"STEP78 FULL_VISITED/{norm:26s} " +
            "  ".join(f"{b} {row[b]:.4f}" for b, _ in BL) +
            f"   net {row['vs_SAFE']['net']:+d} p={row['vs_SAFE']['mcnemar_p']:.3g}"
            f"{' SIG' if row['vs_SAFE']['sig'] else ''}")
    for rule in DIRECT:
        SF = [direct_order(qi, rule) for qi in range(nq)]
        ind, churn = EV.evaluate(S, SF + frozen[nq:], "A_F6", (EV.P,))
        ind = ind[EV.P][:nq]
        row = {b: round(float(ind[mk].mean()), 4) for b, mk in BL}
        row["churn"] = round(float(churn[:nq].mean()), 3)
        row["vs_SAFE"] = PP.mcnemar(ind, i_safe)
        for b, mk in BL[1:]:
            row[f"vs_SAFE_{b}"] = PP.mcnemar(ind[mk], i_safe[mk])
        OUT["STEP78"][f"FULL_VISITED/{rule}"] = row
        log(f"STEP78 FULL_VISITED/{rule:26s} " +
            "  ".join(f"{b} {row[b]:.4f}" for b, _ in BL) +
            f"   net {row['vs_SAFE']['net']:+d} p={row['vs_SAFE']['mcnemar_p']:.3g}"
            f"{' SIG' if row['vs_SAFE']['sig'] else ''}")
    OUT["STEP78"]["FROZEN_M64/S4 (SAFE)"] = {**OUT["SAFE"],
                                             "churn": round(float(ch_safe[:nq].mean()), 3)}

    fp = f"{PC.PCD}/diag/calib_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    nqm = next((int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("--nq=")), None)
    for d in (a or ["metaqa"]):
        run(d, nq_max=nqm)
