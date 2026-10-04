"""G2 B1.5 — QUERY-LOCAL ADAPTIVE RESERVE POLICY (3-target pilot; STOP before six-way).

Frozen from B1.4/B1.2 (NOT changed here): candidate universe, verification channel definitions
{BASE_FUSED,S0,S1_PRIME}, structural reserve orderer s_dir, two_channel_pool composition, TOP_POOL=50.
The ONLY new component is a map  query -> reserve size R in {0,4,8,16}.

Design choices (documented, faithful to directive):
  * Verification channel is FIXED to BASE_FUSED for the adaptive policy and the FIXED_R baselines. Rationale:
    the directive says the ONLY new component is query->R; fixing the channel isolates R as the sole variable.
    BASE_FUSED is the simplest frozen channel and won 2/3 B1.4 source-selections; learned channels added little
    and S1_PRIME caused the SQuAD toxicity. (B1.4's own source-selected channel+R is reported as a baseline row,
    loaded verbatim from its JSON.)
  * Training supervision (TRAIN queries only, gold ALLOWED): for each source query evaluate the two-channel pool
    under each R in {0,4,8,16}; per-R utility = new relevant golds admitted - old relevant golds evicted (vs the
    R=0 = fused-top50 base). Conservative preferred label = smallest R attaining the max utility, and R>0 is chosen
    ONLY if it strictly beats R=0 (max utility>0) -> R=0 is a legitimate, common label. No target gold used for
    labels/selection.
  * Policy features P0 = query-local, inference-safe, bounded/relative ONLY (no dataset ID, no hop, no gold at
    runtime): retrieval confidence, structural novelty, structural confidence. P1 = P0 + available runtime/graph-
    regime proxies (candidate count, structural-generation exhaustion, local neighborhood density). True per-query
    frontier/edge-budget counters are NOT persisted (stat json has aggregates only) -> P1 uses the available proxies
    and this limitation is stated. P1 is an ABLATION; P0 preferred.
  * First model = linear 4-way softmax (formulation A), class-balanced, source-TRAIN only norm/weights, frozen
    before target; smaller-R margin bias at inference. NOT an MLP.

True source-only 3-fold pilot: metaqa<-2wiki+squad, 2wiki<-metaqa+squad, squad<-metaqa+2wiki. Policy weights /
thresholds / normalization / class weighting = source TRAIN only, frozen before target. TEST untouched; 0 encoder
passes; no graph rebuild; no LLM. STOP after B1.5 (do NOT launch six-way).
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from _b14_twochannel import (load, per_query_idx, two_channel_pool, summarize, q3,
                             TOP_POOL, FUSED_COL, R_GRID, DSES)

R_VALUES = [0, 4, 8, 16]                 # output space (fixed; no continuous, no dataset-specific R)
R2C = {r: i for i, r in enumerate(R_VALUES)}; C2R = {i: r for r, i in R2C.items()}
K = len(R_VALUES)
SMALLER_R_MARGIN = 0.10                   # a-priori conservative tie: pick smallest R whose prob >= max - margin
SEED_K = 5                                # SEED_K from build; multi-seed support = seed_support_frac >= 0.4 (>=2 seeds)
B14_JSON = "results/GENERALIZATION/_g2_b14_twochannel.json"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)


# ------------------------------------------------------------------ per-query column accessors (magnitude-carrying)
# S0:  0 dense_pct 1 splade_pct 2 fused 3 retriever_support_frac 4 dense_splade_agree
# S1new: 1 seed_support_frac 3 degree_local_ratio
# S2new: 1 dir_margin 4 dir_vs_retrieval_adv
# S2old: 5 dir_support_max_pct
def _safe_mean(v):
    return float(v.mean()) if len(v) else 0.0


def p0_features(d, a, b):
    """Query-local, inference-safe, bounded features. Verification = BASE_FUSED (order by fused)."""
    n = b - a
    fused = d["fused"][a:b]
    dr = d["dense_rank"][a:b].astype(np.float64); sr = d["splade_rank"][a:b].astype(np.float64)
    dagg = d["S0"][a:b, 4]; rsupp = d["S0"][a:b, 3]
    added = d["added"][a:b]
    dirm = d["S2new"][a:b, 1]; dirvr = d["S2new"][a:b, 4]
    seeds = d["S1new"][a:b, 1]; dsupp = d["S2old"][a:b, 5]; degl = d["S1new"][a:b, 3]
    order = np.argsort(-fused, kind="stable")
    t10 = order[:10]; t20 = order[:20]; t50 = order[:TOP_POOL]
    in50 = np.zeros(n, bool); in50[t50] = True
    minr = np.minimum(dr, sr)

    # RETRIEVAL CONFIDENCE
    rc1 = _safe_mean(dagg[t20])                                        # dense/splade agreement (top pool)
    rc2 = _safe_mean(rsupp[t20])                                       # retriever-support concentration
    rc3 = _safe_mean(((dr[t10] <= 2) & (sr[t10] <= 2)).astype(float))  # top-rank peakedness (both retrievers top-3)
    rc4 = _safe_mean(minr[t10]) / TOP200                               # depth of confident core (low=confident)
    rc5 = _safe_mean(((dr[t50] < 50) & (sr[t50] < 50)).astype(float))  # cross-retriever agreement over pool
    rc6 = _safe_mean((minr[t50] >= TOP200).astype(float))             # retrieval-invisible fraction of verification pool

    # STRUCTURAL NOVELTY
    na = int(added.sum())
    add_novel = added & (~in50)
    sn1 = min(int(add_novel.sum()), 256) / 256.0                       # # unique structural OUTSIDE verification (bounded)
    sn2 = int(add_novel.sum()) / max(na, 1)                            # fraction of structural that is novel vs V
    sn3 = int((added & in50).sum()) / max(na, 1)                       # structural / verifier overlap fraction
    sn4 = na / max(n, 1)                                               # fraction of universe structural

    # STRUCTURAL CONFIDENCE (top-16 structural discoveries by s_dir)
    if na:
        aidx = np.where(added)[0]
        sd = d["sdir"][a:b][aidx]
        tk = aidx[np.argsort(-sd, kind="stable")[:16]]
        sc1 = float(dirm[aidx].max())                                 # top directional margin
        sc2 = _safe_mean(dirm[tk])                                    # top-k directional concentration
        sc3 = _safe_mean(seeds[tk])                                   # fraction supported by retrieval seeds
        sc4 = _safe_mean((seeds[tk] >= 0.4).astype(float))           # multi-seed support (>=2 of 5 seeds)
        sc5 = _safe_mean(dirvr[tk])                                   # directional advantage over retrieval
        sc6 = _safe_mean((minr[tk] >= TOP200).astype(float))         # top structural with weak dense/splade support
        sc7 = _safe_mean(dsupp[tk])                                   # anchor-support concentration
    else:
        sc1 = sc2 = sc3 = sc4 = sc5 = sc6 = sc7 = 0.0
    return np.array([rc1, rc2, rc3, rc4, rc5, rc6, sn1, sn2, sn3, sn4,
                     sc1, sc2, sc3, sc4, sc5, sc6, sc7], np.float64)


P0_NAMES = ["rc_agree", "rc_support", "rc_peaked", "rc_coredepth", "rc_pool_agree", "rc_pool_invis",
            "sn_unique_out", "sn_novel_frac", "sn_overlap_frac", "sn_univ_frac",
            "sc_dirmargin_max", "sc_dirmargin_topk", "sc_seed_supp", "sc_multiseed", "sc_dirvsret",
            "sc_weak_support", "sc_anchor"]


def p1_extra(d, a, b):
    """Runtime / graph-regime proxies actually available per-query (frontier/edge counters not persisted)."""
    n = b - a; added = d["added"][a:b]; na = int(added.sum())
    degl = d["S1new"][a:b, 3]
    p1a = min(n, 320) / 320.0                                         # candidate_count (near-saturated ~300)
    p1b = min(na, 256) / 256.0                                        # structural-generation exhaustion (M_MAX=256)
    p1c = _safe_mean(degl[np.where(added)[0]]) if na else 0.0         # query-local neighborhood density
    p1d = 1.0 if na >= 250 else 0.0                                   # exhaustion indicator
    return np.array([p1a, p1b, p1c, p1d], np.float64)


P1_NAMES = P0_NAMES + ["p1_candcount", "p1_struct_exhaust", "p1_neigh_density", "p1_exhaust_ind"]


def query_matrix(d, use_p1):
    """Feature matrix over ALL queries (policy predicts per query regardless of gold)."""
    rows = []
    for i, a, b in per_query_idx(d):
        f = p0_features(d, a, b)
        if use_p1:
            f = np.concatenate([f, p1_extra(d, a, b)])
        rows.append(f)
    return np.array(rows, np.float64)


# ------------------------------------------------------------------ oracle labels (gold; TRAIN supervision / eval-only diag)
def oracle_R_per_query(d, sc):
    """For each query: utility(R)=gold-count(pool_R)-gold-count(fused-top50). Conservative label = smallest R at max
    utility, R>0 only if it strictly beats R=0. Uses gold (TRAIN supervision on source; eval-only on target)."""
    nq = len(d["groups"]); lab = np.zeros(nq, np.int64); util = np.zeros((nq, K), np.float64)
    for i, a, b in per_query_idx(d):
        yy = d["gold"][a:b]; golds = set(np.where(yy)[0].tolist())
        base = set(np.argsort(-d["fused"][a:b], kind="stable")[:TOP_POOL].tolist())
        gb = len(golds & base)
        u = [0.0] * K
        for j, R in enumerate(R_VALUES):
            if R == 0:
                u[j] = 0.0; continue
            top, _ = two_channel_pool(sc[a:b], d["sdir"][a:b], d["added"][a:b], R)
            u[j] = len(golds & set(top.tolist())) - gb
        util[i] = u; mx = max(u)
        if mx <= 0:
            lab[i] = 0
        else:
            for j, R in enumerate(R_VALUES):
                if u[j] == mx:
                    lab[i] = R; break
    return lab, util


# ------------------------------------------------------------------ linear 4-way softmax policy (formulation A)
def fit_softmax(X, y, iters=600, lr=0.3, l2=2e-3):
    n, dd = X.shape; W = np.zeros((dd, K)); b = np.zeros(K)
    cnt = np.array([max((y == k).sum(), 1) for k in range(K)], np.float64)
    cw = (n / (K * cnt))                                             # balanced class weights
    sw = cw[y]; Y = np.eye(K)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        W -= lr * (X.T @ G / n + l2 * W); b -= lr * G.mean(0)
    return W, b


def predict_R(X, W, b):
    Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
    Rq = np.zeros(len(X), np.int64)
    for i in range(len(X)):
        p = P[i]; mx = p.max()
        for j in range(K):                                          # smallest R within margin of best prob
            if p[j] >= mx - SMALLER_R_MARGIN:
                Rq[i] = R_VALUES[j]; break
    return Rq, P


# ------------------------------------------------------------------ adaptive evaluation (per-query R)
def evaluate_adaptive(d, sc, Rq):
    """Same metric accounting as B1.4 evaluate() but R varies per query (Rq: array length nq)."""
    agg = dict(nq=0, ANY=0, RECn=0, RECd=0, ALL=0, go_adm=0, go_avail=0, rv_ret=0, rv_base=0,
               new=0, evict=0, churn=[], comp=dict(inserted=[], overlap=[], unused=[], sfrac=[]), per_hop={})
    for i, a, b in per_query_idx(d):
        yy = d["gold"][a:b]; ng = int(yy.sum())
        if ng == 0:
            continue
        R = int(Rq[i])
        base = set(np.argsort(-d["fused"][a:b], kind="stable")[:TOP_POOL].tolist())
        top, comp = two_channel_pool(sc[a:b], d["sdir"][a:b], d["added"][a:b], R)
        tset = set(top.tolist()); golds = set(np.where(yy)[0].tolist())
        go_gold = set(np.where(d["recg"][a:b] & d["graph_only"][a:b])[0].tolist()) & golds
        rv_gold = golds - go_gold
        gt = golds & tset; gb = golds & base; rv_in_base = rv_gold & base
        agg["nq"] += 1; agg["ANY"] += int(len(gt) >= 1); agg["ALL"] += int(len(gt) == ng)
        agg["RECn"] += len(gt); agg["RECd"] += ng
        agg["go_adm"] += len(go_gold & tset); agg["go_avail"] += len(go_gold)
        agg["rv_ret"] += len(rv_in_base & tset); agg["rv_base"] += len(rv_in_base)
        agg["new"] += len(gt - gb); agg["evict"] += len(gb - gt)
        agg["churn"].append(len(tset ^ base) / (2 * TOP_POOL))
        agg["comp"]["inserted"].append(comp["inserted"]); agg["comp"]["overlap"].append(comp["overlap"])
        agg["comp"]["unused"].append(comp["unused"]); agg["comp"]["sfrac"].append(comp["struct_frac"])
        h = int(d["qhop"][i]); ph = agg["per_hop"].setdefault(h, dict(nq=0, RECn=0, RECd=0, go_adm=0, go_avail=0, new=0, evict=0))
        ph["nq"] += 1; ph["RECn"] += len(gt); ph["RECd"] += ng
        ph["go_adm"] += len(go_gold & tset); ph["go_avail"] += len(go_gold)
        ph["new"] += len(gt - gb); ph["evict"] += len(gb - gt)
    return agg


def r_distribution(Rq):
    Rq = np.asarray(Rq); n = len(Rq)
    return {str(R): round(float((Rq == R).mean()), 4) for R in R_VALUES}


def policy_vs_oracle(Rq, orc):
    """chosen R vs (eval-only) target-oracle R: accuracy, MAE, over/under-reserve rates."""
    Rq = np.asarray(Rq, np.float64); orc = np.asarray(orc, np.float64); n = len(Rq)
    return {"R_accuracy": round(float((Rq == orc).mean()), 4),
            "mean_abs_reserve_error": round(float(np.abs(Rq - orc).mean()), 3),
            "OVER_RESERVE_RATE": round(float((Rq > orc).mean()), 4),
            "UNDER_RESERVE_RATE": round(float((Rq < orc).mean()), 4),
            "mean_chosen_R": round(float(Rq.mean()), 3), "mean_oracle_R": round(float(orc.mean()), 3)}


# ------------------------------------------------------------------ identity (dataset-fingerprint) diagnostic
def identity_diag(FEAT, mu, sd):
    """3-way dataset classifier on standardized query features (DIAGNOSTIC ONLY; not used by policy)."""
    Xs = []; ys = []
    for di, ds in enumerate(DSES):
        X = (FEAT[ds] - mu) / sd; Xs.append(X); ys.append(np.full(len(X), di))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    m = min((y == k).sum() for k in range(len(DSES)))               # balance
    idx = np.concatenate([np.random.permutation(np.where(y == k)[0])[:m] for k in range(len(DSES))])
    X = X[idx]; y = y[idx]; perm = np.random.permutation(len(y)); X = X[perm]; y = y[perm]
    ntr = len(y) // 2
    global K
    Ksave = K; K = len(DSES)
    W, b = fit_softmax(X[:ntr], y[:ntr], iters=500, lr=0.3)
    Z = X[ntr:] @ W + b; pred = Z.argmax(1)
    K = Ksave
    return round(float((pred == y[ntr:]).mean()), 4)


# ------------------------------------------------------------------ main
def main():
    log(f"=== B1.5 QUERY-LOCAL ADAPTIVE RESERVE POLICY  datasets={DSES}  R={R_VALUES}  margin={SMALLER_R_MARGIN} ===")
    DATA = {ds: load(ds) for ds in DSES}
    SC = {ds: DATA[ds]["fused"].copy() for ds in DSES}              # verification = BASE_FUSED -> sc = fused
    for ds in DSES:
        d = DATA[ds]
        log(f"{ds}: q={len(d['groups'])} added={int(d['added'].sum())} "
            f"graph_only_gold={int((d['recg']&d['graph_only']).sum())}")

    # oracle labels + utilities per dataset (gold used; TRAIN supervision on source, eval-only diagnostic on target)
    ORC = {}; UTIL = {}
    for ds in DSES:
        ORC[ds], UTIL[ds] = oracle_R_per_query(DATA[ds], SC[ds])
        log(f"  oracle-R dist {ds}: {r_distribution(ORC[ds])}  (label uses gold; TRAIN-only for source folds)")

    # feature matrices (P0 and P1) per dataset
    FP0 = {ds: query_matrix(DATA[ds], use_p1=False) for ds in DSES}
    FP1 = {ds: query_matrix(DATA[ds], use_p1=True) for ds in DSES}

    folds = {}
    for target in DSES:
        train_ds = [x for x in DSES if x != target]
        log(f"--- FOLD target={target}  SOURCE={train_ds} ---")
        res = {"train": train_ds}

        # ---- baselines: FIXED R (BASE_FUSED verification), evaluated on target
        base_rows = {}
        for R in R_VALUES:
            Rq = np.full(len(DATA[target]["groups"]), R, np.int64)
            base_rows[f"FIXED_R{R}"] = summarize(evaluate_adaptive(DATA[target], SC[target], Rq))
        # B1.4 source-selected fixed-R (loaded verbatim)
        b14_row = None
        if os.path.exists(B14_JSON):
            j = json.load(open(B14_JSON))
            tf = j["FOLDS"].get(target, {})
            b14_row = {"channel": tf.get("SELECTED", {}).get("channel"), "R": tf.get("SELECTED", {}).get("R"),
                       "TARGET_METRICS": tf.get("TARGET_METRICS", {})}

        # ---- adaptive policies P0 / P1 : source-trained, frozen, predicted on target
        pol = {}
        for name, FEAT in [("P0", FP0), ("P1", FP1)]:
            Xtr = np.concatenate([FEAT[dd] for dd in train_ds])
            ytr = np.concatenate([ORC[dd] for dd in train_ds]).astype(np.int64)
            ytr_c = np.array([R2C[int(r)] for r in ytr])
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            W, b = fit_softmax((Xtr - mu) / sd, ytr_c)               # source TRAIN only
            Rq_t, _ = predict_R((FEAT[target] - mu) / sd, W, b)     # freeze -> predict target
            m = summarize(evaluate_adaptive(DATA[target], SC[target], Rq_t))
            diag = policy_vs_oracle(Rq_t, ORC[target])
            pol[name] = {"metrics": m, "R_distribution": r_distribution(Rq_t),
                         "policy_vs_target_oracle": diag,
                         "source_label_dist": r_distribution(ytr)}
            log(f"  {name}: NET={m['NET_GOLD_GAIN@50']} GO_adm={m['GRAPH_ONLY_GOLD_ADMISSION@50']} "
                f"RVret={m['RETRIEVAL_VISIBLE_GOLD_RETENTION@50']} R@50={m['GOLD_RECALL@50']} "
                f"Rdist={pol[name]['R_distribution']} over_reserve={diag['OVER_RESERVE_RATE']} "
                f"Racc={diag['R_accuracy']}")

        # ---- target-oracle adaptive R (eval-only headroom): per-query oracle R
        Rq_orc = ORC[target]
        orc_m = summarize(evaluate_adaptive(DATA[target], SC[target], Rq_orc))

        res["FIXED_R_baselines"] = base_rows
        res["B14_source_selected_fixed_R"] = b14_row
        res["QUERY_LOCAL_P0_LINEAR"] = pol["P0"]
        res["QUERY_LOCAL_P1_LINEAR"] = pol["P1"]
        res["TARGET_ORACLE_ADAPTIVE_R_eval_only"] = {"metrics": orc_m, "R_distribution": r_distribution(Rq_orc)}
        folds[target] = res

    # ---- identity diagnostic (dataset fingerprinting) on P0 and P1 (diagnostic only)
    muP0 = np.concatenate([FP0[d] for d in DSES]).mean(0); sdP0 = np.concatenate([FP0[d] for d in DSES]).std(0) + 1e-6
    muP1 = np.concatenate([FP1[d] for d in DSES]).mean(0); sdP1 = np.concatenate([FP1[d] for d in DSES]).std(0) + 1e-6
    ident = {"P0_dataset_pred_acc": identity_diag(FP0, muP0, sdP0),
             "P1_dataset_pred_acc": identity_diag(FP1, muP1, sdP1),
             "chance": round(1.0 / len(DSES), 4),
             "note": "diagnostic only; not optimized against. Flag if P1 passes only via higher fingerprinting."}

    # ---- verdict (same success conditions as B1.4; primary = P0)
    def cond(target, mm):
        if target == "metaqa":
            return (mm["GRAPH_ONLY_GOLD_ADMISSION@50"] > 0) and (mm["NET_GOLD_GAIN@50"] > 0)
        if target == "2wiki_clean":
            return (mm["NET_GOLD_GAIN@50"] >= 0) and (mm["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= 0.95)
        return (mm["NET_GOLD_GAIN@50"] >= -5) and (mm["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= 0.99)

    def squad_over(target, pol_key):
        return folds["squad_clean"][pol_key]["policy_vs_target_oracle"]["OVER_RESERVE_RATE"]

    b14_squad_over = None  # B1.4 fixed R=8 on squad => over-reserve ~1.0 among (near-all) R=0-oracle queries
    verdicts = {}
    for pol_key in ["QUERY_LOCAL_P0_LINEAR", "QUERY_LOCAL_P1_LINEAR"]:
        c = {t: cond(t, folds[t][pol_key]["metrics"]) for t in DSES}
        verdicts[pol_key] = {"conditions": c, "PASS": bool(all(c.values())),
                             "squad_over_reserve_rate": squad_over("squad_clean", pol_key)}
    # B1.4 squad over-reserve (fixed R vs target oracle R): compute from its fixed R against squad oracle
    b14_sq = folds["squad_clean"]["B14_source_selected_fixed_R"]
    if b14_sq and b14_sq.get("R") is not None:
        Rq_b14 = np.full(len(DATA["squad_clean"]["groups"]), int(b14_sq["R"]), np.int64)
        b14_squad_over = policy_vs_oracle(Rq_b14, ORC["squad_clean"])["OVER_RESERVE_RATE"]

    p0_pass = verdicts["QUERY_LOCAL_P0_LINEAR"]["PASS"]
    p1_pass = verdicts["QUERY_LOCAL_P1_LINEAR"]["PASS"]
    p0_over = verdicts["QUERY_LOCAL_P0_LINEAR"]["squad_over_reserve_rate"]
    reduces_squad_over = (b14_squad_over is not None) and (p0_over < b14_squad_over - 1e-9)
    # target-oracle adaptive headroom (does per-query oracle R satisfy all conds?)
    orc_cond = {t: cond(t, folds[t]["TARGET_ORACLE_ADAPTIVE_R_eval_only"]["metrics"]) for t in DSES}
    oracle_headroom = all(orc_cond.values())

    # --- honesty guards on the literal PASS ---------------------------------------------------------------
    # (1) Is SQuAD NET even sensitive to R under BASE_FUSED? If FIXED_R{0..16} all give NET~0 & RVret>=0.99, then the
    #     SQuAD-NET success criterion is VACUOUS (satisfied by ANY R) -> the ONLY meaningful SQuAD criterion is the
    #     over-reservation rate. The NET-safety is a verification-channel property, not evidence of policy skill.
    sq_fixed = folds["squad_clean"]["FIXED_R_baselines"]
    squad_net_R_insensitive = all(sq_fixed[f"FIXED_R{R}"]["NET_GOLD_GAIN@50"] == 0 and
                                  sq_fixed[f"FIXED_R{R}"]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= 0.99
                                  for R in R_VALUES)
    # (2) Discovery-headroom capture on the graph-rich targets (P0 NET / target-oracle adaptive NET).
    def cap(t):
        on = folds[t]["TARGET_ORACLE_ADAPTIVE_R_eval_only"]["metrics"]["NET_GOLD_GAIN@50"]
        pn = folds[t]["QUERY_LOCAL_P0_LINEAR"]["metrics"]["NET_GOLD_GAIN@50"]
        return round(pn / on, 3) if on else None
    headroom_capture = {t: cap(t) for t in ["metaqa", "2wiki_clean"]}
    squad_over_high = p0_over > 0.5                                   # still over-reserves majority of SQuAD queries
    meta_hollow = (headroom_capture["metaqa"] is not None) and (headroom_capture["metaqa"] < 0.25)

    if not p0_pass:
        if p1_pass:
            flag = "P1_ONLY"
            interp = ("Only P1 passes the literal conditions -> inspect fingerprinting (identity diagnostic). Do NOT "
                      "adopt P1 if it wins mainly via dataset fingerprinting; then the P0 signal is insufficient.")
        elif squad_over_high:
            flag = "SIGNAL_INSUFFICIENT"
            interp = ("QUERY_LOCAL_RESERVE_SIGNAL_INSUFFICIENT=YES — source-trained query-local policy still "
                      "over-reserves SQuAD and fails a literal condition. Do NOT increase capacity yet.")
        else:
            flag = "LINEAR_FAIL"
            interp = "Linear policy fails a literal condition with SQuAD over-reservation controlled."
    else:
        # Literal PASS. But qualify it: SQuAD NET is vacuous here and over-reservation is still high; MetaQA hollow.
        if squad_over_high or meta_hollow:
            flag = "QUALIFIED_PASS_SIGNAL_PARTIAL"
            interp = (
                "QUALIFIED PASS. The P0 linear policy clears the LETTER of all four success conditions (MetaQA "
                f"GO_adm>0 & NET+{folds['metaqa']['QUERY_LOCAL_P0_LINEAR']['metrics']['NET_GOLD_GAIN@50']}>0; 2Wiki "
                f"NET+{folds['2wiki_clean']['QUERY_LOCAL_P0_LINEAR']['metrics']['NET_GOLD_GAIN@50']}>=0 & RVret high; "
                "SQuAD NET~0 & RVret~1; SQuAD over-reservation materially reduced 0.999->%.3f). BUT the query-local "
                "reserve signal is only PARTIAL: (a) SQuAD NET is R-INSENSITIVE under BASE_FUSED (every fixed R gives "
                "NET 0 / RVret~1) so the SQuAD-NET criterion is VACUOUS — the meaningful SQuAD metric is "
                "over-reservation, still %.3f (majority of queries reserved wastefully; the policy even picks R=16 on "
                "~%.0f%%). (b) MetaQA capture is HOLLOW: NET %s of target-oracle %s (~%s of headroom); the policy is "
                "right that ~85%% of MetaQA is R=0 but under-reserves the deep multi-hop minority where graph-only "
                "golds live. (c) It over-reserves the graph-rich targets too (2Wiki over %.3f), winning there only "
                "because over-reservation is tolerated. Identity: features fingerprint dataset at %.3f (chance %.3f) — "
                "the separability EXISTS but the R-label mapping does not (correctly) exploit dataset ID, so the "
                "inference-safe signal does not cleanly separate useful structural novelty from noise. VERDICT is "
                "deferred to the user: read as a LITERAL PASS (prefer P0, proceed to six-way) OR as "
                "QUERY_LOCAL_RESERVE_SIGNAL=PARTIAL/INSUFFICIENT (inspect signals before six-way). Six-way NOT "
                "launched. Do NOT increase model capacity on the basis of the vacuous SQuAD-NET pass."
            ) % (p0_over, p0_over, folds["squad_clean"]["QUERY_LOCAL_P0_LINEAR"]["R_distribution"]["16"] * 100,
                 folds["metaqa"]["QUERY_LOCAL_P0_LINEAR"]["metrics"]["NET_GOLD_GAIN@50"],
                 folds["metaqa"]["TARGET_ORACLE_ADAPTIVE_R_eval_only"]["metrics"]["NET_GOLD_GAIN@50"],
                 headroom_capture["metaqa"],
                 folds["2wiki_clean"]["QUERY_LOCAL_P0_LINEAR"]["policy_vs_target_oracle"]["OVER_RESERVE_RATE"],
                 ident["P0_dataset_pred_acc"], ident["chance"])
        else:
            flag = "P0_PASS"
            interp = ("ADAPTIVE_TWO_CHANNEL_ADMISSION=PROMISING via P0 linear (prefer P0; STOP model complexity). "
                      "Next step (separate directive) = six-way LODO. NOT launched here.")

    verdict = {"primary_policy": "QUERY_LOCAL_P0_LINEAR", "P0_literal_PASS": bool(p0_pass), "P1_literal_PASS": bool(p1_pass),
               "P0_conditions": verdicts["QUERY_LOCAL_P0_LINEAR"]["conditions"],
               "P1_conditions": verdicts["QUERY_LOCAL_P1_LINEAR"]["conditions"],
               "squad_net_R_insensitive_under_BASE_FUSED": bool(squad_net_R_insensitive),
               "squad_NET_criterion_is_vacuous": bool(squad_net_R_insensitive),
               "squad_over_reserve": {"P0": p0_over, "P1": verdicts["QUERY_LOCAL_P1_LINEAR"]["squad_over_reserve_rate"],
                                      "B14_fixed_R": b14_squad_over,
                                      "P0_materially_reduces_vs_B14": bool(reduces_squad_over),
                                      "still_majority_over_reserved": bool(squad_over_high)},
               "discovery_headroom_capture_P0_over_oracle": headroom_capture,
               "metaqa_capture_hollow": bool(meta_hollow),
               "target_oracle_adaptive_headroom_all_conditions": bool(oracle_headroom),
               "target_oracle_conditions": orc_cond, "P0_ge_P1_prefer_P0": bool(p0_pass or not p1_pass),
               "FLAG": flag, "INTERPRETATION": interp,
               "reference_success": ("MetaQA: GO_adm>0 & NET>0 | 2Wiki: NET>=0 & RVret>=0.95 | SQuAD: NET>=-5 & "
                                     "RVret>=0.99 ; AND SQuAD over-reservation materially reduced vs B1.4, from ONE "
                                     "source-trained query-local policy (no dataset ID / no target R / no gold at runtime).")}
    log(f"VERDICT P0_PASS={p0_pass} P1_PASS={p1_pass} squad_over P0={p0_over} B14={b14_squad_over} "
        f"reduces={reduces_squad_over} oracle_headroom={oracle_headroom} FLAG={flag}")

    out = {"phase": "G2 B1.5 query-local adaptive reserve policy; 3-target source-only pilot; TEST untouched; STOP before six-way",
           "datasets": DSES, "R_values": R_VALUES, "verification_channel_fixed": "BASE_FUSED",
           "structural_reserve_orderer": "s_dir (frozen)", "policy_model": "linear 4-way softmax (formulation A), class-balanced",
           "smaller_R_margin": SMALLER_R_MARGIN, "P0_features": P0_NAMES, "P1_features": P1_NAMES,
           "P1_limitation": "true per-query frontier/edge-budget counters not persisted (stat json aggregates only); "
                            "P1 uses available runtime proxies (candidate count, structural-generation exhaustion, "
                            "local neighborhood density).",
           "leakage_checks": {"target_excluded_from_training_and_selection": True,
                              "policy_norm_and_weights_source_train_only": True,
                              "gold_used_only_for_TRAIN_labels_and_eval_only_oracle": True,
                              "no_gold_in_pool_construction": True, "no_dataset_id_feature": True,
                              "no_hop_or_benchmark_metadata_feature": True, "verification_channel_frozen": True,
                              "structural_orderer_frozen_s_dir": True, "no_new_scorer_or_encoder": True},
           "IDENTITY_DIAGNOSTIC": ident, "FOLDS": folds, "VERDICT": verdict,
           "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO", "SIX_WAY_LODO_LAUNCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b15_adaptive.json", "w"), indent=1, default=str)
    log("B15_DONE -> results/GENERALIZATION/_g2_b15_adaptive.json")


TOP200 = 200
if __name__ == "__main__":
    main()
