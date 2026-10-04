"""G2 B1.4 — SOURCE-SELECTED TWO-CHANNEL ADMISSION (final 3-target pilot before six-way).

Reuses scratchpad/_b12/{metaqa,2wiki_clean,squad_clean}.npz ONLY. No new model/encoder/graph/LLM/directional-MLP.

Two-channel admission interface (deterministic, NO gold in construction):
  CHANNEL A (verification): one of {BASE_FUSED, S0, S1_PRIME} — the already-valid source-trained/LODO-safe
                            base-anchored-residual definitions from B1.2 (S1_PRIME == B1.2 'S1new' ladder).
  CHANNEL B (structural discovery reserve): geometry-added candidates ordered ONLY by s_dir (frozen orderer;
                            S2old[:,0]). No learned directional score. No dataset-specific orderer.
  POOL: take top(50-R) from verification V; add up to R highest-s_dir UNIQUE structural discoveries (not already
        in V); if fewer than R unique structural exist, fill remaining slots from the next V candidates. Exactly 50.

Reserve grid R in {0,4,8,16} (R=32 excluded per directive). Held-out target D: choose (channel, R) using the
OTHER TWO datasets only (SOURCE); all normalizers/weights (logreg w/mu/sd, beta) source-trained/frozen; evaluate D
once. Report source-selected pair per fold, target metrics, TARGET-ORACLE gap (diagnostic, NOT valid selection),
channel composition, PASS/FAIL. TEST untouched; 0 encoder passes. STOP after B1.4 (no six-way).
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
TOP_POOL = 50; TOP200 = 200; FUSED_COL = 2
CHANNELS = ["BASE_FUSED", "S0", "S1_PRIME"]
CHAN_LADDER = {"S0": ["S0"], "S1_PRIME": ["S0", "S1new"]}   # verification feature blocks (B1.2 definitions)
CHAN_SIMPLICITY = {"BASE_FUSED": 0, "S0": 1, "S1_PRIME": 2}  # tie-break: prefer simpler
R_GRID = [0, 4, 8, 16]
BETAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35]                    # B1.2 grid (incl 0.0); beta = source-trained normalizer
RET_MIN = 0.95                                              # strong retention constraint (source-side, fixed a priori)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)


# ------------------------------------------------------------------ data
def load(ds):
    z = np.load(f"scratchpad/_b12/{ds}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["blocks"] = {b: d[b] for b in ["S0", "S1old", "S2old", "S1new", "S2new"]}
    d["fused"] = d["S0"][:, FUSED_COL].astype(np.float64)
    d["sdir"] = d["S2old"][:, 0].astype(np.float64)          # FROZEN structural reserve orderer = s_dir
    d["added"] = d["S2old"][:, 6] == 1                        # geometry-added (structural discovery) indicator
    d["graph_only"] = (d["dense_rank"] >= TOP200) & (d["splade_rank"] >= TOP200)
    d["gold"] = d["y"] == 1
    d["recg"] = (d["recov"] == 1) & (d["y"] == 1)
    return d


def feat(d, tag):
    return np.concatenate([d["blocks"][b] for b in CHAN_LADDER[tag]], 1)


def per_query_idx(d):
    g = d["groups"]; st = np.concatenate([[0], np.cumsum(g)])
    for i in range(len(g)):
        yield i, int(st[i]), int(st[i + 1])


# ------------------------------------------------------------------ verification model (B1.2 logreg + hard sample)
def fit_logreg(X, y, iters=400, lr=0.5, l2=1e-3):
    n, dd = X.shape; w = np.zeros(dd); b = 0.0
    pos = max(y.sum(), 1); wpos = (n - pos) / pos
    sw = np.where(y == 1, wpos, 1.0)
    for _ in range(iters):
        z = X @ w + b; p = 1 / (1 + np.exp(-z)); grad = (p - y) * sw
        w -= lr * (X.T @ grad / n + l2 * w); b -= lr * grad.mean()
    return w, b


def hard_sample(d, tag, per_q=60):
    F = feat(d, tag); width = F.shape[1]; y = d["y"]; Xs = []; Ys = []
    for i, a, b in per_query_idx(d):
        f = F[a:b]; yy = y[a:b]; n = len(yy)
        fused = f[:, FUSED_COL]; order = np.argsort(-fused, kind="stable")
        sel = set(np.where(yy == 1)[0].tolist())
        sel |= set(order[:20].tolist())
        sel |= set(order[max(TOP_POOL - 10, 0):TOP_POOL + 10].tolist())
        if width >= 10:
            sel |= set(np.argsort(-f[:, 5:10].sum(1))[:10].tolist())
        pool = list(sel); rest = [j for j in range(n) if j not in sel]
        if rest:
            pool += list(np.random.choice(rest, size=min(len(rest), max(per_q - len(pool), 0)), replace=False))
        pool = pool[:per_q] if len(pool) > per_q else pool
        idx = np.array(pool, np.int64); Xs.append(f[idx]); Ys.append(yy[idx])
    return np.concatenate(Xs), np.concatenate(Ys)


def build_train(DATA, train_ds, tag):
    pX = []; pY = []
    for dd in train_ds:
        Xs, Ys = hard_sample(DATA[dd], tag); pX.append(Xs); pY.append(Ys)
    m = min(len(a) for a in pY)
    X = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in pX])
    Y = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in pY])
    return X, Y


def _zq(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def channel_scores(d, channel, models):
    """Per-candidate verification score sc (higher=better), computed per query (base-anchored residual)."""
    if channel == "BASE_FUSED":
        return d["fused"].copy()
    tag = channel; score_fn, beta = models[channel]
    F = feat(d, tag); sc = np.empty(len(F), np.float64)
    for i, a, b in per_query_idx(d):
        f = F[a:b]; raw = score_fn(f)
        sc[a:b] = f[:, FUSED_COL] + beta * _zq(raw)
    return sc


def single_channel_net(d, sc):
    """B1.2-style NET_GOLD_GAIN@50 of a single-channel top50 admission vs fused-top50 base (for beta selection)."""
    new = 0; evict = 0
    for i, a, b in per_query_idx(d):
        yy = d["y"][a:b]; ng = int(yy.sum())
        if ng == 0:
            continue
        top = set(np.argsort(-sc[a:b], kind="stable")[:TOP_POOL].tolist())
        base = set(np.argsort(-d["fused"][a:b], kind="stable")[:TOP_POOL].tolist())
        golds = set(np.where(yy == 1)[0].tolist())
        gt = golds & top; gb = golds & base
        new += len(gt - gb); evict += len(gb - gt)
    return new - evict


def select_beta_and_fit(DATA, train_ds, channel):
    """Fit logreg on SOURCE (hard-sampled, train-only mu/sd) then select beta on SOURCE single-channel NET."""
    tag = channel
    Xtr, Ytr = build_train(DATA, train_ds, tag)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    w, b = fit_logreg((Xtr - mu) / sd, Ytr)
    score_fn = (lambda W, B, MU, SD: (lambda F: ((F - MU) / SD) @ W + B))(w, b, mu, sd)
    best = (0.0, -1e18)
    for beta in BETAS:
        nets = []
        for dd in train_ds:
            sc = channel_scores(DATA[dd], channel, {channel: (score_fn, beta)})
            nets.append(single_channel_net(DATA[dd], sc))
        mn = float(np.mean(nets))
        if mn > best[1]:
            best = (beta, mn)
    return score_fn, best[0]


# ------------------------------------------------------------------ two-channel pool + evaluation
def two_channel_pool(sc_q, sdir_q, am_q, R):
    """Deterministic. top(50-R) from V=argsort(sc) UNION up-to-R highest-s_dir UNIQUE structural; fill from V.
    Returns (top_indices, composition dict). NO gold used."""
    n = len(sc_q)
    forder = np.argsort(-sc_q, kind="stable")
    if R <= 0:
        top = forder[:TOP_POOL]
        return top, {"R": R, "inserted": 0, "overlap": 0, "unused": 0, "struct_frac": 0.0}
    keep = TOP_POOL - R
    verify = forder[:keep]
    vset = set(int(x) for x in verify)
    top = [int(x) for x in verify]
    dorder = np.argsort(-sdir_q, kind="stable")             # structural discoveries by s_dir desc
    inserted = 0; overlap = 0
    for c in dorder:
        if inserted >= R:
            break
        c = int(c)
        if not am_q[c]:
            continue
        if c in vset:
            overlap += 1                                    # structural discovery already held by verification
            continue
        top.append(c); vset.add(c); inserted += 1
    unused = R - inserted
    if len(top) < TOP_POOL:                                 # fewer than R unique structural: fill from next V
        for c in forder:
            if len(top) >= TOP_POOL:
                break
            c = int(c)
            if c not in vset:
                top.append(c); vset.add(c)
    top = np.array(top[:TOP_POOL], np.int64)
    return top, {"R": R, "inserted": inserted, "overlap": overlap, "unused": unused,
                 "struct_frac": inserted / TOP_POOL}


def evaluate(d, sc, R):
    agg = dict(nq=0, ANY=0, RECn=0, RECd=0, ALL=0, go_adm=0, go_avail=0, rv_ret=0, rv_base=0,
               new=0, evict=0, churn=[], comp=dict(inserted=[], overlap=[], unused=[], sfrac=[]),
               per_hop={})
    for i, a, b in per_query_idx(d):
        yy = d["gold"][a:b]; ng = int(yy.sum())
        if ng == 0:
            continue
        base = set(np.argsort(-d["fused"][a:b], kind="stable")[:TOP_POOL].tolist())
        top, comp = two_channel_pool(sc[a:b], d["sdir"][a:b], d["added"][a:b], R)
        tset = set(top.tolist())
        golds = set(np.where(yy)[0].tolist())
        go_gold = set(np.where(d["recg"][a:b] & d["graph_only"][a:b])[0].tolist()) & golds
        rv_gold = golds - go_gold
        gt = golds & tset; gb = golds & base
        rv_in_base = rv_gold & base
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


def q3(x):
    x = np.array(x, np.float64) if len(x) else np.array([0.0])
    return {"mean": round(float(x.mean()), 3), "median": round(float(np.median(x)), 3),
            "p95": round(float(np.percentile(x, 95)), 3)}


def summarize(agg):
    nq = max(agg["nq"], 1)
    o = {"nq": agg["nq"], "GOLD_RECALL@50": round(agg["RECn"] / max(agg["RECd"], 1), 4),
         "ALL@50": round(agg["ALL"] / nq, 4), "ANY@50": round(agg["ANY"] / nq, 4),
         "GRAPH_ONLY_GOLD_ADMISSION@50": round(agg["go_adm"] / max(agg["go_avail"], 1), 4),
         "graph_only_gold_available": agg["go_avail"],
         "RETRIEVAL_VISIBLE_GOLD_RETENTION@50": round(agg["rv_ret"] / max(agg["rv_base"], 1), 4),
         "retrieval_visible_gold_base": agg["rv_base"], "retrieval_visible_gold_retained": agg["rv_ret"],
         "NEW_GRAPH_ONLY_GOLDS_ADMITTED": agg["go_adm"],
         "NEW_GOLDS_ADMITTED": agg["new"], "OLD_GOLDS_EVICTED": agg["evict"],
         "NET_GOLD_GAIN@50": agg["new"] - agg["evict"],
         "POOL_CHURN_mean": round(float(np.mean(agg["churn"])) if agg["churn"] else 0.0, 4),
         "CHANNEL_COMPOSITION": {"nominal_R_unique_inserted": q3(agg["comp"]["inserted"]),
                                 "structural_overlap_with_verification": q3(agg["comp"]["overlap"]),
                                 "unused_structural_slots_to_verification": q3(agg["comp"]["unused"]),
                                 "fraction_final_pool_structural": q3(agg["comp"]["sfrac"])}}
    ph = {}
    for h, dd in sorted(agg["per_hop"].items()):
        ph[str(h)] = {"nq": dd["nq"], "GOLD_RECALL@50": round(dd["RECn"] / max(dd["RECd"], 1), 4),
                      "GRAPH_ONLY_GOLD_ADMISSION@50": round(dd["go_adm"] / max(dd["go_avail"], 1), 4),
                      "graph_only_gold_available": dd["go_avail"],
                      "NET_GOLD_GAIN@50": dd["new"] - dd["evict"]}
    o["PER_HOP"] = ph
    return o


# ------------------------------------------------------------------ main LODO
def main():
    log(f"=== B1.4 SOURCE-SELECTED TWO-CHANNEL ADMISSION  datasets={DSES} R_grid={R_GRID} RET_MIN={RET_MIN} ===")
    DATA = {ds: load(ds) for ds in DSES}
    for ds in DSES:
        d = DATA[ds]
        log(f"{ds}: q={len(d['groups'])} cands={len(d['y'])} added={int(d['added'].sum())} "
            f"graph_only_gold={int((d['recg']&d['graph_only']).sum())}")

    SELECTION_RULE = (
        "For held-out D, SOURCE = other two datasets. Fit verification channels (logreg w + train-only mu/sd; "
        "beta selected on SOURCE single-channel NET) FROZEN. Grid = {BASE_FUSED,S0,S1_PRIME} x R{0,4,8,16}. "
        f"FEASIBLE = configs with RETRIEVAL_VISIBLE_GOLD_RETENTION@50 >= {RET_MIN} on BOTH source datasets "
        "(strong retention constraint: may not buy graph-only golds by destroying retrieval-visible golds). "
        "Among FEASIBLE choose max mean-over-source NET_GOLD_GAIN@50; tie-break (eps 1e-9) -> smaller R -> simpler "
        "channel (BASE_FUSED<S0<S1_PRIME). If FEASIBLE empty -> fallback (BASE_FUSED, R=0). D never used to select "
        "channel/R/beta/threshold/normalization.")

    folds = {}
    for target in DSES:
        train_ds = [x for x in DSES if x != target]
        log(f"--- FOLD target={target}  SOURCE={train_ds} ---")
        # frozen source-trained verification models
        models = {}
        for ch in ["S0", "S1_PRIME"]:
            score_fn, beta = select_beta_and_fit(DATA, train_ds, ch)
            models[ch] = (score_fn, beta)
            log(f"  fit {ch}: beta_source={beta}")
        # precompute channel scores on source + target (frozen)
        sc_cache = {ds: {ch: channel_scores(DATA[ds], ch, models) for ch in CHANNELS} for ds in DSES}

        grid = [(ch, R) for ch in CHANNELS for R in R_GRID]
        # source metrics per config per source dataset
        src_metrics = {}   # (ch,R) -> {ds: summary}
        for ch, R in grid:
            src_metrics[(ch, R)] = {dd: summarize(evaluate(DATA[dd], sc_cache[dd][ch], R)) for dd in train_ds}
        # target metrics per config (for oracle + reporting)
        tgt_metrics = {(ch, R): summarize(evaluate(DATA[target], sc_cache[target][ch], R)) for ch, R in grid}

        def feasible_src(cfg):
            return all(src_metrics[cfg][dd]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= RET_MIN for dd in train_ds)

        def mean_src_net(cfg):
            return float(np.mean([src_metrics[cfg][dd]["NET_GOLD_GAIN@50"] for dd in train_ds]))

        feas = [cfg for cfg in grid if feasible_src(cfg)]
        fallback = False
        if feas:
            best = max(mean_src_net(c) for c in feas)
            cands = [c for c in feas if mean_src_net(c) >= best - 1e-9]
            cands.sort(key=lambda c: (c[1], CHAN_SIMPLICITY[c[0]]))    # smaller R, then simpler channel
            sel = cands[0]
        else:
            sel = ("BASE_FUSED", 0); fallback = True
        sel_ch, sel_R = sel

        # TARGET-ORACLE (diagnostic only; selected on TARGET) — NOT valid inference.
        # (a) CONSTRAINED: same objective as source selection (max target NET s.t. target retention>=RET_MIN, tie-break).
        # (b) MAXNET: absolute best target NET in grid (ceiling; ignores retention) -> clean >=0 "NET left on table" gap.
        def pick(cfgs):
            best = max(tgt_metrics[c]["NET_GOLD_GAIN@50"] for c in cfgs)
            cc = [c for c in cfgs if tgt_metrics[c]["NET_GOLD_GAIN@50"] >= best - 1e-9]
            cc.sort(key=lambda c: (c[1], CHAN_SIMPLICITY[c[0]])); return cc[0]
        ofeas = [c for c in grid if tgt_metrics[c]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= RET_MIN]
        orc_con = pick(ofeas) if ofeas else ("BASE_FUSED", 0)
        orc_max = pick(grid)
        sel_target_retention = tgt_metrics[sel]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"]
        gap_maxnet = tgt_metrics[orc_max]["NET_GOLD_GAIN@50"] - tgt_metrics[sel]["NET_GOLD_GAIN@50"]
        folds[target] = {
            "train": train_ds, "beta_source": {ch: models[ch][1] for ch in ["S0", "S1_PRIME"]},
            "SELECTED": {"channel": sel_ch, "R": sel_R, "fallback_used": fallback,
                         "mean_source_NET": round(mean_src_net(sel), 3) if not fallback else 0.0,
                         "source_retention": {dd: src_metrics[sel][dd]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] for dd in train_ds},
                         "satisfies_target_retention_floor": bool(sel_target_retention >= RET_MIN)},
            "TARGET_METRICS": tgt_metrics[sel],
            "TARGET_ORACLE_DIAGNOSTIC_NOT_VALID_SELECTION": {
                "MAXNET_ceiling": {"channel": orc_max[0], "R": orc_max[1], "target_NET": tgt_metrics[orc_max]["NET_GOLD_GAIN@50"],
                                   "target_retention": tgt_metrics[orc_max]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"],
                                   "ORACLE_GAP_NET_left_on_table": gap_maxnet},
                "CONSTRAINED_same_objective": {"channel": orc_con[0], "R": orc_con[1],
                                               "target_NET": tgt_metrics[orc_con]["NET_GOLD_GAIN@50"],
                                               "target_retention": tgt_metrics[orc_con]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"]}},
            "SOURCE_CONFIG_TABLE": {f"{ch}|R{R}": {dd: {"NET": src_metrics[(ch, R)][dd]["NET_GOLD_GAIN@50"],
                                                        "RVret": src_metrics[(ch, R)][dd]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"]}
                                                   for dd in train_ds} for ch, R in grid},
            "TARGET_GRID_NET": {f"{ch}|R{R}": {"NET": tgt_metrics[(ch, R)]["NET_GOLD_GAIN@50"],
                                               "RVret": tgt_metrics[(ch, R)]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"],
                                               "GO_adm": tgt_metrics[(ch, R)]["GRAPH_ONLY_GOLD_ADMISSION@50"]}
                                for ch, R in grid}}
        tm = tgt_metrics[sel]
        log(f"  SELECTED {sel_ch} R={sel_R} (fallback={fallback}) -> target NET={tm['NET_GOLD_GAIN@50']} "
            f"GO_adm={tm['GRAPH_ONLY_GOLD_ADMISSION@50']} RVret={tm['RETRIEVAL_VISIBLE_GOLD_RETENTION@50']} "
            f"R@50={tm['GOLD_RECALL@50']} churn={tm['POOL_CHURN_mean']}")
        log(f"  ORACLE maxNET {orc_max[0]} R={orc_max[1]} NET={tgt_metrics[orc_max]['NET_GOLD_GAIN@50']} "
            f"(gap_left={gap_maxnet}) | constrained {orc_con[0]} R={orc_con[1]} NET={tgt_metrics[orc_con]['NET_GOLD_GAIN@50']} "
            f"| sel_target_retention={sel_target_retention} feas={sel_target_retention>=RET_MIN}")

    # ---- PASS/FAIL verdict per success condition (source-selected), + oracle-based failure interpretation
    def cond(target, mm):
        if target == "metaqa":
            return (mm["GRAPH_ONLY_GOLD_ADMISSION@50"] > 0) and (mm["NET_GOLD_GAIN@50"] > 0)
        if target == "2wiki_clean":
            return (mm["NET_GOLD_GAIN@50"] >= 0) and (mm["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= RET_MIN)
        return (mm["NET_GOLD_GAIN@50"] >= -5) and (mm["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"] >= 0.99)  # squad

    sel_cond = {t: cond(t, folds[t]["TARGET_METRICS"]) for t in DSES}
    passed = all(sel_cond.values())
    # constrained target-oracle pass (uses GO_adm/NET/RVret stored in TARGET_GRID_NET)
    orc_cond = {}
    for t in DSES:
        oc = folds[t]["TARGET_ORACLE_DIAGNOSTIC_NOT_VALID_SELECTION"]["CONSTRAINED_same_objective"]
        key = f"{oc['channel']}|R{oc['R']}"; g = folds[t]["TARGET_GRID_NET"][key]
        mm = {"GRAPH_ONLY_GOLD_ADMISSION@50": g["GO_adm"], "NET_GOLD_GAIN@50": g["NET"],
              "RETRIEVAL_VISIBLE_GOLD_RETENTION@50": g["RVret"]}
        orc_cond[t] = cond(t, mm)
    oracle_passed = all(orc_cond.values())
    if passed:
        interp = "TWO_CHANNEL_UNIVERSAL_ADMISSION=PROMISING (proceed next to six-way LODO)"
    elif oracle_passed:
        interp = ("RESERVE_POLICY_GENERALIZATION=NO — target-oracle pairs satisfy all conditions but the SOURCE-SELECTED "
                  "policy does not; the open problem is POLICY SELECTION (choosing channel/R without target), not "
                  "candidate scoring. Per directive: do NOT enlarge the MLP; do NOT proceed to six-way.")
    else:
        interp = "TWO_CHANNEL_ADMISSION=NO — even target-oracle pairs fail a condition; revisit the structural bridge."
    verdict = {
        "success_conditions_source_selected": sel_cond, "PASS": bool(passed),
        "target_oracle_would_pass_all": bool(oracle_passed), "oracle_conditions": orc_cond,
        "INTERPRETATION": interp,
        "reference_success_condition": ("MetaQA: GO_adm>0 & NET>0 | 2Wiki: NET>=0 & RVret>=0.95 | "
                                        "SQuAD: NET>=-5 & RVret>=0.99 — all WITHOUT target-specific R / dataset-ID / "
                                        "target calibration / learned directional verifier.")}
    log(f"VERDICT PASS={passed} src_cond={sel_cond} oracle_pass={oracle_passed} orc_cond={orc_cond}")

    out = {"phase": "G2 B1.4 source-selected two-channel admission; 3-target pilot; TEST untouched; STOP before six-way",
           "datasets": DSES, "channels": CHANNELS, "R_grid": R_GRID, "structural_reserve_orderer": "s_dir (frozen)",
           "RET_MIN": RET_MIN, "SELECTION_RULE": SELECTION_RULE,
           "leakage_checks": {"target_excluded_from_selection": True, "beta_and_norm_source_trained": True,
                              "no_gold_in_pool_construction": True, "no_dataset_id": True,
                              "no_learned_directional_verifier": True, "structural_orderer_is_frozen_s_dir": True},
           "FOLDS": folds, "VERDICT": verdict,
           "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO", "SIX_WAY_LODO_LAUNCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b14_twochannel.json", "w"), indent=1, default=str)
    log("B14_DONE -> results/GENERALIZATION/_g2_b14_twochannel.json")


if __name__ == "__main__":
    main()
