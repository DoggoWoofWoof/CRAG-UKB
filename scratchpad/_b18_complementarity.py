"""B1.8 — CROSS-CHANNEL COMPLEMENTARITY ANALYSIS (diagnostic-first).

FROZEN: candidate universe, verification channel, s_dir orderer, two_channel_pool, TOP_POOL=50,
R in {0,4,8,16}. ZERO new encoder passes (reuses scratchpad/_b12/*.npz only). No dataset id,
no target statistics, no target calibration, no hop label in any model input, no TEST.

Order of work (cheapest-decisive-first, per directive AUTONOMY clause):
  STEP 0  channel inventory: available / missing / degenerate
  STEP 1  distribution construction + EMPIRICAL degeneracy demo (why JS/KL/entropy are ill-posed here)
  STEP 2  scale-free (order-based) complementarity features -- well-defined on percentile cache
  STEP 3  exact TRAIN channel Shapley (needs only orderings + gold => valid on cache)
  STEP 4  cross-domain SIGN stability of feature -> phi_structural and feature -> true D
  STEP 5  calibration transfer / conditional-shift reduction vs the B1.7 P0 baseline
"""
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _b14_twochannel import load, per_query_idx, two_channel_pool, TOP_POOL, FUSED_COL
from _b17_marginal import utilities

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
TOP200 = 200
K_RRF = 60
OUT = "results/GENERALIZATION/_g2_b18_complementarity.json"

# ---------------------------------------------------------------- STEP 0: channel inventory

CHANNEL_MAP = {
    "SEMANTIC":   {"dense_pct": ("S0", 0), "splade_pct": ("S0", 1), "fused_rrf_pct": ("S0", 2),
                   "retriever_support_frac": ("S0", 3), "dense_splade_agree": ("S0", 4)},
    "RELATIONAL": {},   # offset / mixture / relation -- see inventory() note
    "STRUCTURAL": {"s_dir_pct": ("S2old", 0), "dir_exp_rank_pct": ("S2old", 1), "min_exp_hop_norm": ("S2old", 2),
                   "n_struct_anchors_pct": ("S2old", 3), "struct_dist_pct": ("S2old", 4),
                   "dir_support_max_pct": ("S2old", 5), "geometry_added_ind": ("S2old", 6),
                   "seed_support_frac": ("S1new", 1), "ppr_share_pct": ("S1new", 2),
                   "degree_local_ratio": ("S1new", 3), "dir_margin": ("S2new", 1)},
}


def modal_tie_share(d, block, col, nq_cap=400):
    """Per-query share of candidates sitting on the single most common value (1.0 = fully degenerate)."""
    A = d[block][:, col]
    sh = []
    for i, a, b in per_query_idx(d):
        if i >= nq_cap:
            break
        c = A[a:b]
        if len(c) < 5:
            continue
        _, cnt = np.unique(c, return_counts=True)
        sh.append(cnt.max() / len(c))
    return float(np.mean(sh)), float(np.percentile(sh, 95))


def inventory(DATA):
    inv = {}
    for fam, sigs in CHANNEL_MAP.items():
        inv[fam] = {}
        if not sigs:
            inv[fam] = {"_status": "MISSING_OVER_EXPANDED_UNIVERSE",
                        "_detail": ("offset/mixture/relation expert scores are NOT cached over the B1.x expanded "
                                    "universe (_b12 builder emits only S0 semantic + S1/S2 structural). Deterministic "
                                    "abstention applied; no relational evidence fabricated.")}
            continue
        for nm, (blk, col) in sigs.items():
            m, p95 = modal_tie_share(DATA["metaqa"], blk, col)
            allds = {}
            for ds, d in DATA.items():
                A = d[blk][:, col]
                allds[ds] = {"mean": round(float(A.mean()), 4), "frac_zero": round(float((A == 0).mean()), 4)}
            inv[fam][nm] = {"status": "DEGENERATE" if m > 0.5 else "AVAILABLE",
                            "modal_tie_share_mean": round(m, 4), "modal_tie_share_p95": round(p95, 4),
                            "per_ds": allds}
    return inv


# ---------------------------------------------------------------- STEP 1: distributions + degeneracy demo

def rank_softmax_entropy(n, tau):
    """H of a rank-based softmax over n items -- depends ONLY on n (and tau), never on the query's evidence."""
    r = np.arange(n, dtype=np.float64)
    w = np.exp(-r / tau); p = w / w.sum()
    return float(-(p * np.log(p + 1e-300)).sum())


def degeneracy_demo(DATA, tau=10.0):
    """Empirically show the two reasons JS/KL/entropy are ill-posed on this cache."""
    out = {}
    # (a) percentile columns have query-independent shape => entropy carries no query information
    ent_pct, ent_rank, sizes = [], [], []
    for ds, d in DATA.items():
        A = d["S0"][:, 0]  # dense_pct (a within-query percentile by construction)
        e1, e2 = [], []
        for i, a, b in per_query_idx(d):
            if i >= 300:
                break
            c = A[a:b].astype(np.float64); n = len(c)   # float64: 1e-300 underflows to 0 in float32
            if n < 5:
                continue
            p = c / (c.sum() + 1e-12)
            e1.append(-(p * np.log(p + 1e-300)).sum())
            e2.append(rank_softmax_entropy(n, tau))
            sizes.append(n)
        ent_pct.append((ds, float(np.mean(e1)), float(np.std(e1))))
        ent_rank.append((ds, float(np.mean(e2)), float(np.std(e2))))
    out["entropy_of_percentile_channel"] = {ds: {"mean": round(m, 4), "std": round(s, 5)} for ds, m, s in ent_pct}
    out["entropy_of_rank_softmax"] = {ds: {"mean": round(m, 4), "std": round(s, 5)} for ds, m, s in ent_rank}
    out["universe_size"] = {"mean": round(float(np.mean(sizes)), 1), "std": round(float(np.std(sizes)), 2)}
    out["VERDICT"] = ("Both candidate constructions give query-independent entropy: a percentile column has a fixed "
                      "shape by construction, and a rank-softmax's entropy is a function of universe size only "
                      "(sizes are near-constant ~300). Raw channel SCORES are not cached, so H / JS / KL over "
                      "p_sem,p_struct cannot carry query-level evidence. Scale-free ORDER-based complementarity is "
                      "used instead; JS is reported below on the rank-softmax purely as the null ablation.")
    return out


# ---------------------------------------------------------------- STEP 2: scale-free complementarity features

def rbo(A, B, p=0.9, depth=50):
    """Rank-Biased Overlap (truncated): weighted top-heavy agreement between two orderings."""
    sa, sb, s, acc = set(), set(), 0.0, 0.0
    for k in range(depth):
        if k < len(A):
            sa.add(int(A[k]))
        if k < len(B):
            sb.add(int(B[k]))
        acc += (len(sa & sb) / (k + 1)) * (p ** k)
        s += p ** k
    return float(acc / s) if s > 0 else 0.0


CF_NAMES = ["ovl10", "ovl50", "rbo", "str_out_sem50", "sem_core10", "sem_core50",
            "sem_blind_frac", "str_conc", "added_frac", "js_rank_null"]


def comp_features(d):
    """Query-local, inference-safe, scale-free complementarity summaries. No gold, no dataset id."""
    fused = d["S0"][:, FUSED_COL].astype(np.float64)
    densep = d["S0"][:, 0].astype(np.float64)
    sdir = d["sdir"]; added = d["added"]
    dm = d["S2new"][:, 1].astype(np.float64)          # dir_margin: z-scored s_dir (spacing-preserving)
    dr = d["dense_rank"]; sr = d["splade_rank"]
    rows = []
    for i, a, b in per_query_idx(d):
        n = b - a
        f = fused[a:b]; dp = densep[a:b]; sd = sdir[a:b]; am = added[a:b]
        # semantic ordering: fused primary, exact-dense percentile as deterministic tiebreak
        sem_ord = np.lexsort((-dp, -f))
        str_ord = np.argsort(-np.where(am, sd, -np.inf), kind="stable")
        n_add = int(am.sum())
        str_ord = str_ord[:max(n_add, 1)]
        s10, s50 = set(sem_ord[:10].tolist()), set(sem_ord[:50].tolist())
        t10 = set(str_ord[:10].tolist()); t16 = set(str_ord[:16].tolist())
        ovl10 = len(s10 & t10) / 10.0
        ovl50 = len(s50 & set(str_ord[:50].tolist())) / 50.0
        r = rbo(sem_ord, str_ord, 0.9, 50)
        str_out = len(t16 - s50) / max(len(t16), 1)                  # directional complementarity
        core10 = float((dr[a:b] < 10).sum()) / n
        core50 = float((dr[a:b] < 50).sum()) / n
        blind = float(((dr[a:b] >= TOP200) & (sr[a:b] >= TOP200)).sum()) / n
        if n_add >= 4:
            dma = dm[a:b][am]
            k = max(int(0.1 * len(dma)), 1)
            top = np.sort(dma)[::-1][:k]
            str_conc = float(top.mean() - dma.mean())                # concentration of the structural signal
        else:
            str_conc = 0.0
        # null ablation: JS between rank-softmaxes (carries no query evidence -- see degeneracy_demo)
        rr = np.arange(n, dtype=np.float64)
        ps = np.exp(-rr / 10.0); ps /= ps.sum()
        inv = np.empty(n, np.int64); inv[str_ord] = np.arange(len(str_ord))
        if len(str_ord) < n:
            miss = np.setdiff1d(np.arange(n), str_ord, assume_unique=False)
            inv[miss] = np.arange(len(str_ord), n)
        pg = np.exp(-inv / 10.0); pg /= pg.sum()
        mm = 0.5 * (ps + pg)
        js = 0.5 * (ps * np.log((ps + 1e-300) / (mm + 1e-300))).sum() + \
             0.5 * (pg * np.log((pg + 1e-300) / (mm + 1e-300))).sum()
        rows.append([ovl10, ovl50, r, str_out, core10, core50, blind, str_conc,
                     n_add / n, float(js)])
    return np.array(rows, np.float64)


# ---------------------------------------------------------------- STEP 3: exact TRAIN channel Shapley

def shapley(d):
    """Exact channel-family Shapley. Players present over this universe: S(emantic), G(structural).
    RELATIONAL abstains deterministically (absent from universe) => phi_R == 0 by construction, NOT evidence
    of irrelevance in general (see Q2 for its behaviour on the recovery population).

    Coalition utility, BUDGET FIXED AT 50 FOR EVERY COALITION (no silent budget change):
        U(0)   = 50 * n_gold_in_universe / n_universe      (expected golds under uniform admission)
        U(S)   = #golds in top-50 of the semantic ordering
        U(G)   = #golds in top-50 of the structural ordering
        U(SG)  = #golds in top-50 of RRF(semantic, structural), K=60
    Uses TRAIN relevance labels ONLY. Never computed at inference.
    """
    fused = d["S0"][:, FUSED_COL].astype(np.float64)
    densep = d["S0"][:, 0].astype(np.float64)
    sdir = d["sdir"]; added = d["added"]; gold = d["gold"]
    phiS, phiG, us = [], [], []
    for i, a, b in per_query_idx(d):
        n = b - a; yy = gold[a:b]; ng = int(yy.sum())
        if ng == 0:
            phiS.append(np.nan); phiG.append(np.nan); us.append([np.nan] * 4); continue
        f = fused[a:b]; dp = densep[a:b]; sd = sdir[a:b]; am = added[a:b]
        sem_ord = np.lexsort((-dp, -f))
        str_ord = np.argsort(-np.where(am, sd, -np.inf), kind="stable")
        rk_s = np.empty(n, np.float64); rk_s[sem_ord] = np.arange(n)
        rk_g = np.empty(n, np.float64); rk_g[str_ord] = np.arange(n)
        rk_g[~am] = n                                          # abstain: structural proposes only discoveries
        rrf = 1.0 / (K_RRF + rk_s) + 1.0 / (K_RRF + rk_g)
        u0 = TOP_POOL * ng / n
        uS = float(yy[sem_ord[:TOP_POOL]].sum())
        uG = float(yy[str_ord[:TOP_POOL]].sum())
        uSG = float(yy[np.argsort(-rrf, kind="stable")[:TOP_POOL]].sum())
        phiS.append(0.5 * ((uS - u0) + (uSG - uG)))
        phiG.append(0.5 * ((uG - u0) + (uSG - uS)))
        us.append([u0, uS, uG, uSG])
    return np.array(phiS), np.array(phiG), np.array(us, np.float64)


# ---------------------------------------------------------------- STEP 4/5 helpers

def safe_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def bin_calibration(zs, ds_, zt, dt, nb=5):
    """Fit E[D|bin] on SOURCE, freeze bin edges, apply to TARGET (analysis only)."""
    ed = np.quantile(zs, np.linspace(0, 1, nb + 1))
    ed[0] -= 1e-9; ed[-1] += 1e-9
    bs = np.clip(np.digitize(zs, ed[1:-1]), 0, nb - 1)
    bt = np.clip(np.digitize(zt, ed[1:-1]), 0, nb - 1)
    src = np.array([ds_[bs == k].mean() if (bs == k).sum() >= 10 else np.nan for k in range(nb)])
    tgt = np.array([dt[bt == k].mean() if (bt == k).sum() >= 10 else np.nan for k in range(nb)])
    pred = np.where(np.isfinite(src[bt]), src[bt], np.nanmean(src))
    cal_err = float(np.abs(np.nanmean(pred) - np.nanmean(dt)))
    return {"src_bin_meanD": [None if not np.isfinite(v) else round(float(v), 4) for v in src],
            "tgt_bin_meanD": [None if not np.isfinite(v) else round(float(v), 4) for v in tgt],
            "tgt_frac_by_bin": [round(float((bt == k).mean()), 3) for k in range(nb)],
            "pred_meanD_target": round(float(np.nanmean(pred)), 4),
            "true_meanD_target": round(float(np.nanmean(dt)), 4),
            "calibration_abs_err": round(cal_err, 4)}


def main():
    print("[b18] loading ...", flush=True)
    DATA = {ds: load(ds) for ds in DSES}
    res = {"CONTRACT": {
        "frozen": ["candidate universe", "verification channel BASE_FUSED", "s_dir structural orderer",
                   "two_channel_pool", "TOP_POOL=50", "R in {0,4,8,16}"],
        "encoder_passes": 0, "uses_dataset_id": False, "uses_target_stats": False,
        "uses_hop_label_as_input": False, "uses_TEST": False,
        "shapley_gold_usage": "TRAIN relevance labels only; never computed at inference"}}

    print("[b18] STEP 0 inventory ...", flush=True)
    res["STEP0_INVENTORY"] = inventory(DATA)

    print("[b18] STEP 1 degeneracy demo ...", flush=True)
    res["STEP1_DISTRIBUTION_CONSTRUCTION"] = degeneracy_demo(DATA)

    print("[b18] STEP 2/3 features + shapley ...", flush=True)
    FE, PHI_G, PHI_S, D, U = {}, {}, {}, {}, {}
    for ds in DSES:
        d = DATA[ds]
        FE[ds] = comp_features(d)
        ps, pg, us = shapley(d)
        PHI_S[ds], PHI_G[ds], U[ds] = ps, pg, us
        UR, DD = utilities(d)
        D[ds] = DD
        print(f"   {ds}: feats {FE[ds].shape} phiG mean {np.nanmean(pg):.4f} D4 mean {DD[:,0].mean():.4f}",
              flush=True)

    res["STEP3_SHAPLEY"] = {}
    for ds in DSES:
        pg, ps, us = PHI_G[ds], PHI_S[ds], U[ds]
        fin = np.isfinite(pg)
        res["STEP3_SHAPLEY"][ds] = {
            "n_queries_with_gold": int(fin.sum()),
            "phi_semantic_mean": round(float(np.nanmean(ps)), 4),
            "phi_structural_mean": round(float(np.nanmean(pg)), 4),
            "phi_relational_mean": 0.0,
            "phi_relational_note": "player ABSENT from universe (deterministic abstention) -- not identifiable here",
            "frac_phi_struct_positive": round(float((pg[fin] > 1e-9).mean()), 4),
            "frac_phi_struct_zero": round(float((np.abs(pg[fin]) <= 1e-9).mean()), 4),
            "coalition_U_means": {k: round(float(np.nanmean(us[:, j])), 4)
                                  for j, k in enumerate(["U_empty", "U_S", "U_G", "U_SG"])},
            "structural_share_of_total": round(float(np.nanmean(pg) / max(np.nanmean(pg) + np.nanmean(ps), 1e-9)), 4)}

    print("[b18] STEP 4 cross-domain sign stability ...", flush=True)
    step4 = {"corr_feature_vs_phi_structural": {}, "corr_feature_vs_trueD4": {}, "SIGN_STABILITY": {}}
    for j, nm in enumerate(CF_NAMES):
        cg, cd = {}, {}
        for ds in DSES:
            cg[ds] = safe_corr(FE[ds][:, j], PHI_G[ds])
            cd[ds] = safe_corr(FE[ds][:, j], D[ds][:, 0])
        step4["corr_feature_vs_phi_structural"][nm] = {k: (None if v is None else round(v, 4)) for k, v in cg.items()}
        step4["corr_feature_vs_trueD4"][nm] = {k: (None if v is None else round(v, 4)) for k, v in cd.items()}
        sg = [v for v in cg.values() if v is not None and abs(v) > 0.05]
        step4["SIGN_STABILITY"][nm] = {
            "phi_signs": [None if v is None else int(np.sign(v)) for v in cg.values()],
            "stable_sign_phi": bool(len(sg) >= 2 and len(set(int(np.sign(v)) for v in sg)) == 1),
            "n_datasets_with_signal": len(sg)}
    res["STEP4_CROSS_DOMAIN_STABILITY"] = step4

    print("[b18] STEP 5 calibration transfer ...", flush=True)
    # B1.7 P0 baseline conditional-shift (predicted vs true marginal utility, summed |err| over 3 blocks)
    B17_PRED = {"metaqa": [0.020, 0.003, 0.002], "2wiki_clean": [0.099, 0.054, 0.094],
                "squad_clean": [0.246, 0.148, 0.324]}
    step5 = {"B17_P0_baseline_conditional_shift": {}, "complementarity_calibration": {}}
    tot17 = 0.0
    for ds in DSES:
        tr = [float(D[ds][:, k].mean()) for k in range(3)]
        e = sum(abs(B17_PRED[ds][k] - tr[k]) for k in range(3))
        tot17 += e
        step5["B17_P0_baseline_conditional_shift"][ds] = {
            "pred": B17_PRED[ds], "true": [round(v, 4) for v in tr], "sum_abs_err": round(e, 4)}
    step5["B17_P0_baseline_conditional_shift"]["TOTAL_sum_abs_err"] = round(tot17, 4)

    best_feats = ["sem_core10", "sem_blind_frac", "str_out_sem50", "str_conc", "ovl10"]
    for nm in best_feats:
        j = CF_NAMES.index(nm)
        step5["complementarity_calibration"][nm] = {}
        for tgt in DSES:
            src = [s for s in DSES if s != tgt]
            zs = np.concatenate([FE[s][:, j] for s in src])
            dsv = np.concatenate([D[s][:, 0] for s in src])
            step5["complementarity_calibration"][nm][tgt] = bin_calibration(
                zs, dsv, FE[tgt][:, j], D[tgt][:, 0])
    res["STEP5_CALIBRATION_TRANSFER"] = step5

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("[b18] wrote", OUT, flush=True)

    # ---- console summary
    print("\n=== STEP 3 Shapley ===")
    for ds in DSES:
        s = res["STEP3_SHAPLEY"][ds]
        print(f"  {ds:13s} phiS {s['phi_semantic_mean']:+.4f} phiG {s['phi_structural_mean']:+.4f} "
              f"fracG>0 {s['frac_phi_struct_positive']:.3f}  U: {s['coalition_U_means']}")
    print("\n=== STEP 4 sign stability (corr vs phi_structural) ===")
    for nm in CF_NAMES:
        r = step4["corr_feature_vs_phi_structural"][nm]; st = step4["SIGN_STABILITY"][nm]
        print(f"  {nm:15s} " + " ".join(f"{k[:6]}:{('  None' if v is None else f'{v:+.3f}')}" for k, v in r.items())
              + f"   stable={st['stable_sign_phi']}")
    print("\n=== STEP 5 calibration (D4) ===")
    for nm in best_feats:
        for tgt in DSES:
            c = step5["complementarity_calibration"][nm][tgt]
            print(f"  {nm:15s} -> {tgt:13s} pred {c['pred_meanD_target']:.4f} true {c['true_meanD_target']:.4f} "
                  f"err {c['calibration_abs_err']:.4f}")
    print(f"\n  B1.7 P0 baseline TOTAL conditional-shift err = {tot17:.4f}")


if __name__ == "__main__":
    main()
