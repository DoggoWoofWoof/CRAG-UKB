"""Q2 — does the FROZEN C11a reranker convert newly co-scoped multi-hop chains into better downstream ranking?

Evaluates the EXACT frozen source backbone (C7b soft-archetype fusion + C8c XGBRanker + C11a interaction-MLP,
ZERO_SHOT source weights + source E-standardization) on MetaQA VAL under FIVE scopes:
  M=0    : the original frozen P50 scope (must reproduce _g1_eval_metaqa.json ZERO_SHOT_C11A per-hop = GATE)
  M=32/64/128/256 : P50  UNION  parameter-free directional geometry expansion (Track-A, structural-only).

C11a is NOT modified. The 18-feature contract is re-expressed EXACTLY (SH.contribs -> C7b weights -> C8c ->
C11a top-20), only the candidate SET is augmented. For each added geometry candidate we compute the expert
raw scores deterministically:
  dense    = cos(q, node)                                   [exact]
  offset   = node . universal_offset_head(q, dense_top1)    [exact, frozen universal head]
  mixture  = max_k node . universal_mixture_head_k(...)     [exact, frozen universal head]
  splade   = per-query MIN-1 (conservative: added multi-hop cands rank last in the lexical expert) [APPROX]
  relation = ABSTAIN (mask=False -> RRF contribution exactly 0; the correct default with no
             title-mention connecting-sentence edge for the added candidate)                       [default]
Only SPLADE is a genuine approximation; it is CONSERVATIVE (under-credits added candidates on the lexical
axis). Both universal relational heads — the experts most able to point at a 2nd-hop target — are exact.
If the conservative smoke shows fusion cannot surface geometry candidates, the targeted follow-up is to add
real SPLADE (and the KB relation edge) for added candidates. M=0 is unaffected by any approximation.

Reports per hop: cand count, ANY, ALL coverage, R@5, R@20, NDCG@5, FullCov@5/@20, and the A/B decomposition
  A. L1_MISSING     = expected(in-corpus) gold NOT in the augmented scope
  B. L2_RANK_FAIL   = gold present in augmented scope but final rank >= K (C11a fails to promote it)

Env: Q2_BACKBONE_DIR (frozen bundle), Q2_M (comma list, default "0,32,64,128,256"),
     Q2_SUBSET (per-hop query cap for the smoke; 0 = full val), Q2_OUT (json path).
CPU-only. No new encoder passes. TEST never touched.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c8c as C8C, l2_c9 as C9, l2_c6 as C6
import l2_shapley as SH
import l2_lib as L
from l2_lib import OffsetHead, MixtureHead
import l2_c11 as M11
import _run_c11 as R11
from _run_c9 import starts_of
from _run_c11_prep import EIDX
import g2_l1_geo as G

DS = "metaqa"; SPLIT = "val"; CAP = 20; CORP = "data/l2_corpus"
HEADS = "results/L2/_heads"
H_BUDGET = 3; B_CAP = 2000
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)


# ----------------------------------------------------------------- frozen backbone
def load_frozen(bdir):
    base_full = joblib.load(f"{bdir}/base_full.joblib")
    c8c = joblib.load(f"{bdir}/c8c.joblib")
    sd11 = joblib.load(f"{bdir}/C11_models.joblib")
    ma = M11.C11a(); ma.load_state_dict(sd11["C11a"])
    return base_full, c8c, ma, sd11["emean"], sd11["estd"]


# ----------------------------------------------------------------- raw corpus (full P50 scope)
def load_raw():
    R = C6._load_raw(DS, SPLIT)
    R["cand_ids"] = np.load(f"{CORP}/{DS}/{SPLIT}/cand_ids.npy")
    R["query_meta"] = json.load(open(f"{CORP}/{DS}/{SPLIT}/query_meta.json"))
    return R


# ----------------------------------------------------------------- geometry (Track-A, structural-only)
def residual(qvec, seed_rows, Xn):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not seed_rows: return q
    E = Xn[seed_rows]
    U, s, _ = np.linalg.svd(E.T, full_matrices=False); U = U[:, s > 1e-6]
    if U.shape[1] == 0: return q
    r = q - U @ (U.T @ q); n = np.linalg.norm(r)
    return r / n if n > 1e-6 else q


def expand_dir(seed_rows, r_q, adj, M, Xn, B=B_CAP):
    scope = set(seed_rows); frontier = list(seed_rows); scored = []
    for hop in range(H_BUDGET):
        cand = []; seen = set()
        for e in frontier:
            xe = Xn[e]
            for (v, rel, dirn) in adj.get(e, []):
                if v in scope or (e, v) in seen: continue
                seen.add((e, v))
                delta = Xn[v] - xe; nd = np.linalg.norm(delta)
                if nd < 1e-9: continue
                cand.append((float(r_q @ (delta / nd)), v))
        if not cand: break
        cand.sort(key=lambda x: -x[0]); keep = cand[:M]
        nf = []
        for sdir, v in keep:
            if v not in scope: scope.add(v); nf.append(v); scored.append((sdir, v))
        frontier = nf
        if len(scope) >= B * 2: break
    out = []
    seen = set(seed_rows)
    for sdir, v in sorted(scored, key=lambda x: -x[0]):
        if len(out) >= B: break
        if v not in seen: out.append(v); seen.add(v)
    return out  # node rows, s_dir order, excludes seeds


# ----------------------------------------------------------------- augmented bundle (mirrors C9.build_bundle EXACTLY)
def build_aug(qi_list, base_full, R, W, ra, di, added_by_q, added_sc, diag_out=None, force_pool_golds=False):
    off = R["query_offsets"]
    X28 = []; Xex = []; Y = []; Gg = []; meta = []; cand_by_q = []
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1])
        dense = R["dense_score"][s:e].astype(np.float64)
        splade = R["splade_scope_score"][s:e].astype(np.float64)
        offs = R["offset_score"][s:e].astype(np.float64)
        mix = R["mixture_score"][s:e].astype(np.float64)
        rel = R["relation_qwen_score"][s:e].astype(np.float64)
        relm = (R["relation_mask"][s:e] > 0)
        lab = (R["labels"][s:e] == 1)
        candq = R["cand_ids"][s:e].astype(np.int64)
        add = added_by_q.get(qi, [])
        if len(add):
            a = added_sc[qi]
            dense = np.concatenate([dense, a["dense"]]); splade = np.concatenate([splade, a["splade"]])
            offs = np.concatenate([offs, a["offset"]]); mix = np.concatenate([mix, a["mixture"]])
            rel = np.concatenate([rel, np.zeros(len(add))]); relm = np.concatenate([relm, np.zeros(len(add), bool)])
            lab = np.concatenate([lab, a["label"].astype(bool)]); candq = np.concatenate([candq, np.asarray(add, np.int64)])
        n = len(dense)
        C = SH.contribs(dense, splade, offs, mix, rel, relm)     # (5,n) — recomputes ranks over aug scope
        w = W[r]
        base = (w[:, None] * C).sum(0); brank = C8.ranks_from_score(base)
        rk = np.stack([C8.ranks_from_score(C[i]) for i in range(5)]).astype(np.float64)
        gold = set(np.where(lab)[0].tolist())
        pool = np.argsort(-base, kind="stable")[:min(C8C.POOL, n)]
        forced_local = []
        if force_pool_golds and len(add):
            # ORACLE_TOP50_ADMISSION: force recovered golds (added & gold) into the pool, replacing the
            # lowest-base NON-gold. Gold info used ONLY to construct the diagnostic window (NOT inference-safe).
            orig_n = e - s; added_gold_local = [orig_n + jj for jj in range(len(add)) if a["label"][jj]]
            poolset = set(int(x) for x in pool)
            for gl in added_gold_local:
                if gl in poolset: continue
                # lowest-base non-gold currently in pool
                cand_out = [int(x) for x in pool if int(x) not in gold]
                if not cand_out: break
                worst = min(cand_out, key=lambda li: base[li])
                pool = np.array([gl if int(x) == worst else int(x) for x in pool], np.int64)
                poolset.discard(worst); poolset.add(gl); forced_local.append(gl)
            pool = pool[np.argsort(-base[pool], kind="stable")]   # keep pool sorted by base
        min4 = rk[:4].min(0); mean4 = rk[:4].mean(0); maxc = C.max(0); votes10 = (rk < 10).sum(0)
        for pp, li in enumerate(pool):
            X28.append([brank[li], base[li], pp, rk[0, li], rk[1, li], rk[2, li], rk[3, li], rk[4, li],
                        C[0, li], C[1, li], C[2, li], C[3, li], C[4, li], (1.0 if relm[li] else 0.0), rel[li],
                        min4[li], mean4[li], maxc[li], rk[0, li] - rk[1, li], rk[2, li] - rk[3, li], votes10[li],
                        w[0], w[1], w[2], w[3], w[4], float(ra[r]), float(di[r])])
            Y.append(1 if int(li) in gold else 0)
        Xex.append(C9._extra_cols(C[:, pool], rk[:, pool], w))
        Gg.append(len(pool))
        added_gold_local = [ (e - s) + jj for jj in range(len(add)) if a["label"][jj] ] if len(add) else []
        meta.append({"qi": int(qi), "pool": pool.astype(np.int64), "brank": brank,
                     "gold": np.array(sorted(gold), np.int64), "n": n,
                     "n_added": int(len(add)), "n_gold_added": int(sum(a["label"])) if len(add) else 0,
                     "added_gold_local": np.array(added_gold_local, np.int64),
                     "forced_local": np.array(forced_local, np.int64)})
        cand_by_q.append(candq)
        # --- recovered-gold diagnostic: for golds that were ADDED (not in P50), where do the REAL experts
        #     (dense/offset/mixture) rank them within the augmented scope, and do they enter the top-50 pool? ---
        if diag_out is not None and len(add):
            orig_n = e - s; poolset = set(int(x) for x in pool)
            for jj in range(len(add)):
                if not a["label"][jj]:
                    continue
                li = orig_n + jj                                    # local index of this recovered gold
                diag_out.append({"qi": int(qi), "base_rank": int(brank[li]),
                                 "dense_rank": int(rk[0, li]), "offset_rank": int(rk[2, li]),
                                 "mixture_rank": int(rk[3, li]), "in_pool50": bool(li in poolset)})
    B = {"X28": np.asarray(X28, np.float32), "Xex": np.concatenate(Xex).astype(np.float32),
         "y": np.asarray(Y, np.int8), "groups": np.asarray(Gg, np.int64), "meta": meta}
    return B, cand_by_q


def build_sp_aug(B, cand_by_q, s_c8c, nodes, qall, query_meta):
    X = C9.assemble(B, C9.resid_feats(s_c8c, B["groups"])); st = starts_of(B["groups"])
    QV = []; DV = []; E = []; C8b = []; Yl = []; Gw = []
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; sc = s_c8c[a:a + g]; pool = m["pool"]
        win = np.argsort(-sc, kind="stable")[:min(CAP, g)]; pool_win = pool[win]
        row_all = int(query_meta[m["qi"]]["row_all"])
        QV.append(np.asarray(qall[row_all], np.float32))
        gids = cand_by_q[i][pool_win].astype(np.int64); DV.append(np.asarray(nodes[gids], np.float32))
        E.append(X[a + win][:, EIDX]); C8b.append(sc[win])
        gs = set(m["gold"].tolist()); Yl.append(np.array([1 if int(pl) in gs else 0 for pl in pool_win], np.int8)); Gw.append(len(win))
    Gw = np.asarray(Gw, np.int64)
    return {"qv": np.asarray(QV, np.float32), "dv": np.concatenate(DV).astype(np.float32),
            "E": np.concatenate(E).astype(np.float32), "c8c": np.concatenate(C8b).astype(np.float32),
            "y": np.concatenate(Yl).astype(np.int64), "g": Gw, "st": starts_of(Gw)}


# ----------------------------------------------------------------- per-hop metrics + A/B decomposition
def metrics(B, res, query_meta, Ks=(5, 20)):
    """From eval_ranking result (res._goldranks over FULL aug scope, res._perq). Per-hop table."""
    gr = res["_goldranks"]                                  # {(qi, gold_local): final_rank}
    meta_by_qi = {m["qi"]: m for m in B["meta"]}
    rows = {}
    for m in B["meta"]:
        qi = m["qi"]; qm = query_meta[qi]; h = qm.get("hop")
        if h is None: continue
        exp = int(qm.get("N_GOLD_EXPECTED_INCORP", 0))
        if exp <= 0: continue
        d = rows.setdefault(h, {"nq": 0, "cand": [], "n_added": [], "exp_golds": 0,
                                "in_scope_golds": 0, "any": 0, "all": 0,
                                "rk": {k: 0 for k in Ks}, "ndcg5_num": 0.0,
                                "fullcov": {k: 0 for k in Ks}, "A_l1_missing": 0,
                                "B_rankfail": {k: 0 for k in Ks}})
        d["nq"] += 1; d["cand"].append(m["n"]); d["n_added"].append(m["n_added"])
        d["exp_golds"] += exp
        ng_scope = len(m["gold"]); d["in_scope_golds"] += ng_scope
        d["any"] += int(ng_scope >= 1)
        d["all"] += int(ng_scope == exp)                     # all expected golds co-scoped
        d["A_l1_missing"] += (exp - ng_scope)
        ranks = [gr[(qi, int(gl))] for gl in m["gold"]]      # final ranks of in-scope golds
        for k in Ks:
            in_k = sum(1 for rr in ranks if rr < k)
            d["rk"][k] += in_k                               # numerator of micro recall@k (over expected)
            d["fullcov"][k] += int(exp == ng_scope and all(rr < k for rr in ranks))
            d["B_rankfail"][k] += sum(1 for rr in ranks if rr >= k)   # present but not in top-k
        # NDCG@5 over expected golds (missing golds contribute 0 to DCG; IDCG over exp)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(exp, 5)))
        dcg = sum(1.0 / np.log2(rr + 2) for rr in ranks if rr < 5)
        d["ndcg5_num"] += (dcg / idcg) if idcg > 0 else 0.0
    out = {}
    for h, d in sorted(rows.items()):
        nq = d["nq"]; exp = max(d["exp_golds"], 1)
        out[str(h)] = {
            "n_val_queries": nq, "cand_count_mean": round(float(np.mean(d["cand"])), 1),
            "added_mean": round(float(np.mean(d["n_added"])), 1),
            "ANY_coverage": round(d["any"] / nq, 4), "ALL_coverage": round(d["all"] / nq, 4),
            "R@5": round(d["rk"][5] / exp, 4), "R@20": round(d["rk"][20] / exp, 4),
            "NDCG@5": round(d["ndcg5_num"] / nq, 4),
            "FullCov@5": round(d["fullcov"][5] / nq, 4), "FullCov@20": round(d["fullcov"][20] / nq, 4),
            "expected_golds": d["exp_golds"], "in_scope_golds": d["in_scope_golds"],
            "A_L1_MISSING_golds": d["A_l1_missing"],
            "B_L2_RANKFAIL@5_golds": d["B_rankfail"][5], "B_L2_RANKFAIL@20_golds": d["B_rankfail"][20],
        }
    # overall
    return out


# ----------------------------------------------------------------- one scope evaluation
def compute_added(M, qi_list, R, adj, Xn, qall, qtext, d2i, golds_g, ids_g, robust):
    """Parameter-free directional geometry expansion (Track-A) for scope M; returns added_by_q / added_sc with
    deterministic expert scores for the added candidates (dense/offset/mixture exact, splade conservative,
    relation abstain). M=0 -> empty (reproduces frozen P50)."""
    added_by_q = {}; added_sc = {}
    if M <= 0:
        for qi in qi_list: added_by_q[qi] = []
        return added_by_q, added_sc
    for qi in qi_list:
        i = int(R["query_meta"][qi]["row_all"])
        seeds = G.resolve_seeds(qtext[ids_g[i]], d2i)
        r_q = residual(np.asarray(qall[i], np.float64), seeds, Xn)
        exp = expand_dir(seeds, r_q, adj, M, Xn)
        s, e = int(R["query_offsets"][qi]), int(R["query_offsets"][qi + 1])
        present = set(int(x) for x in R["cand_ids"][s:e])
        add = [v for v in exp if v not in present]
        if not add:
            added_by_q[qi] = []; continue
        add = np.asarray(add, np.int64)
        gold_rows = set(d2i[g] for g in golds_g[i] if g in d2i)
        qn = np.asarray(qall[i], np.float32); qn = qn / (np.linalg.norm(qn) + 1e-9)
        seed_idx = int(robust["dtop"][i, 0])
        with torch.no_grad():
            qt = torch.from_numpy(qn).unsqueeze(0)
            seedt = robust["Dn"][seed_idx:seed_idx + 1]
            po = robust["oh"](qt, seedt)[0]; pm = robust["mh"](qt, seedt)[0]
            Cadd = robust["Dn"][torch.from_numpy(add)]
            dsc = (Cadd @ torch.from_numpy(qn)).numpy().astype(np.float64)
            osc = (Cadd @ po).numpy().astype(np.float64)
            msc = (Cadd @ pm.T).max(dim=1).values.numpy().astype(np.float64)
        sp_min = float(R["splade_scope_score"][s:e].min()) - 1.0
        added_by_q[qi] = add
        added_sc[qi] = {"dense": dsc, "offset": osc, "mixture": msc,
                        "splade": np.full(len(add), sp_min, np.float64),
                        "label": np.array([1 if int(v) in gold_rows else 0 for v in add], np.int8)}
    return added_by_q, added_sc


def eval_scope(M, qi_list, base_full, c8c, ma, em11, es11, R, W, ra, di,
               adj, Xn, nodes, qall, qtext, d2i, golds_g, ids_g, robust):
    added_by_q, added_sc = compute_added(M, qi_list, R, adj, Xn, qall, qtext, d2i, golds_g, ids_g, robust)
    diag = []
    B, cand_by_q = build_aug(qi_list, base_full, R, W, ra, di, added_by_q, added_sc, diag_out=diag)
    sv = c8c.predict(B["X28"]).astype(np.float64)
    spv = build_sp_aug(B, cand_by_q, sv, nodes, qall, R["query_meta"])
    spv["_t"] = R11.prep_tensors(spv, em11, es11)
    res = R11.evaluate(ma, spv, B, sv)                          # ZERO_SHOT_C11A over aug scope
    per_hop = metrics(B, res, R["query_meta"])
    agg = {"ndcg5": round(res["ndcg5"], 4), "recall5_macro": round(res["recall5_macro"], 4),
           "all5": round(res["all5"], 4), "all50": round(res["all50"], 4), "mrr": round(res["mrr"], 4),
           "n": res["n"], "added_total": int(sum(len(added_by_q[qi]) for qi in qi_list)),
           "gold_added_total": int(sum(int(added_sc[qi]["label"].sum()) for qi in qi_list if qi in added_sc))}
    # aggregate recovered-gold diagnostic per hop (real dense/offset/mixture ranks; splade/relation NOT computed)
    rgd = {}
    hop_of = {qi: R["query_meta"][qi].get("hop") for qi in qi_list}
    for e_ in diag:
        h = hop_of.get(e_["qi"]);
        if h is None: continue
        d = rgd.setdefault(str(h), {"n": 0, "in_pool50": 0, "base_rank": [], "dense_rank": [], "offset_rank": [], "mixture_rank": []})
        d["n"] += 1; d["in_pool50"] += int(e_["in_pool50"])
        for k in ("base_rank", "dense_rank", "offset_rank", "mixture_rank"): d[k].append(e_[k])
    RGD = {}
    for h, d in rgd.items():
        RGD[h] = {"n_recovered_golds": d["n"], "frac_in_pool50": round(d["in_pool50"] / max(d["n"], 1), 4),
                  "best_expert_rank_median": int(np.median([min(a, b, c) for a, b, c in
                                                            zip(d["dense_rank"], d["offset_rank"], d["mixture_rank"])])) if d["n"] else None,
                  "median": {k: int(np.median(d[k])) for k in ("base_rank", "dense_rank", "offset_rank", "mixture_rank")} if d["n"] else {}}
    return {"M": M, "AGG": agg, "PER_HOP": per_hop, "RECOVERED_GOLD_DIAG": RGD, "_res": res}


# ----------------------------------------------------------------- Q2.2 ORACLE EXPOSURE (NOT INFERENCE SAFE)
def _c11_window_final(ma, em11, es11, E_win, qv, dv_win, c8c_win):
    """C11a within-window final score = zscore(c8c) + beta*delta(qv,dv,standardize(E)). Higher = better.
    Mirrors _run_c11.prep_tensors/evaluate EXACTLY: E standardized by source (em11,es11); c8c z-scored over
    the window; qv/dv RAW embeddings; delta from frozen C11a. Window = the candidate set passed in."""
    Es = (E_win - em11) / es11
    c = c8c_win.astype(np.float64); c8cz = (c - c.mean()) / (c.std() + 1e-6)
    with torch.no_grad():
        qvr = torch.from_numpy(np.tile(qv.astype(np.float32), (len(dv_win), 1)))
        delta = ma.delta(qvr, torch.from_numpy(dv_win.astype(np.float32)),
                         torch.from_numpy(Es.astype(np.float32))).numpy()
    return c8cz + float(ma.beta) * delta


def run_oracle(qi_list, base_full, c8c, ma, em11, es11, R, W, ra, di,
               added_by_q, added_sc, nodes, qall, EIDX_):
    """ORACLE EXPOSURE DIAGNOSTIC (Q2.2) — NOT INFERENCE SAFE (uses gold identity to construct windows).
    Force each recovered gold into the top-50 pool (build_aug force_pool_golds) so it gets pool-level C8c
    features, then measure two oracles:
      A. ORACLE_TOP50_ADMISSION : gold admitted to pool, run the NORMAL C8c->C11a top-20 window; report its
         final rank (does the frozen reranker promote it once it's poolable?).
      B. ORACLE_C11A_EXPOSURE   : force the gold into a 20-cand C11a window = {gold} U top-19 pool by C8c;
         C11a scores it against the strongest competitors; report its within-window rank.
    Classify each recovered gold (per MetaQA hop):
      1. SCOPE_RECOVERED_BUT_POOL_REJECTED  = not naturally in the top-50 pool (baseline reality)
      2. POOL_EXPOSED_BUT_C11A_REJECTED     = force-exposed to C11a (B) but NOT ranked top-5
      3. C11A_EXPOSED_AND_PROMOTED_TOP5     = force-exposed to C11a (B) and ranked top-5
    """
    Bf, cand_by_q = build_aug(qi_list, base_full, R, W, ra, di, added_by_q, added_sc, force_pool_golds=True)
    sv = c8c.predict(Bf["X28"]).astype(np.float64)
    X = C9.assemble(Bf, C9.resid_feats(sv, Bf["groups"]))
    st = starts_of(Bf["groups"])
    # ORACLE A: normal C8c->C11a window over the forced pool
    spv = build_sp_aug(Bf, cand_by_q, sv, nodes, qall, R["query_meta"]); spv["_t"] = R11.prep_tensors(spv, em11, es11)
    resA = R11.evaluate(ma, spv, Bf, sv); grA = resA["_goldranks"]     # {(qi, gold_local): final rank}
    hop_of = {qi: R["query_meta"][qi].get("hop") for qi in qi_list}
    cat = {}
    for i, m in enumerate(Bf["meta"]):
        qi = m["qi"]; h = hop_of.get(qi)
        agl = m["added_gold_local"]
        if h is None or len(agl) == 0: continue
        a0 = st[i]; g = int(Bf["groups"][i]); pool = m["pool"]; sc_pool = sv[a0:a0 + g]
        row_all = int(R["query_meta"][qi]["row_all"]); qv = np.asarray(qall[row_all], np.float32)
        cq = cand_by_q[i]
        pool_pos = {int(li): p for p, li in enumerate(pool)}          # local scope idx -> pool position
        forced = set(int(x) for x in m["forced_local"])
        order = np.argsort(-sc_pool, kind="stable")                   # pool positions by C8c
        d = cat.setdefault(str(h), {"n": 0, "natural_in_pool": 0,
                                    "A_in_window": 0, "A_top5": 0, "A_top20": 0,
                                    "B_top5": 0, "B_top20": 0,
                                    "cat1_pool_rejected": 0, "cat2_c11a_rejected": 0, "cat3_promoted": 0})
        for gl in agl:
            gl = int(gl)
            if gl not in pool_pos: continue                          # forcing guarantees membership; guard
            d["n"] += 1
            nat = gl not in forced                                   # naturally in pool (not forced)
            d["natural_in_pool"] += int(nat)
            d["cat1_pool_rejected"] += int(not nat)
            gp = pool_pos[gl]                                         # gold's position within the pool
            # ORACLE A: final rank from the normal C8c->C11a window run; window = top-20 pool by C8c
            c8c_rank_in_pool = int((sc_pool > sc_pool[gp]).sum())     # gold's rank among pool by C8c
            rkA = grA.get((qi, gl), 10 ** 9)
            d["A_in_window"] += int(c8c_rank_in_pool < CAP)
            d["A_top5"] += int(rkA < 5); d["A_top20"] += int(rkA < 20)
            # ORACLE B: window = {gl} U top-19 pool by C8c (gold forced into the C11a scoring window)
            top19 = [int(p) for p in order if int(p) != gp][:19]
            win_pos = [gp] + top19                                    # gold first
            local_ids = np.array([int(pool[p]) for p in win_pos], np.int64)
            dv_win = np.asarray(nodes[cq[local_ids].astype(np.int64)], np.float32)
            E_win = X[a0 + np.asarray(win_pos, np.int64)][:, EIDX_]
            c8c_win = sc_pool[np.asarray(win_pos, np.int64)]
            fin = _c11_window_final(ma, em11, es11, E_win, qv, dv_win, c8c_win)
            rankB = int((fin > fin[0]).sum())                        # gold is index 0
            d["B_top5"] += int(rankB < 5); d["B_top20"] += int(rankB < 20)
            d["cat3_promoted"] += int(rankB < 5); d["cat2_c11a_rejected"] += int(rankB >= 5)
    out = {}
    for h, d in sorted(cat.items()):
        n = max(d["n"], 1)
        out[h] = {"n_recovered_golds": d["n"], "natural_in_pool": d["natural_in_pool"],
                  "ORACLE_A_TOP50_ADMISSION": {"n_in_C11a_window": d["A_in_window"],
                        "top5": d["A_top5"], "top20": d["A_top20"],
                        "frac_top5": round(d["A_top5"] / n, 4), "frac_top20": round(d["A_top20"] / n, 4)},
                  "ORACLE_B_C11A_EXPOSURE": {"top5": d["B_top5"], "top20": d["B_top20"],
                        "frac_top5": round(d["B_top5"] / n, 4), "frac_top20": round(d["B_top20"] / n, 4)},
                  "CLASSIFICATION": {"1_SCOPE_RECOVERED_BUT_POOL_REJECTED": d["cat1_pool_rejected"],
                        "2_POOL_EXPOSED_BUT_C11A_REJECTED": d["cat2_c11a_rejected"],
                        "3_C11A_EXPOSED_AND_PROMOTED_TOP5": d["cat3_promoted"]},
                  "LABEL": "ORACLE / DIAGNOSTIC / NOT INFERENCE SAFE"}
    return out


def main():
    bdir = os.environ.get("Q2_BACKBONE_DIR", "results/GENERALIZATION/_g1_backbone")
    Ms = [int(x) for x in os.environ.get("Q2_M", "0,32,64,128,256").split(",")]
    subset = int(os.environ.get("Q2_SUBSET", "0"))
    outp = os.environ.get("Q2_OUT", "results/GENERALIZATION/_q2_metaqa.json")
    log(f"=== Q2 MetaQA VAL  M={Ms} subset={subset} backbone={bdir} ===")
    base_full, c8c, ma, em11, es11 = load_frozen(bdir); log("frozen backbone loaded")

    R = load_raw(); off = R["query_offsets"]; nq = len(off) - 1
    query_meta = R["query_meta"]; log(f"raw val loaded nq={nq} total_cands={len(R['cand_ids'])}")

    # geometry artifacts
    A = G.load_artifacts(); d2i = A["doc_id_to_idx"]; j = A["j"]
    golds_g = j["golds"]; ids_g = j["ids"]
    from src.pipeline.standardizer import load_nodes
    qtext = {n.node_id: n.content for n in load_nodes(f"data/processed/master_nodes_{DS}.json")
             if n.metadata.get("type") == "question"}
    adj, sinfo = G.build_structural_adj(d2i); log(f"structural adj edges={sinfo['n_edges']}")
    Dn, Qn, dtop = L.load_embeddings(DS)                        # normalized nodes/queries (torch)
    nodes = np.load(f"{G.BASE}nodes.npy"); Xn = nodes / (np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-9)
    qall = np.load(f"{G.BASE}queries_all.npy")
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"{HEADS}/universal_offset_src_gteqwen.pt", map_location="cpu")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"{HEADS}/universal_mixture_src_gteqwen.pt", map_location="cpu")); mh.eval()
    robust = {"Dn": Dn, "dtop": dtop, "oh": oh, "mh": mh}
    log("geometry + heads ready")

    # query universe = the frozen valid-query set (>=1 in-scope gold under P50) — the EXACT population G1's
    # ZERO_SHOT_C11A bundle covers, so M=0 reproduces G1. Covers all L1_PARTIAL cases (where geometry adds the
    # MISSING golds); fully-L1-failed queries (0 in-scope golds) need a C6-feat-cache extension (full-run TODO).
    valid = set(int(x) for x in C8.precompute_arch(DS, SPLIT)["qi"].tolist())
    if subset > 0:
        hop_of = {qi: query_meta[qi].get("hop") for qi in range(nq)}
        qi_list = []
        for h in (1, 2, 3):
            cnt = 0
            for qi in range(nq):
                if qi in valid and hop_of[qi] == h:
                    qi_list.append(qi); cnt += 1
                    if cnt >= subset: break
        qi_list = sorted(qi_list)
    else:
        qi_list = sorted(valid)
    log(f"eval queries: {len(qi_list)} (valid universe={len(valid)})")

    # query-level frozen inputs (augmentation-invariant): soft weights + qmeta scalars
    qi_arr = np.array(qi_list, np.int64)
    W = C8.soft_weights(*base_full[:3], DS, SPLIT, qi_arr)
    ra, di = C9._qmeta(DS, SPLIT, qi_arr)

    results = {}
    for M in Ms:
        t = time.time()
        r = eval_scope(M, qi_list, base_full, c8c, ma, em11, es11, R, W, ra, di,
                       adj, Xn, nodes, qall, qtext, d2i, golds_g, ids_g, robust)
        results[f"M{M}"] = {"M": M, "AGG": r["AGG"], "PER_HOP": r["PER_HOP"], "RECOVERED_GOLD_DIAG": r["RECOVERED_GOLD_DIAG"]}
        log(f"M={M} agg={r['AGG']} ({time.time()-t:.0f}s)")
        for h, hp in r["PER_HOP"].items():
            log(f"   hop{h}: cand~{hp['cand_count_mean']:.0f}(+{hp['added_mean']:.0f}) "
                f"ANY={hp['ANY_coverage']:.3f} ALL={hp['ALL_coverage']:.3f} "
                f"R@5={hp['R@5']:.3f} NDCG@5={hp['NDCG@5']:.3f} FullCov@5={hp['FullCov@5']:.3f} "
                f"A(L1miss)={hp['A_L1_MISSING_golds']} B(rankfail@5)={hp['B_L2_RANKFAIL@5_golds']}")
        for h, dg in r["RECOVERED_GOLD_DIAG"].items():
            log(f"     recov-gold hop{h}: n={dg['n_recovered_golds']} in_pool50={dg['frac_in_pool50']:.3f} "
                f"best_expert_rank_med={dg['best_expert_rank_median']} medians={dg.get('median')}")

    # ---- Q2.2 ORACLE EXPOSURE DIAGNOSTIC (NOT INFERENCE SAFE) at the max M ----
    oracle = {}
    if os.environ.get("Q2_ORACLE", "0") == "1":
        Mmax = max(Ms)
        log(f"=== ORACLE EXPOSURE DIAGNOSTIC (NOT INFERENCE SAFE) at M={Mmax} ===")
        t = time.time()
        added_by_q, added_sc = compute_added(Mmax, qi_list, R, adj, Xn, qall, qtext, d2i, golds_g, ids_g, robust)
        oracle = run_oracle(qi_list, base_full, c8c, ma, em11, es11, R, W, ra, di,
                            added_by_q, added_sc, nodes, qall, EIDX)
        oracle = {"M": Mmax, "PER_HOP": oracle}
        log(f"ORACLE done ({time.time()-t:.0f}s)")
        for h, d in oracle["PER_HOP"].items():
            log(f"   ORACLE hop{h}: n={d['n_recovered_golds']} natural_in_pool={d['natural_in_pool']} "
                f"A_top5={d['ORACLE_A_TOP50_ADMISSION']['top5']}/{d['ORACLE_A_TOP50_ADMISSION']['top20']} "
                f"B_top5={d['ORACLE_B_C11A_EXPOSURE']['top5']}/{d['ORACLE_B_C11A_EXPOSURE']['top20']} "
                f"CLASS={d['CLASSIFICATION']}")

    # M=0 gate vs G1 (if present)
    gate = {}
    try:
        g1 = json.load(open("results/GENERALIZATION/_g1_eval_metaqa.json"))
        ph = g1.get("PER_HOP", {})
        m0 = results.get("M0", {}).get("PER_HOP", {})
        for h in ("1", "2", "3"):
            zs = ph.get(h, {}).get("systems", {}).get("ZERO_SHOT_C11A", {}).get("CONDITIONAL_ON_P50", {})
            gate[h] = {"g1_zs_cond_ndcg5": zs.get("ndcg5"), "g1_zs_cond_all5": zs.get("all5"),
                       "q2_M0_ndcg5": m0.get(h, {}).get("NDCG@5"), "q2_M0_R@5": m0.get(h, {}).get("R@5"),
                       "note": "q2 M0 uses expected-gold denominator (end-to-end); g1 CONDITIONAL_ON_P50 uses feasible denom — compare trend not identity unless subset=0"}
    except Exception as ex:
        gate = {"error": repr(ex)}

    out = {"phase": "Q2 — frozen C11a (ZERO_SHOT) on P50 vs P50-union-geometry; MetaQA VAL; TEST untouched",
           "M_values": Ms, "subset_per_hop": subset, "n_eval_queries": len(qi_list),
           "APPROXIMATIONS": {"splade_added": "per-query MIN-1 (conservative bottom-rank); genuine approx",
                              "relation_added": "ABSTAIN mask=False (correct default, no connecting-sentence edge)",
                              "dense_added": "exact cosine", "offset_added": "exact frozen universal head",
                              "mixture_added": "exact frozen universal head",
                              "M0_exactness": "M=0 uses ONLY on-disk scores -> reproduces frozen G1 path exactly"},
           "RESULTS": results, "M0_GATE_vs_G1": gate,
           "ORACLE_EXPOSURE_DIAGNOSTIC": oracle,
           "DECOMPOSITION_LEGEND": {"A_L1_MISSING": "expected in-corpus gold NOT in augmented scope",
                                    "B_L2_RANKFAIL@K": "gold present in augmented scope but final rank >= K (C11a fails to promote)",
                                    "ORACLE_A_TOP50_ADMISSION": "recovered gold force-admitted to top-50 pool, then NORMAL C8c->C11a top-20 window; NOT inference safe",
                                    "ORACLE_B_C11A_EXPOSURE": "recovered gold forced into a 20-cand C11a window={gold}+top19 pool; within-window rank; NOT inference safe"},
           "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1, default=str)
    log(f"Q2_DONE -> {outp}")


if __name__ == "__main__":
    main()
