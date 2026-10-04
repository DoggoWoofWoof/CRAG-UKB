"""Exact query-level Shapley over the 5 frozen experts (Dense, SPLADE, Offset, Mixture, Masked-Qwen-Relation).

n=5 -> 2^5=32 coalitions/query, EXACT enumeration (no Monte-Carlo).
Value function v_q(S) = utility of the ranking induced by the frozen masked-RRF fusion of coalition S.

LOCKED semantics:
  - RRF K0=60; non-relation expert contributes 1/(K0+rank) to EVERY candidate.
  - Relation contributes 1/(K0+rank_among_eligible) on mask=1, and EXACTLY 0 on mask=0 (ABSTAIN).
  - A candidate whose total coalition contribution is 0 is NOT RETRIEVED (rank=+inf) — no tie-order fallback.
    => v_q(EMPTY)=0 automatically; relation-only with no signal => 0.
  - Utilities: PRIMARY nDCG@50 (multi-positive), SECONDARY MRR / ALL@10 / ALL@50. Signed phi (never clamped).
  - L1_ANY_FAIL (no in-scope gold) -> SHAPLEY_VALID=False, skipped.
"""
import os, sys, json, time
from math import factorial, log2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import l2_lib as L

PILOTS = ("2wiki_clean", "musique_clean")
SPLITS = ("train", "val")
CORP = "data/l2_corpus"
OUTD = "results/L2/_shapley"
EXPERTS = ("dense", "splade", "offset", "mixture", "relation")
N = 5
K0 = 60
INF = 1 << 30
UTILS = ("ndcg50", "mrr", "all10", "all50")
_W = {s: factorial(s) * factorial(N - s - 1) / factorial(N) for s in range(N)}   # Shapley coalition weights
_POP = [bin(b).count("1") for b in range(1 << N)]


def _ranks(score):
    order = np.argsort(-score, kind="stable")
    rk = np.empty(len(order), np.int64); rk[order] = np.arange(len(order))
    return rk


def contribs(dense, splade, offset, mixture, relscore, relmask):
    """(5,m) RRF contribution vectors. Relation masked (0 on mask=0)."""
    m = len(dense); C = np.zeros((N, m), dtype=np.float64)
    for idx, sc in enumerate((dense, splade, offset, mixture)):
        C[idx] = 1.0 / (K0 + _ranks(np.asarray(sc, dtype=np.float32)))
    ie = np.where(relmask)[0]
    if len(ie):
        loc = np.argsort(-relscore[ie], kind="stable")     # rank AMONG eligible only
        ranks_e = np.empty(len(ie), np.int64); ranks_e[loc] = np.arange(len(ie))
        C[4][ie] = 1.0 / (K0 + ranks_e)
    return C


def _gold_ranks(total, golds):
    gr = np.empty(len(golds), np.int64)
    for k, g in enumerate(golds):
        tg = total[g]
        if tg <= 0:
            gr[k] = INF
        else:
            gr[k] = int((total > tg).sum() + (total[:g] == tg).sum())   # stable/dense tie-break
    return gr


def _utils(gr, ng):
    best = int(gr.min()); worst = int(gr.max())
    mrr = 1.0 / (best + 1) if best < INF else 0.0
    all10 = 1.0 if worst < 10 else 0.0
    all50 = 1.0 if worst < 50 else 0.0
    dcg = sum(1.0 / log2(r + 2) for r in gr if r < 50)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(ng, 50)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return np.array([ndcg, mrr, all10, all50], dtype=np.float64)


def shapley_query(C, golds):
    """Return phi (5,4) signed Shapley + v_full (4,). golds = local indices of in-scope golds."""
    ng = len(golds)
    V = np.zeros((1 << N, 4), dtype=np.float64)
    for b in range(1, 1 << N):
        members = [i for i in range(N) if (b >> i) & 1]
        total = C[members].sum(0)
        V[b] = _utils(_gold_ranks(total, golds), ng)
    phi = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        bit = 1 << i
        for b in range(1 << N):
            if b & bit:
                continue
            w = _W[_POP[b]]
            phi[i] += w * (V[b | bit] - V[b])
    return phi, V[(1 << N) - 1]


# ------------------------------------------------------------------ split driver
def _load_split_arrays(ds, split):
    d = f"{CORP}/{ds}/{split}"
    return dict(
        off=np.load(f"{d}/query_offsets.npy"), labels=np.load(f"{d}/labels.npy"),
        dense=np.load(f"{d}/dense_score.npy"), splade=np.load(f"{d}/splade_scope_score.npy"),
        offset=np.load(f"{d}/offset_score.npy"), mixture=np.load(f"{d}/mixture_score.npy"),
        relscore=np.load(f"{d}/relation_qwen_score.npy"), relmask=np.load(f"{d}/relation_mask.npy"),
        meta=np.load(f"{d}/expert_meta.npz"))


def compute_split(ds, split, qsubset=None, verbose=True):
    A = _load_split_arrays(ds, split); off = A["off"]; labels = A["labels"]; M = A["meta"]
    nq = len(off) - 1
    rows = list(range(nq)) if qsubset is None else list(qsubset)
    PHI = {u: np.full((nq, N), np.nan, np.float64) for u in UTILS}
    VFULL = {u: np.full(nq, np.nan, np.float64) for u in UTILS}
    valid = np.zeros(nq, bool)
    t0 = time.time()
    for c, qi in enumerate(rows):
        s, e = int(off[qi]), int(off[qi + 1])
        lab = labels[s:e]; golds = np.where(lab == 1)[0]
        if len(golds) == 0:
            continue                                        # L1_ANY_FAIL -> SHAPLEY_VALID=False
        C = contribs(A["dense"][s:e], A["splade"][s:e], A["offset"][s:e], A["mixture"][s:e],
                     A["relscore"][s:e], A["relmask"][s:e])
        phi, vfull = shapley_query(C, golds)
        for ui, u in enumerate(UTILS):
            PHI[u][qi] = phi[:, ui]; VFULL[u][qi] = vfull[ui]
        valid[qi] = True
        if verbose and (c + 1) % 500 == 0:
            print(f"[shapley/{ds}/{split}] {c+1}/{len(rows)} t={time.time()-t0:.0f}s", flush=True)
    out = dict(query_row_all=M["row_all"], valid=valid, gold_count=M["gold_count"], L1_status=M["L1_status"],
               relation_any_signal_present=M["relation_any_signal_present"],
               relation_gold_present=M["relation_gold_present"],
               relation_num_candidates=M["relation_num_candidates"],
               dense_splade_disagreement=M["dense_splade_disagreement"])
    for u in UTILS:
        for i, en in enumerate(EXPERTS):
            out[f"phi_{u}_{en}"] = PHI[u][:, i]
        out[f"v_full_{u}"] = VFULL[u]
    out["_runtime_s"] = time.time() - t0; out["_n_computed"] = int(valid.sum())
    return out


# ------------------------------------------------------------------ unit tests
def unit_tests():
    rng = np.random.default_rng(0); m = 200
    res = {}
    # base random experts
    D = rng.standard_normal(m); S = rng.standard_normal(m); O = rng.standard_normal(m); Mx = rng.standard_normal(m)
    rmask = np.zeros(m, bool); rmask[rng.choice(m, 6, replace=False)] = True
    rsc = rng.standard_normal(m)
    C = contribs(D, S, O, Mx, rsc, rmask)
    golds = np.array([3, 50, 120])
    phi, vfull = shapley_query(C, golds)
    # EFFICIENCY: sum phi == v_full (v_empty=0)
    eff_err = np.abs(phi.sum(0) - vfull).max()
    res["efficiency_max_error"] = float(eff_err)
    # DUMMY: 6th expert contributing 0 everywhere -> extend to n=6 quickly via manual check:
    # add a zero contribution vector; its marginal is always 0 -> phi=0
    C6 = np.vstack([C, np.zeros((1, m))])
    # recompute a 6-expert shapley just for the dummy/symmetry tests
    def shap6(C6, golds, n):
        from math import factorial as f
        W = {s: f(s) * f(n - s - 1) / f(n) for s in range(n)}
        V = np.zeros(1 << n)
        for b in range(1, 1 << n):
            members = [i for i in range(n) if (b >> i) & 1]
            V[b] = _utils(_gold_ranks(C6[members].sum(0), golds), len(golds))[0]  # ndcg
        ph = np.zeros(n)
        for i in range(n):
            bit = 1 << i
            for b in range(1 << n):
                if b & bit: continue
                ph[i] += W[bin(b).count("1")] * (V[b | bit] - V[b])
        return ph
    ph6 = shap6(C6, golds, 6)
    res["dummy_phi"] = float(ph6[5]); res["dummy_pass"] = abs(ph6[5]) < 1e-12
    # SYMMETRY: duplicate dense as 6th expert -> equal shapley to dense
    C6s = np.vstack([C, C[0:1]])
    ph6s = shap6(C6s, golds, 6)
    res["symmetry_gap"] = float(abs(ph6s[0] - ph6s[5])); res["symmetry_pass"] = abs(ph6s[0] - ph6s[5]) < 1e-9
    # RELATION ABSTENTION: mask all false -> relation phi exactly 0 (all utils)
    Cabs = contribs(D, S, O, Mx, rsc, np.zeros(m, bool))
    phabs, _ = shapley_query(Cabs, golds)
    res["relation_abstain_maxabs"] = float(np.abs(phabs[4]).max()); res["relation_abstain_pass"] = np.abs(phabs[4]).max() < 1e-15
    # NEGATIVE: construct an expert that ranks a gold LAST while others rank it well -> its phi can be negative
    Dn = np.full(m, 0.0); Dn[golds] = 5.0                 # good expert: golds on top
    Bad = rng.standard_normal(m); Bad[golds] = -10.0       # bad expert: golds at bottom
    Cneg = contribs(Dn, Bad, O, Mx, rsc, np.zeros(m, bool))
    phneg, _ = shapley_query(Cneg, golds)
    res["negative_splade_phi_mrr"] = float(phneg[1, 1]); res["negative_found"] = bool((phneg < -1e-9).any())
    res["EFFICIENCY_PASS"] = bool(eff_err < 1e-9)
    return res


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "test"
    os.makedirs(OUTD, exist_ok=True)
    if stage == "test":
        print(json.dumps(unit_tests(), indent=1, default=str))
    elif stage == "full":
        ds = sys.argv[2]; sp = sys.argv[3]
        out = compute_split(ds, sp)
        np.savez(f"{OUTD}/{ds}_{sp}.npz", **{k: v for k, v in out.items() if not k.startswith("_")})
        print(f"SAVED {ds}/{sp} n={out['_n_computed']} runtime={out['_runtime_s']:.0f}s "
              f"({1000*out['_runtime_s']/max(1,out['_n_computed']):.1f} ms/q)")
