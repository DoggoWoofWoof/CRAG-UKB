"""FINAL Relation-correctness phase (PART A): MASKED relation semantics.

The legacy relation path (l2_relation.py) represented "expert absent" by score=-1 and then ranked the
ENTIRE P50 vector with a stable argsort. Because the cand array is stored in canonical dense-desc order,
the tied -1 candidates receive deterministic DENSE-CORRELATED ranks and therefore an accidental RRF
contribution 1/(k0+rank). That both (a) made relation-only best-gold-rank look dense-correlated and
(b) let no-relation candidates leak a dense-shaped vote into E1a+relation / E3+relation.

This module fixes that with an explicit sparse/abstain interface:
    relation_score[pair], relation_mask[pair]
    mask=1  <=> a valid topic->candidate connecting relation sentence exists  -> expert votes
    mask=0  <=> expert ABSTAINS (contributes 0 to RRF, NO pseudo-rank)

The eligibility mask is PURE GRAPH TOPOLOGY (cand_sent>=0) and is therefore ENCODER-INDEPENDENT:
identical for MiniLM proxy and canonical gte-Qwen2. Only the ordering AMONG eligible candidates changes
with the encoder.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import l2_lib as L
import l2_relation as R

PILOTS = ("2wiki_clean", "musique_clean")
KS_RESCUE = (5, 10, 20, 50)


# ---------------------------------------------------------------- masked RRF interface
def _ranks_full_stable(score):
    return L._ranks_from_scores(score)                      # dense expert: rank over full scope (stable)


def masked_ranks(score, mask):
    """0-indexed ranks AMONG masked-in candidates only (by score desc, stable). masked-out -> -1."""
    rk = np.full(len(score), -1, dtype=np.int64)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return rk
    order = idx[np.argsort(-score[idx], kind="stable")]
    rk[order] = np.arange(len(order))
    return rk


def rrf_contrib(score, mask=None, k0=L.RRF_K0):
    """RRF contribution vector. mask=None -> full/dense expert (every cand ranked). mask given ->
    sparse expert: masked-in cands contribute 1/(k0+rank_among_eligible); masked-out contribute 0."""
    n = len(score); c = np.zeros(n, dtype=np.float64)
    if mask is None:
        c = 1.0 / (k0 + _ranks_full_stable(score))
    else:
        rk = masked_ranks(score, mask); inn = rk >= 0
        c[inn] = 1.0 / (k0 + rk[inn])
    return c


def masked_rrf_fuse(experts, k0=L.RRF_K0):
    """experts: list of (score_vec, mask_or_None). Sum of RRF contributions (higher=better).
    Non-masked experts reproduce the generic rrf_fuse() exactly; the masked expert ABSTAINS on mask=0.
    Generic L.rrf_fuse is left untouched — this is a separate masked-expert interface."""
    fused = np.zeros(len(experts[0][0]), dtype=np.float64)
    for sc, mask in experts:
        fused += rrf_contrib(sc, mask, k0)
    return fused


# ---------------------------------------------------------------- per-split relation (score + mask)
def load_relation(ds, split, encoder="minilm"):
    """Return (rscore[pairs], rmask[pairs]) aligned to S['cand']. mask = topology eligibility."""
    B = R.build_edges_for_split(ds, split, verbose=False)
    mask = (B["cand_sent"] >= 0)
    rscore = np.load(f"scratchpad/_relscore_{ds}_{split}.npy")   # -1 sentinel on masked-out (unused now)
    assert len(mask) == len(rscore)
    return rscore, mask, B


# ---------------------------------------------------------------- AUDIT: prove the tie-order leak
def audit_tie_leak(ds, split="val"):
    """Prove that the LEGACY full-vector ranking gives no-relation candidates deterministic dense-order
    ranks (and hence accidental RRF contribution). Encoder-independent check."""
    S = L.load_split(ds, split); off = S["off"]
    rscore, mask, _ = load_relation(ds, split)
    # pick queries with many mask=0 candidates
    examples = []
    dense_tie_match = 0; checked = 0
    for qi in range(len(off) - 1):
        s, e = off[qi], off[qi + 1]
        m = mask[s:e]; sc = rscore[s:e]
        n_absent = int((~m).sum())
        if n_absent < 5:
            continue
        # legacy rank vector (full stable argsort of -score, absent all tie at -1)
        legacy_rk = L._ranks_from_scores(sc)
        # dense-desc canonical order rank = identity position (cand stored dense-desc) -> compare tie order
        # among absent candidates, legacy assigns ranks in ascending array index (stable) after eligibles
        absent_idx = np.where(~m)[0]
        n_elig = int(m.sum())
        expected_absent_ranks = np.arange(n_elig, n_elig + n_absent)   # dense-order positions of absent cands
        got = np.sort(legacy_rk[absent_idx])
        if np.array_equal(got, expected_absent_ranks):
            dense_tie_match += 1
        checked += 1
        if len(examples) < 3:
            examples.append({"qi": int(qi), "n_elig": n_elig, "n_absent": n_absent,
                             "absent_rank_min": int(legacy_rk[absent_idx].min()),
                             "absent_rank_max": int(legacy_rk[absent_idx].max()),
                             "elig_rank_max": int(legacy_rk[np.where(m)[0]].max()) if n_elig else -1})
    return {
        "dataset": ds, "split": split, "queries_checked": checked,
        "RELATION_ABSENT_CANDIDATES_RECEIVE_RANKS": "YES" if checked and dense_tie_match == checked else ("PARTIAL" if dense_tie_match else "NO"),
        "RELATION_TIE_ORDER_SOURCE": "stable argsort of -score over FULL scope; absent cands all tie at -1 -> "
                                     "placed after eligibles in original array order == canonical dense-desc scope order",
        "absent_ranks_equal_dense_order_positions_frac": round(dense_tie_match / checked, 4) if checked else None,
        "examples": examples,
    }


# ---------------------------------------------------------------- masked relation-only + coverage
def masked_relation_only(ds, split="val"):
    """Relation-only ranking respecting the mask. best_gold_rank_relation_masked = rank AMONG eligible
    (masked-in) golds, -1 (NO_RELATION_RANK) if relation abstains on every gold. Plus coverage + the
    conditional ranking denominators the future regime matrix needs."""
    S = L.load_split(ds, split); off = S["off"]; labels = S["labels"]; meta = S["meta"]
    nq = len(off) - 1
    rscore, mask, B = load_relation(ds, split)
    best_masked = np.full(nq, -1, dtype=np.int64)      # NO_RELATION_RANK = -1
    gold_present = np.zeros(nq, dtype=bool)
    any_signal = np.zeros(nq, dtype=bool)
    num_cand = np.zeros(nq, dtype=np.int64)
    # coverage accumulators
    nq_gold = 0; any_gold_cov = 0; all_gold_cov = 0
    # conditional ranking metric accumulators
    cond_sig_rr = []; cond_gold_rr = []
    cond_gold_any = {K: [] for K in KS_RESCUE}
    for qi in range(nq):
        s, e = off[qi], off[qi + 1]
        m = mask[s:e]; sc = rscore[s:e]; lab = labels[s:e]
        num_cand[qi] = int(m.sum()); any_signal[qi] = m.any()
        goldmask = lab == 1; ng = int(goldmask.sum())
        elig_gold = m & goldmask
        gold_present[qi] = bool(elig_gold.any())
        # rank among eligible
        rk = masked_ranks(sc, m)                      # -1 for masked-out
        if elig_gold.any():
            br = int(rk[elig_gold].min()); best_masked[qi] = br
        if ng > 0:
            nq_gold += 1
            gp = int(elig_gold.sum())
            if gp > 0: any_gold_cov += 1
            if gp == ng: all_gold_cov += 1
        # conditional metrics
        if m.any():                                    # relation has >=1 eligible candidate
            # relation-only best gold rank among eligible (or +inf if no eligible gold)
            if elig_gold.any():
                cond_sig_rr.append(1.0 / (best_masked[qi] + 1))
            else:
                cond_sig_rr.append(0.0)
        if elig_gold.any():                            # relation has >=1 eligible GOLD
            cond_gold_rr.append(1.0 / (best_masked[qi] + 1))
            for K in KS_RESCUE:
                cond_gold_any[K].append(1.0 if best_masked[qi] < K else 0.0)

    def frac(a, b): return round(a / b, 4) if b else 0.0
    cov = {
        "RELATION_QUERY_COVERAGE": frac(int(any_signal.sum()), nq),
        "ANY_GOLD_RELATION_COVERAGE": frac(any_gold_cov, nq_gold),
        "ALL_GOLD_RELATION_COVERAGE": frac(all_gold_cov, nq_gold),
        "n_queries": nq, "n_gold_queries": nq_gold,
        "mean_eligible_candidates_per_query": round(float(num_cand.mean()), 3),
        "mean_eligible_candidates_where_signal": round(float(num_cand[any_signal].mean()), 3) if any_signal.any() else 0.0,
        "p95_eligible_candidates": int(np.percentile(num_cand, 95)),
        "max_eligible_candidates": int(num_cand.max()),
    }
    condm = {
        "COND_RELATION_SIGNAL_PRESENT": {"n": len(cond_sig_rr), "MRR": round(float(np.mean(cond_sig_rr)), 4) if cond_sig_rr else None},
        "COND_RELATION_GOLD_PRESENT": {"n": len(cond_gold_rr), "MRR": round(float(np.mean(cond_gold_rr)), 4) if cond_gold_rr else None,
                                       **{f"ANY@{K}": round(float(np.mean(cond_gold_any[K])), 4) for K in KS_RESCUE}},
    }
    feats = {"best_gold_rank_relation_masked": best_masked, "relation_gold_present": gold_present,
             "relation_any_signal_present": any_signal, "relation_num_candidates": num_cand}
    return cov, condm, feats


# ---------------------------------------------------------------- fix regime raw material
def fix_regime_raw(ds, split="val"):
    p = f"results/L2/_regime_raw/{ds}_{split}.npz"
    z = np.load(p); cols = {k: z[k] for k in z.files}
    _, _, feats = masked_relation_only(ds, split)
    # preserve legacy under explicit name; drop the raw ambiguous key
    if "best_gold_rank_relation" in cols:
        cols["best_gold_rank_relation_proxy_legacy"] = cols.pop("best_gold_rank_relation")
    cols.update(feats)
    np.savez(p, **cols)
    return {"path": p, "columns": sorted(cols.keys())}


# ---------------------------------------------------------------- masked VAL rescue (E1a / E3 +- relation)
def masked_rescue(ds, split="val", encoder="minilm"):
    S = L.load_split(ds, split); off = S["off"]
    rscore, rmask, _ = load_relation(ds, split, encoder)
    Dn, Qn, dtop = L.load_embeddings(ds)
    import torch
    from l2_lib import OffsetHead, MixtureHead
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
    dense = S["dense_score"]; splade = S["splade_scope_score"]

    def sl(vec, qi): s, e = off[qi], off[qi + 1]; return vec[s:e]
    def masked_expert(qi): s, e = off[qi], off[qi + 1]; return rscore[s:e], rmask[s:e]

    def make(name):
        def f(qi):
            d = (sl(dense, qi), None); sp = (sl(splade, qi), None)
            o = (sl(oh_sc, qi), None); m = (sl(mh_sc, qi), None)
            rel = masked_expert(qi)
            if name == "E1a": return masked_rrf_fuse([d, sp])
            if name == "E1a+relation": return masked_rrf_fuse([d, sp, rel])
            if name == "E3": return masked_rrf_fuse([d, sp, o, m])
            if name == "E3+relation": return masked_rrf_fuse([d, sp, o, m, rel])
        return f

    metrics = {}; bestranks = {}
    for name in ("E1a", "E1a+relation", "E3", "E3+relation"):
        res, per_best, _ = L.eval_ranking(S, make(name), collect_regime=True)
        metrics[name] = {k: res["COND_ANY_GOLD_IN_SCOPE"].get(k) for k in
                         ("MRR", "R@5", "R@10", "R@20", "R@50", "ANY@10", "ANY@20", "ANY@50",
                          "ALL@10", "ALL@20", "ALL@50")}
        bestranks[name] = per_best

    # relation-only masked best gold rank (for SOLO test)
    _, _, feats = masked_relation_only(ds, split)
    rel_best = feats["best_gold_rank_relation_masked"]      # -1 = abstain on gold
    reg = np.load(f"results/L2/_regime_raw/{ds}_{split}.npz")
    cheap_best = np.minimum.reduce([reg["best_gold_rank_dense"], reg["best_gold_rank_splade"],
                                    reg["best_gold_rank_offset"], reg["best_gold_rank_mixture"]])

    rescue = {}
    for base, plus in (("E1a", "E1a+relation"), ("E3", "E3+relation")):
        b0 = bestranks[base]; b1 = bestranks[plus]
        vv = (b0 >= 0) & (b1 >= 0)
        for K in KS_RESCUE:
            resc = vv & (b0 >= K) & (b1 < K)
            hurt = vv & (b0 < K) & (b1 >= K)
            rescue[f"{base}: RELATION_RESCUE_top{K}"] = int(resc.sum())
            rescue[f"{base}: RELATION_HURT_top{K}"] = int(hurt.sum())
            if base == "E3":
                # Decompose E3 rescues by how deep the BASE STACK (cheap_stack = min over the 4 base
                # experts) had the rescued gold. rel_best is always 0/1 here (tiny eligible sets), so a
                # rel-only<K test is degenerate; base-depth is the meaningful SOLO/COOP discriminator.
                # SOLO_RESCUE     = base stack essentially MISSED it (cheap_best>=50) -> relation carried it.
                # COOPERATIVE     = base had partial signal (K<=cheap_best<50) -> relation's vote tipped it in.
                solo = resc & (cheap_best >= 50)
                coop = resc & (cheap_best >= K) & (cheap_best < 50)
                rescue[f"E3: SOLO_RESCUE_top{K} (base_missed_>=50)"] = int(solo.sum())
                rescue[f"E3: COOPERATIVE_RESCUE_top{K} (base_partial)"] = int(coop.sum())
                if resc.any():
                    rescue[f"E3: rescued_top{K}_cheap_best_median"] = float(np.median(cheap_best[resc]))
    return {"dataset": ds, "split": split, "encoder": encoder, "metrics": metrics, "rescue": rescue}


def _rel_experts_and_bases(ds, split, encoder="minilm"):
    """Shared setup: returns S, off, per-query slicers for the 4 base experts + a relation (score,mask)
    getter, plus the raw relation score/mask arrays."""
    S = L.load_split(ds, split); off = S["off"]
    rscore, rmask, _ = load_relation(ds, split, encoder)
    Dn, Qn, dtop = L.load_embeddings(ds)
    import torch
    from l2_lib import OffsetHead, MixtureHead
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
    return S, off, S["dense_score"], S["splade_scope_score"], oh_sc, mh_sc, rscore, rmask


def ordering_ablation(ds, split="val"):
    """ENCODER-INVARIANCE BOUND. The eligibility mask is encoder-independent; an encoder can ONLY reorder
    the (tiny) eligible set. We bracket every possible encoder (incl. gte-Qwen2) by recomputing the masked
    E3+relation result under 4 orderings of the eligible relation scores:
      minilm      : the actual MiniLM cosine order
      tied_dense  : all eligible tied -> stable(dense) order among eligible (pure eligibility, NO encoder)
      reverse     : reverse MiniLM order among eligible (adversarial)
      random      : seeded random permutation among eligible
    If E3+relation metrics + gold rescue barely move across these, the KEEP decision is encoder-invariant
    and gte-Qwen2 must land inside this band."""
    S, off, dense, splade, oh_sc, mh_sc, rscore, rmask = _rel_experts_and_bases(ds, split)
    labels = S["labels"]
    rng = np.random.default_rng(1234)

    def variant_scores(kind):
        out = rscore.copy()
        for qi in range(len(off) - 1):
            s, e = off[qi], off[qi + 1]
            idx = np.where(rmask[s:e])[0]
            if len(idx) <= 1:
                continue
            loc = s + idx
            if kind == "tied_dense":
                out[loc] = 0.0                                   # all equal -> stable argsort = array(dense) order
            elif kind == "reverse":
                out[loc] = rscore[loc][::-1]
            elif kind == "random":
                out[loc] = rscore[loc][rng.permutation(len(idx))]
        return out

    def sl(v, qi): s, e = off[qi], off[qi + 1]; return v[s:e]
    rk = L._ranks_from_scores
    res = {}
    for kind in ("minilm", "tied_dense", "reverse", "random"):
        rsc = rscore if kind == "minilm" else variant_scores(kind)
        def E3r(qi): s, e = off[qi], off[qi + 1]; return masked_rrf_fuse(
            [(sl(dense, qi), None), (sl(splade, qi), None), (sl(oh_sc, qi), None), (sl(mh_sc, qi), None),
             (rsc[s:e], rmask[s:e])])
        m, per_best, _ = L.eval_ranking(S, E3r, collect_regime=True)
        # gold-level rescue vs E3 (E3 base ranks recomputed once)
        res[kind] = {k: round(m["COND_ANY_GOLD_IN_SCOPE"][k], 4) for k in
                     ("MRR", "R@5", "R@10", "R@20", "R@50", "ALL@10", "ALL@20", "ALL@50")}
    # spread across encoder-orderings
    keys = ("MRR", "R@5", "R@10", "R@20", "R@50", "ALL@10", "ALL@20", "ALL@50")
    spread = {k: round(max(res[o][k] for o in res) - min(res[o][k] for o in res), 4) for k in keys}
    return {"dataset": ds, "split": split, "orderings": res, "max_spread_across_encoders": spread}


def gold_level_rescue(ds, split="val", encoder="minilm"):
    """GOLD-level rescue (the correct metric for a sparse bridge expert). For every relation-ELIGIBLE
    gold, compare its E3 rank vs E3+relation rank, and characterize how deep the base stack had it.
    Query-best metrics are confounded by the easy first-hop gold in multi-gold queries; this isolates
    the second-hop bridge gold relation is designed to recover."""
    S = L.load_split(ds, split); off = S["off"]; labels = S["labels"]
    rscore, rmask, _ = load_relation(ds, split, encoder)
    Dn, Qn, dtop = L.load_embeddings(ds)
    import torch
    from l2_lib import OffsetHead, MixtureHead
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
    dense = S["dense_score"]; splade = S["splade_scope_score"]

    def sl(v, qi): s, e = off[qi], off[qi + 1]; return v[s:e]
    def me(qi): s, e = off[qi], off[qi + 1]; return rscore[s:e], rmask[s:e]
    def E3(qi): return M_rrf([(sl(dense, qi), None), (sl(splade, qi), None), (sl(oh_sc, qi), None), (sl(mh_sc, qi), None)])
    def E3r(qi): return M_rrf([(sl(dense, qi), None), (sl(splade, qi), None), (sl(oh_sc, qi), None), (sl(mh_sc, qi), None), me(qi)])
    M_rrf = masked_rrf_fuse

    rk = L._ranks_from_scores
    # per eligible gold: base_min_rank, E3 rank, E3r rank
    base_ranks = []; e3_ranks = []; e3r_ranks = []
    for qi in range(len(off) - 1):
        s, e = off[qi], off[qi + 1]; lab = labels[s:e]; gl = np.where(lab == 1)[0]
        elig = rmask[s:e]
        eg = [g for g in gl if elig[g]]
        if not eg:
            continue
        rd = rk(sl(dense, qi)); rs = rk(sl(splade, qi)); ro = rk(sl(oh_sc, qi)); rm = rk(sl(mh_sc, qi))
        r3 = rk(E3(qi)); r3r = rk(E3r(qi))
        for g in eg:
            base_ranks.append(int(min(rd[g], rs[g], ro[g], rm[g])))
            e3_ranks.append(int(r3[g])); e3r_ranks.append(int(r3r[g]))
    base_ranks = np.array(base_ranks); e3_ranks = np.array(e3_ranks); e3r_ranks = np.array(e3r_ranks)
    out = {"n_eligible_golds": int(len(base_ranks))}
    for K in KS_RESCUE:
        resc = (e3_ranks >= K) & (e3r_ranks < K)
        hurt = (e3_ranks < K) & (e3r_ranks >= K)
        out[f"GOLD_RESCUE_top{K}"] = int(resc.sum())
        out[f"GOLD_HURT_top{K}"] = int(hurt.sum())
        # of the rescued bridge golds, how many did the base stack genuinely miss (min base rank>=50 / >=K)
        out[f"GOLD_RESCUE_top{K}_base_missed>=50"] = int((resc & (base_ranks >= 50)).sum())
        out[f"GOLD_RESCUE_top{K}_base_missed>=K"] = int((resc & (base_ranks >= K)).sum())
        if resc.any():
            out[f"GOLD_RESCUE_top{K}_base_rank_median"] = float(np.median(base_ranks[resc]))
    out["eligible_gold_base_rank_median"] = float(np.median(base_ranks)) if len(base_ranks) else None
    out["eligible_gold_base_missed>=50_frac"] = round(float((base_ranks >= 50).mean()), 4) if len(base_ranks) else None
    return {"dataset": ds, "split": split, "encoder": encoder, "gold_level": out}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    for ds in PILOTS:
        block = {}
        if stage in ("all", "audit"):
            block["audit"] = audit_tie_leak(ds)
        if stage in ("all", "coverage", "regime"):
            cov, condm, _ = masked_relation_only(ds)
            block["masked_coverage"] = cov; block["masked_conditional"] = condm
        if stage in ("all", "regime"):
            block["regime_fix"] = fix_regime_raw(ds)
        if stage in ("all", "rescue"):
            block["masked_rescue"] = masked_rescue(ds)
        out[ds] = block
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open("scratchpad/_relmasked_partA.json", "w"), indent=1, default=str)
