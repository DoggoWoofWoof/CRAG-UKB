"""Offset SCALING sweep, DATASET-AGNOSTIC (one shared model across all corpora). Open-corpus R@k per dataset
(rank golds among ALL entities — the real metric). WebQSP is the headline (multi-answer -> the "5 slots").

Two questions:
  (1) WHICH OFFSET TYPE SCALES?  single-offset scaled by MLP width h vs a K-SLOT set-offset scaled by slot
      count K. WebQSP single peak to beat = 35.12 (offset_beam.py, dense-top1 seed, webqsp-only).
  (2) Can K slots be trained so each finds a DISTINCT gold INDIVIDUALLY (the "5 perfect combos")? -> set-
      prediction: bipartite-match slots to golds, give EACH matched slot its own contrastive gradient toward
      its assigned gold. Modes: 'hungarian' (optimal) and 'topk' (slot j -> j-th nearest gold).

TRAINING mirrors the proven universal head (_train_universal): ONE shared model across datasets; per epoch we
shuffle DATASET order and, within each dataset, shuffle examples; negatives are CORPUS-WIDE HARD negatives
(GPU top-k over that dataset's Xt) + in-batch negatives — not a fixed random pool. Single-offset trains batched
on (q, seed, gold) triples (in-batch + hard negs). Set-offset trains per-query (needs all golds at once for the
matching) with per-slot hard negs.

RETRIEVAL help: per-slot dense-cos fused with the query's SPLADE score (z-scored CombSUM, w swept) then slots
merge by max; SPLADE's own top-M is unioned into the pool. Reported offset-only (w=0) AND fused. Seed = dense-
top1 (agnostic) by default; --seed-mode topic uses the NER-linkable topic entity where available (WebQSP).
"""
import json
import logging
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
TAU = 0.07                                                        # contrastive temperature (matches universal head scale)


def _assign_topk(cos, K, g):
    """slot j -> j-th nearest gold to the query-composed vectors. Deterministic, collapse-prone.
    cos: (K,g). Returns [(slot, gold_local)] one per slot."""
    order = torch.argsort(cos.max(0).values, descending=True).tolist()
    return [(j, order[min(j, g - 1)]) for j in range(K)]


def _assign_hungarian(cost_np, K, g):
    """Optimal bipartite match of min(K,g) (slot,gold) pairs; leftover slots -> their own nearest gold (soft
    'backup' supervision so they don't drift to noise). Returns (matched_pairs, extra_pairs)."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_np)
        matched = list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        matched = []; used_s, used_g = set(), set()
        for _, j, gg in sorted((cost_np[j, gg], j, gg) for j in range(K) for gg in range(g)):
            if j in used_s or gg in used_g:
                continue
            used_s.add(j); used_g.add(gg); matched.append((j, gg))
            if len(matched) == min(K, g):
                break
    ms = {j for j, _ in matched}
    extra = [(j, int(np.argmin(cost_np[j]))) for j in range(K) if j not in ms]
    return matched, extra


def _load_dataset(d, seed_mode, te_cap, tr_cap, DEV, M):
    """Per-dataset bundle: corpus on GPU, train (qvec/seed/golds), test (with SPLADE caches)."""
    from src.experiments.l1_universal_head import _load
    dd = _load(d, "gte_qwen", 8000, tr_cap, te_cap)
    qtr, str_tr, gtr = dd["train"]; qte, ste, gte = dd["test"]
    tr_txt, te_txt = dd["train_texts"], dd["test_texts"]
    X = dd["X"].astype("float32"); N, DIM = X.shape
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); del X

    qtr_t = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    qte_t = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    gold_tr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    gold_te = [[int(x) for x in gte[i]] for i in range(len(gte))]

    # ---- SEED = the anchor the relational offset shifts FROM ----
    #   dense: dense-top1 doc (wrong ~97% on this KB -> a poor anchor)
    #   topic: centroid of the TOPIC ENTITY's docs. q_entity is the entity extracted from the question
    #          (NER/entity-linking, not oracle) -> anchor the shift at the topic region, then offset to the answer.
    import collections
    q2ents = {}
    if d == "webqsp":
        import pandas as pd
        df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
        q2ents = {str(r["question"]): [str(e).strip() for e in r["q_entity"]] for _, r in df.iterrows()}

    def dense_seed(vec_t, sd):
        return {"idx": [int(sd)]}                                          # anchor = dense-top1 row

    title2rows = None
    if seed_mode == "topic" and q2ents:
        id2idx = dd["id2idx"]
        mn = json.load(open(f"data/processed/master_nodes_{d}.json", encoding="utf-8"))
        title2rows = collections.defaultdict(list)
        for n in mn:
            nid = n["node_id"]
            if nid in id2idx:
                title2rows[n.get("metadata", {}).get("title", "").strip()].append(id2idx[nid])

    def seed_vec(txt, sd):
        if title2rows is not None:
            rows = [r for e in q2ents.get(txt, []) for r in title2rows.get(e, [])]
            if rows:
                return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0)   # topic centroid
        return Efull[int(sd)]                                             # fallback: dense-top1

    seed_tr_t = torch.stack([seed_vec(tr_txt[i], str_tr[i]) for i in range(len(str_tr))])
    seed_te_t = torch.stack([seed_vec(te_txt[i], ste[i]) for i in range(len(te_txt))])
    TR = [(i, gold_tr[i]) for i in range(len(gold_tr)) if gold_tr[i]]      # (query_idx, golds)

    TE = []
    for i in range(len(te_txt)):
        if gold_te[i]:
            TE.append((qte_t[i], seed_te_t[i], gold_te[i], te_txt[i]))

    # SPLADE: query vecs + per-query top-M doc scores (one full sparse dot per query, cached)
    splade = dd.get("splade"); spl_scorer, spl_mat = (splade if splade else (None, None))
    te_qs = te_spl_top = None
    if spl_scorer is not None:
        te_qs = np.stack([spl_scorer.encode_query(e[3]).astype("float32") for e in TE])
        te_spl_top = []
        for qs in te_qs:
            sf = spl_mat.dot(qs)
            top = np.argpartition(-sf, M)[:M] if M < N else np.argsort(-sf)
            te_spl_top.append((top.astype(np.int64), sf[top].astype("float32")))
    negpool = torch.randint(0, N, (min(60000, N),), device=DEV)
    log.info("  loaded %s: N=%d dim=%d train=%d test=%d splade=%s golds/q(mean)=%.1f",
             d, N, DIM, len(TR), len(TE), bool(splade), float(np.mean([len(g) for *_, g, _ in TE])))
    return {"Efull": Efull, "N": N, "DIM": DIM, "qtr": qtr_t, "seed_tr": seed_tr_t, "TR": TR, "TE": TE,
            "spl_mat": spl_mat, "te_qs": te_qs, "te_spl_top": te_spl_top, "negpool": negpool}


def _run(epochs=40, te_cap=2000, tr_cap=3000, assign="both", families=("baselines", "single", "set"),
         ks=(5, 8, 12, 16), hs=(768, 1536, 3072), w_splades=(0.0, 0.5, 1.0, 2.0),
         seed_mode="dense", extra_weight=0.3, M=2000, datasets=("webqsp",), hard_negs=True,
         bs=256, hn_k=16, sym_single=False, out_suffix="", single_loss="triple", seed=0, n_seeds=1):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = {d: _load_dataset(d, seed_mode, te_cap, tr_cap, DEV, M) for d in datasets}
    DIM = D[datasets[0]]["DIM"]
    log.info("[offset-scale] datasets=%s dim=%d hard_negs=%s seed=%s on %s",
             list(datasets), DIM, hard_negs, seed_mode, DEV)

    # ---------------- models ----------------
    class Dense(nn.Module):
        def paths(s, q, seed): return q, None
    class Single(nn.Module):
        def __init__(s, h=768):
            super().__init__(); s.m = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU(), nn.Linear(h, DIM))
        def paths(s, q, seed):
            return F.normalize(q + s.m(torch.cat([q, seed], -1)), dim=-1), None
    class SetOffset(nn.Module):
        def __init__(s, K=5, h=1024):
            super().__init__(); s.K = K
            s.trunk = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU())
            s.heads = nn.Linear(h, K * DIM)
        def paths(s, q, seed):
            z = s.trunk(torch.cat([q, seed], -1))
            offs = s.heads(z).view(q.shape[0], s.K, DIM)
            return F.normalize(q[:, None, :] + offs, dim=-1).reshape(-1, DIM), None   # (B*K,d), B=1 at train

    # ---------------- joint training (one shared model across datasets) ----------------
    def train_single(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        if single_loss == "multipos":            # single's OWN best recipe (== offset_beam 35.12): one vector pulled
            for ep in range(epochs):             # toward the CENTROID of all golds (multi-positive) + random negs.
                order = list(datasets); _random.Random(ep).shuffle(order)                       # naturally sibling-safe
                for d in order:
                    Ef = D[d]["Efull"]; qtr = D[d]["qtr"]; sdt = D[d]["seed_tr"]
                    negpool = D[d]["negpool"]; np_ = negpool.shape[0]
                    trq = D[d]["TR"][:]; _random.Random(ep * 131 + len(d)).shuffle(trq)
                    for qi, gold in trq:
                        cand = torch.cat([torch.tensor(gold, device=DEV), negpool[torch.randint(0, np_, (256,), device=DEV)]])
                        V, _ = model.paths(qtr[qi][None], sdt[qi][None])                         # (1,d)
                        sc = (Ef[cand] @ V.T).squeeze(-1)
                        tgt = torch.zeros(len(cand), device=DEV); tgt[:len(gold)] = 1.0 / len(gold)
                        loss = -(F.log_softmax(sc, 0) * tgt).sum()
                        opt.zero_grad(); loss.backward(); opt.step()
            model.eval(); return model
        trips = {d: [(i, g) for i, gl in D[d]["TR"] for g in gl] for d in datasets}   # (query_idx, one-gold)
        for ep in range(epochs):
            order = list(datasets); _random.Random(ep).shuffle(order)                          # shuffle DATASET order
            for d in order:
                Ef = D[d]["Efull"]; qtr = D[d]["qtr"]; sdt = D[d]["seed_tr"]; trip = trips[d][:]
                _random.Random(ep * 131 + len(d)).shuffle(trip)                                # shuffle within dataset
                for st in range(0, len(trip), bs):
                    b = trip[st:st + bs]
                    qn = qtr[[t[0] for t in b]]; seedv = sdt[[t[0] for t in b]]                 # topic/dense anchor
                    own = torch.tensor([t[1] for t in b], device=DEV); goldv = Ef[own]
                    V, _ = model.paths(qn, seedv)                                              # (B,d)
                    logits = V @ goldv.T / TAU                                                 # in-batch negatives
                    if sym_single:                                                            # symmetric w/ set: don't
                        qi = torch.tensor([t[0] for t in b], device=DEV)                       # negate a co-answer of the
                        sib = (qi[:, None] == qi[None, :]); sib.fill_diagonal_(False)          # SAME multi-answer query
                        logits = logits.masked_fill(sib, -1e4)
                    if hard_negs:
                        with torch.no_grad():
                            neg = torch.topk(V @ Ef.T, hn_k, dim=1).indices.reshape(-1)        # corpus-wide hard negs
                        hard = (V @ Ef[neg].T / TAU).masked_fill(neg.unsqueeze(0) == own.unsqueeze(1), -1e4)
                        logits = torch.cat([logits, hard], dim=1)
                    loss = F.cross_entropy(logits, torch.arange(len(b), device=DEV))
                    opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); return model

    def train_set(model, mode):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            order = list(datasets); _random.Random(ep).shuffle(order)
            for d in order:
                Ef = D[d]["Efull"]; qtr = D[d]["qtr"]; sdt = D[d]["seed_tr"]
                negpool = D[d]["negpool"]; np_ = negpool.shape[0]
                trq = D[d]["TR"][:]; _random.Random(ep * 131 + len(d)).shuffle(trq)
                for qi, gold in trq:
                    V, _ = model.paths(qtr[qi][None], sdt[qi][None])                           # (K,d)
                    G = torch.tensor(gold, device=DEV); Eg = Ef[G]; cos = V @ Eg.T             # (K,g)
                    g = len(gold)
                    if mode == "topk":
                        pairs = [(j, gg, 1.0) for j, gg in _assign_topk(cos, model.K, g)]
                    else:
                        matched, extra = _assign_hungarian((-cos).detach().cpu().numpy(), model.K, g)
                        pairs = [(j, gg, 1.0) for j, gg in matched] + [(j, gg, extra_weight) for j, gg in extra]
                    negs = negpool[torch.randint(0, np_, (256,), device=DEV)]
                    if hard_negs:
                        with torch.no_grad():
                            hn = torch.topk(V @ Ef.T, hn_k, dim=1).indices.reshape(-1)         # per-slot hard negs
                        negs = torch.cat([negs, hn])
                    gold_set = set(gold)
                    keep = torch.tensor([n.item() not in gold_set for n in negs], device=DEV)  # never negate a true gold
                    negs = negs[keep]
                    neg_sc = V @ Ef[negs].T                                                    # (K,nneg)
                    js = [j for j, _, _ in pairs]; ggs = [gg for _, gg, _ in pairs]
                    pos = (V[js] * Eg[ggs]).sum(-1)                                            # (npair,)
                    rows = torch.cat([pos[:, None], neg_sc[js]], 1) / TAU                      # InfoNCE per matched pair
                    wt = torch.tensor([w for *_, w in pairs], device=DEV)
                    ce = F.cross_entropy(rows, torch.zeros(len(pairs), dtype=torch.long, device=DEV), reduction="none")
                    loss = (wt * ce).sum() / max(len(pairs), 1)
                    opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); return model

    # ---------------- eval (open-corpus, per dataset) ----------------
    def _z(x):
        x = np.asarray(x, dtype="float64"); return (x - x.mean()) / (x.std() + 1e-8)

    @torch.no_grad()
    def dense_pool(model, d):
        Ef = D[d]["Efull"]; N = D[d]["N"]; TE = D[d]["TE"]
        Vs = [model.paths(e[0][None], e[1][None])[0] for e in TE]         # e[1] is the precomputed seed vector
        P = Vs[0].shape[0]; Vflat = torch.cat(Vs, 0); nq = len(TE); m = min(M, N)
        chunk = max(10000, min(N, int(3e8 / (nq * max(P, 1)))))
        topv = torch.full((nq, m), -1e9, device=DEV); top = torch.full((nq, m), -1, dtype=torch.long, device=DEV)
        for st0 in range(0, N, chunk):
            ce = Ef[st0:st0 + chunk]
            sc = (ce @ Vflat.T).view(ce.shape[0], nq, P).max(2).values
            v, ix = sc.topk(min(m, sc.shape[0]), dim=0)
            cat_v = torch.cat([topv, v.T], 1); cat_i = torch.cat([top, (ix.T + st0)], 1)
            sel = cat_v.topk(m, 1); topv = sel.values; top = torch.gather(cat_i, 1, sel.indices)
        return top.cpu().numpy(), topv.cpu().numpy(), Vs

    @torch.no_grad()
    def eval_model(model, d, ks=(1, 5, 20, 50)):
        Ef = D[d]["Efull"]; TE = D[d]["TE"]; spl_mat = D[d]["spl_mat"]
        te_qs = D[d]["te_qs"]; te_spl_top = D[d]["te_spl_top"]; has_spl = spl_mat is not None
        ids, cos, Vs = dense_pool(model, d)                       # ONE scan/model, reused across w
        pools = []
        for i, entry in enumerate(TE):
            gold = set(entry[2]); v = ids[i] >= 0; d_ids = ids[i][v]; d_cos = cos[i][v]
            dense_order = d_ids[np.argsort(-d_cos)]
            if has_spl:
                s_top, s_sc = te_spl_top[i]
                pool = np.unique(np.concatenate([d_ids, s_top]))
                cos_map = {int(a): float(b) for a, b in zip(d_ids, d_cos)}
                extra = np.array([p for p in pool if int(p) not in cos_map], dtype=np.int64)
                if len(extra):
                    ec = (Ef[torch.tensor(extra, device=DEV)] @ Vs[i].T).max(1).values.cpu().numpy()
                    for a, b in zip(extra, ec):
                        cos_map[int(a)] = float(b)
                spl_map = {int(a): float(b) for a, b in zip(s_top, s_sc)}
                miss = np.array([p for p in pool if int(p) not in spl_map], dtype=np.int64)
                if len(miss):
                    for a, b in zip(miss, spl_mat[miss].dot(te_qs[i])):
                        spl_map[int(a)] = float(b)
                c = np.array([cos_map[int(p)] for p in pool]); sv = np.array([spl_map[int(p)] for p in pool])
                pools.append((dense_order, pool, _z(c), _z(sv), gold))
            else:
                pools.append((dense_order, None, None, None, gold))
        def bucket(g):                                           # gold-count strata: is beam's value hiding in multi-answer?
            return "g1" if g == 1 else ("g2_4" if g <= 4 else "g5plus")
        res = {}
        for w in ([0.0] if not has_spl else w_splades):
            rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
            strat = {bk: {k: [] for k in ks} for bk in ("g1", "g2_4", "g5plus")}
            for dense_order, pool, cz, sz, gold in pools:
                row = (dense_order if (w == 0 or pool is None) else pool[np.argsort(-(cz + w * sz))]).tolist()
                bk = bucket(len(gold))
                for k in ks:
                    r = len(set(row[:k]) & gold) / len(gold)
                    rec[k].append(r); hit[k].append(1.0 if set(row[:k]) & gold else 0.0); strat[bk][k].append(r)
            strata = {bk: {"n": len(v[ks[0]]), **{f"R@{k}": round(100 * np.mean(v[k]), 2) for k in ks}}
                      for bk, v in strat.items() if v[ks[0]]}                # only non-empty buckets
            res[w] = ({f"R@{k}": round(100 * np.mean(rec[k]), 2) for k in ks},
                      {f"hit@{k}": round(100 * np.mean(hit[k]), 2) for k in ks}, strata)
        return res

    @torch.no_grad()
    def slot_oracle(model, d, k=5):
        """Separate COVERAGE from SELECTION. For each of the K slots take its top-k; report:
          slot_cover@1  = % queries where SOME slot's top-1 is a gold (do the vectors point right at all?)
          oracle_bestslot_R@5 = R@5 if we could pick the single best slot per query (selection upper bound)
          golds_in_Kslot_top1 = mean distinct golds among the K slot top-1s.
        If oracle >> merged R@5 -> SELECTION/weighting fails (vectors are right, ranking buries them).
        If oracle ~ merged (both low) -> COVERAGE/encoder fails (no slot reaches the gold)."""
        Ef = D[d]["Efull"]; TE = D[d]["TE"]
        cov1 = []; ncov = []; best = []
        for e in TE:
            V, _ = model.paths(e[0][None], e[1][None])                        # (P,d)
            if V.shape[0] < 2:
                return None
            topk = (V @ Ef.T).topk(k, dim=1).indices                          # (P,k) each slot's top-k
            gold = set(e[2]); g = len(gold); t1 = set(topk[:, 0].tolist())
            cov1.append(1.0 if (t1 & gold) else 0.0); ncov.append(len(t1 & gold))
            best.append(max(len(set(topk[j].tolist()) & gold) / g for j in range(V.shape[0])))
        return {"slot_cover@1": round(100 * np.mean(cov1), 2),
                "oracle_bestslot_R@5": round(100 * np.mean(best), 2),
                "golds_in_Kslot_top1": round(float(np.mean(ncov)), 2)}

    def report(name, builder):
        """Train+eval `builder()` over n_seeds fresh random seeds; report mean±std per (dataset,w).
        Tiny 159-query webqsp test => single runs are noisy; error bars are the honest instrument."""
        per_seed = []                                            # list over seeds: {tag: (recall, hit, strata)}
        for s in range(n_seeds):
            torch.manual_seed(seed + s); np.random.seed(seed + s)
            model = builder()
            run = {}
            for d in datasets:
                for w, (r, hh, strata) in eval_model(model, d).items():
                    tag = f"{name}|{d}" if w == 0 else f"{name}|{d}+splade(w={w})"
                    run[tag] = (r, hh, strata)
                if s == 0:                                       # COVERAGE-vs-SELECTION diagnostic (set models only)
                    diag = slot_oracle(model, d)
                    if diag is not None:
                        log.info("%-34s ORACLE %s", f"{name}|{d}", diag)
                        out[f"{name}|{d}|slot_oracle"] = diag
            per_seed.append(run)
        for tag in per_seed[0]:
            rs = [ps[tag][0] for ps in per_seed]; hs_ = [ps[tag][1] for ps in per_seed]
            ks_ = list(rs[0]); hk = list(hs_[0])
            mean = {k: round(float(np.mean([r[k] for r in rs])), 2) for k in ks_}
            std = {k: round(float(np.std([r[k] for r in rs])), 2) for k in ks_}
            hmean = {k: round(float(np.mean([h[k] for h in hs_])), 2) for k in hk}
            log.info("%-34s %s  ±%s  %s", tag, mean, {k: std[k] for k in ks_}, hmean)
            out[tag] = {"recall": mean, "recall_std": std, "hit": hmean,
                        "strata": per_seed[0][tag][2], "n_seeds": n_seeds}

    log.info("=== OPEN-CORPUS R@k (agnostic; webqsp test=159, n_seeds=%d -> mean±std) ===", n_seeds)
    out = {}
    if "baselines" in families:
        report("dense", lambda: Dense().to(DEV))
    if "single" in families:
        for h in hs:
            report(f"single_h{h}", (lambda h=h: train_single(Single(h=h).to(DEV))))
    if "set" in families:
        for mode in (["hungarian", "topk"] if assign == "both" else [assign]):
            for K in ks:
                report(f"set_{mode}_K{K}", (lambda K=K, mode=mode: train_set(SetOffset(K=K).to(DEV), mode)))

    import json, os
    os.makedirs("results/L2", exist_ok=True)
    path = f"results/L2/offset_scale_webqsp{out_suffix}.json"
    json.dump(out, open(path, "w"), indent=2, default=str)
    log.info("-> %s  (%d configs)", path, len(out))
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--te-cap", type=int, default=2000)
    p.add_argument("--tr-cap", type=int, default=3000)
    p.add_argument("--assign", choices=["hungarian", "topk", "both"], default="both")
    p.add_argument("--families", nargs="+", default=["baselines", "single", "set"])
    p.add_argument("--ks", type=int, nargs="+", default=[5, 8, 12, 16])
    p.add_argument("--hs", type=int, nargs="+", default=[768, 1536, 3072])
    p.add_argument("--w-splades", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--seed-mode", choices=["dense", "topic"], default="dense")
    p.add_argument("--datasets", nargs="+", default=["webqsp"],
                   help="one shared model across these (dataset-agnostic). Dev: webqsp. Joint: add text corpora.")
    p.add_argument("--no-hard-negs", action="store_true", help="use random negatives only (ablation)")
    p.add_argument("--sym-single", action="store_true",
                   help="mask same-query co-answers from single's in-batch negatives (fully symmetric w/ set)")
    p.add_argument("--out-suffix", default="", help="suffix for the results filename (parallel runs)")
    p.add_argument("--single-loss", choices=["triple", "multipos"], default="triple",
                   help="single's training loss: 'triple' (in-batch+hard negs) or 'multipos' (centroid-of-golds, the proven 35.12 recipe)")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (reproducible runs)")
    p.add_argument("--n-seeds", type=int, default=1, help="repeat each config over N seeds -> report mean±std (needed: 159-query test)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(epochs=a.epochs, te_cap=a.te_cap, tr_cap=a.tr_cap, assign=a.assign, families=tuple(a.families),
         ks=tuple(a.ks), hs=tuple(a.hs), w_splades=tuple(a.w_splades), seed_mode=a.seed_mode,
         datasets=tuple(a.datasets), hard_negs=not a.no_hard_negs, sym_single=a.sym_single,
         out_suffix=a.out_suffix, single_loss=a.single_loss, seed=a.seed, n_seeds=a.n_seeds)


if __name__ == "__main__":
    main()
