"""Experiment 2 — LEARNED doc reranker over a high-recall candidate pool.

Exp 1 showed candidates aren't the problem (graph reaches 86% at doc level) — ranking them at doc granularity
is. `max(dense, offset)` gets only ~9-30 R@5. So learn it: build a budgeted pool = dense-top-Nd UNION graph
2-hop neighbourhood of the dense seeds (high recall, ~few hundred docs), compute cheap per-candidate features,
and train a tiny MLP to score relevance. The pool's structurally-related-but-wrong docs (Jamaica->capital,
Jamaica->currency for a language question) are the hard negatives — exactly what forces the reranker to read
the requested relationship. Open-corpus WebQSP; reports pool recall (ceiling) vs reranked R@k vs baselines.

Features per candidate doc (all cheap scalars, masked to 0 when a signal is absent -> dataset-agnostic):
  dense q·d | offset v·d | prototype q·centroid(entity(d)) | graph hop (0/1/2) | splade lexical q·d.
"""
import json
import logging
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
TAU = 0.07


def _run(off_epochs=25, rr_epochs=8, n_dense=100, n_seed=10, hops=2, cap=32, tr_cap=1104, te_cap=2000,
         add_proto=False, proto_M=20, dataset="webqsp"):
    import os, re
    from src.experiments.l1_universal_head import CoreEngine, load_docs_and_encoder, _load
    from src.experiments.l1l3_recall import _graph
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = _load(dataset, "gte_qwen", 8000, tr_cap, te_cap)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]; splade = d.get("splade")
    qtr, str_tr, gtr = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); N, DIM = X.shape; del X, d

    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    _, A, _ = _graph(eng, N, id2idx, sources=("struct", "syn")); A = A.tocsr()

    # entity/prototype centroids from doc titles (dataset-agnostic: title groups docs; degrades to 1-doc groups)
    mfile = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mfile):
        mfile = "data/processed/master_nodes.json"                       # metaqa lives in the canonical master
    t2r = collections.defaultdict(list); t2r_lc = collections.defaultdict(list)
    for n in json.load(open(mfile, encoding="utf-8")):
        nid = n["node_id"]
        if nid in id2idx:
            ti = n.get("metadata", {}).get("title", "").strip()
            if ti:
                t2r[ti].append(id2idx[nid]); t2r_lc[ti.lower()].append(id2idx[nid])
    ents = [e for e in t2r if e]; ent2i = {e: i for i, e in enumerate(ents)}
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for e, i in ent2i.items():
        doc2ent[torch.tensor(t2r[e], device=DEV)] = i
    m = doc2ent >= 0
    Eent = torch.zeros(len(ents), DIM, device=DEV).index_add_(0, doc2ent[m], Efull[m])
    c = torch.zeros(len(ents), device=DEV).index_add_(0, doc2ent[m], torch.ones(m.sum(), device=DEV))
    Eent = F.normalize(Eent / c.clamp(min=1)[:, None], dim=1)

    # topic-entity rows per question (dataset-specific extraction; dense-top1 fallback otherwise)
    q2rows = {}
    if dataset == "webqsp":
        import pandas as pd
        df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
        for _, r in df.iterrows():
            q2rows[str(r["question"])] = [rr for e in r["q_entity"] for rr in t2r.get(str(e).strip(), [])]
    elif dataset == "metaqa":
        for txt in list(tr_txt) + list(te_txt):                          # topic entity is in [brackets]
            mm = re.findall(r"\[(.*?)\]", txt)
            q2rows[txt] = [rr for e in mm for rr in t2r_lc.get(e.strip().lower(), [])]
    def topic(txt, sd):
        rows = q2rows.get(txt, [])
        return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0) if rows else Efull[int(sd)]

    Qtr = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    Qte = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    Htr = torch.stack([topic(tr_txt[i], str_tr[i]) for i in range(len(tr_txt))])
    Hte = torch.stack([topic(te_txt[i], ste[i]) for i in range(len(te_txt))])
    Gtr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    Gte = [[int(x) for x in gte[i]] for i in range(len(gte))]

    # splade doc scores (aligned) — optional feature
    spl_scorer, spl_mat = (splade if splade else (None, None))
    def splade_scores(txt, ids):
        if spl_scorer is None:
            return np.zeros(len(ids), "float32")
        qv = spl_scorer.encode_query(txt)
        return spl_mat[ids].dot(qv).astype("float32")

    # a small QA offset for the "offset" feature: v = norm(topic + MLP([topic,q])), trained doc-level
    class Off(nn.Module):
        def __init__(s):
            super().__init__(); s.m = nn.Sequential(nn.Linear(2 * DIM, 1024), nn.GELU(), nn.Linear(1024, DIM))
        def forward(s, hd, q):
            return F.normalize(hd + s.m(torch.cat([hd, q], -1)), dim=-1)
    off = Off().to(DEV); oo = torch.optim.Adam(off.parameters(), lr=1e-3)
    negp = torch.randint(0, N, (60000,), device=DEV)
    idx = [i for i in range(len(Gtr)) if Gtr[i]]
    for ep in range(off_epochs):
        np.random.shuffle(idx)
        for i in idx:
            v = off(Htr[i][None], Qtr[i][None])
            cand = torch.cat([torch.tensor(Gtr[i], device=DEV), negp[torch.randint(0, 60000, (256,), device=DEV)]])
            sc = (Efull[cand] @ v.T).squeeze(-1)
            tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Gtr[i])] = 1.0 / len(Gtr[i])
            loss = -(F.log_softmax(sc, 0) * tgt).sum()
            oo.zero_grad(); loss.backward(); oo.step()
    off.eval()

    @torch.no_grad()
    def build_pool(q, h, txt):
        sim = q @ Efull.T
        order = torch.topk(sim, max(n_dense, n_seed)).indices.cpu().numpy()
        seeds = [int(x) for x in order[:n_seed]]
        dense_pool = [int(x) for x in order[:n_dense]]
        hop = {}
        for s in dense_pool:
            hop[s] = 0
        frontier = set(seeds)
        for hh in range(1, hops + 1):
            nxt = set()
            for s in frontier:
                nb = A.indices[A.indptr[s]:A.indptr[s + 1]][:cap]
                for x in nb:
                    x = int(x)
                    if x not in hop:
                        hop[x] = hh; nxt.add(x)
            frontier = nxt
        v = off(h[None], q[None])[0]
        if add_proto:                                                    # Exp 3: union in prototype-expanded docs
            top_e = torch.topk(Eent @ v, proto_M).indices.tolist()
            for e in top_e:
                for x in t2r[ents[e]]:
                    if x not in hop:
                        hop[x] = 1                                       # treat proto-expanded as near
        pool = list(hop.keys())
        P = torch.tensor(pool, device=DEV)
        dp = Efull[P]
        feats = torch.stack([
            dp @ q,                                                        # dense
            dp @ v,                                                        # offset
            (Eent[doc2ent[P].clamp(min=0)] * q).sum(1) * (doc2ent[P] >= 0),# prototype (0 if no entity)
            torch.tensor([hop[p] for p in pool], device=DEV, dtype=torch.float32) / hops,  # graph hop (norm)
            torch.tensor(splade_scores(txt, np.array(pool)), device=DEV),  # splade lexical
        ], 1)                                                             # (|pool|, 5)
        return pool, feats

    # ---- cache pools ONCE (offset is fixed), then seed-loop the reranker for error bars ----
    residual = globals().get("_RESIDUAL", False); n_seeds = globals().get("_N_SEEDS", 1)
    ks = (1, 5, 20, 50)
    log.info("[kg-rerank] building train pools (%d) ...", len(idx))
    train_pools = []
    for i in idx:
        pool, feats = build_pool(Qtr[i], Htr[i], tr_txt[i])
        y = torch.tensor([1.0 if p in set(Gtr[i]) else 0.0 for p in pool], device=DEV)
        if y.sum() > 0:
            train_pools.append((feats, y))
    allf = torch.cat([f for f, _ in train_pools], 0)
    mu, sd = allf.mean(0), allf.std(0).clamp(min=1e-6)
    test_pools = []                                                        # (pool_ids, feats, gold_set)
    ceil = []
    for i in range(len(Gte)):
        if not Gte[i]:
            continue
        pool, feats = build_pool(Qte[i], Hte[i], te_txt[i])
        gs = set(Gte[i]); ceil.append(len(gs & set(pool)) / len(gs))
        test_pools.append((pool, feats, gs))
    ceil = round(100 * np.mean(ceil), 2)

    class RR(nn.Module):
        def __init__(s, f=5):
            super().__init__(); s.m = nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Linear(64, 1))
            if residual:                                                   # start AS dense: correction=0 at init
                nn.init.zeros_(s.m[-1].weight); nn.init.zeros_(s.m[-1].bias)
        def forward(s, x, dense):
            corr = s.m(x).squeeze(-1)
            return dense + corr if residual else corr                     # residual: score = dense + learned Δ

    def run_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        rr = RR().to(DEV); ro = torch.optim.Adam(rr.parameters(), lr=1e-3)
        order = list(range(len(train_pools)))
        for ep in range(rr_epochs):
            np.random.shuffle(order)
            for j in order:
                feats, y = train_pools[j]
                sco = rr((feats - mu) / sd, feats[:, 0])
                loss = -(F.log_softmax(sco, 0) * (y / y.sum())).sum()
                ro.zero_grad(); loss.backward(); ro.step()
        rr.eval()
        rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        with torch.no_grad():
            for pool, feats, gs in test_pools:
                sco = rr((feats - mu) / sd, feats[:, 0]).cpu().numpy()
                row = [pool[j] for j in np.argsort(-sco)]
                for k in ks:
                    t = set(row[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        return ({k: 100 * np.mean(rec[k]) for k in ks}, {k: 100 * np.mean(hit[k]) for k in ks})

    seed_res = [run_seed(s) for s in range(n_seeds)]
    rmean = {f"R@{k}": round(float(np.mean([sr[0][k] for sr in seed_res])), 2) for k in ks}
    rstd = {f"R@{k}": round(float(np.std([sr[0][k] for sr in seed_res])), 2) for k in ks}
    hmean = {f"hit@{k}": round(float(np.mean([sr[1][k] for sr in seed_res])), 2) for k in ks}
    # dense-over-pool baseline (fixed)
    drec = {k: [] for k in ks}
    for pool, feats, gs in test_pools:
        row = [pool[j] for j in np.argsort(-feats[:, 0].cpu().numpy())]
        for k in ks:
            drec[k].append(len(set(row[:k]) & gs) / len(gs))
    dbase = {f"R@{k}": round(100 * np.mean(drec[k]), 2) for k in ks}

    out = {"pool_ceiling": ceil, "residual": residual, "n_seeds": n_seeds,
           "reranker": {"recall": rmean, "recall_std": rstd, "hit": hmean}, "dense_over_pool": dbase,
           "n_dense": n_dense, "n_seed": n_seed, "hops": hops, "cap": cap}
    log.info("[kg-rerank] ceiling=%.1f residual=%s | LEARNED %s ±%s | dense-pool %s",
             ceil, residual, rmean, rstd, dbase)
    import os
    os.makedirs("results/L2", exist_ok=True)
    suf = globals().get("_OUT_SUFFIX", "") or ""
    path = f"results/L2/kg_rerank_{dataset}{suf}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--rr-epochs", type=int, default=8)
    p.add_argument("--n-dense", type=int, default=100)
    p.add_argument("--n-seed", type=int, default=10)
    p.add_argument("--cap", type=int, default=32)
    p.add_argument("--add-proto", action="store_true")
    p.add_argument("--proto-M", type=int, default=20)
    p.add_argument("--dataset", default="webqsp")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--residual", action="store_true", help="score = dense + learned Δ (never underperforms dense)")
    p.add_argument("--n-seeds", type=int, default=1, help="repeat reranker train/eval over N seeds -> mean±std")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    global _OUT_SUFFIX, _RESIDUAL, _N_SEEDS
    _OUT_SUFFIX = a.out_suffix; _RESIDUAL = a.residual; _N_SEEDS = a.n_seeds
    _run(off_epochs=a.off_epochs, rr_epochs=a.rr_epochs, n_dense=a.n_dense, n_seed=a.n_seed,
         cap=a.cap, add_proto=a.add_proto, proto_M=a.proto_M, dataset=a.dataset)


if __name__ == "__main__":
    main()
