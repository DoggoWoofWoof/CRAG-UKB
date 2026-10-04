"""Dataset-agnostic GATED JOINT reranker — engineering queue changes 1-4.

Findings so far: a free reranker over a dense∪graph pool BEATS the benchmark on KBs (WebQSP 29.7->35.9,
MetaQA R@50 77.5->88.9) but HURTS text (Hotpot 61.6 < dense 69.9), and a hard dense-residual fixes text but
BREAKS KB (MetaQA 69->59, because dense is a weak base there). So:

  Change 1 GATED scorer     : score = (1-g)·dense + g·MLP(features), g = sigmoid(gate(pool-summary)).
                              Text (dense strong) -> g→0 -> fall back to dense; KB (dense weak) -> g→1 -> free MLP.
  Change 2 JOINT training   : one reranker over WebQSP+MetaQA+HotpotQA pools -> the gate learns the regime.
  Change 3 SOURCE features  : from_dense / from_graph / from_proto / n_sources -> agreement becomes evidence.
  Change 4 LOCAL proto      : prototype -> score its docs by q -> keep top-3 (not all) -> retain recall w/o dilution.

Loads one corpus at a time, builds pool features, frees it (memory-safe across 3 corpora). Reports per-dataset
reranked R@k (mean±std over seeds) vs dense-over-pool and the benchmark.
"""
import json
import logging
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
NF = 9   # dense, offset, proto, hop, splade, from_dense, from_graph, from_proto, n_sources


def _build_pools(dataset, off_epochs, n_dense, n_seed, hops, cap, proto_M, proto_top, tr_cap, te_cap, DEV):
    """Load one corpus, train its offset, build train/test pool FEATURES, free the corpus. Returns CPU tensors."""
    import os, re
    from src.experiments.l1_universal_head import CoreEngine, load_docs_and_encoder, _load
    from src.experiments.l1l3_recall import _graph
    d = _load(dataset, "gte_qwen", 8000, tr_cap, te_cap)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]; splade = d.get("splade")
    qtr, str_tr, gtr = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); N, DIM = X.shape; del X, d
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    _, A, _ = _graph(eng, N, id2idx, sources=("struct", "syn")); A = A.tocsr()

    mfile = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mfile):
        mfile = "data/processed/master_nodes.json"
    t2r = collections.defaultdict(list); t2r_lc = collections.defaultdict(list)
    for n in json.load(open(mfile, encoding="utf-8")):
        if n["node_id"] in id2idx:
            ti = n.get("metadata", {}).get("title", "").strip()
            if ti:
                t2r[ti].append(id2idx[n["node_id"]]); t2r_lc[ti.lower()].append(id2idx[n["node_id"]])
    ents = [e for e in t2r if e]; ent2i = {e: i for i, e in enumerate(ents)}
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for e, i in ent2i.items():
        doc2ent[torch.tensor(t2r[e], device=DEV)] = i
    m = doc2ent >= 0
    Eent = torch.zeros(len(ents), DIM, device=DEV).index_add_(0, doc2ent[m], Efull[m])
    cc = torch.zeros(len(ents), device=DEV).index_add_(0, doc2ent[m], torch.ones(m.sum(), device=DEV))
    Eent = F.normalize(Eent / cc.clamp(min=1)[:, None], dim=1)

    q2rows = {}
    if dataset == "webqsp":
        import pandas as pd
        df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
        for _, r in df.iterrows():
            q2rows[str(r["question"])] = [rr for e in r["q_entity"] for rr in t2r.get(str(e).strip(), [])]
    elif dataset == "metaqa":
        for txt in list(tr_txt) + list(te_txt):
            q2rows[txt] = [rr for e in re.findall(r"\[(.*?)\]", txt) for rr in t2r_lc.get(e.strip().lower(), [])]

    def topic(txt, sd):
        rows = q2rows.get(txt, [])
        return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0) if rows else Efull[int(sd)]
    Qtr = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    Qte = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    Htr = torch.stack([topic(tr_txt[i], str_tr[i]) for i in range(len(tr_txt))])
    Hte = torch.stack([topic(te_txt[i], ste[i]) for i in range(len(te_txt))])
    Gtr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    Gte = [[int(x) for x in gte[i]] for i in range(len(gte))]
    spl_scorer, spl_mat = (splade if splade else (None, None))

    class Off(nn.Module):
        def __init__(s):
            super().__init__(); s.m = nn.Sequential(nn.Linear(2 * DIM, 1024), nn.GELU(), nn.Linear(1024, DIM))
        def forward(s, hd, q): return F.normalize(hd + s.m(torch.cat([hd, q], -1)), dim=-1)
    off = Off().to(DEV); oo = torch.optim.Adam(off.parameters(), lr=1e-3)
    negp = torch.randint(0, N, (60000,), device=DEV); tri = [i for i in range(len(Gtr)) if Gtr[i]]
    for ep in range(off_epochs):
        np.random.shuffle(tri)
        for i in tri:
            v = off(Htr[i][None], Qtr[i][None])
            cand = torch.cat([torch.tensor(Gtr[i], device=DEV), negp[torch.randint(0, 60000, (256,), device=DEV)]])
            sc = (Efull[cand] @ v.T).squeeze(-1)
            tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Gtr[i])] = 1.0 / len(Gtr[i])
            loss = -(F.log_softmax(sc, 0) * tgt).sum(); oo.zero_grad(); loss.backward(); oo.step()
    off.eval()

    @torch.no_grad()
    def feats_for(q, h, txt):
        sim = q @ Efull.T
        order = torch.topk(sim, max(n_dense, n_seed)).indices.cpu().numpy()
        dense_set = set(int(x) for x in order[:n_dense]); seeds = [int(x) for x in order[:n_seed]]
        src = collections.defaultdict(lambda: [0, 0, 0])                   # doc -> [from_dense, from_graph, from_proto]
        hop = {}
        for x in dense_set:
            src[x][0] = 1; hop[x] = 0
        frontier = set(seeds)
        for hh in range(1, hops + 1):
            nxt = set()
            for s in frontier:
                for x in A.indices[A.indptr[s]:A.indptr[s + 1]][:cap]:
                    x = int(x)
                    if x not in hop:
                        hop[x] = hh; nxt.add(x)
                    src[x][1] = 1
            frontier = nxt
        v = off(h[None], q[None])[0]
        if proto_M > 0:                                                    # Change 4: local proto expansion (top-`proto_top` docs/entity)
            for e in torch.topk(Eent @ v, proto_M).indices.tolist():
                rws = t2r[ents[e]]
                if not rws:
                    continue
                rt = torch.tensor(rws, device=DEV)
                top = rt[torch.argsort(-(Efull[rt] @ q))[:proto_top]].tolist()
                for x in top:
                    src[x][2] = 1
                    if x not in hop:
                        hop[x] = 1
        pool = list(hop.keys()); P = torch.tensor(pool, device=DEV); dp = Efull[P]
        S = torch.tensor([src[p] for p in pool], device=DEV, dtype=torch.float32)   # (n,3)
        feats = torch.stack([
            dp @ q, dp @ v,
            (Eent[doc2ent[P].clamp(min=0)] * q).sum(1) * (doc2ent[P] >= 0),
            torch.tensor([hop[p] for p in pool], device=DEV, dtype=torch.float32) / max(hops, 1),
            torch.tensor(spl_mat[np.array(pool)].dot(spl_scorer.encode_query(txt)) if spl_scorer else np.zeros(len(pool)),
                         device=DEV, dtype=torch.float32),
            S[:, 0], S[:, 1], S[:, 2], S.sum(1) / 3.0,
        ], 1)
        return pool, feats

    tr_data = []
    for i in tri:
        pool, feats = feats_for(Qtr[i], Htr[i], tr_txt[i])
        y = torch.tensor([1.0 if p in set(Gtr[i]) else 0.0 for p in pool], device=DEV)
        if y.sum() > 0:
            tr_data.append((feats.cpu(), y.cpu()))
    te_data = []
    for i in range(len(Gte)):
        if not Gte[i]:
            continue
        pool, feats = feats_for(Qte[i], Hte[i], te_txt[i])
        gm = torch.tensor([1.0 if p in set(Gte[i]) else 0.0 for p in pool])
        te_data.append((feats.cpu(), gm, len(Gte[i])))
    del Efull, Eent, eng, A
    import gc; gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    ceil = round(100 * float(np.mean([gm.sum().item() / ng for _, gm, ng in te_data])), 2)
    log.info("[kg-joint] %s: train pools=%d test pools=%d pool_ceiling=%.1f", dataset, len(tr_data), len(te_data), ceil)
    return tr_data, te_data, ceil


class Gated(nn.Module):
    """score = (1-g)·dense_z + g·MLP(feats); g = sigmoid(gate(pool summary)). Text->g≈0 (dense), KB->g≈1 (MLP)."""
    def __init__(s, f=NF):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Linear(64, 1))
        s.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(s, x, mu, sd):
        z = (x - mu) / sd
        dense_z = z[:, 0]
        summ = torch.stack([z[:, 0].max(), z[:, 0].mean(), torch.tensor(float(len(x)), device=x.device).log()])
        g = torch.sigmoid(s.gate(summ))
        return (1 - g) * dense_z + g * s.m(z).squeeze(-1)


def _run(off_epochs=25, rr_epochs=8, n_dense=100, n_seed=20, hops=2, cap=64, proto_M=20, proto_top=3,
         tr_cap=1104, te_cap=2000, datasets=("webqsp", "metaqa", "hotpotqa_clean"), n_seeds=3):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA = {}
    for ds in datasets:
        DATA[ds] = _build_pools(ds, off_epochs, n_dense, n_seed, hops, cap, proto_M, proto_top, tr_cap, te_cap, DEV)
    allf = torch.cat([f for ds in datasets for f, _ in DATA[ds][0]], 0)
    mu = allf.mean(0).to(DEV); sd = allf.std(0).clamp(min=1e-6).to(DEV)
    ks = (1, 5, 20, 50)

    def one_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        rr = Gated().to(DEV); opt = torch.optim.Adam(rr.parameters(), lr=1e-3)
        pool = [(f.to(DEV), y.to(DEV)) for ds in datasets for f, y in DATA[ds][0]]   # joint: all datasets' train pools
        for ep in range(rr_epochs):
            np.random.shuffle(pool)
            for f, y in pool:
                s = rr(f, mu, sd)
                loss = -(F.log_softmax(s, 0) * (y / y.sum())).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        rr.eval()
        res = {}
        with torch.no_grad():
            for ds in datasets:
                rec = {k: [] for k in ks}
                for f, gm, ng in DATA[ds][1]:
                    s = rr(f.to(DEV), mu, sd).cpu().numpy()
                    ordr = np.argsort(-s); gmn = gm.numpy()
                    for k in ks:
                        rec[k].append(gmn[ordr[:k]].sum() / ng)
                res[ds] = {k: 100 * np.mean(rec[k]) for k in ks}
        return res

    seed_res = [one_seed(s) for s in range(n_seeds)]
    out = {}
    for ds in datasets:
        mean = {f"R@{k}": round(float(np.mean([sr[ds][k] for sr in seed_res])), 2) for k in ks}
        std = {f"R@{k}": round(float(np.std([sr[ds][k] for sr in seed_res])), 2) for k in ks}
        # dense-over-pool baseline for this dataset
        drec = {k: [] for k in ks}
        for f, gm, ng in DATA[ds][1]:
            ordr = np.argsort(-f[:, 0].numpy()); gmn = gm.numpy()
            for k in ks:
                drec[k].append(gmn[ordr[:k]].sum() / ng)
        out[ds] = {"pool_ceiling": DATA[ds][2], "reranker": {"recall": mean, "recall_std": std},
                   "dense_over_pool": {f"R@{k}": round(100 * np.mean(drec[k]), 2) for k in ks}}
        log.info("[kg-joint] %s: ceiling=%.1f | JOINT-GATED %s ±%s | dense-pool %s",
                 ds, DATA[ds][2], mean, std, out[ds]["dense_over_pool"])
    import os
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open("results/L2/kg_joint.json", "w"), indent=2)
    log.info("-> results/L2/kg_joint.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--rr-epochs", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--datasets", nargs="+", default=["webqsp", "metaqa", "hotpotqa_clean"])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(off_epochs=a.off_epochs, rr_epochs=a.rr_epochs, n_seeds=a.n_seeds, datasets=tuple(a.datasets))


if __name__ == "__main__":
    main()
