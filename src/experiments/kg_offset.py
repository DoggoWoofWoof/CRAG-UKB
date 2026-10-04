"""KG relation-offset — learn the graph's relational structure INTO an MLP's weights, from the TRIPLES
themselves (2.27M WebQSP edges), not from sparse QA pairs.

    tail_emb  ~=  normalize( head_emb + MLP([head_emb, rel_emb]) )

Scalable where a GNN is not: O(1) at inference, no per-query message passing, no oversmoothing. The MLP
amortizes "from head, along relation r, reach tail" over every edge; multi-hop = COMPOSE the offset (apply
it twice). Decisive test = LINK PREDICTION on held-out edges: does head+offset rank the true tail among all
entities? If yes, the graph is encodable in the weights (the whole thesis); then bridge to QA.

Entity embedding = centroid of that entity's fact-docs in the frozen gte-Qwen2 space (entities are shattered
into "<entity>. <relation> <value>" docs; the centroid is the entity's location). Relations = a learned
embedding table. Reports single-hop link-pred R@k and a 2-hop COMPOSITION probe (apply offset twice).
"""
import json
import logging
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
TAU = 0.05


def _run(epochs=8, bs=1024, h=1024, hn_k=16, test_frac=0.1, max_ent=0):
    import pandas as pd
    from src.experiments.l1_universal_head import _load
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = _load("webqsp", "gte_qwen", 8000, 3000, 2000)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]; del d
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); del X

    # entity -> corpus rows (by fact-doc title), then entity centroid embedding
    mn = json.load(open("data/processed/master_nodes_webqsp.json", encoding="utf-8"))
    title2rows = collections.defaultdict(list)
    for n in mn:
        nid = n["node_id"]
        if nid in id2idx:
            title2rows[n.get("metadata", {}).get("title", "").strip()].append(id2idx[nid])
    del mn
    ents = [e for e in title2rows if e]                                    # entities that have >=1 doc
    ent2i = {e: i for i, e in enumerate(ents)}
    Eent = torch.stack([F.normalize(Efull[torch.tensor(title2rows[e], device=DEV)].mean(0), dim=0) for e in ents])
    log.info("[kg-offset] entities-with-docs=%d  (of corpus %d docs)", len(ents), Efull.shape[0])
    del Efull

    # triples with head+tail both resolvable to an entity centroid
    df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
    rel2i = {}; triples = []
    for g in df["graph"]:
        for t in g:
            hn, rn, tn = str(t[0]).strip(), str(t[1]), str(t[2]).strip()
            if hn in ent2i and tn in ent2i:
                r = rel2i.setdefault(rn, len(rel2i))
                triples.append((ent2i[hn], r, ent2i[tn]))
    triples = np.array(sorted(set(triples)), dtype=np.int64)
    n_rel = len(rel2i); n_ent = len(ents)
    rng = np.random.default_rng(0); rng.shuffle(triples)
    n_te = int(len(triples) * test_frac)
    te = triples[:n_te]; tr = triples[n_te:]
    log.info("[kg-offset] triples: train=%d test=%d | entities=%d relations=%d", len(tr), len(te), n_ent, n_rel)

    class RelOffset(nn.Module):
        def __init__(s, dim):
            super().__init__()
            s.rel = nn.Embedding(n_rel, dim)
            s.mlp = nn.Sequential(nn.Linear(2 * dim, h), nn.GELU(), nn.Linear(h, dim))
            nn.init.normal_(s.rel.weight, std=0.02)
        def forward(s, head, rel_idx):                                     # head:(B,d) rel_idx:(B,)
            r = s.rel(rel_idx)
            return F.normalize(head + s.mlp(torch.cat([head, r], -1)), dim=-1)

    dim = Eent.shape[1]; model = RelOffset(dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trt = torch.tensor(tr, device=DEV)
    for ep in range(epochs):
        perm = torch.randperm(len(trt), device=DEV)
        tot = 0.0
        for st in range(0, len(trt), bs):
            b = trt[perm[st:st + bs]]
            head, rel, tail = Eent[b[:, 0]], b[:, 1], b[:, 2]
            v = model(head, rel)                                           # (B,d)
            logits = v @ Eent[b[:, 2]].T / TAU                             # in-batch negatives (cols = batch tails)
            with torch.no_grad():
                neg = torch.topk(v @ Eent.T, hn_k, dim=1).indices.reshape(-1)   # corpus-wide hard negs
            hard = (v @ Eent[neg].T / TAU).masked_fill(neg.unsqueeze(0) == b[:, 2:3], -1e4)
            logits = torch.cat([logits, hard], 1)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        log.info("[kg-offset] epoch %d  loss %.4f", ep, tot / max(1, len(trt) // bs))
    model.eval()

    @torch.no_grad()
    def linkpred(pairs, compose2=False, chunk=200000):
        """Rank the true tail among ALL entities for each (head,rel[,rel2]) — held-out edges."""
        ks = (1, 5, 10, 50); hit = {k: 0 for k in ks}; n = 0
        for st in range(0, len(pairs), 4096):
            b = torch.tensor(pairs[st:st + 4096], device=DEV)
            v = model(Eent[b[:, 0]], b[:, 1])
            if compose2:
                v = model(v, b[:, 3])                                      # apply offset again (2-hop)
            tail = b[:, 2]
            ranks = torch.zeros(len(b), device=DEV)
            true_score = (v * Eent[tail]).sum(1)
            for c0 in range(0, n_ent, chunk):                             # count entities scoring above the true tail
                ranks += (Eent[c0:c0 + chunk] @ v.T > true_score[None, :]).sum(0)
            for k in ks:
                hit[k] += (ranks < k).sum().item()
            n += len(b)
        return {f"R@{k}": round(100 * hit[k] / n, 2) for k in ks}

    r1 = linkpred(te)
    log.info("[kg-offset] SINGLE-HOP link-pred (held-out edges, rank tail among %d): %s", n_ent, r1)

    # 2-hop composition probe: paths h -r1-> m -r2-> t (m has a doc); can offset-compose reach t?
    by_head = collections.defaultdict(list)
    for hh, rr, tt in tr[:400000]:
        by_head[hh].append((rr, tt))
    paths = []
    for hh, rr, mm in te:
        for rr2, tt in by_head.get(mm, [])[:2]:
            paths.append((hh, rr, tt, rr2))
        if len(paths) >= 5000:
            break
    out = {"entities": n_ent, "relations": n_rel, "n_train": len(tr), "n_test": len(te), "single_hop": r1}
    if paths:
        r2 = linkpred(np.array(paths, dtype=np.int64), compose2=True)
        out["two_hop_compose"] = r2; out["n_2hop_paths"] = len(paths)
        log.info("[kg-offset] TWO-HOP compose (%d paths): %s", len(paths), r2)

    import os
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open("results/L2/kg_offset_webqsp.json", "w"), indent=2)
    log.info("-> results/L2/kg_offset_webqsp.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=1024)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(epochs=a.epochs, bs=a.bs)


if __name__ == "__main__":
    main()
