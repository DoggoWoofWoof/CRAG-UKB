"""Offset-head architecture sweep for WebQSP, on GPU. Validates on OPEN-corpus R@5 (rank golds among ALL
781k entities — the real metric; peak to beat = 29.7). Compares:
  - dense (no head)                         seeds: dense-top1 / topic-entity
  - single-offset (== rel_hard)             seeds: dense-top1 / topic-entity
  - BEAM multi-fusion (K branches/level, L levels -> beam of DISTINCT predicted vectors, weighted-MAX fuse;
    NOT summed/smudged) — the multi-hop, multi-answer head.
Topic entity is NER-linkable from the question (q_entity), not an oracle. GPU makes the 12-vec beam trivial.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _run(epochs=40, seeds_negpool=60000):
    import pandas as pd
    from src.experiments.l1_universal_head import _load
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATHS = ["data/raw/full/kb/webqsp_test0.parquet", "data/raw/full/kb/webqsp_test1.parquet"]

    df = pd.concat([pd.read_parquet(p) for p in PATHS], ignore_index=True)
    tri = set()
    for g in df["graph"]:
        for t in g:
            tri.add((str(t[0]), str(t[2])))
    ents = set()
    for h, tl in tri:
        ents.add(h); ents.add(tl)
    ent2idx = {e: i for i, e in enumerate(sorted(ents))}
    q2topic = {str(r["question"]): [ent2idx[str(e)] for e in r["q_entity"] if str(e) in ent2idx]
               for _, r in df.iterrows()}

    d = _load("webqsp", "gte_qwen", 8000, 3000, 2000)
    qtr, str_tr, gtr = d["train"]; qte, str_te, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    qtr = qtr.astype("float32"); qte = qte.astype("float32")
    X = d["X"].astype("float32"); del d
    N, DIM = X.shape
    Efull = torch.tensor(X, device=DEV); Efull = F.normalize(Efull, dim=1)     # whole corpus on GPU
    del X
    log.info("[offset-beam] corpus N=%d on %s", N, DEV)

    def pack(texts, qv, sd, gd, cap):
        out = []
        for i in range(min(len(texts), cap)):
            topic = q2topic.get(texts[i], []); g = [int(x) for x in gd[i]]
            if not g:
                continue
            out.append((F.normalize(torch.tensor(qv[i], device=DEV), dim=0),
                        topic[0] if topic else int(sd[i]), int(sd[i]), g))
        return out
    TR = pack(tr_txt, qtr, str_tr, gtr, 3000); TE = pack(te_txt, qte, str_te, gte, 2000)
    negpool = torch.randint(0, N, (seeds_negpool,), device=DEV)
    log.info("[offset-beam] train=%d test=%d", len(TR), len(TE))

    class Dense(nn.Module):                                    # no head; seed-independent (dense retrieval)
        def paths(s, q, seed): return q, torch.zeros(1, device=q.device)
    class Single(nn.Module):
        def __init__(s, d=DIM, h=768):
            super().__init__(); s.m = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, d))
        def paths(s, q, seed):                                 # q,seed:(1,d) -> V:(1,d), logW:(1,)
            v = F.normalize(q + s.m(torch.cat([q, seed], -1)), dim=-1)
            return v, torch.zeros(1, device=q.device)

    class Beam(nn.Module):
        def __init__(s, d=DIM, h=512, K=3, L=2):
            super().__init__(); s.K, s.L = K, L
            s.off = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, d))
                                                  for _ in range(K)]) for _ in range(L)])
        def paths(s, q, seed):                                 # -> V:(nvec,d), logW:(nvec,) uniform (0)
            out = []; beam = [q]
            for l in range(s.L):
                nb = []
                for v in beam:
                    for k in range(s.K):
                        vk = F.normalize(v + s.off[l][k](torch.cat([v, seed], -1)), dim=-1)
                        nb.append(vk); out.append(vk)
                beam = nb
            V = torch.cat(out, 0)
            return V, torch.zeros(V.shape[0], device=q.device)

    class LearnedBeam(nn.Module):
        """The full architecture (VECTORIZED): at each level a query-conditioned MLP emits ALL K branch offsets
        at once; fan out from each kept path -> B*K candidate vectors; a scorer PRUNES to beam width B (learned
        pruning); COLLECT survivors from EVERY level (variable effective depth N — a gold reachable in 1/2/3
        hops is kept); retrieve independently with each; FUSE by weighted-max with a learned query-conditioned
        weight per path (so a bad path can't dilute @5). Paths stay SEPARATE vectors — never summed/smudged.
        The MLP is literally learning to DECOMPOSE the query into K meaningful sub-vectors on the fly."""
        def __init__(s, d=DIM, h=512, K=3, L=3, B=6):
            super().__init__(); s.K, s.L, s.B, s.d = K, L, B, d
            s.off = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, K * d)) for _ in range(L)])
            s.score = nn.Sequential(nn.Linear(3 * d, h), nn.GELU(), nn.Linear(h, 1))   # (q, seed, path_vec) -> weight
        def paths(s, q, seed):                                                         # q,seed:(1,d) -> V:(P,d), logW:(P,)
            d = s.d; beam = q                                                          # (Bn,d), start Bn=1
            allV = []; allW = []
            for l in range(s.L):
                Bn = beam.shape[0]
                inp = torch.cat([beam, seed.expand(Bn, -1)], -1)                       # (Bn,2d)
                offs = s.off[l](inp).view(Bn, s.K, d)                                  # (Bn,K,d): K branch offsets
                vk = F.normalize(beam[:, None, :] + offs, dim=-1).reshape(Bn * s.K, d) # (Bn*K,d) composed paths
                w = s.score(torch.cat([q.expand(vk.shape[0], -1), seed.expand(vk.shape[0], -1), vk], -1)).squeeze(-1)
                keep = min(s.B, vk.shape[0])
                topw, topi = w.topk(keep)                                              # learned pruning
                beam = vk[topi]; allV.append(beam); allW.append(topw)                 # survivors -> next level + collect
            V = torch.cat(allV, 0); W = torch.cat(allW, 0)
            return V, F.log_softmax(W, 0)

    def train(model):
        if not list(model.parameters()):
            return model
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        idx = list(range(len(TR)))
        for ep in range(epochs):
            np.random.shuffle(idx)
            for i in idx:
                q, st, sd, gold = TR[i]
                seed = Efull[sd]                                                # dense-top1 seed (dataset-agnostic, no oracle)
                cand = torch.cat([torch.tensor(gold, device=DEV), negpool[torch.randint(0, seeds_negpool, (256,), device=DEV)]])
                V, logW = model.paths(q.unsqueeze(0), seed.unsqueeze(0))         # (P,d),(P,)
                sc = (Efull[cand] @ V.T + logW[None, :]).max(1).values          # weighted-max over paths
                tgt = torch.zeros(len(cand), device=DEV); tgt[:len(gold)] = 1.0 / len(gold)
                loss = -(F.log_softmax(sc, 0) * tgt).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    @torch.no_grad()
    def open_recall(model, ks=(1, 5, 20, 50), chunk=100000):
        Vs, Ws = [], []
        for q, st, sd, _ in TE:
            V, logW = model.paths(q.unsqueeze(0), Efull[sd].unsqueeze(0))
            Vs.append(V); Ws.append(logW)
        P = Vs[0].shape[0]; Vflat = torch.cat(Vs, 0)                            # (nq*P,d)
        Wf = torch.stack(Ws, 0)                                                 # (nq,P)
        nq = len(TE); K = max(ks)
        topv = torch.full((nq, K), -1e9, device=DEV); top = torch.full((nq, K), -1, dtype=torch.long, device=DEV)
        for st0 in range(0, N, chunk):
            ce = Efull[st0:st0 + chunk]
            sc = ((ce @ Vflat.T).view(ce.shape[0], nq, P) + Wf[None, :, :]).max(2).values   # weighted-max
            v, ix = sc.topk(min(K, sc.shape[0]), dim=0)
            cat_v = torch.cat([topv, v.T], 1); cat_i = torch.cat([top, (ix.T + st0)], 1)
            sel = cat_v.topk(K, 1); topv = sel.values; top = torch.gather(cat_i, 1, sel.indices)
        top = top.cpu().tolist()
        rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        for i, (_, _, _, gold) in enumerate(TE):
            gs = set(gold); row = top[i]
            for k in ks:
                t = set(row[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        return ({f"R@{k}": round(100 * np.mean(rec[k]), 2) for k in ks},
                {f"hit@{k}": round(100 * np.mean(hit[k]), 2) for k in ks})

    log.info("=== OPEN-CORPUS webqsp R@5 (all %d entities, dense-top1 seed, agnostic) — peak to beat 29.7 ===", N)
    out = {}
    for name, mk in [("dense", lambda: Dense()), ("single", lambda: Single()),
                     ("beamK3L2", lambda: Beam(K=3, L=2)), ("beamK3L3", lambda: Beam(K=3, L=3)),
                     ("learnedBeam_K3L3B6", lambda: LearnedBeam(K=3, L=3, B=6)),
                     ("learnedBeam_K4L3B8", lambda: LearnedBeam(K=4, L=3, B=8))]:
        r, h = open_recall(train(mk().to(DEV)))
        out[name] = (r, h)
        log.info("%-20s %s  %s", name, r, h)
    import json, os
    os.makedirs("results/L2", exist_ok=True)
    json.dump({k: v for k, v in out.items()}, open("results/L2/offset_beam_webqsp.json", "w"), indent=2, default=str)
    log.info("-> results/L2/offset_beam_webqsp.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(epochs=a.epochs)


if __name__ == "__main__":
    main()
