"""KG-offset QA BRIDGE — does the graph learned into MLP weights (from edges) transfer to ANSWERING QUESTIONS?

Model:  answer ~= normalize( head + MLP([head, context]) )   where context is a gte-Qwen2 TEXT embedding.
  - EDGE-pretrain: head = head-entity centroid, context = the relation's TEXT embedding, target = tail entity.
  - QA-finetune:   head = topic-entity centroid (q_entity, NER), context = the QUESTION embedding, target =
                   answer docs. Because relation-text and questions share the SAME gte-Qwen2 space, the edge-
                   pretrained MLP already accepts a question as context — the bridge needs no new parameters.

Three variants, to find the best:
  qa_only        : train on the ~1104 QA pairs only (the data-starved baseline).
  pretrain_zero  : edge-pretrained, applied to QA with NO finetune (pure zero-shot transfer of graph knowledge).
  pretrain_ft    : edge-pretrained THEN QA-finetuned (the full model).
Plus a 2-hop compose eval. Open-corpus WebQSP R@k over the 781k docs (topic-entity head, dense-top1 fallback).
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


def _reltext(r):
    """Freebase relation id -> readable phrase: 'location.country.official_language' -> 'official language'."""
    return r.split(".")[-1].replace("_", " ")


def _run(pre_epochs=6, ft_epochs=40, bs=1024, h=1024, hn_k=16, seeds_negpool=60000):
    import pandas as pd
    from src.experiments.l1_universal_head import (CoreEngine, load_docs_and_encoder, _load,
                                                   _splits, _hard_membership)
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = _load("webqsp", "gte_qwen", 8000, 3000, 2000)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]
    qtr, str_tr, gtr = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1)                # 781k docs (QA retrieval space)
    N, DIM = X.shape; del X, d

    eng = CoreEngine(source="webqsp", index_subdir="gte_qwen")
    _, eq, _ = load_docs_and_encoder(eng, "webqsp", "gte_qwen")            # encoder for relation TEXT

    # entity centroids (vectorized) + title->rows
    mn = json.load(open("data/processed/master_nodes_webqsp.json", encoding="utf-8"))
    title2rows = collections.defaultdict(list)
    for n in mn:
        nid = n["node_id"]
        if nid in id2idx:
            title2rows[n.get("metadata", {}).get("title", "").strip()].append(id2idx[nid])
    del mn
    ents = [e for e in title2rows if e]; ent2i = {e: i for i, e in enumerate(ents)}
    n_ent = len(ents)
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for e, i in ent2i.items():
        doc2ent[torch.tensor(title2rows[e], device=DEV)] = i
    Eent = torch.zeros(n_ent, DIM, device=DEV)
    Eent.index_add_(0, doc2ent[doc2ent >= 0], Efull[doc2ent >= 0])
    cnt = torch.zeros(n_ent, device=DEV).index_add_(0, doc2ent[doc2ent >= 0], torch.ones((doc2ent >= 0).sum(), device=DEV))
    Eent = F.normalize(Eent / cnt.clamp(min=1)[:, None], dim=1)
    log.info("[kg-qa] entities=%d docs=%d", n_ent, N)

    # triples + relation-text embeddings (encoded once, same space as questions)
    df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
    rel2i = {}; triples = []
    for g in df["graph"]:
        for t in g:
            hn, rn, tn = str(t[0]).strip(), str(t[1]), str(t[2]).strip()
            if hn in ent2i and tn in ent2i:
                r = rel2i.setdefault(rn, len(rel2i)); triples.append((ent2i[hn], r, ent2i[tn]))
    triples = np.array(sorted(set(triples)), dtype=np.int64)
    rels = sorted(rel2i, key=lambda k: rel2i[k])
    Rtext = F.normalize(torch.tensor(eq([_reltext(r) for r in rels]).astype("float32"), device=DEV), dim=1)  # (n_rel,d)
    log.info("[kg-qa] triples=%d relations=%d", len(triples), len(rels))

    # topic-entity head per question (q_entity, NER; dense-top1 doc fallback)
    q2ent = {str(r["question"]): [str(e).strip() for e in r["q_entity"]] for _, r in df.iterrows()}
    def topic_head(txt, sd):
        rows = [rr for e in q2ent.get(txt, []) for rr in title2rows.get(e, [])]
        return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0) if rows else Efull[int(sd)]
    Qtr = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    Qte = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    Htr = torch.stack([topic_head(tr_txt[i], str_tr[i]) for i in range(len(tr_txt))])
    Hte = torch.stack([topic_head(te_txt[i], ste[i]) for i in range(len(te_txt))])
    Gtr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    Gte = [[int(x) for x in gte[i]] for i in range(len(gte))]
    negpool = torch.randint(0, N, (seeds_negpool,), device=DEV)

    class RelOffset(nn.Module):
        def __init__(s):
            super().__init__(); s.mlp = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU(), nn.Linear(h, DIM))
        def forward(s, head, ctx):
            return F.normalize(head + s.mlp(torch.cat([head, ctx], -1)), dim=-1)

    class MultiRel(nn.Module):
        """GROUNDED multi-relation: emit K relation-hypothesis contexts from the query, push each through the
        EDGE-PRETRAINED offset transform -> K answer predictions. Hedges the question->relation uncertainty."""
        def __init__(s, K=5):
            super().__init__(); s.K = K
            s.ctx = nn.Sequential(nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, K * DIM))   # question -> K contexts
            s.mlp = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU(), nn.Linear(h, DIM))   # shared offset (edge-pretrained)
        def forward(s, head, q):                                                          # head,q:(B,d) -> (B,K,d)
            ctx = s.ctx(q).view(q.shape[0], s.K, DIM)
            hb = head[:, None, :].expand(-1, s.K, -1)
            return F.normalize(hb + s.mlp(torch.cat([hb, ctx], -1)), dim=-1)

    def pretrain(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        T = torch.tensor(triples, device=DEV)
        for ep in range(pre_epochs):
            perm = torch.randperm(len(T), device=DEV)
            for st in range(0, len(T), bs):
                b = T[perm[st:st + bs]]
                v = model(Eent[b[:, 0]], Rtext[b[:, 1]])
                logits = v @ Eent[b[:, 2]].T / TAU
                with torch.no_grad():
                    neg = torch.topk(v @ Eent.T, hn_k, dim=1).indices.reshape(-1)
                hard = (v @ Eent[neg].T / TAU).masked_fill(neg.unsqueeze(0) == b[:, 2:3], -1e4)
                loss = F.cross_entropy(torch.cat([logits, hard], 1), torch.arange(len(b), device=DEV))
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    def finetune(model):                                                   # QA: head=topic, ctx=question, tgt=answer docs
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        idx = list(range(len(Gtr)))
        for ep in range(ft_epochs):
            np.random.shuffle(idx)
            for i in idx:
                if not Gtr[i]:
                    continue
                v = model(Htr[i][None], Qtr[i][None])                      # (1,d)
                cand = torch.cat([torch.tensor(Gtr[i], device=DEV), negpool[torch.randint(0, seeds_negpool, (256,), device=DEV)]])
                sc = (Efull[cand] @ v.T).squeeze(-1)
                tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Gtr[i])] = 1.0 / len(Gtr[i])
                loss = -(F.log_softmax(sc, 0) * tgt).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    def finetune_multi(model):                                             # MultiRel QA finetune (max over K slots)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        idx = list(range(len(Gtr)))
        for ep in range(ft_epochs):
            np.random.shuffle(idx)
            for i in idx:
                if not Gtr[i]:
                    continue
                V = model(Htr[i][None], Qtr[i][None])[0]                    # (K,d)
                cand = torch.cat([torch.tensor(Gtr[i], device=DEV), negpool[torch.randint(0, seeds_negpool, (256,), device=DEV)]])
                sc = (Efull[cand] @ V.T).max(1).values                     # (cand,) weighted-max over K
                tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Gtr[i])] = 1.0 / len(Gtr[i])
                loss = -(F.log_softmax(sc, 0) * tgt).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    @torch.no_grad()
    def qa_recall_multi(model, ks=(1, 5, 20, 50), chunk=100000):
        Vm = model(Hte, Qte)                                               # (nq,K,d)
        nq, Kn = Vm.shape[0], Vm.shape[1]; K = max(ks)
        Vflat = Vm.reshape(nq * Kn, DIM)
        topv = torch.full((nq, K), -1e9, device=DEV); top = torch.full((nq, K), -1, dtype=torch.long, device=DEV)
        for st in range(0, N, chunk):
            sc = (Efull[st:st + chunk] @ Vflat.T).view(-1, nq, Kn).max(2).values      # (chunk,nq) max over K
            v, ix = sc.topk(min(K, sc.shape[0]), dim=0)
            cv = torch.cat([topv, v.T], 1); ci = torch.cat([top, ix.T + st], 1)
            sel = cv.topk(K, 1); topv = sel.values; top = torch.gather(ci, 1, sel.indices)
        top = top.cpu().tolist(); rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        for i, g in enumerate(Gte):
            if not g:
                continue
            gs = set(g); row = top[i]
            for k in ks:
                t = set(row[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        return ({f"R@{k}": round(100 * np.mean(rec[k]), 2) for k in ks},
                {f"hit@{k}": round(100 * np.mean(hit[k]), 2) for k in ks})

    @torch.no_grad()
    def qa_recall(model, ks=(1, 5, 20, 50), chunk=100000):
        V = model(Hte, Qte)                                                # (nq,d)
        nq = len(V); K = max(ks)
        topv = torch.full((nq, K), -1e9, device=DEV); top = torch.full((nq, K), -1, dtype=torch.long, device=DEV)
        for st in range(0, N, chunk):
            sc = Efull[st:st + chunk] @ V.T                                # (chunk,nq)
            v, ix = sc.topk(min(K, sc.shape[0]), dim=0)
            cv = torch.cat([topv, v.T], 1); ci = torch.cat([top, ix.T + st], 1)
            sel = cv.topk(K, 1); topv = sel.values; top = torch.gather(ci, 1, sel.indices)
        top = top.cpu().tolist(); rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        for i, g in enumerate(Gte):
            if not g:
                continue
            gs = set(g); row = top[i]
            for k in ks:
                t = set(row[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        return ({f"R@{k}": round(100 * np.mean(rec[k]), 2) for k in ks},
                {f"hit@{k}": round(100 * np.mean(hit[k]), 2) for k in ks})

    out = {}
    log.info("=== KG-QA bridge, webqsp open-corpus R@5 (benchmark joint head = 29.7) ===")
    torch.manual_seed(0); r, hh = qa_recall(finetune(RelOffset().to(DEV)));                 out["qa_only"] = {"recall": r, "hit": hh}; log.info("qa_only        %s %s", r, hh)
    torch.manual_seed(0); pt = pretrain(RelOffset().to(DEV))
    r, hh = qa_recall(pt);                                                                   out["pretrain_zero"] = {"recall": r, "hit": hh}; log.info("pretrain_zero  %s %s", r, hh)
    r, hh = qa_recall(finetune(pt));                                                         out["pretrain_ft"] = {"recall": r, "hit": hh}; log.info("pretrain_ft    %s %s", r, hh)
    # grounded MULTI-relation: K relation-hypotheses through the EDGE-PRETRAINED offset (hedge relation inference)
    torch.manual_seed(0); pt2 = pretrain(RelOffset().to(DEV))
    mr = MultiRel(K=5).to(DEV); mr.mlp.load_state_dict(pt2.mlp.state_dict())                  # share edge-pretrained offset
    r, hh = qa_recall_multi(finetune_multi(mr));                                             out["pretrain_ft_multirel"] = {"recall": r, "hit": hh}; log.info("pretrain_ft_multirel %s %s", r, hh)

    import os
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open("results/L2/kg_qa_webqsp.json", "w"), indent=2)
    log.info("-> results/L2/kg_qa_webqsp.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pre-epochs", type=int, default=6)
    p.add_argument("--ft-epochs", type=int, default=40)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(pre_epochs=a.pre_epochs, ft_epochs=a.ft_epochs)


if __name__ == "__main__":
    main()
