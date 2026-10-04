"""Experiment 1 — ENTITY-FIRST (prototype) retrieval.

The dossier's killer result: (head,relation)->tail works at the ENTITY level (link-pred R@50 86), but forcing
that prediction onto the shattered 781k DOC index gives R@5 7.4. Fix (the prototype-layer proposal): let the
offset retrieve the thing it was trained on — ENTITY PROTOTYPES — then EXPAND the top entities to their docs
and rerank. This tests the load-bearing assumption: does (topic, QUESTION) -> answer-ENTITY work at the
entity level? If yes, entity-first + expand beats the ~34 doc ceiling; if no, the bottleneck just moved up.

Reports (WebQSP):
  ENTITY-level R@k     : rank the 322k prototypes; is a gold answer-entity in top-k?  (question head; + oracle-relation ceiling)
  DOC-level R@5 (expand): top-M entities -> their docs -> rerank by dense/offset -> R@k over gold DOCS.
Baselines for context: direct-doc oracle_rel 7.4, qa_only ~34.
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
    return r.split(".")[-1].replace("_", " ")


def _run(pre_epochs=6, ft_epochs=40, bs=1024, h=1024, hn_k=16, expand_M=20, seeds_negpool=60000):
    import pandas as pd
    from src.experiments.l1_universal_head import CoreEngine, load_docs_and_encoder, _load
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = _load("webqsp", "gte_qwen", 8000, 3000, 2000)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]
    qtr, str_tr, gtr = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); N, DIM = X.shape; del X, d

    eng = CoreEngine(source="webqsp", index_subdir="gte_qwen")
    _, eq, _ = load_docs_and_encoder(eng, "webqsp", "gte_qwen")

    mn = json.load(open("data/processed/master_nodes_webqsp.json", encoding="utf-8"))
    title2rows = collections.defaultdict(list)
    for n in mn:
        if n["node_id"] in id2idx:
            title2rows[n.get("metadata", {}).get("title", "").strip()].append(id2idx[n["node_id"]])
    del mn
    ents = [e for e in title2rows if e]; ent2i = {e: i for i, e in enumerate(ents)}; n_ent = len(ents)
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    rows_of = [torch.tensor(title2rows[e], device=DEV) for e in ents]
    for i, rws in enumerate(rows_of):
        doc2ent[rws] = i
    m = doc2ent >= 0
    Eent = torch.zeros(n_ent, DIM, device=DEV).index_add_(0, doc2ent[m], Efull[m])
    cnt = torch.zeros(n_ent, device=DEV).index_add_(0, doc2ent[m], torch.ones(m.sum(), device=DEV))
    Eent = F.normalize(Eent / cnt.clamp(min=1)[:, None], dim=1)

    df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
    rel2i = {}; triples = []
    for g in df["graph"]:
        for t in g:
            hn, rn, tn = str(t[0]).strip(), str(t[1]), str(t[2]).strip()
            if hn in ent2i and tn in ent2i:
                r = rel2i.setdefault(rn, len(rel2i)); triples.append((ent2i[hn], r, ent2i[tn]))
    triples = np.array(sorted(set(triples)), dtype=np.int64)
    rels = sorted(rel2i, key=lambda k: rel2i[k])
    Rtext = F.normalize(torch.tensor(eq([_reltext(r) for r in rels]).astype("float32"), device=DEV), dim=1)

    def q_rels(row):
        tops = set(str(e).strip() for e in row["q_entity"]); ans = set(str(e).strip() for e in row["a_entity"])
        adj = collections.defaultdict(list)
        for t in row["graph"]:
            adj[str(t[0]).strip()].append((str(t[1]), str(t[2]).strip()))
        return sorted({rel2i[rel] for tp in tops for (rel, tl) in adj[tp] if tl in ans and rel in rel2i})
    q2rels = {str(r["question"]): q_rels(r) for _, r in df.iterrows()}
    q2ent = {str(r["question"]): [str(e).strip() for e in r["q_entity"]] for _, r in df.iterrows()}

    def head(txt, sd):
        rows = [rr for e in q2ent.get(txt, []) for rr in title2rows.get(e, [])]
        return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0) if rows else Efull[int(sd)]
    Qtr = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    Qte = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    Htr = torch.stack([head(tr_txt[i], str_tr[i]) for i in range(len(tr_txt))])
    Hte = torch.stack([head(te_txt[i], ste[i]) for i in range(len(te_txt))])
    # gold entities per question = entities of the gold docs (doc -> entity via doc2ent)
    def gold_ents(gdocs):
        return sorted({int(doc2ent[int(g)]) for g in gdocs if int(g) < N and doc2ent[int(g)] >= 0})
    Gdoc_te = [[int(x) for x in gte[i]] for i in range(len(gte))]
    Gdoc_tr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    Ge_te = [gold_ents(Gdoc_te[i]) for i in range(len(Gdoc_te))]
    Ge_tr = [gold_ents(Gdoc_tr[i]) for i in range(len(Gdoc_tr))]
    log.info("[kg-proto] ents=%d triples=%d | test w/ gold-entity: %d/%d",
             n_ent, len(triples), sum(1 for g in Ge_te if g), len(Ge_te))

    class RelOffset(nn.Module):
        def __init__(s):
            super().__init__(); s.mlp = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU(), nn.Linear(h, DIM))
        def forward(s, hd, ctx):
            return F.normalize(hd + s.mlp(torch.cat([hd, ctx], -1)), dim=-1)

    # edge-pretrain over entity prototypes (context = relation text)
    pt = RelOffset().to(DEV); opt = torch.optim.Adam(pt.parameters(), lr=1e-3)
    T = torch.tensor(triples, device=DEV)
    for ep in range(pre_epochs):
        perm = torch.randperm(len(T), device=DEV)
        for st in range(0, len(T), bs):
            b = T[perm[st:st + bs]]; v = pt(Eent[b[:, 0]], Rtext[b[:, 1]])
            logits = v @ Eent[b[:, 2]].T / TAU
            with torch.no_grad():
                neg = torch.topk(v @ Eent.T, hn_k, dim=1).indices.reshape(-1)
            hard = (v @ Eent[neg].T / TAU).masked_fill(neg.unsqueeze(0) == b[:, 2:3], -1e4)
            loss = F.cross_entropy(torch.cat([logits, hard], 1), torch.arange(len(b), device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    pt.eval()

    # QA-proto finetune: (topic, question) -> answer ENTITY (contrastive over prototypes)
    negpool = torch.randint(0, n_ent, (seeds_negpool,), device=DEV)

    def finetune_proto(model):
        o = torch.optim.Adam(model.parameters(), lr=1e-3)
        idx = [i for i in range(len(Ge_tr)) if Ge_tr[i]]
        for ep in range(ft_epochs):
            np.random.shuffle(idx)
            for i in idx:
                v = model(Htr[i][None], Qtr[i][None])
                cand = torch.cat([torch.tensor(Ge_tr[i], device=DEV), negpool[torch.randint(0, seeds_negpool, (256,), device=DEV)]])
                sc = (Eent[cand] @ v.T).squeeze(-1)
                tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Ge_tr[i])] = 1.0 / len(Ge_tr[i])
                loss = -(F.log_softmax(sc, 0) * tgt).sum()
                o.zero_grad(); loss.backward(); o.step()
        return model

    @torch.no_grad()
    def entity_recall(vec_of, ks=(1, 5, 20, 50)):
        rec = {k: [] for k in ks}
        denom = sum(1 for g in Ge_te if g)
        for i in range(len(Ge_te)):
            if not Ge_te[i] or vec_of(i) is None:
                continue
            v = vec_of(i)                                                  # (P,d)
            sc = (Eent @ v.T).max(1).values                               # max over P
            order = torch.topk(sc, max(ks)).indices.cpu().tolist()
            gs = set(Ge_te[i])
            for k in ks:
                rec[k].append(len(set(order[:k]) & gs) / len(gs))
        return {f"ent_R@{k}": round(100 * sum(rec[k]) / denom, 2) for k in ks}

    @torch.no_grad()
    def doc_recall_via_expand(vec_of, ks=(1, 5, 20, 50)):
        """top-M entities -> their docs -> rerank by max(dense q.doc, offset v.doc) -> gold-DOC recall."""
        rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        denom = sum(1 for g in Gdoc_te if g)
        for i in range(len(Gdoc_te)):
            if not Gdoc_te[i] or vec_of(i) is None:
                continue
            v = vec_of(i)
            ent_order = torch.topk((Eent @ v.T).max(1).values, expand_M).indices.tolist()
            docs = torch.cat([rows_of[e] for e in ent_order]) if ent_order else torch.tensor([], dtype=torch.long, device=DEV)
            if len(docs) == 0:
                continue
            dv = Efull[docs]
            score = torch.maximum(dv @ Qte[i], (dv @ v.T).max(1).values)   # dense OR offset
            order = docs[torch.argsort(-score)].cpu().tolist()
            gs = set(Gdoc_te[i])
            for k in ks:
                t = set(order[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        return ({f"R@{k}": round(100 * sum(rec[k]) / denom, 2) for k in ks},
                {f"hit@{k}": round(100 * sum(hit[k]) / denom, 2) for k in ks})

    out = {}
    Rte = [q2rels.get(te_txt[i], []) for i in range(len(te_txt))]
    # (a) ORACLE-relation, entity level (ceiling) + doc expand
    vo = lambda i: pt(Hte[i][None].expand(max(len(Rte[i]), 1), -1), Rtext[torch.tensor(Rte[i] or [0], device=DEV)]) if Rte[i] else None
    out["oracle_rel_entity"] = entity_recall(vo)
    out["oracle_rel_doc_expand"] = dict(zip(("recall", "hit"), doc_recall_via_expand(vo)))
    log.info("ORACLE-rel  entity %s | doc-expand %s", out["oracle_rel_entity"], out["oracle_rel_doc_expand"]["recall"])
    # (b) QUESTION -> proto (the load-bearing test) + doc expand
    qp = finetune_proto(RelOffset().to(DEV))
    vq = lambda i: qp(Hte[i][None], Qte[i][None])
    out["question_entity"] = entity_recall(vq)
    out["question_doc_expand"] = dict(zip(("recall", "hit"), doc_recall_via_expand(vq)))
    log.info("QUESTION    entity %s | doc-expand %s", out["question_entity"], out["question_doc_expand"]["recall"])

    import os
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open("results/L2/kg_proto_webqsp.json", "w"), indent=2)
    log.info("-> results/L2/kg_proto_webqsp.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pre-epochs", type=int, default=6)
    p.add_argument("--ft-epochs", type=int, default=40)
    p.add_argument("--expand-M", type=int, default=20)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(pre_epochs=a.pre_epochs, ft_epochs=a.ft_epochs, expand_M=a.expand_M)


if __name__ == "__main__":
    main()
