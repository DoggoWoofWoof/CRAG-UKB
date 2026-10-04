"""KG relation-inference QA — the honest test of "infer the relation, then apply the proven edge-offset".

Decompose WebQSP QA into two solvable halves instead of one hard end-to-end map:
  (1) INFER relation from question  -> supervised classifier question -> relation (only ~218 relations used
      in QA; label = the topic->answer edge in the KB, extractable for 65% of questions).
  (2) APPLY relation via the EDGE-PRETRAINED offset  answer ~= topic + MLP([topic, rel_text])  (link-pred 86).

Variants (open-corpus WebQSP R@5 over 781k docs):
  qa_only            : end-to-end question->answer offset (baseline).
  oracle_rel         : feed the TRUE relation into the edge-offset (the CEILING — how much is relation-inference
                       worth? isolates inference from application).
  pred_rel  / _top3  : supervised question->relation classifier, apply edge-offset with top-1 / union of top-3.
If oracle_rel >> qa_only, relation inference is the lever and the classifier is the whole game.
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


def _run(pre_epochs=6, cls_epochs=60, bs=1024, h=1024, hn_k=16):
    import pandas as pd
    from src.experiments.l1_universal_head import CoreEngine, load_docs_and_encoder, _load
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = _load("webqsp", "gte_qwen", 8000, 3000, 2000)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]
    qtr, str_tr, _ = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); N, DIM = X.shape; del X, d

    eng = CoreEngine(source="webqsp", index_subdir="gte_qwen")
    _, eq, _ = load_docs_and_encoder(eng, "webqsp", "gte_qwen")

    # entity centroids (vectorized)
    mn = json.load(open("data/processed/master_nodes_webqsp.json", encoding="utf-8"))
    title2rows = collections.defaultdict(list)
    for n in mn:
        if n["node_id"] in id2idx:
            title2rows[n.get("metadata", {}).get("title", "").strip()].append(id2idx[n["node_id"]])
    del mn
    ents = [e for e in title2rows if e]; ent2i = {e: i for i, e in enumerate(ents)}; n_ent = len(ents)
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for e, i in ent2i.items():
        doc2ent[torch.tensor(title2rows[e], device=DEV)] = i
    m = doc2ent >= 0
    Eent = torch.zeros(n_ent, DIM, device=DEV).index_add_(0, doc2ent[m], Efull[m])
    cnt = torch.zeros(n_ent, device=DEV).index_add_(0, doc2ent[m], torch.ones(m.sum(), device=DEV))
    Eent = F.normalize(Eent / cnt.clamp(min=1)[:, None], dim=1)

    # triples + QA relation labels (topic -r-> answer, 1-hop)
    df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
    rel2i = {}; triples = []
    for g in df["graph"]:
        for t in g:
            hn, rn, tn = str(t[0]).strip(), str(t[1]), str(t[2]).strip()
            if hn in ent2i and tn in ent2i:
                r = rel2i.setdefault(rn, len(rel2i)); triples.append((ent2i[hn], r, ent2i[tn]))
    triples = np.array(sorted(set(triples)), dtype=np.int64)
    rels = sorted(rel2i, key=lambda k: rel2i[k]); n_rel = len(rels)
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
    Rtr = [q2rels.get(tr_txt[i], []) for i in range(len(tr_txt))]
    Rte = [q2rels.get(te_txt[i], []) for i in range(len(te_txt))]
    Gte = [[int(x) for x in gte[i]] for i in range(len(gte))]
    log.info("[kg-relqa] ents=%d rels=%d triples=%d | test w/ rel-label: %d/%d",
             n_ent, n_rel, len(triples), sum(1 for r in Rte if r), len(Rte))

    class RelOffset(nn.Module):
        def __init__(s):
            super().__init__(); s.mlp = nn.Sequential(nn.Linear(2 * DIM, h), nn.GELU(), nn.Linear(h, DIM))
        def forward(s, hd, ctx):
            return F.normalize(hd + s.mlp(torch.cat([hd, ctx], -1)), dim=-1)

    # EDGE-pretrain the offset on triples (context = relation text)
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

    @torch.no_grad()
    def recall_from_vecs(Vlist, ks=(1, 5, 20, 50), chunk=100000):
        """Vlist[i] = (Pi,d) predicted vectors for query i; retrieve union (max over Pi) over all docs."""
        idx = [i for i, v in enumerate(Vlist) if v is not None and Gte[i]]
        flat = torch.cat([Vlist[i] for i in idx], 0); sizes = [Vlist[i].shape[0] for i in idx]
        offs = np.cumsum([0] + sizes); nq = len(idx); K = max(ks)
        topv = torch.full((nq, K), -1e9, device=DEV); top = torch.full((nq, K), -1, dtype=torch.long, device=DEV)
        for st in range(0, N, chunk):
            sc = Efull[st:st + chunk] @ flat.T                             # (chunk, sum P)
            per = torch.stack([sc[:, offs[j]:offs[j + 1]].max(1).values for j in range(nq)], 1)  # (chunk,nq)
            v, ix = per.topk(min(K, per.shape[0]), dim=0)
            cv = torch.cat([topv, v.T], 1); ci = torch.cat([top, ix.T + st], 1)
            sel = cv.topk(K, 1); topv = sel.values; top = torch.gather(ci, 1, sel.indices)
        top = top.cpu().tolist(); rec = {k: [] for k in ks}; hit = {k: [] for k in ks}
        for r, i in enumerate(idx):
            gs = set(Gte[i]); row = top[r]
            for k in ks:
                t = set(row[:k]); rec[k].append(len(t & gs) / len(gs)); hit[k].append(1.0 if t & gs else 0.0)
        # report over ALL test (questions without a vector count as 0 recall)
        denom = sum(1 for i in range(len(Gte)) if Gte[i])
        return ({f"R@{k}": round(100 * sum(rec[k]) / denom, 2) for k in ks},
                {f"hit@{k}": round(100 * sum(hit[k]) / denom, 2) for k in ks}, f"{len(idx)}/{denom} answered")

    out = {}
    # ORACLE relation: feed the TRUE relation(s) into the edge-offset (ceiling)
    with torch.no_grad():
        Vo = [pt(Hte[i][None].expand(len(Rte[i]), -1), Rtext[torch.tensor(Rte[i], device=DEV)]) if Rte[i] else None
              for i in range(len(Rte))]
    r, hh, cov = recall_from_vecs(Vo); out["oracle_rel"] = {"recall": r, "hit": hh, "cov": cov}
    log.info("oracle_rel   %s %s  (%s)", r, hh, cov)

    # SUPERVISED question -> relation classifier (multi-label over the QA relations)
    clf = nn.Sequential(nn.Linear(DIM, h), nn.GELU(), nn.Linear(h, n_rel)).to(DEV)
    co = torch.optim.Adam(clf.parameters(), lr=1e-3)
    tri = [i for i in range(len(Rtr)) if Rtr[i]]
    for ep in range(cls_epochs):
        np.random.shuffle(tri)
        for st in range(0, len(tri), 256):
            bi = tri[st:st + 256]
            y = torch.zeros(len(bi), n_rel, device=DEV)
            for j, i in enumerate(bi):
                y[j, Rtr[i]] = 1.0 / len(Rtr[i])
            loss = -(F.log_softmax(clf(Qtr[bi]), 1) * y).sum(1).mean()
            co.zero_grad(); loss.backward(); co.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Qte)                                                    # (nq, n_rel)
        for topk, name in [(1, "pred_rel"), (3, "pred_rel_top3")]:
            pk = pred.topk(topk, 1).indices                               # (nq, topk)
            Vp = [pt(Hte[i][None].expand(topk, -1), Rtext[pk[i]]) for i in range(len(Rte))]
            r, hh, cov = recall_from_vecs(Vp); out[name] = {"recall": r, "hit": hh}
            log.info("%-12s %s %s", name, r, hh)
        # relation classification accuracy (top-1 in the true set?)
        acc = np.mean([1.0 if pred[i].argmax().item() in Rte[i] else 0.0 for i in range(len(Rte)) if Rte[i]])
        out["rel_top1_acc"] = round(100 * float(acc), 2); log.info("relation top1 acc: %.2f", 100 * acc)

    import os
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open("results/L2/kg_relqa_webqsp.json", "w"), indent=2)
    log.info("-> results/L2/kg_relqa_webqsp.json")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pre-epochs", type=int, default=6)
    p.add_argument("--cls-epochs", type=int, default=60)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(pre_epochs=a.pre_epochs, cls_epochs=a.cls_epochs)


if __name__ == "__main__":
    main()
