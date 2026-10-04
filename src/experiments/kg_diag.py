"""WebQSP ranking-gap DIAGNOSTIC (engineering-queue §16-18) — decide whether to invest in the RERANKER or in
CANDIDATE GENERATION, before building bigger models.

For the dense∪graph-2hop pool (the one the reranker sees), measure:
  1. per-signal R@5           : rank the pool by dense / splade / offset / prototype / graph alone.
  2. gold-rank histogram      : for golds IN the pool, where do they sit (1-5 / 6-10 / 11-20 / 21-50 / 51+),
                                under the BEST signal per gold -> how much is "just below top-5".
  3. ORACLE fusion R@5        : per query pick the single best signal -> upper bound if fusion were perfect.
                                If oracle >> learned (35.9), the info is present -> fix the reranker.
                                If oracle ~ 39, the info isn't there -> fix candidate generation/representation.
  4. hop breakdown            : 1-hop / 2-hop / unreachable (topic->answer in the KB subgraph) x pool-hit & R@5.
"""
import json
import logging
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _run(off_epochs=25, n_dense=200, n_seed=20, hops=2, cap=64, tr_cap=1104, te_cap=2000, dataset="webqsp"):
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

    # hop bucket per test question (topic -> answer in the KB subgraph); webqsp only
    hopbucket = {}
    q2rows = {}
    if dataset == "webqsp":
        import pandas as pd
        df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
        for _, r in df.iterrows():
            q = str(r["question"]); q2rows[q] = [rr for e in r["q_entity"] for rr in t2r.get(str(e).strip(), [])]
            tops = set(str(e).strip() for e in r["q_entity"]); ans = set(str(e).strip() for e in r["a_entity"])
            adj = collections.defaultdict(list)
            for t in r["graph"]:
                adj[str(t[0]).strip()].append(str(t[2]).strip())
            if any(tl in ans for tp in tops for tl in adj[tp]):
                hopbucket[q] = "1hop"
            elif any(tl2 in ans for tp in tops for mid in adj[tp] for tl2 in adj[mid]):
                hopbucket[q] = "2hop"
            else:
                hopbucket[q] = "unreach"
    elif dataset == "metaqa":
        for txt in te_txt:
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

    SIGS = ["dense", "offset", "prototype", "splade", "graph"]
    KS = (1, 5, 20, 50)
    BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, 10 ** 9)]

    @torch.no_grad()
    def signals(q, h, txt):
        sim = q @ Efull.T
        order = torch.topk(sim, max(n_dense, n_seed)).indices.cpu().numpy()
        dense_set = set(int(x) for x in order[:n_dense]); seeds = [int(x) for x in order[:n_seed]]
        hop = {int(x): 0 for x in dense_set}
        frontier = set(seeds)
        for hh in range(1, hops + 1):
            nxt = set()
            for s in frontier:
                for x in A.indices[A.indptr[s]:A.indptr[s + 1]][:cap]:
                    x = int(x)
                    if x not in hop:
                        hop[x] = hh; nxt.add(x)
            frontier = nxt
        pool = list(hop.keys()); P = torch.tensor(pool, device=DEV); dp = Efull[P]
        v = off(h[None], q[None])[0]
        sc = {
            "dense": (dp @ q).cpu().numpy(),
            "offset": (dp @ v).cpu().numpy(),
            "prototype": ((Eent[doc2ent[P].clamp(min=0)] * q).sum(1) * (doc2ent[P] >= 0)).cpu().numpy(),
            "splade": (spl_mat[np.array(pool)].dot(spl_scorer.encode_query(txt)) if spl_scorer else np.zeros(len(pool))),
            "graph": -np.array([hop[p] for p in pool], dtype="float32"),
        }
        return pool, sc

    # accumulators
    perR = {s: {k: [] for k in KS} for s in SIGS}
    oracle = {k: [] for k in KS}
    best_bucket = collections.Counter(); absent = 0; gold_total = 0
    hop_stat = {b: {"n": 0, "pool_hit": 0, "r5": []} for b in ("1hop", "2hop", "unreach")}

    for i in range(len(Gte)):
        if not Gte[i]:
            continue
        pool, sc = signals(Qte[i], Hte[i], te_txt[i])
        pos = {p: j for j, p in enumerate(pool)}
        gs = set(Gte[i]); ng = len(gs)
        # rank of each gold under each signal
        rank_by_sig = {s: {} for s in SIGS}
        for s in SIGS:
            order = np.argsort(-sc[s])
            ranks = np.empty(len(pool), dtype=np.int64); ranks[order] = np.arange(len(pool))
            for g in gs:
                if g in pos:
                    rank_by_sig[s][g] = int(ranks[pos[g]]) + 1                # 1-indexed
            for k in KS:
                perR[s][k].append(len(gs & set(pool[j] for j in order[:k])) / ng)
        # oracle = per-query best single signal by R@5
        for k in KS:
            oracle[k].append(max(len(gs & set(pool[j] for j in np.argsort(-sc[s])[:k])) / ng for s in SIGS))
        # gold-rank histogram (best signal per gold)
        for g in gs:
            gold_total += 1
            best = min((rank_by_sig[s].get(g, 10 ** 9) for s in SIGS), default=10 ** 9)
            if best >= 10 ** 9:
                absent += 1
            else:
                for name, lo, hi in BUCKETS:
                    if lo <= best <= hi:
                        best_bucket[name] += 1; break
        # hop breakdown
        b = hopbucket.get(te_txt[i])
        if b:
            hop_stat[b]["n"] += 1
            hop_stat[b]["pool_hit"] += 1 if (gs & set(pool)) else 0
            best_order = np.argsort(-np.maximum.reduce([_z(sc[s]) for s in SIGS]))  # crude fused for r5-by-hop
            hop_stat[b]["r5"].append(len(gs & set(pool[j] for j in best_order[:5])) / ng)

    out = {
        "dataset": dataset, "pool_ceiling": round(100 * (1 - absent / max(gold_total, 1)), 2),
        "per_signal_R@5": {s: round(100 * np.mean(perR[s][5]), 2) for s in SIGS},
        "per_signal_R@50": {s: round(100 * np.mean(perR[s][50]), 2) for s in SIGS},
        "oracle_fusion": {f"R@{k}": round(100 * np.mean(oracle[k]), 2) for k in KS},
        "gold_rank_hist_pct": {b: round(100 * best_bucket[b] / max(gold_total, 1), 2) for b, _, _ in BUCKETS},
        "gold_absent_pct": round(100 * absent / max(gold_total, 1), 2),
        "learned_reranker_R@5_ref": 35.9,
    }
    if any(v["n"] for v in hop_stat.values()):
        out["hop_breakdown"] = {b: {"n": v["n"], "pool_hit_pct": round(100 * v["pool_hit"] / max(v["n"], 1), 2),
                                    "fused_R@5": round(100 * np.mean(v["r5"]), 2) if v["r5"] else None}
                                for b, v in hop_stat.items()}
    log.info("[kg-diag/%s] per-signal R@5 %s", dataset, out["per_signal_R@5"])
    log.info("[kg-diag/%s] ORACLE fusion %s | gold-rank hist %s absent %.1f",
             dataset, out["oracle_fusion"], out["gold_rank_hist_pct"], out["gold_absent_pct"])
    if "hop_breakdown" in out:
        log.info("[kg-diag/%s] hop %s", dataset, out["hop_breakdown"])
    import os as _os
    _os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open(f"results/L2/kg_diag_{dataset}.json", "w"), indent=2)
    log.info("-> results/L2/kg_diag_%s.json", dataset)
    return out


def _z(x):
    x = np.asarray(x, dtype="float64"); s = x.std()
    return (x - x.mean()) / (s + 1e-8) if s > 0 else x * 0.0


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="webqsp")
    p.add_argument("--off-epochs", type=int, default=25)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(off_epochs=a.off_epochs, dataset=a.dataset)


if __name__ == "__main__":
    main()
