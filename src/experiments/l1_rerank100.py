"""
L1 toward 100@20 — smart re-ranking of the voted partitions (the gap is PURE ranking).
=====================================================================================
Diagnosis: gold partitions are votable in ~100% of queries and the greedy min-cover of a
query's golds is 1-4 partitions (overlap lets neighbouring golds share a partition), so
oracle FullCov@20 = 100 on all datasets. The ~10pt gap of sum+max RRF is therefore entirely
a RANKING problem: the <=4 covering partitions exist, we just need them in the top-20.

Signals = 6 reciprocal-rank vote matrices: {dense, relhard, rel2hop} x {sum, max} under overlap
membership (MAX rewards a partition holding one strong answer-node that SUM dilutes). Re-rankers:
  oracle    perfect gold-first (ceiling, ~100)
  eqrrf     fixed equal weights over the 6  (~ current sum+max)
  attn      per-query attention a_s(q) over the 6 signals: score(p)=sum_s a_s(q)*rr_s(p)  (pairwise)
  mlp       pairwise MLP over the 6 rr features per partition
Reports partition FullCov@{20,50}. Writes L1_select/rerank100_{ds}.json (isolated dir; safe pull).
"""
import os, json, random, argparse, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, MAXK
from src.experiments.l1_dynamic import _train_hop2
from src.experiments.overlap_retrain import _reconstruct, _hard_membership, _onehop_membership, _splits, INIT_SEED

log = logging.getLogger("experiments.l1_rerank100")
K0 = 60; TOPN = 200; SIGNALS = ["dense_sum", "dense_max", "relh_sum", "relh_max", "rel2_sum", "rel2_max"]


def _feats(order, mem_idx, npart, topn=TOPN, k0=K0):
    nq = len(order); S = np.zeros((nq, npart), np.float32); M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(order[qi][:topn]):
            w = 1.0 / (k0 + r)
            for p in mem_idx[int(nd)]:
                S[qi, p] += w
                if w > M[qi, p]: M[qi, p] = w
    return S, M


def _rr(score):
    nq, npart = score.shape; order = np.argsort(-score, axis=1)
    rank = np.empty((nq, npart), np.int32); rows = np.arange(nq)[:, None]; rank[rows, order] = np.arange(npart)[None, :]
    return (1.0 / (K0 + rank)).astype(np.float32)


def _fullcov(scores, gpl, npart, budgets=(20, 50)):
    order = np.argsort(-scores, axis=1); out = {}
    for b in budgets:
        ok = tot = 0
        for qi, golds in enumerate(gpl):
            if not golds: continue
            tot += 1; rp = {int(p): r for r, p in enumerate(order[qi])}
            if max(min(rp.get(p, npart) for p in gl) for gl in golds) < b: ok += 1
        out[b] = round(100 * ok / tot, 2)
    return out


class Attn(nn.Module):
    def __init__(self, dq, nsig): super().__init__(); self.w = nn.Sequential(nn.Linear(dq, 64), nn.ReLU(), nn.Linear(64, nsig))
    def forward(self, a, q): return (a * F.softmax(self.w(q), dim=-1)).sum(-1)      # a:(npart,nsig) q:(dq,)


class MLP(nn.Module):
    def __init__(self, nsig): super().__init__(); self.net = nn.Sequential(nn.Linear(nsig, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, a, q=None): return self.net(a).squeeze(-1)


def _pairs(A, goldset):
    vs = (A.sum(-1) > 0); out = []
    for qi in range(len(A)):
        g = [p for p in goldset[qi] if vs[qi, p]]
        if not g: continue
        top = np.argsort(-A[qi].sum(-1))[:80]
        neg = [int(p) for p in top if p not in set(goldset[qi]) and vs[qi, p]][:20]
        if neg: out.append((qi, g, neg))
    return out


def _train(model, A_tr, q_tr, pairs, A_va, q_va, gpl_va, npart, device, epochs=40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); best, bs, noimp = -1.0, None, 0
    At = torch.tensor(A_tr, device=device); Qt = torch.tensor(q_tr, device=device)
    Av = torch.tensor(A_va, device=device); Qv = torch.tensor(q_va, device=device)
    for ep in range(epochs):
        model.train(); random.Random(ep).shuffle(pairs)
        for qi, gpos, gneg in pairs:
            s = model(At[qi], Qt[qi])
            sp = s[torch.tensor(gpos, device=device)]; sn = s[torch.tensor(gneg, device=device)]
            loss = -F.logsigmoid(sp.unsqueeze(1) - sn.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = np.stack([model(Av[qi], Qv[qi]).cpu().numpy() for qi in range(len(Av))])
        v = _fullcov(sc, gpl_va, npart)[20]
        if v > best: best, bs, noimp = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else: noimp += 1
        if noimp >= 8: break
    model.load_state_dict(bs); model.eval(); return model


def run(dataset, limit=8000, off_epochs=25, tr_cap=4000, subdir=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.experiments.encoder_swap import load_docs_and_encoder
    eng = CoreEngine(source=dataset, index_subdir=subdir)          # per-encoder graph + partitions
    X, eq, enc_tag = load_docs_and_encoder(eng, dataset, subdir)   # subdir='bge_large' swaps encoder
    X = X.astype("float32")
    n = X.shape[0]; npart = max(int(p) for p in eng.partition_map.values()) + 1
    id2idx = eng.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X); Xt = torch.tensor(X, device=device)
    sp = _splits(eng, _hard_membership(eng)); sp = {s: q[:limit] for s, q in sp.items()}

    def prep(qs, split=None):
        cache = (os.path.join("data", "ukb_storage", dataset, subdir, f"queries_{split}.npy")
                 if subdir and split else None)
        if cache and os.path.exists(cache):                # reuse cached query embeddings (no re-encode)
            q = np.load(cache)[:len(qs)].astype("float32")
        else:
            q = eq([nd.content for nd, _, _ in qs])        # eq -> normalized float32 (MiniLM or BGE)
        seed = np.argmax(q @ X.T, axis=1); gi = [[id2idx[g] for g in gg if g in id2idx] for _, _, gg in qs]
        return q, seed, gi
    qtr, str_, gtr = prep(sp["train"], "train"); qva, sva, gva = prep(sp["val"], "val"); qte, ste, gte = prep(sp["test"], "test")
    g1 = _train_offset("base", qtr, str_, gtr, Xt, index, device, off_epochs)
    gh = _train_offset("hard", qtr, str_, gtr, Xt, index, device, off_epochs)
    g2 = _train_hop2(g1, qtr, str_, gtr, X, Xt, index, device, off_epochs)
    mem = _onehop_membership(eng); mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]

    def orders(q, seed):
        qT = torch.tensor(q, device=device); sT = Xt[torch.tensor(seed, device=device)]
        de = _order(q, index); rh = _order(gh(qT, sT).detach().cpu().numpy(), index)
        ba = _order(g1(qT, sT).detach().cpu().numpy(), index); s1 = ba[:, 0]
        h2 = _order(g2(qT, Xt[torch.tensor(s1, device=device)]).detach().cpu().numpy(), index)
        r2 = _rrf_fuse([_ranks(ba), _ranks(h2)], [1, 1], k=MAXK)
        return {"dense": de, "relh": rh, "rel2": r2}

    def A_of(q, seed, gi):
        od = orders(q, seed); mats = []
        for k in ("dense", "relh", "rel2"):
            S, M = _feats(od[k], mem_idx, npart); mats += [_rr(S), _rr(M)]
        A = np.stack(mats, axis=-1).astype(np.float32)
        gpl = [[mem_idx[g] for g in gg] for gg in gi]
        goldset = [sorted({p for gg in g for p in gg}) for g in gpl]
        return A, gpl, goldset

    A_tr, _, gs_tr = A_of(qtr[:tr_cap], str_[:tr_cap], gtr[:tr_cap])
    A_va, gp_va, _ = A_of(qva, sva, gva); A_te, gp_te, _ = A_of(qte, ste, gte)

    from collections import Counter

    def _ceiling(A, gpl, budgets=(20, 50)):
        """TRUE FullCov ceiling: a query is coverable@b iff its golds' VOTED partitions admit a
        min-cover of size <= b (each gold covered by ANY of its overlap partitions). The old oracle
        forced ALL distinct gold partitions into top-b -> understated ceiling on multi-partition
        datasets (metaqa spans ~92 distinct gold-partitions but min-cover is 1-2)."""
        voted = A.sum(-1) > 0; out = {}
        for b in budgets:
            ok = tot = 0
            for qi, golds in enumerate(gpl):
                if not golds:
                    continue
                tot += 1
                sets = [[p for p in gl if voted[qi, p]] for gl in golds]
                if any(not s for s in sets):
                    continue                                   # a gold with no voted partition (retrieval tail)
                rem = set(range(len(sets))); picks = 0
                while rem and picks < b:
                    c = Counter(p for i in rem for p in sets[i]); best = c.most_common(1)[0][0]; picks += 1
                    rem = {i for i in rem if best not in sets[i]}
                if not rem:
                    ok += 1
            out[b] = round(100 * ok / max(tot, 1), 2)
        return out
    res = {"oracle": _ceiling(A_te, gp_te),
           "eqrrf_sum+max": _fullcov(A_te.sum(-1), gp_te, npart)}
    # topn sweep: deeper voting raises the VOTED oracle (the realistic ceiling) toward 100
    test_od = orders(qte, ste); sweep = {}
    for tn in (200, 350, 500):
        mats = []
        for k in ("dense", "relh", "rel2"):
            S, M = _feats(test_od[k], mem_idx, npart, topn=tn); mats += [_rr(S), _rr(M)]
        Asw = np.stack(mats, axis=-1).astype(np.float32)
        sweep[tn] = {"eqrrf@20": _fullcov(Asw.sum(-1), gp_te, npart)[20],
                     "voted_oracle@20": _ceiling(Asw, gp_te)[20]}
    res["topn_sweep"] = sweep
    # BEST-OF retrievers: per node, MIN rank across dense/relh/rel2 -> max-vote -> augment the 6-way.
    # rel_hard rescues dense-buried answers (rank 256->8); equal-RRF re-buries them by averaging.
    # best-of keeps the single best retriever's rank, so rel_hard's rescue survives.
    nqte = len(gp_te); best_node = np.full((nqte, n), MAXK, np.int32)
    for od in test_od.values():
        for qi in range(nqte):
            row = od[qi]
            for r in range(len(row)):
                nd = int(row[r])
                if r < best_node[qi, nd]: best_node[qi, nd] = r
    Mbo = np.zeros((nqte, npart), np.float32)
    for qi in range(nqte):
        for nd in np.argsort(best_node[qi])[:TOPN]:
            bn = int(best_node[qi, nd])
            if bn >= MAXK: break
            w = 1.0 / (K0 + bn)
            for p in mem_idx[int(nd)]:
                if w > Mbo[qi, p]: Mbo[qi, p] = w
    bo_rr = _rr(Mbo)
    res["eqrrf6+bestof"] = _fullcov(A_te.sum(-1) + bo_rr, gp_te, npart)
    res["eqrrf6+bestof3"] = _fullcov(A_te.sum(-1) + 3 * bo_rr, gp_te, npart)
    pairs = _pairs(A_tr, gs_tr)
    torch.manual_seed(INIT_SEED)
    for name, model in (("attn", Attn(qtr.shape[1], len(SIGNALS))), ("mlp", MLP(len(SIGNALS)))):
        m = _train(model, A_tr, qtr[:tr_cap], list(pairs), A_va, qva, gp_va, npart, device)
        with torch.no_grad():
            Ate = torch.tensor(A_te, device=device); Qte = torch.tensor(qte, device=device)
            sc = np.stack([m(Ate[qi], Qte[qi]).cpu().numpy() for qi in range(len(Ate))])
        res[name] = _fullcov(sc, gp_te, npart)
    out = {"dataset": dataset, "npart": npart, "n_test": len(gp_te), "encoder": enc_tag, "results": res}
    d = os.path.join("data", "ukb_storage", dataset, "results", "L1_select"); os.makedirs(d, exist_ok=True)
    tag = f"_{subdir}" if subdir else ""
    json.dump(out, open(os.path.join(d, f"rerank100_{dataset}{tag}.json"), "w"), indent=2)
    log.info(f"[{dataset}] " + " | ".join(f"{k} @20={v[20]}" for k, v in res.items() if k != "topn_sweep")
             + f" | topn_sweep={res['topn_sweep']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 re-ranking toward 100@20 (attention over signals).")
    p.add_argument("--datasets", nargs="+", default=["musique_clean", "2wiki_clean", "squad_clean", "hotpotqa_clean"])
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--tr-cap", type=int, default=4000)
    p.add_argument("--subdir", default=None, help="encoder subdir (e.g. bge_large) to swap MiniLM -> BGE/E5")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 RERANK-100 (attention){' ['+a.subdir+']' if a.subdir else ''}: {ds.upper()} =====")
        try:
            run(ds, limit=a.limit, off_epochs=a.off_epochs, tr_cap=a.tr_cap, subdir=a.subdir)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
