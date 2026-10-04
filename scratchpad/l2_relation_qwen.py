"""Canonical gte-Qwen2 masked Relation — TRAIN+VAL build for 2wiki_clean + musique_clean.

Pipeline:
  1. build_sparse_edges(ds,split)  -> resumable, stores ONLY eligible (mask=1) pairs + per-split sentence
     vocab. mask is pure topology (encoder-independent), identical to the MiniLM phase.
  2. collect_global_sentences()    -> dedup unique connecting sentences across all 4 splits -> one list to
     encode ONCE on Modal. Deterministic sentence_text / sha1 hash / global id.
  3. (Modal) encode raw sentences with Alibaba-NLP/gte-Qwen2-1.5B-instruct, doc_prefix="" (CANONICAL doc
     semantics == nodes.npy; the old kg_relsig applied the QUERY instruction to the sentence, which is
     NON-canonical and is corrected here).
  4. materialize_relation(ds,split) -> relation_mask.npy + relation_qwen_score.npy aligned to the C/P50 CSR
     pair order. Query side reuses queries_all.npy (RELATION_Q_EMBED_REUSE=PASS).

TEST is never touched.
"""
import os, sys, json, time, hashlib, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import l2_lib as L
import l2_relation as R

PILOTS = ("2wiki_clean", "musique_clean")
SPLITS = ("train", "val")
OUT = "data/l2_corpus"                       # relation arrays live beside the split CSR
SC = "scratchpad"


def _sparse_cache(ds, split): return f"{SC}/_reledges_sparse_{ds}_{split}.pkl"


def build_sparse_edges(ds, split, budget_s=100, verbose=True):
    """Resumable. Stores per-query eligible (local_j, sentence_id) pairs + per-split sentence texts.
    mask=1 pairs only (sparse ~2/query). Returns dict when complete, else None (resume)."""
    cache = _sparse_cache(ds, split); done = cache + ".done"
    if os.path.exists(done):
        return pickle.load(open(cache, "rb"))
    st = pickle.load(open(cache, "rb")) if os.path.exists(cache) else \
        {"sent_texts": [], "text2sid": {}, "per_q": {}, "next_qi": 0}
    M = R.load_master(ds); S = L.load_split(ds, split); qtxt = R.qtext_all(ds)
    idx2title = M["idx2title"]; off = S["off"]; cand = S["cand"]; meta = S["meta"]
    nq = len(off) - 1
    t0 = time.time(); qi = st["next_qi"]
    while qi < nq:
        s, e = off[qi], off[qi + 1]
        q = qtxt[meta[qi]["row_all"]]
        edges = R.query_edges(M, q)                       # title -> connecting sentence (deterministic)
        if edges:
            lst = []
            for j in range(s, e):
                ti = idx2title.get(int(cand[j]), "")
                if ti and ti in edges:
                    sent = edges[ti]
                    sid = st["text2sid"].get(sent)
                    if sid is None:
                        sid = st["text2sid"][sent] = len(st["sent_texts"]); st["sent_texts"].append(sent)
                    lst.append((j - s, sid))               # local candidate index within scope
            if lst:
                st["per_q"][qi] = lst
        qi += 1
        if time.time() - t0 > budget_s and qi < nq:
            st["next_qi"] = qi; pickle.dump(st, open(cache, "wb"), protocol=4)
            if verbose: print(f"[edges/{ds}/{split}] progress {qi}/{nq} sents={len(st['sent_texts'])} (saved, resume)")
            return None
    st["next_qi"] = nq; pickle.dump(st, open(cache, "wb"), protocol=4); open(done, "w").write("1")
    if verbose:
        neg = sum(len(v) for v in st["per_q"].values())
        print(f"[edges/{ds}/{split}] DONE queries_with_edges={len(st['per_q'])} eligible_pairs={neg} unique_sents={len(st['sent_texts'])}")
    return st


def collect_global_sentences():
    """Dedup unique connecting sentences across ALL 4 splits -> single deterministic encode list.
    Writes scratchpad/_qwen_sentences.json: {sentences:[...], sha1:[...], per_split:{ds/split:{local2global:[...]}}}."""
    global_t2g = {}; sentences = []; per_split = {}
    counts = {}
    for ds in PILOTS:
        for sp in SPLITS:
            st = pickle.load(open(_sparse_cache(ds, sp), "rb"))
            l2g = []
            for txt in st["sent_texts"]:
                g = global_t2g.get(txt)
                if g is None:
                    g = global_t2g[txt] = len(sentences); sentences.append(txt)
                l2g.append(g)
            per_split[f"{ds}/{sp}"] = {"n_local_sents": len(st["sent_texts"]), "local2global": l2g}
            counts[f"{ds}/{sp}"] = len(st["sent_texts"])
    sha1 = [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in sentences]
    obj = {"n_unique_global": len(sentences), "per_split_counts": counts,
           "sentences": sentences, "sha1": sha1, "per_split": per_split}
    json.dump({k: v for k, v in obj.items() if k != "sentences"}, open(f"{SC}/_qwen_sentences_meta.json", "w"), indent=1)
    pickle.dump(obj, open(f"{SC}/_qwen_sentences.pkl", "wb"), protocol=4)
    print(f"GLOBAL unique sentences = {len(sentences)}")
    print(json.dumps(counts, indent=1))
    return obj


def materialize_relation(ds, split, sent_emb_path=f"{SC}/_qwen_sent_emb.npy"):
    """relation_qwen_score[pair] = cos(Qwen(query), Qwen(sentence)) for mask=1 pairs, else 0 (ABSTAIN).
    Writes data/l2_corpus/<ds>/<split>/relation_mask.npy (bool) + relation_qwen_score.npy (float32)."""
    st = pickle.load(open(_sparse_cache(ds, split), "rb"))
    g = pickle.load(open(f"{SC}/_qwen_sentences.pkl", "rb"))
    l2g = g["per_split"][f"{ds}/{split}"]["local2global"]
    Sent = np.load(sent_emb_path)                          # (n_global, 1536) normalized
    S = L.load_split(ds, split); off = S["off"]; npairs = len(S["cand"])
    # query embeddings (canonical, reused)
    Q = np.load(f"data/ukb_storage/{ds}/gte_qwen/queries_all.npy").astype("float32")
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    mask = np.zeros(npairs, dtype=bool); score = np.zeros(npairs, dtype=np.float32)
    meta = S["meta"]
    for qi, lst in st["per_q"].items():
        base = off[qi]; qrow = meta[qi]["row_all"]; qv = Q[qrow]
        for (j, sid) in lst:
            gsid = l2g[sid]; p = base + j
            mask[p] = True; score[p] = float(Sent[gsid] @ qv)
    d = f"{OUT}/{ds}/{split}"
    np.save(f"{d}/relation_mask.npy", mask); np.save(f"{d}/relation_qwen_score.npy", score)
    return {"split": split, "eligible_pairs": int(mask.sum()), "path": d}


_L1CODE = {"L1_ANY_FAIL": 0, "L1_PARTIAL": 1, "L1_ALL_SUCCESS": 2}


def build_expert_matrix(ds, split):
    """Freeze the 5-expert aligned matrix for Shapley. Per-pair score arrays aligned to the C/P50 CSR
    (dense/splade already on disk; offset/mixture materialized f16; relation f32 + mask). Plus per-query
    behavioral metadata. Does NOT pre-fuse; keeps raw scores. Returns manifest fragment."""
    import torch
    from l2_lib import OffsetHead, MixtureHead
    S = L.load_split(ds, split); off = S["off"]; labels = S["labels"]; meta = S["meta"]
    Dn, Qn, dtop = L.load_embeddings(ds)
    d = f"{OUT}/{ds}/{split}"
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn).astype("float16")
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn).astype("float16")
    np.save(f"{d}/offset_score.npy", oh_sc); np.save(f"{d}/mixture_score.npy", mh_sc)
    rq, mask = load_relation_qwen(ds, split)
    # per-query behavioral metadata
    nq = len(off) - 1
    row_all = np.array([meta[qi]["row_all"] for qi in range(nq)], dtype=np.int64)
    gold_count = np.array([int((labels[off[qi]:off[qi + 1]] == 1).sum()) for qi in range(nq)], dtype=np.int64)
    l1 = np.array([_L1CODE.get(meta[qi]["L1_STATUS"], -1) for qi in range(nq)], dtype=np.int8)
    dense = S["dense_score"]; splade = S["splade_scope_score"]
    rk = L._ranks_from_scores
    dsg = np.zeros(nq, dtype=np.float32)                   # 1 - Jaccard@20(dense,splade) : dense/splade disagreement
    rel_any = np.zeros(nq, bool); rel_goldp = np.zeros(nq, bool); rel_nc = np.zeros(nq, np.int64)
    for qi in range(nq):
        s, e = off[qi], off[qi + 1]
        dd = dense[s:e]; sp = splade[s:e]
        k = min(20, e - s)
        td = set(np.argpartition(-dd, k - 1)[:k].tolist()); ts = set(np.argpartition(-sp, k - 1)[:k].tolist())
        u = len(td | ts); dsg[qi] = 1.0 - (len(td & ts) / u if u else 0.0)
        m = mask[s:e]; rel_any[qi] = m.any(); rel_nc[qi] = int(m.sum())
        rel_goldp[qi] = bool((m & (labels[s:e] == 1)).any())
    np.savez(f"{d}/expert_meta.npz", row_all=row_all, gold_count=gold_count, L1_status=l1,
             dense_splade_disagreement=dsg, relation_any_signal_present=rel_any,
             relation_gold_present=rel_goldp, relation_num_candidates=rel_nc)
    return {"split": split, "n_pairs": int(len(S["cand"])), "n_queries": int(nq),
            "files": {"dense_score": "dense_score.npy(f16)", "splade_score": "splade_scope_score.npy(f32)",
                      "offset_score": "offset_score.npy(f16)", "mixture_score": "mixture_score.npy(f16)",
                      "relation_qwen_score": "relation_qwen_score.npy(f32)", "relation_mask": "relation_mask.npy(bool)",
                      "labels": "labels.npy", "cand_ids": "cand_ids.npy", "query_offsets": "query_offsets.npy",
                      "per_query_meta": "expert_meta.npz"}}


def load_relation_qwen(ds, split):
    d = f"{OUT}/{ds}/{split}"
    return np.load(f"{d}/relation_qwen_score.npy").astype("float32"), np.load(f"{d}/relation_mask.npy")


def mask_parity_with_minilm(ds, split="val"):
    """QWEN_RELATION_MASK_PARITY: the topology mask must be identical to the MiniLM-phase mask."""
    _, mq = load_relation_qwen(ds, split)
    import l2_relation_masked as MK
    _, mm, _ = MK.load_relation(ds, split)                # minilm-phase cand_sent>=0
    return {"split": split, "n_pairs": int(len(mq)), "qwen_eligible": int(mq.sum()),
            "minilm_eligible": int(mm.sum()),
            "QWEN_RELATION_MASK_PARITY": "PASS" if np.array_equal(mq, mm) else "FAIL"}


def qwen_eval(ds, split="val"):
    """E3 / E3+Qwen-relation masked fusion + gold-level rescue/hurt, and Qwen-vs-MiniLM ordering agreement."""
    import l2_relation_masked as MK
    from scipy.stats import kendalltau
    S = L.load_split(ds, split); off = S["off"]; labels = S["labels"]
    rq, mask = load_relation_qwen(ds, split)
    rmin, mmin, _ = MK.load_relation(ds, split)
    Dn, Qn, dtop = L.load_embeddings(ds)
    import torch
    from l2_lib import OffsetHead, MixtureHead
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
    dense = S["dense_score"]; splade = S["splade_scope_score"]
    def sl(v, qi): s, e = off[qi], off[qi + 1]; return v[s:e]

    def fuse(qi, rel):
        s, e = off[qi], off[qi + 1]
        experts = [(sl(dense, qi), None), (sl(splade, qi), None), (sl(oh_sc, qi), None), (sl(mh_sc, qi), None)]
        if rel is not None: experts.append(rel)
        return MK.masked_rrf_fuse(experts)
    E3 = lambda qi: fuse(qi, None)
    E3q = lambda qi: (lambda s, e: fuse(qi, (rq[s:e], mask[s:e])))(off[qi], off[qi + 1])

    metrics = {}
    for name, fn in (("E3", E3), ("E3+relation_qwen", E3q)):
        res = L.eval_ranking(S, fn)
        metrics[name] = {k: round(res["COND_ANY_GOLD_IN_SCOPE"][k], 4) for k in
                         ("MRR", "R@5", "R@10", "R@20", "R@50", "ALL@10", "ALL@20", "ALL@50")}
    # gold-level rescue/hurt (E3 vs E3+qwen) + base reachability
    rk = L._ranks_from_scores
    base_ranks = []; e3_ranks = []; e3q_ranks = []; per_gold = []
    kt = []; sole = 0; n_multi = 0
    for qi in range(len(off) - 1):
        s, e = off[qi], off[qi + 1]; lab = labels[s:e]; gl = np.where(lab == 1)[0]
        elig = mask[s:e]; eg = [g for g in gl if elig[g]]
        # ordering agreement among eligible (where >=2 eligible)
        idx = np.where(elig)[0]
        if len(idx) >= 2:
            n_multi += 1
            tau, _ = kendalltau(rq[s:e][idx], rmin[s:e][idx])
            if not np.isnan(tau): kt.append(tau)
        elif len(idx) == 1:
            sole += 1
        if not eg: continue
        rd = rk(sl(dense, qi)); rs = rk(sl(splade, qi)); ro = rk(sl(oh_sc, qi)); rm = rk(sl(mh_sc, qi))
        r3 = rk(E3(qi)); r3q = rk(E3q(qi))
        for g in eg:
            b = int(min(rd[g], rs[g], ro[g], rm[g]))
            base_ranks.append(b); e3_ranks.append(int(r3[g])); e3q_ranks.append(int(r3q[g]))
    base_ranks = np.array(base_ranks); e3_ranks = np.array(e3_ranks); e3q_ranks = np.array(e3q_ranks)
    gold = {"n_eligible_golds": int(len(base_ranks)),
            "eligible_gold_base_missed>=50_frac": round(float((base_ranks >= 50).mean()), 4) if len(base_ranks) else None}
    for K in (5, 10, 20, 50):
        resc = (e3_ranks >= K) & (e3q_ranks < K); hurt = (e3_ranks < K) & (e3q_ranks >= K)
        gold[f"GOLD_RESCUE@{K}"] = int(resc.sum()); gold[f"GOLD_HURT@{K}"] = int(hurt.sum())
        gold[f"GOLD_RESCUE@{K}_base_missed>=50"] = int((resc & (base_ranks >= 50)).sum())
    agree = {"n_queries_multi_eligible": n_multi, "n_queries_sole_eligible": sole,
             "kendall_tau_eligible_order_mean": round(float(np.mean(kt)), 4) if kt else None,
             "note": "sole-eligible queries (rank fixed=0) are encoder-invariant by construction"}
    return {"dataset": ds, "split": split, "metrics": metrics, "gold_level": gold, "ordering_agreement": agree}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "edges"
    if stage == "edges":
        ds = sys.argv[2]; sp = sys.argv[3]
        r = build_sparse_edges(ds, sp)
        print("EDGES_RESUME" if r is None else "EDGES_DONE")
    elif stage == "collect":
        collect_global_sentences()
    elif stage == "materialize":
        for ds in PILOTS:
            for sp in SPLITS:
                print(materialize_relation(ds, sp))
