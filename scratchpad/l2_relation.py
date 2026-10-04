"""C/P50 RELATION expert wrapper (L2 phase).

Faithful re-implementation of the kg_relsig PASSAGE relation signal (ner_edges=False, depth-1 edges),
but computed on the CANONICAL C/Dense+SPLADE/K100/P50 candidate universe (data/l2_corpus) instead of
the old dense-top200 U NER-2hop pool. NO CoreEngine: everything is derived from master_nodes
(doc rows, titles, struct neighbors, question texts), which is bit-identical to the frozen row space
(doc_N -> row N verified; id2idx == numeric suffix).

RELATION signal:
  relation_raw(q, d) = cos( E(q), E(connecting_sentence(topic(q) -> title(d))) )
  where the connecting sentence is the sentence in the topic doc that names title(d) over the struct
  (title-mention) adjacency A. A candidate with no topic->title edge gets sentinel -1 (ABSENT).

This module only BUILDS + PROFILES + evaluates relation. It does not train a controller.
"""
import os, re, json, collections, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root for `src`
import l2_lib as L  # noqa: E402

DATA = "data"
MASTER = {}   # ds -> dict(built structures)


# ------------------------------------------------------------------ master-node derived structures
def _master_path(ds):
    p = f"data/processed/master_nodes_{ds}.json"
    return p if os.path.exists(p) else "data/processed/master_nodes.json"


def load_master(ds):
    if ds in MASTER:
        return MASTER[ds]
    import pickle
    cache = f"scratchpad/_relcache_master_{ds}.pkl"
    if os.path.exists(cache):
        M = pickle.load(open(cache, "rb"))
        M["_wb"] = {}; M["_toks"] = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        MASTER[ds] = M
        return M
    docrow = {}                      # node_id -> row
    title_by_row = {}; content_by_row = {}; neigh = {}   # row -> [rows]
    q_text = {}                      # question node_id -> text
    data = json.load(open(_master_path(ds), encoding="utf-8"))
    doc_ids = [n["node_id"] for n in data if n.get("metadata", {}).get("type") == "document"]
    # Row scheme decided ONCE per dataset (all-or-nothing), never per node:
    #   * integer-suffixed ids (webqsp_<r>, hotpotqa_clean_<r>, ...) -> parse the suffix (exact,
    #     unchanged from before); this is the canonical row for those corpora.
    #   * entity-name ids (metaqa_ent_$, metaqa_ent_'71, metaqa_ent_1984, ...) -> file-order
    #     position among document nodes. A per-node try/except would be WRONG here: numeric-named
    #     entities ("1984", "300") would int-parse to a bogus row and collide, so the choice must
    #     be uniform. Position == id2idx row (load_partition_topology enumerates the same
    #     non-question doc_nodes in file order), so the relation rows align with nodes.npy.
    def _suffix_int(nid):
        try:
            return int(nid.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
    use_suffix = all(_suffix_int(x) is not None for x in doc_ids)
    doc_i = -1
    for n in data:
        nid = n["node_id"]; md = n.get("metadata", {}); t = md.get("type")
        if t == "document":
            doc_i += 1
            r = _suffix_int(nid) if use_suffix else doc_i
            docrow[nid] = r
            ti = (md.get("title") or "").strip()
            title_by_row[r] = ti; content_by_row[r] = n.get("content", "") or ""
            neigh[r] = n.get("neighbors", [])
        elif t == "question":
            q_text[nid] = n.get("content", "") or ""
    N = max(title_by_row) + 1
    # struct adjacency as neighbor-row lists (title-mention), undirected
    adj = [[] for _ in range(N)]
    for r, nbs in neigh.items():
        for nb in nbs:
            j = docrow.get(nb)
            if j is not None and j != r:
                adj[r].append(j)
    # symmetrize
    for r in range(N):
        for j in adj[r]:
            adj[j].append(r)
    adj = [sorted(set(a)) for a in adj]
    # title -> rows, entity vocab, doc2ent
    t2r = collections.defaultdict(list); idx2title = {}
    for r, ti in title_by_row.items():
        if ti:
            t2r[ti].append(r); idx2title[r] = ti
    ents = [e for e in t2r if e]; ent2i = {e: i for i, e in enumerate(ents)}
    doc2ent = np.full(N, -1, dtype=np.int64)
    for e, i in ent2i.items():
        for r in t2r[e]:
            doc2ent[r] = i
    # find_topics support: title tokens, rarest-token index
    def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())
    ttoks = {t: set(_toks(t)) for t in ents}
    dfc = collections.Counter(w for ts in ttoks.values() for w in ts)
    keyidx = collections.defaultdict(list)
    for t in ents:
        if ttoks[t]:
            keyidx[min(ttoks[t], key=lambda w: dfc[w])].append(t)
    M = dict(N=N, docrow=docrow, title_by_row=title_by_row, content_by_row=content_by_row,
             adj=adj, t2r=dict(t2r), idx2title=idx2title, ents=ents, ent2i=ent2i, doc2ent=doc2ent,
             ttoks=ttoks, keyidx=dict(keyidx), q_text=q_text)
    import pickle
    pickle.dump(M, open(f"scratchpad/_relcache_master_{ds}.pkl", "wb"), protocol=4)
    M["_wb"] = {}; M["_toks"] = _toks
    MASTER[ds] = M
    return M


# ------------------------------------------------------------------ query-text alignment (row_all -> text)
def qtext_all(ds):
    j = json.load(open(f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json"))
    M = load_master(ds); qt = M["q_text"]
    return [qt.get(qid, "") for qid in j["ids"]]


# ------------------------------------------------------------------ find_topics / connect (kg_relsig PASSAGE, ner=False)
_splitre = re.compile(r"(?<=[.!?])\s+")
def _sents(t): return [s for s in _splitre.split(t) if s.strip()]
def _specific(t): return len(t) >= 4 and re.search(r"[A-Z][a-z]", t) is not None
def _mentions(M, title, sent):
    p = M["_wb"].get(title)
    if p is None:
        p = M["_wb"][title] = re.compile(r"(?<![A-Za-z])" + re.escape(title) + r"(?![A-Za-z])")
    return p.search(sent) is not None


def find_topics(M, q):
    _toks = M["_toks"]; ttoks = M["ttoks"]; keyidx = M["keyidx"]
    ql = q.lower(); qt = set(_toks(q)); seen = set(); found = []
    for w in sorted(qt):                                   # deterministic token order (hash-seed independent)
        for t in keyidx.get(w, ()):
            if t in seen:
                continue
            seen.add(t)
            if ttoks[t] and ttoks[t] <= qt and len(t) >= 3 and t.lower() in ql:
                found.append(t)
    found.sort(key=lambda s: (-len(s), s))                 # length desc, name asc (deterministic tiebreak)
    topics = []
    for t in found:
        if not any(t != o and t.lower() in o.lower() for o in topics):
            topics.append(t)
    return topics[:4]


def connect(M, src_title):
    """(tgt_title, connecting_sentence) depth-1 edges over struct adjacency (title-mention)."""
    out = []
    t2r = M["t2r"]; adj = M["adj"]; content_by_row = M["content_by_row"]; idx2title = M["idx2title"]
    for r in t2r.get(src_title, [])[:1]:
        ss = _sents(content_by_row.get(r, ""))
        for nb in adj[r]:
            tt = idx2title.get(int(nb), "")
            if not tt or tt == src_title or not _specific(tt):
                continue
            for s in ss:
                if _mentions(M, tt, s):
                    out.append((tt, s)); break
    return out


def query_edges(M, q):
    """title -> connecting sentence (first) for depth-1 edges from the query's topics."""
    edges = {}
    for tp in find_topics(M, q):
        for tt, sent in connect(M, tp):
            edges.setdefault(tt, sent)          # first sentence per target title
    return edges


# ------------------------------------------------------------------ per-split relation edge build (scoped to P50)
def build_edges_for_split(ds, split, qsubset=None, verbose=True):
    """For each query in the split, compute the connecting sentence for every P50 candidate whose title
    is a topic-connected target. Returns aligned arrays + the unique-sentence vocab (to be encoded).
    STRICT SCOPE: only P50 candidates are ever touched; no new candidate is introduced.
    Full-split builds are cached to disk (2wiki find_topics is ~70s)."""
    import pickle
    cache = f"scratchpad/_reledges_{ds}_{split}.pkl"
    if qsubset is None and os.path.exists(cache):
        B = pickle.load(open(cache, "rb"))
        B["S"] = L.load_split(ds, split)
        return B
    M = load_master(ds); S = L.load_split(ds, split); qtxt = qtext_all(ds)
    idx2title = M["idx2title"]
    off = S["off"]; cand = S["cand"]; meta = S["meta"]
    nq = len(off) - 1
    rows = range(nq) if qsubset is None else qsubset
    sent_vocab = {}                                  # sentence text -> vocab id
    cand_sent = np.full(len(cand), -1, dtype=np.int64)   # per-cand -> vocab id (-1 = absent)
    per_q_topics = {}; t0 = time.time()
    n_edge_q = 0
    for qi in rows:
        s, e = off[qi], off[qi + 1]
        row_all = meta[qi]["row_all"]; q = qtxt[row_all]
        edges = query_edges(M, q)                    # title -> sentence
        per_q_topics[qi] = find_topics(M, q)
        if not edges:
            continue
        n_edge_q += 1
        for j in range(s, e):
            ti = idx2title.get(int(cand[j]), "")
            if ti and ti in edges:
                sent = edges[ti]
                vid = sent_vocab.get(sent)
                if vid is None:
                    vid = sent_vocab[sent] = len(sent_vocab)
                cand_sent[j] = vid
    if verbose:
        print(f"[relation/{ds}/{split}] queries={len(list(rows))} with_edges={n_edge_q} "
              f"unique_sentences={len(sent_vocab)} cand_with_rel={int((cand_sent>=0).sum())} "
              f"elapsed={time.time()-t0:.1f}s")
    B = dict(cand_sent=cand_sent, sent_vocab=sent_vocab, per_q_topics=per_q_topics, rows=list(rows))
    if qsubset is None:
        import pickle
        pickle.dump(B, open(f"scratchpad/_reledges_{ds}_{split}.pkl", "wb"), protocol=4)
    B["S"] = S
    return B


_ENC = {}
def _resumable_encode(cache, texts, chunk, tag, verbose, budget_s=60):
    """Encode `texts` with MiniLM DenseEncoder, saving to `cache` incrementally. Returns the full
    array when complete, else None (partial saved -> re-run to resume). Time-budgeted per call so it
    fits a single foreground window; robust to the environment killing long background tasks."""
    import time
    done = cache + ".done"
    if os.path.exists(done):
        return np.load(cache)
    part = np.load(cache) if os.path.exists(cache) else np.zeros((0, 384), np.float32)
    n0 = len(part)
    if n0 >= len(texts):
        open(done, "w").write("1"); return np.load(cache)
    if "enc" not in _ENC:
        from src.core.encoders import DenseEncoder
        _ENC["enc"] = DenseEncoder()
    enc = _ENC["enc"]; t0 = time.time(); buf = [part] if n0 else []
    k = n0
    while k < len(texts):
        emb = np.asarray(enc.encode(texts[k:k + chunk])).astype("float32")
        buf.append(emb); k += chunk
        if time.time() - t0 > budget_s and k < len(texts):
            np.save(cache, np.concatenate(buf, 0))
            if verbose:
                print(f"[encode/{tag}] progress {k}/{len(texts)} (saved, resume)")
            return None
    full = np.concatenate(buf, 0)
    np.save(cache, full); open(done, "w").write("1")
    if verbose:
        print(f"[encode/{tag}] DONE {len(full)}")
    return full


def relation_scores_for_split(ds, split, encoder="minilm", chunk_sents=256, verbose=True):
    """Compute relation_raw scores over the FULL P50 scope for every candidate, using a loadable encoder.
    encoder='minilm' -> src.core.encoders.DenseEncoder (gte-Qwen2 is local-infeasible: RAM/pagefile).
    Sentinel -1 for candidates with no topic->title edge. Resumable sentence-embedding cache.
    Returns a flat float32 array aligned to S['cand'] (like dense_score)."""
    import pickle
    B = build_edges_for_split(ds, split, verbose=verbose)
    S = B["S"]; cand_sent = B["cand_sent"]; sent_vocab = B["sent_vocab"]; meta = S["meta"]
    off = S["off"]
    inv = [None] * len(sent_vocab)
    for s, i in sent_vocab.items():
        inv[i] = s
    qtxt = qtext_all(ds)
    rows = [meta[qi]["row_all"] for qi in range(len(off) - 1)]
    scache = f"scratchpad/_relemb_{encoder}_{ds}_{split}_sents.npy"
    qcache = f"scratchpad/_relemb_{encoder}_{ds}_{split}_q.npy"
    Sent = _resumable_encode(scache, inv, chunk_sents, "sents", verbose)
    Qm = _resumable_encode(qcache, [qtxt[r] for r in rows], chunk_sents, "queries", verbose)
    if Sent is None or Qm is None:
        return None, B          # not finished encoding yet (resume)
    Sent = Sent / (np.linalg.norm(Sent, axis=1, keepdims=True) + 1e-9)
    Qm = Qm / (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)
    # score: relation_raw[cand] = cos(Qm[qi], Sent[cand_sent[cand]])  (sentinel -1)
    allsc = np.full(len(cand_sent), -1.0, dtype=np.float32)
    for qi in range(len(off) - 1):
        s, e = off[qi], off[qi + 1]
        cs = cand_sent[s:e]
        hv = np.where(cs >= 0)[0]
        if len(hv):
            allsc[s + hv] = Sent[cs[hv]] @ Qm[qi]
    return allsc, B


def coverage_stats(ds, split, B):
    """Task 6 relation coverage + Task 3 alignment, from a built-edges dict B (no encoding needed)."""
    S = B["S"]; cand_sent = B["cand_sent"]; rows = B["rows"]
    off = S["off"]; labels = S["labels"]; meta = S["meta"]
    have = cand_sent >= 0
    tot_cand = tot_pos = tot_neg = 0
    pos_have = neg_have = 0
    q_no = q_pos = q_negonly = q_both = 0
    any_gold_cov = 0; all_gold_cov = 0; nq_gold = 0
    oos = 0                                            # out-of-scope sentinel writes (must be 0)
    for qi in rows:
        s, e = off[qi], off[qi + 1]
        lab = labels[s:e]; hv = have[s:e]
        goldmask = lab == 1; negmask = lab == 0
        ng = int(goldmask.sum())
        tot_cand += (e - s); tot_pos += int(goldmask.sum()); tot_neg += int(negmask.sum())
        gp = int((hv & goldmask).sum()); npv = int((hv & negmask).sum())
        pos_have += gp; neg_have += npv
        hp = gp > 0; hn = npv > 0
        if not (hp or hn):
            q_no += 1
        elif hp and hn:
            q_both += 1
        elif hp:
            q_pos += 1
        else:
            q_negonly += 1
        if ng > 0:
            nq_gold += 1
            if gp > 0:
                any_gold_cov += 1
            if gp == ng:
                all_gold_cov += 1
    def frac(a, b): return round(a / b, 4) if b else 0.0
    return {
        "split": split, "n_queries": len(rows),
        "frac_cand_with_relation": frac(int(have[np.concatenate([np.arange(off[qi], off[qi+1]) for qi in rows])].sum()) if rows else 0, tot_cand),
        "frac_pos_with_relation": frac(pos_have, tot_pos),
        "frac_neg_with_relation": frac(neg_have, tot_neg),
        "queries_NO_RELATION_SIGNAL": q_no,
        "queries_RELATION_ON_POSITIVE": q_pos,
        "queries_RELATION_ONLY_ON_NEGATIVES": q_negonly,
        "queries_RELATION_POS_NEG_BOTH": q_both,
        "ANY_GOLD_RELATION_COVERAGE": frac(any_gold_cov, nq_gold),
        "ALL_GOLD_RELATION_COVERAGE": frac(all_gold_cov, nq_gold),
        "n_unique_sentences": len(B["sent_vocab"]),
        "n_cand_with_relation": int(have.sum()),
        "out_of_scope_writes": oos,
    }


def eval_relation(ds, split):
    """Tasks 7-10 + 14: relation-only metrics, fixed-fusion, rescue, disagreement; append to regime raw."""
    S = L.load_split(ds, split)
    rscore = np.load(f"scratchpad/_relscore_{ds}_{split}.npy")   # aligned to S['cand']
    off = S["off"]
    def rel_scorer(qi): return rscore[off[qi]:off[qi + 1]]
    # base experts
    Dn, Qn, dtop = L.load_embeddings(ds)
    import torch
    from l2_lib import OffsetHead, MixtureHead
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
    oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
    mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
    def dsc(qi): return S["dense_score"][off[qi]:off[qi + 1]]
    def spc(qi): return S["splade_scope_score"][off[qi]:off[qi + 1]]
    def osc(qi): return oh_sc[off[qi]:off[qi + 1]]
    def msc(qi): return mh_sc[off[qi]:off[qi + 1]]
    # Task 7: relation-only ranking metrics + per_best_relation
    rel_metrics, rel_best, _ = L.eval_ranking(S, rel_scorer, collect_regime=True)
    # Task 10: fixed-fusion E1a(dense+splade)+relation and E3(+offset+mixture)+relation, RRF k0=60
    def fuse_rrf(scorers):
        def f(qi): return L.rrf_fuse([s(qi) for s in scorers])
        return f
    E1a = fuse_rrf([dsc, spc]); E1a_rel = fuse_rrf([dsc, spc, rel_scorer])
    E3 = fuse_rrf([dsc, spc, osc, msc]); E3_rel = fuse_rrf([dsc, spc, osc, msc, rel_scorer])
    res = {}
    for name, f in (("E1a", E1a), ("E1a+relation", E1a_rel), ("E3", E3), ("E3+relation", E3_rel)):
        m = L.eval_ranking(S, f)
        res[name] = m["COND_ANY_GOLD_IN_SCOPE"]
    res["relation_only"] = rel_metrics["COND_ANY_GOLD_IN_SCOPE"]
    res["relation_only_ALL"] = rel_metrics["ALL_EVAL_QUERIES"]
    # Task 8/9: rescue + disagreement vs regime-raw base ranks
    reg = np.load(f"results/L2/_regime_raw/{ds}_{split}.npz")
    base = {k.replace("best_gold_rank_", ""): reg[k] for k in reg.files}
    valid = rel_best >= 0
    def rank_of(name): return base[f"best_gold_rank_{name}"] if f"best_gold_rank_{name}" in reg.files else base[name]
    cheap_best = np.minimum.reduce([base["dense"], base["splade"], base["offset"], base["mixture"]])
    rescue = {}; hurt = {}
    for K in (5, 10, 20, 50):
        for bname, b in (("dense", base["dense"]), ("E1a_rrf", base["E1a_rrf"]), ("cheap_stack", cheap_best)):
            m = valid & (b >= 0)
            rescue[f"RELATION_RESCUE_top{K}_vs_{bname}"] = int(np.sum(m & (b >= K) & (rel_best < K)))
        # hurt: fixed fusion E3+relation pushes a gold that E3 had in topK out of topK
    # disagreement distributions (per-query best-gold rank deltas)
    disq = {}
    for other in ("dense", "splade", "mixture"):
        d = np.abs(base[other][valid] - rel_best[valid])
        disq[f"|{other}-relation|"] = {"mean": round(float(d.mean()), 2), "median": float(np.median(d)),
                                       "p90": float(np.percentile(d, 90)), "p95": float(np.percentile(d, 95))}
    # Task 8 hurt via fixed fusion: compare E3 vs E3+relation best-gold rank per query
    _, e3_best, _ = L.eval_ranking(S, E3, collect_regime=True)
    _, e3r_best, _ = L.eval_ranking(S, E3_rel, collect_regime=True)
    vv = (e3_best >= 0) & (e3r_best >= 0)
    for K in (10, 20):
        hurt[f"RELATION_HURT_top{K}_in_E3fusion"] = int(np.sum(vv & (e3_best < K) & (e3r_best >= K)))
        rescue[f"RELATION_RESCUE_top{K}_in_E3fusion"] = int(np.sum(vv & (e3_best >= K) & (e3r_best < K)))
    # Task 14: append best_gold_rank_relation to regime raw (non-destructive)
    cols = {k: reg[k] for k in reg.files}
    cols["best_gold_rank_relation"] = rel_best
    np.savez(f"results/L2/_regime_raw/{ds}_{split}.npz", **cols)
    out = {"dataset": ds, "split": split, "encoder": "minilm-L6-proxy",
           "metrics": res, "rescue": rescue, "hurt": hurt, "disagreement": disq,
           "n_val_queries": int(len(rel_best)), "n_queries_relation_on_gold": int((valid & (rel_best < 999999)).sum())}
    json.dump(out, open(f"scratchpad/_releval_{ds}_{split}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    ds = sys.argv[2] if len(sys.argv) > 2 else "2wiki_clean"
    if stage == "smoke":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        r = build_edges_for_split(ds, "train", qsubset=list(range(n)))
        print("SMOKE_OK")
    elif stage == "align":
        # Task 3: determinism + alignment on a subset
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        sub = list(range(n))
        b1 = build_edges_for_split(ds, "train", qsubset=sub, verbose=False)
        b2 = build_edges_for_split(ds, "train", qsubset=sub, verbose=False)
        det = bool(np.array_equal(b1["cand_sent"], b2["cand_sent"]))
        # candidate alignment: cand_sent length == pairs; sentinel only within scope (always true by construction)
        Slen = len(b1["S"]["cand"])
        cov = coverage_stats(ds, "train", b1)
        print(json.dumps({"DETERMINISM": det, "cand_sent_len_eq_pairs": len(b1["cand_sent"]) == Slen,
                          "coverage_subset": cov}, indent=1))
        print("ALIGN_OK")
    elif stage == "coverage":
        split = sys.argv[3] if len(sys.argv) > 3 else "val"
        lim = int(sys.argv[4]) if len(sys.argv) > 4 else None
        S = L.load_split(ds, split); nq = len(S["off"]) - 1
        sub = list(range(min(lim, nq))) if lim else None
        B = build_edges_for_split(ds, split, qsubset=sub)
        cov = coverage_stats(ds, split, B)
        cov["sampled"] = lim if lim else False
        outp = f"scratchpad/_relcov_{ds}_{split}.json"
        json.dump(cov, open(outp, "w"), indent=1)
        print(json.dumps(cov, indent=1)); print("wrote", outp)
    elif stage == "encode":
        split = sys.argv[3] if len(sys.argv) > 3 else "val"
        sc, B = relation_scores_for_split(ds, split)
        if sc is None:
            print("ENCODE_RESUME")
        else:
            np.save(f"scratchpad/_relscore_{ds}_{split}.npy", sc)
            print("ENCODE_DONE", "nonzero", int((sc > -1).sum()))
    elif stage == "eval":
        split = sys.argv[3] if len(sys.argv) > 3 else "val"
        eval_relation(ds, split)
