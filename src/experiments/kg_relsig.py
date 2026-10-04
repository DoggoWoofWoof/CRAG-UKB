"""Relation-semantic signal — the missing representation (engineering queue Track-A, priorities 1-2).

Diagnostic showed WebQSP is SIGNAL-saturated: oracle fusion of {dense,offset,splade,graph,prototype} = 39.1,
learned reranker = 38.8. The signals can't tell "Jamaica--official_language-->English" (GOLD) from its
structural siblings "Jamaica--capital-->Kingston" etc. — none encodes WHICH RELATION the question requests vs
which the candidate represents.

New signal:  s_rel(q,d) = cos(f_q(E(q)), f_r(E(relation_text_of_edge topic->d))).
  - raw     : f = identity (cheapest check — does gte-Qwen2 already separate?).
  - trained : f_q,f_r small heads, InfoNCE with SIBLING relations (other edges off the topic) as hard negatives.

Relation is represented as TEXT ("official language"), NOT a Freebase ID -> dataset-agnostic. This is
DIAGNOSTIC-ONLY: add s_rel to the oracle and ask whether Oracle{...,relation} >> 39.1. If yes, the signal
carries new information and is worth integrating into the reranker.
"""
import json
import logging
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _reltext(r):
    return r.split(".")[-1].replace("_", " ")


def _z(x):
    x = np.asarray(x, dtype="float64"); s = x.std()
    return (x - x.mean()) / (s + 1e-8) if s > 0 else x * 0.0


def _run(off_epochs=25, rel_epochs=40, n_dense=200, n_seed=20, hops=2, cap=64, tr_cap=1104, te_cap=2000,
         dataset="webqsp", path_cap=8, frontier_cap=4000, ner_edges=False, ner_cap=24, ctx_cap=240,
         path_q_cap=200, max_paths=150000):
    import os, re
    CTX_CAP = ctx_cap                              # per-side char cap on NER connecting sentences (memory + strips table dumps)
    from src.experiments.l1_universal_head import CoreEngine, load_docs_and_encoder, _load
    from src.experiments.l1l3_recall import _graph
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PASSAGE = ("2wiki_clean", "musique_clean", "hotpotqa_clean")     # passage corpora: edge = connecting SENTENCE, generic k-hop paths
    suffix = "_ner" if ner_edges else ""           # NER shares-entity dump writes to *_ner.* (never clobbers title-mention)
    d = _load(dataset, "gte_qwen", 8000, tr_cap, te_cap)
    X = d["X"].astype("float32"); id2idx = d["id2idx"]; splade = d.get("splade")
    qtr, str_tr, gtr = d["train"]; qte, ste, gte = d["test"]
    tr_txt, te_txt = d["train_texts"], d["test_texts"]
    Efull = F.normalize(torch.tensor(X, device=DEV), dim=1); N, DIM = X.shape; del X, d
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    _, A, _ = _graph(eng, N, id2idx, sources=("struct", "syn")); A = A.tocsr()
    _, eq, _ = load_docs_and_encoder(eng, dataset, "gte_qwen")

    mfile = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mfile):
        mfile = "data/processed/master_nodes.json"
    t2r = collections.defaultdict(list); t2r_lc = collections.defaultdict(list); id2content = {}
    idx2title = {}; content_by_row = {}
    for n in json.load(open(mfile, encoding="utf-8")):
        if n["node_id"] in id2idx:
            ix = id2idx[n["node_id"]]; ct = n.get("content", "")
            ti = n.get("metadata", {}).get("title", "").strip()
            if ti:
                t2r[ti].append(ix); t2r_lc[ti.lower()].append(ix); idx2title[ix] = ti
            id2content[n["node_id"]] = ct; content_by_row[ix] = ct
    ents = [e for e in t2r if e]; ent2i = {e: i for i, e in enumerate(ents)}
    doc2ent = torch.full((N,), -1, dtype=torch.long, device=DEV)
    for e, i in ent2i.items():
        doc2ent[torch.tensor(t2r[e], device=DEV)] = i
    m = doc2ent >= 0
    Eent = torch.zeros(len(ents), DIM, device=DEV).index_add_(0, doc2ent[m], Efull[m])
    cc = torch.zeros(len(ents), device=DEV).index_add_(0, doc2ent[m], torch.ones(m.sum(), device=DEV))
    Eent = F.normalize(Eent / cc.clamp(min=1)[:, None], dim=1)

    # ---- optional NER shares-entity edges (spaCy en_core_web_sm, no LLM) with two-sentence connecting context ----
    # Built in the SAME id2idx row space as everything else (no cache-alignment risk), faithful to the integrated
    # NER edge definition (2<=df<=25, weight 1/df). Provides: ner_nb (row -> [(nb,w,ent)] rarest-first) for the
    # connect() context path; doc_ent_sent (row -> {ent: sentence}) for the both-sentence context; Aexp (A ∪ NER)
    # for candidate-pool expansion so NER-reachable golds actually enter the pool (reachability, not just context).
    ner_nb = {}; doc_ent_sent = {}; Aexp = A
    if ner_edges and dataset in PASSAGE:
        import spacy
        import scipy.sparse as _sp
        _ENTL = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}
        _splitre = re.compile(r"(?<=[.!?])\s+")
        def _ssplit(t): return [s for s in _splitre.split(t) if s.strip()]
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        rows = sorted(content_by_row); ent_docs = collections.defaultdict(set)
        for r, doc in zip(rows, nlp.pipe([str(content_by_row[r]) for r in rows], batch_size=256)):
            ss = _ssplit(content_by_row[r]); esent = {}
            for e in doc.ents:
                if e.label_ in _ENTL and len(e.text) > 2:
                    el = e.text.lower().strip(); ent_docs[el].add(r)
                    if el not in esent:                                    # first sentence in THIS doc mentioning the entity
                        et = e.text.strip(); esent[el] = next((s for s in ss if et.lower() in s.lower()), et)
            doc_ent_sent[r] = esent
        best = {}                                                          # (u<v) -> (weight, rarest shared entity)
        for el, ds in ent_docs.items():
            df = len(ds)
            if 2 <= df <= 25:                                              # skip singletons and ubiquitous hubs (matches build_ner_edges)
                w = 1.0 / df; dl = sorted(ds)
                for _i in range(len(dl)):
                    for _j in range(_i + 1, len(dl)):
                        key = (dl[_i], dl[_j]); cur = best.get(key)
                        if cur is None or w > cur[0]:                      # keep the rarest shared entity (bridge over hub)
                            best[key] = (w, el)
        adjd = collections.defaultdict(list); r_, c_, v_ = [], [], []
        for (u, v), (w, el) in best.items():
            adjd[u].append((v, w, el)); adjd[v].append((u, w, el)); r_ += [u, v]; c_ += [v, u]; v_ += [w, w]
        for u in adjd:
            ner_nb[u] = sorted(adjd[u], key=lambda t: (-t[1], t[0]))[:ner_cap]  # rarest-entity bridges first (row# tiebreak = deterministic), then bound degree
        Aner = (_sp.csr_matrix((v_, (r_, c_)), shape=(N, N), dtype="float32") if r_
                else _sp.csr_matrix((N, N), dtype="float32"))
        Aexp = (A + Aner).tocsr()
        log.info("[kg-relsig] NER edges: %d undirected, %d docs w/ ents; Aexp nnz %d (A nnz %d)",
                 len(best), len(doc_ent_sent), Aexp.nnz, A.nnz)

    # per-question: topic entity docs (head) + edges topic->(relation, tail entity)
    rel_vocab = {}; path_vocab = {}; rel2pair = {}; rel2mask = {}; rel2shuf = {}; dbg2w = []
    q_edges = {}; q_paths = {}; q_topicrows = {}; q_goldrel = {}; q_hop = {}
    if dataset == "webqsp":                                                 # clean triples from the KB subgraph parquet
        import pandas as pd
        df = pd.concat([pd.read_parquet(f"data/raw/full/kb/webqsp_test{i}.parquet") for i in (0, 1)], ignore_index=True)
        for _, r in df.iterrows():
            q = str(r["question"]); tops = set(str(e).strip() for e in r["q_entity"]); ans = set(str(e).strip() for e in r["a_entity"])
            adj = collections.defaultdict(list)
            for t in r["graph"]:
                adj[str(t[0]).strip()].append((str(t[1]), str(t[2]).strip()))
            edges = collections.defaultdict(list); paths = collections.defaultdict(list); grel = set()
            for tp in tops:
                for rel, tl in adj[tp]:
                    ri = rel_vocab.setdefault(rel, len(rel_vocab)); edges[tl].append(ri)
                    if tl in ans:
                        grel.add(ri)
                    for rel2, tl2 in adj[tl]:
                        ri2 = rel_vocab.setdefault(rel2, len(rel_vocab))
                        paths[tl2].append(path_vocab.setdefault((ri, ri2), len(path_vocab)))
            q_edges[q] = edges; q_paths[q] = paths; q_goldrel[q] = grel
            q_topicrows[q] = [rr for e in tops for rr in t2r.get(e, [])]
            if any(tl in ans for tp in tops for _, tl in adj[tp]):
                q_hop[q] = "1hop"
            elif any(tl2 in ans for tp in tops for _, mid in adj[tp] for _, tl2 in adj[mid]):
                q_hop[q] = "2hop"
            else:
                q_hop[q] = "unreach"
    elif dataset == "metaqa":                                              # relations live in the doc text; parse the topic's facts
        METAREL = ["directed by", "written by", "starred actors", "release year", "in language",
                   "has genre", "has tags", "has imdb rating", "has imdb votes", "starred", "written", "directed"]
        title2content = {}
        for n in json.load(open(mfile, encoding="utf-8")):
            if n.get("metadata", {}).get("source") == "metaqa" and n["node_id"] in id2idx:
                title2content.setdefault(n.get("metadata", {}).get("title", "").strip(), n.get("content", ""))

        def parse_facts(content):
            out = []
            for seg in content.split("|"):
                seg = seg.strip()
                for rel in METAREL:
                    p = seg.find(" " + rel + " ")
                    if p >= 0:
                        obj = seg[p + len(rel) + 2:].strip().strip(";,.")
                        if obj:
                            out.append((rel, obj)); break
            return out
        for txt in list(tr_txt) + list(te_txt):
            tops = [e.strip() for e in re.findall(r"\[(.*?)\]", txt)]
            edges = collections.defaultdict(list)
            for tp in tops:
                for rel, obj in parse_facts(title2content.get(tp, "") or title2content.get(tp.title(), "")):
                    ri = rel_vocab.setdefault(rel, len(rel_vocab)); edges[obj].append(ri); edges[obj.lower()].append(ri)
            q_edges[txt] = edges; q_paths[txt] = {}
            q_topicrows[txt] = [rr for tp in tops for rr in (t2r.get(tp, []) or t2r_lc.get(tp.lower(), []))]
            q_hop[txt] = "1hop"
    elif dataset in PASSAGE:                                                # passage corpus: edge = connecting SENTENCE
        _wb = {}
        def _sents(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        def _specific(t): return len(t) >= 4 and re.search(r"[A-Z][a-z]", t) is not None  # proper-noun-ish only
        def _mentions(title, sent):                                        # case-sensitive whole-title word-boundary
            p = _wb.get(title)
            if p is None:
                p = _wb[title] = re.compile(r"(?<![A-Za-z])" + re.escape(title) + r"(?![A-Za-z])")
            return p.search(sent) is not None
        def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())
        ttoks = {t: set(_toks(t)) for t in ents}
        dfc = collections.Counter(w for ts in ttoks.values() for w in ts)
        keyidx = collections.defaultdict(list)
        for t in ents:
            if ttoks[t]:
                keyidx[min(ttoks[t], key=lambda w: dfc[w])].append(t)                  # index title by rarest token
        def find_topics(q):
            ql = q.lower(); qt = set(_toks(q)); seen = set(); found = []
            for w in (sorted(qt) if ner_edges else qt):                    # NER run: deterministic token order (2wiki record keeps set-order)
                for t in keyidx.get(w, ()):
                    if t in seen:
                        continue
                    seen.add(t)
                    if ttoks[t] and ttoks[t] <= qt and len(t) >= 3 and t.lower() in ql:
                        found.append(t)
            found.sort(key=(lambda s: (-len(s), s)) if ner_edges else (lambda s: -len(s))); topics = []  # NER: alpha tiebreak so topics[:4] is deterministic (gold-independent)
            for t in found:
                if not any(t != o and t.lower() in o.lower() for o in topics):
                    topics.append(t)
            return topics[:4]
        def connect(src_title):                                           # (tgt_title, connecting context, src_sentences)
            out = []
            for r in t2r.get(src_title, [])[:1]:
                ss = _sents(content_by_row.get(r, ""))
                for nb in A.indices[A.indptr[r]:A.indptr[r + 1]]:          # title-mention edge: the source sentence naming the target
                    tt = idx2title.get(int(nb), "")
                    if not tt or tt == src_title or not _specific(tt):
                        continue
                    for s in ss:
                        if _mentions(tt, s):
                            out.append((tt, s, ss)); break
                if ner_edges:                                             # shares-entity edge: BOTH connecting sentences (src[E] [SEP] tgt[E])
                    for nb, _w, el in ner_nb.get(r, []):
                        tt = idx2title.get(int(nb), "")
                        if not tt or tt == src_title:
                            continue
                        sa = doc_ent_sent.get(r, {}).get(el, ""); sb = doc_ent_sent.get(int(nb), {}).get(el, "")
                        if sa and sb:                                     # cap each side: MuSiQue has table-dump "sentences" (OOM + noise)
                            out.append((tt, f"{sa[:CTX_CAP]} [SEP] {sb[:CTX_CAP]}", ss))
            return out
        def _mask(sent, src, tgt):                                        # replace entity names -> [SRC]/[TGT] (relation words only)
            return sent.replace(tgt, "[TGT]").replace(src, "[SRC]")
        def _register(sent, src, tgt, ss):
            ri = rel_vocab.setdefault(sent, len(rel_vocab))
            rel2pair.setdefault(ri, f"{src} — {tgt}")
            rel2mask.setdefault(ri, _mask(sent, src, tgt))
            wrong = [x for x in ss if x != sent]                          # control: a WRONG sentence from the same source passage
            rel2shuf.setdefault(ri, max(wrong, key=len) if wrong else "n a")
            return ri
        path_capped = 0                                                     # count paths dropped by the global max_paths ceiling (logged, not silent)
        pathstat = collections.defaultdict(lambda: {"nq": 0, "gen": 0, "ret": 0, "capq": 0, "goldreach": 0})  # per-hop-bucket cap diagnostics (test only)
        for txts, gl in ((tr_txt, gtr), (te_txt, gte)):                    # leakage inspection: topics come ONLY from find_topics(txt)
            is_te = txts is te_txt
            for i in range(len(txts)):
                txt = txts[i]; golds = set(int(x) for x in gl[i]); topics = find_topics(txt); tset = set(topics)
                edges = collections.defaultdict(list); paths = collections.defaultdict(list)
                # GENERIC bounded k-hop enumeration from each topic (depth 1 = edge, depth>=2 = ordered path).
                # frontier item = (current_title, rel-index tuple, titles on THIS path). shortest paths first (BFS).
                # reduces EXACTLY to the old 2-hop nesting at hops=2 for sparse graphs (2wiki): the path_cap/per-question
                # ceilings are never reached, so every path is registered+expanded identically. Dense NER edges WOULD
                # explode the global path_vocab (-> multi-hour encode), so registration+expansion happen ONLY for paths
                # kept under path_cap (per candidate) and path_q_cap (PER QUESTION, so late questions aren't starved by
                # early ones — a global cap unfairly zeroed the path signal for later queries); max_paths is a global
                # safety ceiling. Discarded paths are neither encoded nor walked deeper (they cannot enter any bounded P_K).
                frontier = [(tp, (), {tp}) for tp in topics]; qpc = 0; qgen = 0
                for depth in range(1, hops + 1):
                    if not frontier:
                        break
                    nxt = []
                    for cur, rpath, ptitles in frontier:
                        for nb, sent, ss in connect(cur):
                            if nb in ptitles or (depth >= 2 and nb in tset):
                                continue
                            if depth == 1:                                 # edge: register + expand (unchanged)
                                ri = _register(sent, cur, nb, ss)
                                edges[nb].append(ri); edges[nb.lower()].append(ri)
                                nxt.append((nb, (ri,), ptitles | {nb}))
                            else:                                          # depth>=2: a path candidate (count BEFORE any cap for cap-sensitivity diagnostics)
                                qgen += 1
                                if len(paths[nb]) < path_cap and qpc < path_q_cap and len(path_vocab) < max_paths:  # per-candidate + per-question + global
                                    ri = _register(sent, cur, nb, ss); newpath = rpath + (ri,)
                                    pr = path_vocab.setdefault(newpath, len(path_vocab))
                                    paths[nb].append(pr); paths[nb.lower()].append(pr); qpc += 1
                                    nxt.append((nb, newpath, ptitles | {nb}))
                                elif len(path_vocab) >= max_paths:
                                    path_capped += 1
                    frontier = nxt[:frontier_cap]                          # bound construction-time frontier explosion
                q_edges[txt] = edges; q_paths[txt] = paths
                q_topicrows[txt] = [rr for tp in topics for rr in t2r.get(tp, [])]
                ge = {idx2title.get(g) for g in golds}                     # gold titles used ONLY here (bucket) + eval, never in edge build
                if dataset == "musique_clean":                            # #gold docs (2/3/4) = intrinsic hop depth (diagnostic label only)
                    q_hop[txt] = f"{len(golds)}hop" if golds else "unreach"
                else:
                    q_hop[txt] = "1hop" if (ge & set(edges)) else ("2hop" if (ge & set(paths)) else "unreach")
                if ner_edges and is_te:                                    # cap-sensitivity diagnostics per hop bucket (does path_q_cap starve the signal?)
                    st = pathstat[q_hop.get(txt, "?")]; st["nq"] += 1; st["gen"] += qgen; st["ret"] += qpc
                    st["capq"] += 1 if qpc >= path_q_cap else 0            # query hit the per-question cap
                    st["goldreach"] += 1 if any(k in ge for k in paths) else 0  # DIAGNOSTIC ONLY: any retained path tail is a gold (golds not used in construction)
                if is_te and len(dbg2w) < 20 and topics:                  # auditable examples (is_gold attached AFTER construction)
                    ce = connect(topics[0])
                    if ce:
                        mt, sent, _ss = ce[0]
                        dbg2w.append({"question": txt, "topics": topics, "source": topics[0], "target": mt,
                                      "edge_sentence": sent, "edge_pair": f"{topics[0]} — {mt}",
                                      "edge_masked": _mask(sent, topics[0], mt), "is_gold": mt in ge})
    rels = sorted(rel_vocab, key=lambda k: rel_vocab[k])
    paths_list = sorted(path_vocab, key=lambda k: path_vocab[k])            # list of variable-length rel-index tuples
    if ner_edges and dataset in PASSAGE:                                    # encode-volume + cap visibility (paths explode with dense NER)
        log.info("[kg-relsig] NER construct: %d rels, %d paths to encode (max_paths=%d, dropped=%d)",
                 len(rels), len(paths_list), max_paths, path_capped)
    Rpair = Rmask = Rshuf = None
    def _enc(txts):
        return F.normalize(torch.tensor(eq(txts).astype("float32"), device=DEV), dim=1) if txts else torch.zeros(0, DIM, device=DEV)
    if dataset in PASSAGE:                                                  # relation text = the raw connecting sentence
        Rtext = _enc(rels)
        if not ner_edges:                                                  # controls are the 2wiki title-mention ablation; skip for NER (3x the rel encode, not needed for the path test)
            Rpair = _enc([rel2pair.get(i, rels[i]) for i in range(len(rels))])            # control: "A — B" entity pair (connectivity only)
            Rmask = _enc([rel2mask.get(i, rels[i]) for i in range(len(rels))])            # control: "[SRC] directed [TGT]" (relation words only)
            Rshuf = _enc([rel2shuf.get(i, "n a") for i in range(len(rels))])              # control: a WRONG sentence from the source passage
        def build_path_context(p): return " [HOP] ".join(rels[r] for r in p)          # generic, arbitrary k: c_1 [HOP] c_2 [HOP] ...
    else:
        Rtext = _enc([_reltext(r) for r in rels])  # (n_rel,d); _enc guards empty rels (avoids faiss normalize_L2 crash)
        def build_path_context(p): return " then ".join(_reltext(rels[r]) for r in p)  # generic, arbitrary k
    ptxts = [build_path_context(p) for p in paths_list]
    Ptext = F.normalize(torch.tensor(eq(ptxts).astype("float32"), device=DEV), dim=1) if ptxts else torch.zeros(0, DIM, device=DEV)  # (n_path,d)

    def topic(txt, sd):
        rows = q_topicrows.get(txt, [])
        return F.normalize(Efull[torch.tensor(rows, device=DEV)].mean(0), dim=0) if rows else Efull[int(sd)]
    Qtr = F.normalize(torch.tensor(qtr.astype("float32"), device=DEV), dim=1)
    Qte = F.normalize(torch.tensor(qte.astype("float32"), device=DEV), dim=1)
    Htr = torch.stack([topic(tr_txt[i], str_tr[i]) for i in range(len(tr_txt))])
    Hte = torch.stack([topic(te_txt[i], ste[i]) for i in range(len(te_txt))])
    Gtr = [[int(x) for x in gtr[i]] for i in range(len(gtr))]
    Gte = [[int(x) for x in gte[i]] for i in range(len(gte))]
    spl_scorer, spl_mat = (splade if splade else (None, None))

    # offset signal
    class Off(nn.Module):
        def __init__(s):
            super().__init__(); s.m = nn.Sequential(nn.Linear(2 * DIM, 1024), nn.GELU(), nn.Linear(1024, DIM))
        def forward(s, hd, q): return F.normalize(hd + s.m(torch.cat([hd, q], -1)), dim=-1)
    torch.manual_seed(0); np.random.seed(0)
    off = Off().to(DEV); oo = torch.optim.Adam(off.parameters(), lr=1e-3)
    negp = torch.randint(0, N, (60000,), device=DEV); tri = [i for i in range(len(Gtr)) if Gtr[i]]
    for ep in range(off_epochs):
        np.random.shuffle(tri)
        for i in tri:
            v = off(Htr[i][None], Qtr[i][None]); cand = torch.cat([torch.tensor(Gtr[i], device=DEV), negp[torch.randint(0, 60000, (256,), device=DEV)]])
            sc = (Efull[cand] @ v.T).squeeze(-1); tgt = torch.zeros(len(cand), device=DEV); tgt[:len(Gtr[i])] = 1.0 / len(Gtr[i])
            loss = -(F.log_softmax(sc, 0) * tgt).sum(); oo.zero_grad(); loss.backward(); oo.step()
    off.eval()

    # TRAINED relation matcher: f_q(question), f_r(relation-text); InfoNCE with SIBLING relations as negatives
    class RelHead(nn.Module):
        def __init__(s):
            super().__init__()
            s.fq = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Linear(512, 256))
            s.fr = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Linear(512, 256))
        def q(s, x): return F.normalize(s.fq(x), dim=-1)
        def r(s, x): return F.normalize(s.fr(x), dim=-1)
    rh = RelHead().to(DEV); ro = torch.optim.Adam(rh.parameters(), lr=1e-3)
    train_q = [i for i in range(len(tr_txt)) if q_goldrel.get(tr_txt[i]) and q_edges.get(tr_txt[i])]
    for ep in range(rel_epochs):
        np.random.shuffle(train_q)
        for i in train_q:
            gr = list(q_goldrel[tr_txt[i]]); sib = list({ri for rl in q_edges[tr_txt[i]].values() for ri in rl})
            if not gr or len(sib) < 2:
                continue
            zq = rh.q(Qtr[i][None])                                         # (1,256)
            cand = torch.tensor(gr + [r for r in sib if r not in gr], device=DEV)
            zr = rh.r(Rtext[cand])                                          # (c,256)
            sc = (zq @ zr.T).squeeze(0)                                     # (c,)
            tgt = torch.zeros(len(cand), device=DEV); tgt[:len(gr)] = 1.0 / len(gr)
            loss = -(F.log_softmax(sc, 0) * tgt).sum(); ro.zero_grad(); loss.backward(); ro.step()
    rh.eval()

    SIGS = ["dense", "offset", "prototype", "splade", "graph"]
    KS = (1, 5, 20, 50)

    @torch.no_grad()
    def per_query(q, h, txt, qvec):
        sim = q @ Efull.T
        order = torch.topk(sim, max(n_dense, n_seed)).indices.cpu().numpy()
        dense_set = set(int(x) for x in order[:n_dense]); seeds = [int(x) for x in order[:n_seed]]
        hop = {int(x): 0 for x in dense_set}; frontier = set(seeds)
        for hh in range(1, hops + 1):
            nxt = set()
            for s in frontier:
                for x in Aexp.indices[Aexp.indptr[s]:Aexp.indptr[s + 1]][:cap]:   # Aexp = A ∪ NER when --ner-edges (else == A)
                    x = int(x)
                    if x not in hop:
                        hop[x] = hh; nxt.add(x)
            frontier = nxt
        pool = list(hop.keys()); P = torch.tensor(pool, device=DEV); dp = Efull[P]
        v = off(h[None], q[None])[0]
        sc = {
            "dense": (dp @ q).cpu().numpy(), "offset": (dp @ v).cpu().numpy(),
            "prototype": ((Eent[doc2ent[P].clamp(min=0)] * q).sum(1) * (doc2ent[P] >= 0)).cpu().numpy(),
            "splade": (spl_mat[np.array(pool)].dot(spl_scorer.encode_query(txt)) if spl_scorer else np.zeros(len(pool))),
            "graph": -np.array([hop[p] for p in pool], dtype="float32"),
        }
        # relation of each candidate = relation on edge topic->candidate-entity (from the KB subgraph)
        edges = q_edges.get(txt, {})
        de = doc2ent[P].cpu().numpy()
        rel_raw = np.full(len(pool), -1.0, dtype="float32"); rel_tr = np.full(len(pool), -1.0, dtype="float32")
        cand_rel_idx = [-1] * len(pool)
        for j, e in enumerate(de):
            if e < 0:
                continue
            rl = edges.get(ents[e], [])
            if rl:
                cand_rel_idx[j] = rl[0]
        have = [j for j in range(len(pool)) if cand_rel_idx[j] >= 0]
        if have:
            ri = torch.tensor([cand_rel_idx[j] for j in have], device=DEV)
            raw = (Rtext[ri] @ qvec).cpu().numpy()                          # raw cos(E(q), E(rel))
            zq = rh.q(qvec[None]); tr = (rh.r(Rtext[ri]) @ zq.squeeze(0)).cpu().numpy()  # trained
            for idx, j in enumerate(have):
                rel_raw[j] = raw[idx]; rel_tr[j] = tr[idx]
        sc["relation_raw"] = rel_raw; sc["relation_trained"] = rel_tr
        for nm, MAT in (("relation_pair_raw", Rpair), ("relation_masked_raw", Rmask), ("relation_shuf_raw", Rshuf)):
            if MAT is None:                                               # controls only exist for 2wiki
                continue
            arr = np.full(len(pool), -1.0, dtype="float32")
            if have:
                v = (MAT[ri] @ qvec).cpu().numpy()
                for idx, j in enumerate(have):
                    arr[j] = v[idx]
            sc[nm] = arr
        # multi-hop PATH signal: for candidates reached only via a >=2-hop path, BOUNDED MAX over its plausible
        # paths  s_path(q,d) = max_{p in P_K(d)} cos(E(q), E(context(p)))  (P_K bounded at construction, not tuned)
        paths = q_paths.get(txt, {}); path_raw = np.full(len(pool), -1.0, dtype="float32")
        if len(Ptext):
            for j, e in enumerate(de):
                if e < 0 or cand_rel_idx[j] >= 0:                           # only path-only candidates (no 1-hop edge)
                    continue
                pl = paths.get(ents[e], [])
                if pl:
                    path_raw[j] = float((Ptext[torch.tensor(pl, device=DEV)] @ qvec).max())
        sc["path_raw"] = path_raw
        return pool, sc

    CONTROLS = (dataset in PASSAGE) and not ner_edges                       # pair/mask/shuf ablations only for the 2wiki title-mention run
    allsig = SIGS + ["relation_raw", "relation_trained", "path_raw"]
    if CONTROLS:
        allsig = allsig + ["relation_pair_raw", "relation_masked_raw", "relation_shuf_raw"]
    perR = {s: {k: [] for k in KS} for s in allsig}
    # oracle sets: old(5) | +relation | +relation+path ; tracked overall and stratified by hop bucket
    OSETS = {"old": SIGS, "+relation": SIGS + ["relation_raw"], "+relation+path": SIGS + ["relation_raw", "path_raw"]}
    if CONTROLS:                                                            # edge_sentence (=+relation) vs edge_pair
        OSETS["+edge_pair"] = SIGS + ["relation_pair_raw"]
    orc = {name: {k: [] for k in KS} for name in OSETS}
    orc_hop = collections.defaultdict(lambda: {name: [] for name in OSETS})   # R@5 by hop bucket (dynamic keys: 1/2/3/4hop/unreach)
    resc = collections.defaultdict(lambda: {name: [] for name in OSETS})      # oracle R@5 by reachability subset: title_reachable / ner_rescued / still_unreachable
    def _reach_all(trows, golds, ADJ):                                        # all golds reachable from topic rows within `hops` over adjacency ADJ (cap-bounded, matches probe)
        if not trows or not golds:
            return False
        seen = set(int(x) for x in trows); frontier = set(seen)
        for _ in range(hops):
            nx = set()
            for s in frontier:
                for x in ADJ.indices[ADJ.indptr[s]:ADJ.indptr[s + 1]][:cap]:
                    x = int(x)
                    if x not in seen:
                        seen.add(x); nx.add(x)
            frontier = nx
            if not frontier:
                break
        return all(g in seen for g in golds)
    psig_hop = collections.defaultdict(lambda: collections.defaultdict(list))  # per-signal R@5 by hop (musique diagnostic table)
    moved = collections.Counter(); nrel_gold = 0
    # topic-gold vs bridge-gold vs overall recall — the bridge subset is where edge semantics should matter
    SUBSIG = ["dense", "offset", "splade", "prototype", "relation_raw", "path_raw"] + (["relation_pair_raw", "relation_masked_raw", "relation_shuf_raw"] if CONTROLS else [])
    sub = {g: {s: [] for s in SUBSIG} for g in ("topic", "bridge", "all")}
    osub = {g: {n: [] for n in OSETS} for g in ("topic", "bridge", "all")}
    edge_gold = []; edge_fp = []                                            # inspect real vs false-positive high-scoring edges
    def r_at(sc, sset, pool, gs, ng, k):
        return max(len(gs & set(pool[j] for j in np.argsort(-sc[s])[:k])) / ng for s in sset)
    for i in range(len(Gte)):
        if not Gte[i]:
            continue
        pool, sc = per_query(Qte[i], Hte[i], te_txt[i], Qte[i]); gs = set(Gte[i]); ng = len(gs)
        pos = {p: j for j, p in enumerate(pool)}
        if dataset in PASSAGE and (len(edge_gold) < 12 or len(edge_fp) < 12):  # reconstruct edge sentence per scored candidate
            edg = q_edges.get(te_txt[i], {}); rr = sc["relation_raw"]
            for j in sorted((j for j in range(len(pool)) if rr[j] > -1), key=lambda j: -rr[j])[:4]:
                ent = idx2title.get(pool[j], ""); rl = edg.get(ent) or edg.get(ent.lower())
                rec = {"question": te_txt[i][:100], "target": ent, "rel_score": round(float(rr[j]), 3),
                       "is_gold": pool[j] in gs, "edge_sentence": (rels[rl[0]][:160] if rl else "")}
                if rec["is_gold"] and len(edge_gold) < 12:
                    edge_gold.append(rec)
                elif not rec["is_gold"] and len(edge_fp) < 12:
                    edge_fp.append(rec)
        if dataset in PASSAGE:
            tg = gs & set(q_topicrows.get(te_txt[i], [])); subg = {"topic": tg, "bridge": gs - tg, "all": gs}
            top5 = {s: set(pool[j] for j in np.argsort(-sc[s])[:5]) for s in SUBSIG}
            for gname, gset in subg.items():
                if not gset:
                    continue
                for s in SUBSIG:
                    sub[gname][s].append(len(gset & top5[s]) / len(gset))
                for name, sset in OSETS.items():
                    osub[gname][name].append(max(len(gset & set(pool[j] for j in np.argsort(-sc[s])[:5])) / len(gset) for s in sset))
        for s in allsig:
            order = np.argsort(-sc[s])
            for k in KS:
                perR[s][k].append(len(gs & set(pool[j] for j in order[:k])) / ng)
            if dataset == "musique_clean":                                  # per-signal R@5 stratified by hop (diagnostic table)
                bkt = q_hop.get(te_txt[i])
                if bkt:
                    psig_hop[bkt][s].append(len(gs & set(pool[j] for j in order[:5])) / ng)
        for name, sset in OSETS.items():
            for k in KS:
                orc[name][k].append(r_at(sc, sset, pool, gs, ng, k))
        b = q_hop.get(te_txt[i])
        if b:
            for name, sset in OSETS.items():
                orc_hop[b][name].append(r_at(sc, sset, pool, gs, ng, 5))
        if ner_edges:                                                       # reachability subset: did A->Aexp change chain availability for THIS question?
            trows = q_topicrows.get(te_txt[i], [])
            lab = ("title_reachable" if _reach_all(trows, gs, A)
                   else ("ner_rescued" if _reach_all(trows, gs, Aexp) else "still_unreachable"))
            for name, sset in OSETS.items():
                resc[lab][name].append(r_at(sc, sset, pool, gs, ng, 5))
        # rank movement
        oldrank = {}
        for s in SIGS:
            ranks = np.empty(len(pool), dtype=np.int64); ranks[np.argsort(-sc[s])] = np.arange(len(pool))
            for g in gs:
                if g in pos:
                    oldrank[g] = min(oldrank.get(g, 10 ** 9), int(ranks[pos[g]]) + 1)
        rr_ranks = np.empty(len(pool), dtype=np.int64); rr_ranks[np.argsort(-sc["relation_trained"])] = np.arange(len(pool))
        for g in gs:
            if g in pos and sc["relation_trained"][pos[g]] > -1:
                nrel_gold += 1
                new = int(rr_ranks[pos[g]]) + 1; old = oldrank.get(g, 10 ** 9)
                if old > 5 >= new:
                    moved["into_top5"] += 1
                elif new <= 5:
                    moved["stayed_top5"] += 1

    # ---- priority 3/5: REAL reranker over signals; ablate none / +relation / +relation+path ----
    FE = ["dense", "offset", "prototype", "splade", "graph", "relation_raw", "path_raw"]
    # FE column indices whose expert can be genuinely ABSENT for a candidate. dense(0)/offset(1)/graph(4)
    # are always computable; prototype(2) needs an entity; splade(3) is dataset-level (real for all or absent
    # for all — masked, NEVER faked-0); relation_raw(5)/path_raw(6) need an edge/path. The mask is authoritative:
    # downstream trusts `avail`, never a threshold on the (z-scored) score.
    SPL_AVAIL = spl_scorer is not None
    def collect(ix, Qs, Hs, txts, Gs):
        data = []
        for i in ix:
            if not Gs[i]:
                continue
            pool, sc = per_query(Qs[i], Hs[i], txts[i], Qs[i])
            x = torch.tensor(np.stack([sc[s] for s in FE], 1), dtype=torch.float32, device=DEV)
            de = doc2ent[torch.tensor(pool, device=DEV)].cpu().numpy()          # entity row per candidate (-1 = none)
            av = np.ones((len(pool), len(FE)), dtype="float32")
            av[:, 2] = (de >= 0)                                                # prototype
            av[:, 3] = 1.0 if SPL_AVAIL else 0.0                                # splade (dataset-level, mask not fake-0)
            av[:, 5] = (sc["relation_raw"] > -0.5)                              # relation
            av[:, 6] = (sc["path_raw"] > -0.5)                                  # path
            y = torch.tensor([1.0 if p in set(Gs[i]) else 0.0 for p in pool], device=DEV)
            if y.sum() > 0:
                data.append((x, y, len(Gs[i]), torch.tensor(av, device=DEV)))
        return data
    trd = collect(tri, Qtr, Htr, tr_txt, Gtr)
    ted = collect([i for i in range(len(Gte)) if Gte[i]], Qte, Hte, te_txt, Gte)
    # PRESENT-ONLY normalization: missing entries (avail=0) are excluded from mu/sd so the sentinel can
    # never become a weak-real z-score. Always-present columns are unchanged (mask all-ones).
    allx = torch.cat([x for x, _, _, _ in trd], 0); allav = torch.cat([a for *_, a in trd], 0)
    cnt = allav.sum(0).clamp(min=1.0)
    mu = (allx * allav).sum(0) / cnt
    sdv = (((allx - mu) ** 2 * allav).sum(0) / cnt).sqrt().clamp(min=1e-6)
    def _mz(x, a):                                                              # masked z-score: standardize present, zero missing (no sentinel leak)
        return ((x - mu) / sdv) * a
    import os as _os2; _os2.makedirs("results/L2", exist_ok=True)                # export cached features for the hybrid
    torch.save({"train": [(x.cpu(), y.cpu(), ng, a.cpu()) for x, y, ng, a in trd],
                "test": [(x.cpu(), y.cpu(), ng, a.cpu()) for x, y, ng, a in ted],
                "mu": mu.cpu(), "sd": sdv.cpu(), "feat_names": FE, "avail_names": FE,
                "splade_available": bool(SPL_AVAIL), "masked": True},
               f"results/L2/relsig_feats_{dataset}{suffix}.pt")

    def train_eval(ncols, seeds=3):
        cols = list(range(ncols)); r5s = []
        for sd in range(seeds):
            torch.manual_seed(sd)
            net = nn.Sequential(nn.Linear(ncols, 64), nn.GELU(), nn.Linear(64, 1)).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3); order = list(range(len(trd)))
            for ep in range(8):
                np.random.shuffle(order)
                for j in order:
                    x, y, _, a = trd[j]; s = net(_mz(x, a)[:, cols]).squeeze(-1)
                    loss = -(F.log_softmax(s, 0) * (y / y.sum())).sum(); opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); rec = []
            with torch.no_grad():
                for x, y, ng, a in ted:
                    s = net(_mz(x, a)[:, cols]).squeeze(-1).cpu().numpy()
                    rec.append(y.cpu().numpy()[np.argsort(-s)[:5]].sum() / ng)
            r5s.append(100 * np.mean(rec))
        return round(float(np.mean(r5s)), 2), round(float(np.std(r5s)), 2)
    rr_none = train_eval(5); rr_rel = train_eval(6); rr_path = train_eval(7)
    log.info("[kg-relsig] REAL reranker R@5: none %s | +relation %s | +relation+path %s", rr_none, rr_rel, rr_path)

    # Branch A: candidate-level RESCUE gate. score = offset + g_rel·relation + g_path·path (g per-candidate in [0,1]),
    # so relation/path only correct the offset where the gate fires (fixes offset-collapse on high-offset datasets).
    class GateRR(nn.Module):
        def __init__(s):
            super().__init__(); s.g = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 2))
        def forward(s, xz):
            g = torch.sigmoid(s.g(xz))
            return xz[:, 1] + g[:, 0] * xz[:, 5] + g[:, 1] * xz[:, 6]      # offset_z + g_rel·rel_z + g_path·path_z
    def _rank(v):                                                          # 0 = best
        r = torch.empty_like(v, dtype=torch.long); r[torch.argsort(-v)] = torch.arange(len(v), device=v.device); return r
    def train_eval_gated(seeds=5, supervise=False, lam=1.0):
        r5 = []; rescued = []; rescuable = []
        for sdd in range(seeds):
            torch.manual_seed(sdd)
            net = GateRR().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); order = list(range(len(trd)))
            for ep in range(8):
                np.random.shuffle(order)
                for j in order:
                    x, y, _, a = trd[j]; xz = _mz(x, a)
                    glog = net.g(xz); g = torch.sigmoid(glog)
                    s = xz[:, 1] + g[:, 0] * xz[:, 5] + g[:, 1] * xz[:, 6]
                    loss = -(F.log_softmax(s, 0) * (y / y.sum())).sum()
                    if supervise:                                          # §7: teach the gate directly on golds
                        gm = y > 0
                        if gm.any():
                            orank = _rank(xz[:, 1]); rrank = _rank(xz[:, 5]); prank = _rank(xz[:, 6])
                            gt = torch.stack([(rrank < orank).float(), (prank < orank).float()], 1)[gm]
                            loss = loss + lam * F.binary_cross_entropy_with_logits(glog[gm], gt)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); rec = []; rc = 0; rb = 0
            with torch.no_grad():
                for x, y, ng, a in ted:
                    xz = _mz(x, a); s = net(xz).cpu().numpy(); yn = y.cpu().numpy()
                    top = set(np.argsort(-s)[:5]); rec.append(sum(yn[j] for j in top) / ng)
                    # rescue subset: golds where offset ranks >5 but relation ranks <=5
                    orank = np.empty(len(yn), np.int64); orank[np.argsort(-xz[:, 1].cpu().numpy())] = np.arange(len(yn))
                    rrank = np.empty(len(yn), np.int64); rrank[np.argsort(-xz[:, 5].cpu().numpy())] = np.arange(len(yn))
                    for j in range(len(yn)):
                        if yn[j] and orank[j] >= 5 > rrank[j]:
                            rb += 1; rc += 1 if j in top else 0
            r5.append(100 * np.mean(rec)); rescued.append(rc); rescuable.append(rb)
        return (round(float(np.mean(r5)), 2), round(float(np.std(r5)), 2),
                {"rescued": int(np.mean(rescued)), "rescuable": int(np.mean(rescuable))})
    rr_gated = train_eval_gated(5)
    rr_gated_sup = train_eval_gated(5, supervise=True)
    log.info("[kg-relsig] GATED R@5: indirect %s | SUPERVISED %s | subset(sup) %s", rr_gated[:2], rr_gated_sup[:2], rr_gated_sup[2])

    # ---- KL SOFT-ROUTER: soft expert mixture, router trained by KL to a per-query expert-quality teacher ----
    EXPERTS = [0, 1, 3, 5, 6]  # dense, offset, splade, relation, path  (columns of the feature tensor)
    def _expert_loss(s, y, M=20):                                        # bounded InfoNCE: gold vs top-M hard negs -> scalar
        g = s[y > 0]
        if len(g) == 0:
            return torch.tensor(8.0, device=s.device)
        neg = s[y == 0]
        if len(neg) > M:
            neg = torch.topk(neg, M).values
        return -(torch.logsumexp(g, 0) - torch.logsumexp(torch.cat([g, neg]), 0))
    def _stats(s):                                                       # per-expert distribution summary for the router
        p = F.softmax(s, 0); ent = -(p * (p + 1e-9).log()).sum()
        tp = torch.topk(s, min(2, len(s))).values
        return torch.stack([ent, (tp[0] - tp[-1]) if len(tp) > 1 else tp[0] * 0, s.max()])
    class Router(nn.Module):
        def __init__(s, nin):
            super().__init__(); s.m = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, len(EXPERTS)))
        def forward(s, z): return s.m(z)
    def train_eval_router(seeds=5, tau=0.1, tau_r=1.0, lam=1.0, drop=0.25):
        r5 = []
        nin = 3 * len(EXPERTS) + 2                                       # stats per expert + relation/path availability
        for sdd in range(seeds):
            torch.manual_seed(sdd)
            R = Router(nin).to(DEV); opt = torch.optim.Adam(R.parameters(), lr=1e-3)
            # rescue oversampling: weight queries where offset misses a gold that relation/path ranks top-5
            order = list(range(len(trd)))
            for ep in range(10):
                np.random.shuffle(order)
                for j in order:
                    x, y, _, a = trd[j]; xz = _mz(x, a)
                    se = [xz[:, c] for c in EXPERTS]
                    with torch.no_grad():
                        Le = torch.stack([_expert_loss(s, y) for s in se])
                        pi = F.softmax(-Le / tau_r, 0)
                    avail = torch.stack([a[:, 5].amax(), a[:, 6].amax()])
                    z = torch.cat([torch.cat([_stats(s) for s in se]), avail])
                    logit = R(z)
                    if np.random.rand() < drop:                          # expert dropout: force reliance off the dominant one
                        logit = logit.clone(); logit[np.random.randint(len(EXPERTS))] = -1e4
                    alpha = F.softmax(logit, 0)
                    S = sum(alpha[k] * se[k] for k in range(len(EXPERTS)))
                    rank = -(F.log_softmax(S, 0) * (y / y.sum())).sum()
                    kl = (pi * ((pi + 1e-9).log() - F.log_softmax(logit, 0))).sum()
                    loss = rank + lam * kl
                    opt.zero_grad(); loss.backward(); opt.step()
            R.eval(); rec = []
            with torch.no_grad():
                for x, y, ng, a in ted:
                    xz = _mz(x, a); se = [xz[:, c] for c in EXPERTS]
                    avail = torch.stack([a[:, 5].amax(), a[:, 6].amax()])
                    z = torch.cat([torch.cat([_stats(s) for s in se]), avail])
                    alpha = F.softmax(R(z), 0)
                    S = sum(alpha[k] * se[k] for k in range(len(EXPERTS))).cpu().numpy()
                    rec.append(y.cpu().numpy()[np.argsort(-S)[:5]].sum() / ng)
            r5.append(100 * np.mean(rec))
        return round(float(np.mean(r5)), 2), round(float(np.std(r5)), 2)
    rr_router = train_eval_router(5)
    log.info("[kg-relsig] KL SOFT-ROUTER R@5: %s", rr_router)

    def m5(d):  # mean R@5 helper
        return round(100 * float(np.mean(d[5])), 2)
    canon = ("2hop", "3hop", "4hop", "unreach") if dataset == "musique_clean" else ("1hop", "2hop", "unreach")
    hop_out = list(canon) + [b for b in orc_hop if b not in canon]          # canonical order; keeps empty buckets as null
    out = {"dataset": dataset, "ner_edges": ner_edges, "hops": hops, "path_cap": path_cap,
           "reranker_R@5": {"none": rr_none, "+relation": rr_rel, "+relation+path": rr_path,
                            "gated_rescue": rr_gated[:2], "gated_rescue_supervised": rr_gated_sup[:2], "gated_rescue_subset": rr_gated_sup[2], "kl_router": rr_router},
           "per_signal_R@5": {s: round(100 * np.mean(perR[s][5]), 2) for s in allsig},
           "oracle_R@5": {name: m5(orc[name]) for name in OSETS},
           "oracle_R@20": {name: round(100 * float(np.mean(orc[name][20])), 2) for name in OSETS},
           "oracle_R@5_by_hop": {b: {name: round(100 * float(np.mean(orc_hop[b][name])), 2) if orc_hop[b][name] else None
                                     for name in OSETS} for b in hop_out},
           "n_by_hop": {b: len(orc_hop[b]["old"]) for b in hop_out},
           "rank_movement": dict(moved), "n_golds_with_relation": nrel_gold,
           "reference": {"prev_reranker_R@5": 38.8, "prev_oracle_R@5": 39.1, "benchmark": 29.7}}
    if dataset in PASSAGE:
        out["per_signal_R@1"] = {s: round(100 * np.mean(perR[s][1]), 2) for s in allsig}       # R@1 makes edge_sentence vs edge_pair discriminative
        out["oracle_R@1"] = {name: round(100 * np.mean(orc[name][1]), 2) for name in OSETS}
        out["subset_R@5"] = {g: {s: round(100 * np.mean(v), 2) if v else None for s, v in sub[g].items()} for g in sub}
        out["oracle_subset_R@5"] = {g: {n: round(100 * np.mean(v), 2) if v else None for n, v in osub[g].items()} for g in osub}
        out["n_subset"] = {g: len(next(iter(sub[g].values()))) for g in sub}
        out["debug_examples_2wiki"] = dbg2w                                # 20 examples for manual leakage inspection
        out["debug_edges_gold"] = edge_gold                                # high-scoring GOLD bridge edges (sentences)
        out["debug_edges_fp"] = edge_fp                                    # high-scoring WRONG bridge edges (sentences)
    if dataset == "musique_clean":                                         # full hop-stratified diagnostic table (per-signal x 2/3/4hop)
        out["per_signal_R@5_by_hop"] = {b: {s: round(100 * float(np.mean(psig_hop[b][s])), 2) if psig_hop[b][s] else None
                                            for s in allsig} for b in hop_out if b in psig_hop}
    if ner_edges:                                                          # DECISIVE table: does +path help where the graph became able to reach the chain?
        rl_out = ["title_reachable", "ner_rescued", "still_unreachable"]
        out["reach_subset_R@5"] = {lab: ({name: round(100 * float(np.mean(resc[lab][name])), 2) for name in OSETS}
                                         if resc[lab]["old"] else None) for lab in rl_out}
        out["reach_subset_n"] = {lab: len(resc[lab]["old"]) for lab in rl_out}
        # cap-sensitivity: if a bucket's queries frequently hit path_q_cap AND gen>>ret, a null +path may be starvation not absence
        out["path_cap_diag"] = {b: {"n": s["nq"], "gen_per_q": round(s["gen"] / max(s["nq"], 1), 1),
                                    "ret_per_q": round(s["ret"] / max(s["nq"], 1), 1),
                                    "frac_hit_cap": round(s["capq"] / max(s["nq"], 1), 3),
                                    "frac_path_reaches_gold": round(s["goldreach"] / max(s["nq"], 1), 3)}
                                for b, s in sorted(pathstat.items())}
        out["path_q_cap"] = path_q_cap; out["max_paths"] = max_paths; out["ner_cap"] = ner_cap
        log.info("[kg-relsig] REACH-SUBSET R@5 %s (n=%s)", out["reach_subset_R@5"], out["reach_subset_n"])
        log.info("[kg-relsig] PATH-CAP diag %s", out["path_cap_diag"])
    log.info("[kg-relsig] ORACLE R@5 %s | by-hop %s", out["oracle_R@5"], out["oracle_R@5_by_hop"])
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open(f"results/L2/kg_relsig_{dataset}{suffix}.json", "w"), indent=2)
    log.info("-> results/L2/kg_relsig_%s%s.json", dataset, suffix)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--rel-epochs", type=int, default=40)
    p.add_argument("--dataset", default="webqsp")
    p.add_argument("--tr-cap", type=int, default=1104)
    p.add_argument("--te-cap", type=int, default=2000)
    p.add_argument("--hops", type=int, default=2)                           # traversal depth for generic k-hop paths
    p.add_argument("--path-cap", type=int, default=8)                       # bounded P_K paths per candidate (not tuned on test)
    p.add_argument("--ner-edges", action="store_true")                      # add spaCy shares-entity edges (two-sentence context) to traversal
    p.add_argument("--ner-cap", type=int, default=24)                       # max NER neighbors per doc (rarest-entity bridges first)
    p.add_argument("--path-q-cap", type=int, default=200)                   # PER-QUESTION path budget (fair across queries)
    p.add_argument("--max-paths", type=int, default=150000)                 # global safety ceiling on unique paths encoded
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(off_epochs=a.off_epochs, rel_epochs=a.rel_epochs, dataset=a.dataset, tr_cap=a.tr_cap, te_cap=a.te_cap,
         hops=a.hops, path_cap=a.path_cap, ner_edges=a.ner_edges, ner_cap=a.ner_cap,
         path_q_cap=a.path_q_cap, max_paths=a.max_paths)


if __name__ == "__main__":
    main()
