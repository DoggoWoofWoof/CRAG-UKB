"""Relation-conditioned structural walk (universal in form: uses only the served relation labels of
the structural family and the query text; a graph without relation labels degenerates to the
untyped walk).

Query -> relation matching is purely lexical + generic English morphology (Porter stem + a small
irregular-verb lemma table; NO dataset-specific synonym list): a relation type matches the query
when one of its label tokens and one of the query tokens (the top retrieval seed's own name masked
out) share a stem prefix.  Transition at a node = 1/2 untyped uniform edge choice + 1/2 uniform
choice among edges whose relation matched (the untyped half alone when nothing matched).
"""
import json
import os
import re
import sys

import numpy as np
import scipy.sparse as sp

import _l1x90_core as X

STOP = set("a an the of in on by to for is are was were be been do does did what which who whom whose when where how "
           "that this these those with as at from and or not it its has have had also same share shares list listed "
           "person people film films movie movies name names".split())

IRREGULAR = {  # generic English irregular past forms -> lemma (not dataset specific)
    "wrote": "write", "written": "write", "spoke": "speak", "spoken": "speak", "sang": "sing", "sung": "sing",
    "ran": "run", "made": "make", "born": "bear", "bore": "bear", "took": "take", "taken": "take", "gave": "give",
    "given": "give", "went": "go", "gone": "go", "came": "come", "saw": "see", "seen": "see", "knew": "know",
    "known": "know", "grew": "grow", "grown": "grow", "drew": "draw", "drawn": "draw", "began": "begin",
    "begun": "begin", "built": "build", "bought": "buy", "brought": "bring", "caught": "catch", "chose": "choose",
    "chosen": "choose", "did": "do", "done": "do", "drove": "drive", "driven": "drive", "fell": "fall", "fallen": "fall",
    "felt": "feel", "found": "find", "flew": "fly", "flown": "fly", "forgot": "forget", "got": "get", "held": "hold",
    "kept": "keep", "led": "lead", "left": "leave", "lost": "lose", "met": "meet", "paid": "pay", "put": "put",
    "read": "read", "rode": "ride", "ridden": "ride", "rose": "rise", "risen": "rise", "said": "say", "sold": "sell",
    "sent": "send", "set": "set", "shot": "shoot", "showed": "show", "shown": "show", "sat": "sit", "slept": "sleep",
    "spent": "spend", "stood": "stand", "swam": "swim", "taught": "teach", "told": "tell", "thought": "think",
    "threw": "throw", "thrown": "throw", "understood": "understand", "wore": "wear", "worn": "wear", "won": "win",
    "woke": "wake", "woken": "wake", "starred": "star", "stars": "star", "acted": "act", "acts": "act", "died": "die",
    "married": "marry", "founded": "found", "played": "play", "produced": "produce",
}


# ---------------------------------------------------------------- Porter stemmer (Porter 1980), compact
def _cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(w):
    n = 0
    i = 0
    L = len(w)
    while i < L and _cons(w, i):
        i += 1
    while i < L:
        while i < L and not _cons(w, i):
            i += 1
        if i >= L:
            break
        n += 1
        while i < L and _cons(w, i):
            i += 1
    return n


def _vowel(w):
    return any(not _cons(w, i) for i in range(len(w)))


def _dbl(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    return len(w) >= 3 and _cons(w, len(w) - 1) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 3) and w[-1] not in "wxy"


def porter(w):
    if len(w) <= 2:
        return w
    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # step 1b
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    else:
        f = None
        if w.endswith("ed") and _vowel(w[:-2]):
            f = w[:-2]
        elif w.endswith("ing") and _vowel(w[:-3]):
            f = w[:-3]
        if f is not None:
            w = f
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif _dbl(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif _m(w) == 1 and _cvc(w):
                w += "e"
    # step 1c
    if w.endswith("y") and _vowel(w[:-1]):
        w = w[:-1] + "i"
    # step 2
    for suf, rep in (("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
                     ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
                     ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
                     ("iviti", "ive"), ("biliti", "ble")):
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # step 3
    for suf, rep in (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")):
        if w.endswith(suf):
            if _m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # step 4
    for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
                "ous", "ive", "ize"):
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if suf == "ion":
                if stem and stem[-1] in "st" and _m(stem) > 1:
                    w = stem
            elif _m(stem) > 1:
                w = stem
            break
    # step 5a
    if w.endswith("e"):
        s = w[:-1]
        if _m(s) > 1 or (_m(s) == 1 and not _cvc(s)):
            w = s
    # step 5b
    if _m(w) > 1 and _dbl(w) and w.endswith("l"):
        w = w[:-1]
    return w


def toks(text):
    out = []
    for t in re.findall(r"[a-z]+", text.lower()):
        t = IRREGULAR.get(t, t)
        if t in STOP or len(t) < 3:
            continue
        out.append(porter(t))
    return out


def tok_match(a, b):
    """generic conflation: equal stems, or one stem (>=3 chars) is a prefix of the other, or a common
    prefix of >= 4 characters."""
    if a == b:
        return True
    if len(a) >= 3 and b.startswith(a):
        return True
    if len(b) >= 3 and a.startswith(b):
        return True
    if len(a) >= 4 and a in b:
        return True
    if len(b) >= 4 and b in a:
        return True
    # compound suffix: 'screenwrit(er)' ~ 'write', 'cowrit' ~ 'write' (stem minus a final e)
    for s_, l_ in ((a, b), (b, a)):
        if len(s_) >= 4 and len(l_) > len(s_) and (l_.endswith(s_) or (s_.endswith("e") and l_.endswith(s_[:-1]))):
            return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 4


class Typed:
    """relation-typed undirected multigraph of a dataset's structural family + the query matcher."""

    def __init__(self, C):
        self.C = C
        cd = C.cd
        ent = cd.graph_manifest["families"]["structural"]
        self.vocab = ent.get("relation_vocabulary") or []
        s, d, r, _ = cd.family("structural")
        s = np.asarray(s, np.int64)
        d = np.asarray(d, np.int64)
        N = C.N
        if r is None or not self.vocab:
            self.has_rel = False
            return
        self.has_rel = True
        r = np.asarray(r, np.int64)
        m = s != d
        s, d, r = s[m], d[m], r[m]
        u = np.concatenate([s, d])
        v = np.concatenate([d, s])
        rr = np.concatenate([r, r])
        key = (u * N + v) * len(self.vocab) + rr
        _, idx = np.unique(key, return_index=True)
        u, v, rr = u[idx], v[idx], rr[idx]
        o = np.lexsort((v, u))
        self.u, self.v, self.r = u[o], v[o], rr[o]
        self.deg = np.bincount(self.u, minlength=N).astype(np.float64)
        self.xadj = np.zeros(N + 1, np.int64)
        self.xadj[1:] = np.cumsum(np.bincount(self.u, minlength=N))
        self.rel_toks = [toks(lbl.replace("_", " ").replace(".", " ")) for lbl in self.vocab]
        self.node_name = None

    def names(self):
        if self.node_name is None:
            self.node_name = []
            with open(os.path.join(self.C.cd.dir, "nodes.jsonl"), encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    self.node_name.append(o.get("title") or o.get("text") or "")
        return self.node_name

    def match(self, question, seed_node):
        """set of relation ids matched by the query (top seed's name tokens masked out)."""
        qt = toks(question)
        if seed_node is not None and seed_node >= 0:
            mask = set(toks(self.names()[int(seed_node)]))
            qt = [t for t in qt if t not in mask]
        out = set()
        for ri, rt in enumerate(self.rel_toks):
            if any(tok_match(a, b) for a in rt for b in qt):
                out.add(ri)
        return out

    def transition(self, matched):
        """row-stochastic P for one query: 1/2 uniform over all incident edges + 1/2 uniform over
        incident edges of matched relations (falls back to the untyped half when a node has none)."""
        N = self.C.N
        w_all = np.repeat(1.0 / np.maximum(self.deg, 1.0), np.diff(self.xadj))
        if not matched:
            return sp.csr_matrix((w_all.astype(np.float32), self.v, self.xadj), shape=(N, N))
        sel = np.isin(self.r, np.fromiter(matched, np.int64))
        cnt = np.bincount(self.u[sel], minlength=N).astype(np.float64)
        w_typ = np.where(sel, np.repeat(1.0 / np.maximum(cnt, 1.0), np.diff(self.xadj)), 0.0)
        has = np.repeat(cnt > 0, np.diff(self.xadj))
        w = np.where(has, 0.5 * w_all + 0.5 * w_typ, w_all)
        return sp.csr_matrix((w.astype(np.float32), self.v, self.xadj), shape=(N, N))


def typed_block_mass(C, T, questions, seed_rows, alpha=0.3, iters=10, seed_weights=None, hard=False, log_every=500, bounded=False):
    """per-query PPR with the query's own relation-conditioned transition; returns (nq, npart) mass
    and the per-query matched relation sets.  hard=True: typed half only (unmatched edges dropped)."""
    B = C.blockmat()
    nq = seed_rows.shape[0]
    out = np.zeros((nq, C.npart), np.float32)
    matched_all = []
    cache = {}
    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        matched = T.match(questions[i], int(sd[0]) if ok.any() else None) if T.has_rel else set()
        matched_all.append(sorted(matched))
        keym = tuple(sorted(matched))
        if keym not in cache:
            if hard and matched:
                P = T.transition_hard(matched)
            else:
                P = T.transition(matched)
            cache[keym] = P.T.tocsr()
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        PT = cache[keym]
        r = np.zeros(C.N, np.float32)
        if ok.any():
            w = np.ones(int(ok.sum()), np.float64) if seed_weights is None else seed_weights[i][ok].astype(np.float64)
            w = w / w.sum()
            np.add.at(r, sd[ok].astype(np.int64), w.astype(np.float32))
        x = r.copy()
        if bounded:                       # hop-bounded walk: every layer 0..iters carries unit mass, no restart
            acc = r.copy()
            for _ in range(iters):
                x = PT @ x
                acc += x
            x = acc
        else:
            for _ in range(iters):
                x = (1.0 - alpha) * (PT @ x) + alpha * r
        out[i] = np.asarray(B.T @ x).ravel()
        if log_every and (i + 1) % log_every == 0:
            X.log("   typed walk %d/%d (distinct relation sets %d)" % (i + 1, nq, len(cache)))
    return out, matched_all


def _transition_hard(self, matched):
    N = self.C.N
    sel = np.isin(self.r, np.fromiter(matched, np.int64))
    cnt = np.bincount(self.u[sel], minlength=N).astype(np.float64)
    w = np.where(sel, np.repeat(1.0 / np.maximum(cnt, 1.0), np.diff(self.xadj)), 0.0)
    return sp.csr_matrix((w.astype(np.float32), self.v, self.xadj), shape=(N, N))


Typed.transition_hard = _transition_hard


def questions_of(C):
    """query text per cache row, from the served query files (query_id -> question)."""
    qdir = os.path.join(C.cd.dir, "queries")
    txt = {}
    for fn in os.listdir(qdir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(qdir, fn), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                txt[o["query_id"]] = o.get("question_plain") or o.get("question") or o.get("text") or ""
    return [txt.get(q, "") for q in C.qids]


# ---------------------------------------------------------------- relation SCHEDULE (order by proximity to the entity mention)
def _positions(text):
    """token index -> stem, over the raw token stream (stopwords kept so positions are honest)."""
    out = []
    for t in re.findall(r"[a-z]+", text.lower()):
        t = IRREGULAR.get(t, t)
        out.append(porter(t))
    return out


def schedule(T, question, seed_node):
    """ordered relation ids for the walk.  A matched relation token is ATTACHED to the entity mention
    when everything between it and the nearest mention token is stopwords (or its own tokens);
    attached relations come first, ordered by distance, ties preferring the token before the mention
    (the relation governing a mention precedes it in English: 'directed by X', 'director of X').
    Returns (order, anchored); anchored=False when the seed name was not found in the question."""
    qs = _positions(question)
    name = T.names()[int(seed_node)] if seed_node is not None and seed_node >= 0 else ""
    ns = [s for s in _positions(name) if s not in STOP and len(s) >= 3]
    hits = []
    if ns:
        hits = [i for i, s in enumerate(qs) if s in set(ns)]
    mask = set(ns)
    best = {}
    for ri, rt in enumerate(T.rel_toks):
        own = set(rt)
        for i, s in enumerate(qs):
            if s in STOP or len(s) < 3 or s in mask:
                continue
            if any(tok_match(a, s) for a in rt):
                if hits:
                    j = min(hits, key=lambda h: abs(h - i))
                    d = abs(i - j)
                    lo, hi = (i + 1, j) if i < j else (j + 1, i)
                    gap = [qs[t] for t in range(lo, hi)]
                    attached = 0 if all((g in STOP or g in mask or g in own or any(tok_match(a, g) for a in rt)) for g in gap) else 1
                    after = 1 if i > j else 0
                else:
                    d, attached, after = i, 1, 0
                key = (attached, d, after)
                if ri not in best or key < best[ri]:
                    best[ri] = key
    order = sorted(best, key=lambda r: (best[r][0], best[r][1], best[r][2], r))
    return order, bool(hits)


def scheduled_block_mass(C, T, questions, seed_rows, H=3, seed_weights=None, mix=False, log_every=500):
    """hop-bounded walk (H steps, unit mass per layer, summed) where step k may only use relations of
    schedule index <= k (relation j becomes usable at step j and stays usable); mix=True blends 1/2 of
    the untyped uniform transition at every step.  Unanchored / unmatched queries fall back to the
    untyped bounded walk."""
    B = C.blockmat()
    nq = seed_rows.shape[0]
    out = np.zeros((nq, C.npart), np.float32)
    info = []
    cache = {}
    N = C.N
    P_all = None

    def P_of(allowed):
        key = tuple(sorted(allowed))
        if key not in cache:
            if not key:
                cache[key] = T.transition(set()).T.tocsr()
            elif mix:
                cache[key] = T.transition(set(key)).T.tocsr()
            else:
                cache[key] = T.transition_hard(set(key)).T.tocsr()
            if len(cache) > 128:
                cache.pop(next(iter(cache)))
        return cache[key]

    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        order, anchored = schedule(T, questions[i], int(sd[0]) if ok.any() else None) if T.has_rel else ([], False)
        info.append({"order": [T.vocab[r] for r in order], "anchored": bool(anchored)})
        r = np.zeros(N, np.float32)
        if ok.any():
            w = np.ones(int(ok.sum()), np.float64) if seed_weights is None else seed_weights[i][ok].astype(np.float64)
            w = w / w.sum()
            np.add.at(r, sd[ok].astype(np.int64), w.astype(np.float32))
        x = r.copy()
        acc = r.copy()
        for k in range(1, H + 1):
            allowed = set(order[:k]) if order else set()
            PT = P_of(allowed)
            x = PT @ x
            acc += x
        out[i] = np.asarray(B.T @ acc).ravel()
        if log_every and (i + 1) % log_every == 0:
            X.log("   scheduled walk %d/%d" % (i + 1, nq))
    return out, info


def scheduled_block_mass2(C, T, questions, seed_rows, H=3, seed_weights=None, mix=False, frontier=True, perm=False, log_every=500):
    """as scheduled_block_mass, plus: frontier=True zeroes mass on nodes already reached in earlier
    layers before each step (no back-tracking: layer k is the k-hop frontier); perm=True sums the
    walk over every permutation of the matched relation list (order-free) instead of trusting the
    proximity schedule.  Returns (mass, info)."""
    import itertools
    B = C.blockmat()
    nq = seed_rows.shape[0]
    out = np.zeros((nq, C.npart), np.float32)
    info = []
    cache = {}
    N = C.N

    def P_of(allowed):
        key = tuple(sorted(allowed))
        if key not in cache:
            if not key:
                cache[key] = T.transition(set()).T.tocsr()
            elif mix:
                cache[key] = T.transition(set(key)).T.tocsr()
            else:
                cache[key] = T.transition_hard(set(key)).T.tocsr()
            if len(cache) > 256:
                cache.pop(next(iter(cache)))
        return cache[key]

    def walk(r, order):
        x = r.copy()
        acc = r.copy()
        seen = r > 0
        for k in range(1, H + 1):
            allowed = set(order[:k]) if order else set()
            x = P_of(allowed) @ x
            if frontier:
                x[seen] = 0.0
                s = float(x.sum())
                if s > 0:
                    x /= s                    # unit mass per frontier layer
                seen |= x > 0
            acc += x
        return acc

    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        order, anchored = schedule(T, questions[i], int(sd[0]) if ok.any() else None) if T.has_rel else ([], False)
        info.append({"order": [T.vocab[r] for r in order], "anchored": bool(anchored)})
        r = np.zeros(N, np.float32)
        if ok.any():
            w = np.ones(int(ok.sum()), np.float64) if seed_weights is None else seed_weights[i][ok].astype(np.float64)
            w = w / w.sum()
            np.add.at(r, sd[ok].astype(np.int64), w.astype(np.float32))
        if perm and len(order) > 1:
            acc = np.zeros(N, np.float32)
            perms = list(itertools.permutations(order))
            for od in perms:
                acc += walk(r, list(od))
            acc /= len(perms)
        else:
            acc = walk(r, order)
        out[i] = np.asarray(B.T @ acc).ravel()
        if log_every and (i + 1) % log_every == 0:
            X.log("   walk2 %d/%d" % (i + 1, nq))
    return out, info


def anchored_seeds(T, question, cand):
    """keep the candidate seeds whose whole name (non-stop tokens) is mentioned in the question; if
    none is, keep all candidates.  cand: 1-D int array (-1 padded).  Returns 1-D int array."""
    cand = [int(c) for c in cand if c >= 0]
    if not T.has_rel or not cand:
        return np.array(cand if cand else [-1], np.int64)
    qs = set(_positions(question))
    keep = []
    for c in cand:
        ns = [t for t in _positions(T.names()[c]) if t not in STOP and len(t) >= 3]
        if ns and all(t in qs for t in ns):
            keep.append(c)
    return np.array(keep if keep else cand, np.int64)


def walk_variants(C, T, questions, seed_rows, H=3, frontier=True, agg="sum"):
    """returns dict of block-mass matrices: 'hard' (proximity schedule, typed only), 'mix'
    (schedule, 1/2 untyped), 'hardperm' (typed only, aggregated over all relation orders with agg
    'sum' or 'max').  Seeds: uniform over the given row."""
    import itertools
    B = C.blockmat()
    nq = seed_rows.shape[0]
    N = C.N
    out = {k: np.zeros((nq, C.npart), np.float32) for k in ("hard", "mix", "hardperm")}
    cache = {}

    def P_of(allowed, mix):
        key = (tuple(sorted(allowed)), mix)
        if key not in cache:
            if not allowed:
                cache[key] = T.transition(set()).T.tocsr()
            elif mix:
                cache[key] = T.transition(set(allowed)).T.tocsr()
            else:
                cache[key] = T.transition_hard(set(allowed)).T.tocsr()
            if len(cache) > 512:
                cache.pop(next(iter(cache)))
        return cache[key]

    def walk(r, order, mix):
        x = r.copy()
        acc = r.copy()
        seen = r > 0
        for k in range(1, H + 1):
            allowed = set(order[:k]) if order else set()
            x = P_of(allowed, mix) @ x
            if frontier:
                x[seen] = 0.0
                s = float(x.sum())
                if s > 0:
                    x /= s
                seen |= x > 0
            acc += x
        return acc

    info = []
    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        order, anchored = schedule(T, questions[i], int(sd[0]) if ok.any() else None) if T.has_rel else ([], False)
        info.append({"order": [T.vocab[r] for r in order], "anchored": bool(anchored)})
        r = np.zeros(N, np.float32)
        if ok.any():
            r[sd[ok].astype(np.int64)] = 1.0 / float(ok.sum())
        out["hard"][i] = np.asarray(B.T @ walk(r, order, False)).ravel()
        out["mix"][i] = np.asarray(B.T @ walk(r, order, True)).ravel()
        if len(order) > 1:
            accs = [np.asarray(B.T @ walk(r, list(od), False)).ravel() for od in itertools.permutations(order)]
            out["hardperm"][i] = np.max(accs, axis=0) if agg == "max" else np.sum(accs, axis=0) / len(accs)
        else:
            out["hardperm"][i] = out["hard"][i]
    return out, info
