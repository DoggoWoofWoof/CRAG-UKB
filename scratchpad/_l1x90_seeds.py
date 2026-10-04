"""Lexical entity anchoring + positional relation schedule (scratch, L1_P90_EXPLOIT lane).
lexical_seeds: nodes whose full name occurs verbatim in the question (longest spans; case-exact spans
preferred when any exists).  schedule3: relation order from token positions with the mention spans masked
positionally (so a name like 'Starred Up' never matches a relation)."""
import re

import _l1x90_relwalk as RW
import _l1x90_typed as TY

_PUNCT = "?,.!;:\"'()"


def build_name_index(names):
    idx = {}
    for k, nm in enumerate(names):
        key = " ".join(nm.split()).lower()
        if key:
            idx.setdefault(key, []).append(k)
    return idx


def _relword(T, tok):
    """alpha token that is a relation-label word (e.g. 'director', 'actors') -> not an entity mention.
    Tokens without an alpha stem ('$', '2012', '.45', 'M') are never relation words."""
    st = TY.toks(tok)
    if not st:
        return False
    return any(TY.tok_match(a, st[0]) for rt in T.rel_toks for a in rt)


def lexical_seeds(question, idx, names, T=None, max_len=12, cap=5):
    words = question.split()
    spans = []
    for i in range(len(words)):
        for L in range(1, max_len + 1):
            if i + L > len(words):
                break
            w = words[i:i + L]
            raw = " ".join(w)
            cands = [raw, raw.strip(_PUNCT) if L > 1 else raw.rstrip("?,!;:")]
            for s in cands:
                key = s.lower()
                if key in idx:
                    exact = any(names[n] == s for n in idx[key])
                    if L == 1 and key in TY.STOP | RW.FUNC and not (exact and s != key):
                        break                      # 'film', 'it', 'in' ... only as a capitalised exact mention
                    if T is not None and all(_relword(T, t) for t in s.split()):
                        break                      # 'director', 'actors', 'writer' ...
                    spans.append((i, i + L, s, idx[key], exact))
                    break
    if not spans:
        return []
    if any(sp[4] for sp in spans):
        spans = [sp for sp in spans if sp[4]]
    keep = []
    for a in spans:
        if not any((b[0] <= a[0] and a[1] <= b[1] and (b[1] - b[0]) > (a[1] - a[0])) for b in spans):
            keep.append(a)
    keep.sort(key=lambda sp: (-(sp[1] - sp[0]), sp[0]))
    out = []
    for sp in keep:
        for n in sp[3]:
            if n not in out:
                out.append(n)
    return out[:cap]


def _find_seq(qs, ns):
    """all start positions where the stem list ns occurs contiguously in qs."""
    L = len(ns)
    if L == 0 or L > len(qs):
        return []
    return [i for i in range(len(qs) - L + 1) if qs[i:i + L] == ns]


def schedule3(T, question, seed_nodes, key="wh"):
    """returns (order, anchored).  seed_nodes: node ids whose names are masked positionally."""
    qs = TY._positions(question)
    masked = set()
    for sn in seed_nodes:
        if sn is None or sn < 0:
            continue
        ns = TY._positions(T.names()[int(sn)])
        starts = _find_seq(qs, ns)
        if starts:
            for st in starts:
                masked.update(range(st, st + len(ns)))
        else:
            nz = [s for s in ns if s not in TY.STOP and len(s) >= 3]
            if nz and all(s in qs for s in nz):
                masked.update(i for i, s in enumerate(qs) if s in set(nz))
    hits = sorted(masked)
    tokrel = {}
    for ri, rt in enumerate(T.rel_toks):
        for i, s in enumerate(qs):
            if i in masked or s in TY.STOP or len(s) < 3:
                continue
            if any(TY.tok_match(a, s) for a in rt):
                tokrel.setdefault(i, set()).add(ri)
    best = {}
    for i, rels in tokrel.items():
        for ri in rels:
            if hits:
                j = min(hits, key=lambda h: abs(h - i))
                d = abs(i - j)
                lo, hi = (i + 1, j) if i < j else (j + 1, i)
                opaque = any(t in tokrel and (tokrel[t] - {ri}) for t in range(lo, hi))
                attached = 1 if opaque else 0
                after = 1 if i > j else 0
                if after and key == "wh":
                    nxt = next((qs[t] for t in range(i + 1, len(qs)) if qs[t] in RW.WH or qs[t] not in RW.FUNC), None)
                    after = 1 if (nxt is None or nxt in RW.WH) else 0
            else:
                d, attached, after = i, 1, 0
            k = (attached, d, after) if key == "dist" else (attached, after, d)
            if ri not in best or k < best[ri]:
                best[ri] = k
    order = sorted(best, key=lambda r: best[r] + (r,))
    return order, bool(hits)
