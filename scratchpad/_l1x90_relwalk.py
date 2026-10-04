"""Relation-decomposed scheduled frontier walk (scratch, L1_P90_EXPLOIT lane).
schedule2: attachment rule where a gap token is opaque only when it matches ANOTHER matched relation;
sort key selectable.  relwalk: per step, the mass pushed along the schedule's LAST relation (the answer
relation) is accumulated separately from the rest, so 'other' transitions (return hops, untyped hedge)
can be down-weighted post hoc:  score = seed + ans + w * oth."""
import itertools

import numpy as np

import _l1x90_typed as TY

WH = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
# closed-class function words (skipped when looking for the token that follows a relation mention)
FUNC = set("a an the of in on by to for at from as with and or is are was were be been do does did has have had that this these those it its not also same".split())


def schedule2(T, question, seed_node, key="after"):
    qs = TY._positions(question)
    name = T.names()[int(seed_node)] if seed_node is not None and seed_node >= 0 else ""
    ns = [s for s in TY._positions(name) if s not in TY.STOP and len(s) >= 3]
    mask = set(ns)
    hits = [i for i, s in enumerate(qs) if s in mask] if ns else []
    # token position -> matched relations (masked / stop tokens excluded)
    tokrel = {}
    for ri, rt in enumerate(T.rel_toks):
        for i, s in enumerate(qs):
            if s in TY.STOP or len(s) < 3 or s in mask:
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
                    # an after-mention relation is the main verb (last hop) only when the next content
                    # token is a question word or the question ends: 'X starred who' vs 'X starred films'
                    nxt = next((qs[t] for t in range(i + 1, len(qs)) if qs[t] in WH or qs[t] not in FUNC), None)
                    after = 1 if (nxt is None or nxt in WH) else 0
            else:
                d, attached, after = i, 1, 0
            k = (attached, d, after) if key == "dist" else (attached, after, d)
            if ri not in best or k < best[ri]:
                best[ri] = k
    order = sorted(best, key=lambda r: best[r] + (r,))
    return order, bool(hits)


def relwalk(C, T, questions, seed_rows, orders, H=3, mix=False, perm=False, log_every=0):
    """returns dict with 'seed', 'ans', 'oth' block-mass arrays (nq, npart).  orders[i] = relation ids in
    schedule order (last = answer relation).  mix: 1/2 untyped hedge (counted in 'oth').  perm: average
    over all orders of the matched relations (each with its own answer relation)."""
    B = C.blockmat().T.tocsr()
    nq = seed_rows.shape[0]
    N = C.N
    u, v, r = T.u, T.v, T.r
    deg = T.deg
    out = {k: np.zeros((nq, C.npart), np.float32) for k in ("seed", "ans", "oth")}
    cnt_cache = {}

    def sel_of(allowed):
        key = tuple(sorted(allowed))
        if key not in cnt_cache:
            sel = np.isin(r, np.fromiter(allowed, np.int64)) if allowed else np.zeros(len(r), bool)
            cnt = np.bincount(u[sel], minlength=N).astype(np.float64)
            cnt_cache[key] = (sel, cnt)
            if len(cnt_cache) > 256:
                cnt_cache.pop(next(iter(cnt_cache)))
        return cnt_cache[key]

    def walk(x0, order):
        x = x0.astype(np.float64)
        seen = x > 0
        ans = np.zeros(N)
        oth = np.zeros(N)
        last = order[-1] if order else None
        for k in range(1, H + 1):
            allowed = set(order[:k]) if order else set()
            sel, cnt = sel_of(allowed)
            has = cnt > 0
            if allowed:
                src = x / np.maximum(cnt, 1.0)
                src[~has] = 0.0
                if mix:
                    src = 0.5 * src
                y_typed = np.bincount(v[sel], weights=src[u[sel]], minlength=N).astype(np.float64)
                sl = sel & (r == last)
                y_ans = np.bincount(v[sl], weights=src[u[sl]], minlength=N).astype(np.float64)
                y_oth = y_typed - y_ans
            else:
                y_ans = np.zeros(N)
                y_oth = np.zeros(N)
            if mix or not allowed:
                src_u = x / np.maximum(deg, 1.0)
                if allowed:
                    src_u = np.where(has, 0.5 * src_u, src_u)
                y_oth = y_oth + np.bincount(v, weights=src_u[u], minlength=N).astype(np.float64)
            y_ans[seen] = 0.0
            y_oth[seen] = 0.0
            s = y_ans.sum() + y_oth.sum()
            if s > 0:
                y_ans /= s
                y_oth /= s
            x = y_ans + y_oth
            seen |= x > 0
            ans += y_ans
            oth += y_oth
        return ans, oth

    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        x0 = np.zeros(N)
        if ok.any():
            x0[sd[ok].astype(np.int64)] = 1.0 / float(ok.sum())
        order = list(orders[i])
        if perm and len(order) > 1:
            perms = list(itertools.permutations(order))
            acc_a = np.zeros(N)
            acc_o = np.zeros(N)
            for od in perms:
                a_, o_ = walk(x0, list(od))
                acc_a += a_
                acc_o += o_
            a_, o_ = acc_a / len(perms), acc_o / len(perms)
        else:
            a_, o_ = walk(x0, order)
        out["seed"][i] = np.asarray(B @ x0).ravel()
        out["ans"][i] = np.asarray(B @ a_).ravel()
        out["oth"][i] = np.asarray(B @ o_).ravel()
        if log_every and (i + 1) % log_every == 0:
            print("  relwalk %d/%d" % (i + 1, nq), flush=True)
    return out
