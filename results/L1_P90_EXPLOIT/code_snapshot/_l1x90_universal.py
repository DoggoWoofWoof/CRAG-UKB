"""Universal typed-path selector (scratch, L1_P90_EXPLOIT lane).

Rule (query-local, label-free, no dataset switches beyond a property of the graph itself):
  * a graph is TYPED when its structural family carries >= 2 distinct relation labels (a single
    universal label such as 'title_mention' carries no relation information -> untyped graph);
  * seeds = corpus nodes whose full name occurs verbatim in the question (longest spans, case-exact
    preferred); fallback = dense top-1 + SPLADE top-1 of the frozen retrieval;
  * schedule = relation labels lexically matched in the question, ordered by attachment to the mention;
  * TYPED MODE fires iff the graph is typed AND the query is anchored (a seed name occurs in the question)
    AND (gate G1) at least one relation label matched / (gate G2) regardless of relation matches;
  * in typed mode the P50 scope is the structural ranking: blocks ordered lexicographically by
    (seed mass + answer-relation mass of the hard scheduled frontier walk, other typed mass + untyped
    walk mass), back-filled by the frozen text ranking beyond the walk's reach;
  * otherwise the frozen text ranking (BASE) is used unchanged.
"""
import os

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_seeds as SE
import _l1x90_typed as TY

EPS = 1e-6      # lexicographic tie-break weight of the hedge channels
H = 3           # walk depth = MAX_HOPS of the frozen expansion


def is_typed_graph(T):
    return bool(T.has_rel) and len(T.vocab) >= 2


def build_channels(C, T, Q, cache_tag=None):
    """seeds, schedules and the walk channels; cached under results/L1_P90_EXPLOIT/_uni_<tag>.npz"""
    path = os.path.join(X.OUT, "_uni_%s.npz" % cache_tag) if cache_tag else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    names = T.names()
    idx = SE.build_name_index(names)
    d1 = C.ret_dense[:, 0]
    s1 = C.ret_splade[:, 0]
    SD = np.full((C.nq, 5), -1, np.int64)
    lexical = np.zeros(C.nq, bool)
    for i in range(C.nq):
        lex = SE.lexical_seeds(Q[i], idx, names, T)
        if lex:
            SD[i, :len(lex)] = lex[:5]
            lexical[i] = True
        else:
            fb = [int(d1[i])] + ([int(s1[i])] if s1[i] != d1[i] else [])
            SD[i, :len(fb)] = fb
    orders, anchored = [], np.zeros(C.nq, bool)
    for i in range(C.nq):
        sd = [int(x) for x in SD[i] if x >= 0]
        o, a = SE.schedule3(T, Q[i], sd, key="wh")
        orders.append(o)
        anchored[i] = a
    k = np.array([len(o) for o in orders])
    Hh = RW.relwalk(C, T, Q, SD, orders, H=H, mix=False)
    U = RW.relwalk(C, T, Q, SD, [[] for _ in range(C.nq)], H=H, mix=False)
    out = {"seeds": SD, "lexical": lexical, "anchored": anchored, "k": k,
           "orders": np.array([",".join(T.vocab[r] for r in o) for o in orders], dtype=object),
           "seed": Hh["seed"], "ans": Hh["ans"], "oth": Hh["oth"], "unt": U["oth"]}
    if path:
        np.savez_compressed(path, **out)
    return out


def universal_select(C, gate="G1", cache_tag=None):
    """returns (sel (nq, P_MAIN) block ids, mode (nq,) 0=BASE 1=TYPED, info dict)"""
    T_rank = C.base_rank.astype(np.int64)
    base_sel = X.fuse(C, T_rank, T_rank, "T")
    T = TY.Typed(C)
    info = {"typed_graph": is_typed_graph(T), "n_rel": len(T.vocab) if T.has_rel else 0}
    if not info["typed_graph"]:
        return base_sel, np.zeros(C.nq, np.int8), info
    Q = TY.questions_of(C)
    ch = build_channels(C, T, Q, cache_tag)
    M = ch["seed"] + ch["ans"] + EPS * (ch["oth"] + ch["unt"])
    S_rank = X.rank_from_scores(M)
    s_sel = X.fuse(C, T_rank, S_rank, "S")
    if gate == "G1":
        typed = ch["anchored"] & (ch["k"] >= 1)
    elif gate == "G2":
        typed = ch["anchored"].copy()
    else:
        raise ValueError(gate)
    sel = np.where(typed[:, None], s_sel, base_sel)
    info.update({"typed_rate": float(typed.mean()), "anchored_rate": float(ch["anchored"].mean()),
                 "lexical_seed_rate": float(ch["lexical"].mean()), "matched_rate": float((ch["k"] >= 1).mean())})
    return sel, typed.astype(np.int8), info
