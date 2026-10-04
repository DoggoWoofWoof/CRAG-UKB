"""PARAMETER-FREE STRUCTURAL BEAM RECOVERY -- the beam prune policy is the ONLY thing that varies.

Frozen and untouched here: the static s_dir edge score, S4, F6, B=6, EXACT P50, and the frozen
constants (SEED_K=5, MAX_HOPS=3, DEG_CAP=300, beam M=64, M_MAX=256, MAX_EDGES_SCORED=400000).

Two invariants keep the comparison honest:

  * `vmeta` and the final `added` ordering always keep FROZEN semantics -- a node's recorded score is
    the max static s_dir over its incoming edges, and `added` is sorted by that score.  A policy
    changes WHICH nodes reach S4, never how S4 reads them.  So S4/F6 stay literally unmodified.
  * `M0_BASELINE` is a line-for-line replay of `TA.expand_dir`, including the per-frontier-node matvec
    (batching a whole hop into one BLAS call can differ in the last float bits), the np.unique dedup
    (ties therefore break by node id ascending), np.maximum.at, the scope/budget breaks and the
    `added` cut.  Parity against the frozen cache is asserted before anything else is trusted.

POLICIES
  M0_BASELINE / B0_GLOBAL     frozen global top-M by static score
  L1_MAX_FUTURE               order by future(v) = best static s_dir among v legal next-hop children
  L2_TOP2_FUTURE              order by the mean of the best two child scores (deterministic, no weight)
  L3_FUTURE_THEN_CURRENT      lexicographic: future primary, current secondary
  L4_CURRENT_THEN_FUTURE      lexicographic: current primary, future secondary
  D0_DELAYED_PRUNE            do not prune at/after the binding hop; cut once at the end
  C1_ENDPOINT_DISPLACEMENT    order by cos(r_q, normalize(x_v - x_seed))
  C2_NORMALIZED_EDGE_SUM      order by cos(r_q, normalize(sum of unit edge displacements))
  B1_PARENT_DIVERSE           one slot per distinct parent path first, then fill globally
  B2_SEED_DIVERSE             one slot per distinct retrieval seed first, then fill globally

The lookahead runs BEFORE the prune, on the same graph, with the same static score, and its edge cost
is counted separately (STEP 9 internal-work accounting).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP

BMD = f"{PP.ROOT}/L1_BEAM_RECOVERY"
BEAM = 64
SMAX = TA.M_MAX
BIND_HOP = 1                      # 0-indexed hop; hop 1 == path position 2, the binding prune
NEG = -1e30
LOOK = ("L1_MAX_FUTURE", "L2_TOP2_FUTURE", "L3_FUTURE_THEN_CURRENT", "L4_CURRENT_THEN_FUTURE")
POLICIES = ["M0_BASELINE"] + list(LOOK) + [
    "D0_DELAYED_PRUNE", "C1_ENDPOINT_DISPLACEMENT", "C2_NORMALIZED_EDGE_SUM",
    "B1_PARENT_DIVERSE", "B2_SEED_DIVERSE"]


def _dim(Xn):
    """embedding width, tolerating the fp16-backed F32View the frozen builder uses on big corpora."""
    return (Xn.A.shape[1] if hasattr(Xn, "A") else Xn.shape[1])


def _children(cand, adjp, adji, deg, cap=TA.DEG_CAP):
    """flattened legal children of every candidate, with an owner index.  No python loop."""
    starts = adjp[cand].astype(np.int64)
    lens = (adjp[cand + 1] - adjp[cand]).astype(np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    owner = np.repeat(np.arange(len(cand), dtype=np.int64), lens)
    off = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    ch = adji[np.repeat(starts, lens) + off].astype(np.int64)
    ok = deg[ch] <= cap
    return owner[ok], ch[ok], tot


def future_scores(cand, rq, adjp, adji, deg, Xn, scope_arr, top2=False):
    """future(v) over v legal next-hop children, using the EXISTING static score unchanged.

    Children already inside the scope at decision time can never be admitted at the next hop, so
    they are excluded: the lookahead answers "what can this node still reach", causally.
    Returns (score per candidate, raw edges inspected).  NEG where a node has no legal child."""
    owner, ch, raw = _children(cand, adjp, adji, deg)
    if len(ch):
        keep = ~np.isin(ch, scope_arr)
        owner, ch = owner[keep], ch[keep]
    out = np.full(len(cand), NEG)
    if not len(ch):
        return out, raw
    D = Xn[ch] - Xn[cand[owner]]
    nd = np.linalg.norm(D, axis=1)
    ok = nd > 1e-9
    owner, D, nd = owner[ok], D[ok], nd[ok]
    if not len(owner):
        return out, raw
    s = ((D / nd[:, None]) @ rq).astype(np.float64)
    np.maximum.at(out, owner, s)
    if not top2:
        return out, raw
    second = np.full(len(cand), NEG)
    lower = s < out[owner]
    if lower.any():
        np.maximum.at(second, owner[lower], s[lower])
    second = np.where(second <= NEG / 2, out, second)          # single-child: best2 = best1
    return np.where(out <= NEG / 2, out, 0.5 * (out + second)), raw


def expand_beam(seed_rows, r_q, adjp, adji, deg, Xn, M=BEAM, policy="M0_BASELINE",
                budget=TA.MAX_EDGES_SCORED, trace_hop=None, keep_scope=False, keep_hops=False):
    """(added, vmeta, stats[, trace]).  (added, vmeta) come back in exactly the frozen format."""
    scope = set(int(x) for x in seed_rows)
    seeds = np.array([int(x) for x in seed_rows], np.int64)
    F = seeds.copy()
    F_seed = np.arange(len(seeds), dtype=np.int64)              # which retrieval seed each came from
    C2 = policy == "C2_NORMALIZED_EDGE_SUM"                     # only C2 carries the accumulator
    F_acc = np.zeros((len(seeds), _dim(Xn)), np.float32) if C2 else None
    rq = r_q.astype(np.float32)
    order, vmeta, trace = [], {}, None
    st = {"edges": 0, "look_edges": 0, "cand_total": 0, "frontier": [], "cands": [], "kept": []}
    for hop in range(TA.MAX_HOPS):
        Ps, Vs, Ss = [], [], []
        for i, e in enumerate(F):                               # per-node loop == frozen numerics
            e = int(e)
            if deg[e] > TA.DEG_CAP:
                continue
            nb = adji[adjp[e]:adjp[e + 1]]
            if len(nb) == 0:
                continue
            st["edges"] += len(nb)
            nb = nb[deg[nb] <= TA.DEG_CAP]
            if len(nb) == 0:
                continue
            delta = Xn[nb] - Xn[e]; nd = np.linalg.norm(delta, axis=1)
            ok = nd > 1e-9
            if not ok.any():
                continue
            Ps.append(np.full(int(ok.sum()), i, np.int64))
            Vs.append(nb[ok]); Ss.append((delta[ok] / nd[ok, None]) @ rq)
        if not Vs:
            break
        PAR = np.concatenate(Ps)
        V = np.concatenate(Vs); S = np.concatenate(Ss).astype(np.float64)

        if policy == "C1_ENDPOINT_DISPLACEMENT":
            E = Xn[V] - Xn[seeds[F_seed[PAR]]]
            en = np.linalg.norm(E, axis=1)
            K = np.where(en > 1e-9, (E / np.maximum(en, 1e-9)[:, None]) @ rq, S).astype(np.float64)
        elif C2:
            D2 = Xn[V] - Xn[F[PAR]]
            n2 = np.maximum(np.linalg.norm(D2, axis=1), 1e-9)
            A = F_acc[PAR] + (D2 / n2[:, None])
            an = np.linalg.norm(A, axis=1)
            K = np.where(an > 1e-9, (A / np.maximum(an, 1e-9)[:, None]) @ rq, S).astype(np.float64)
        else:
            K = S

        uv, inv = np.unique(V, return_inverse=True)
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), NEG); np.maximum.at(best_s, inv, S)
        if K is S:
            best_k = best_s
        else:
            best_k = np.full(len(uv), NEG); np.maximum.at(best_k, inv, K)
        bpar = np.full(len(uv), 1 << 40, np.int64)              # smallest parent index attaining max
        top = K >= best_k[inv]
        np.minimum.at(bpar, inv[top], PAR[top])
        for k in range(len(uv)):                                # frozen vmeta semantics, verbatim
            v = int(uv[k]); m = vmeta.get(v)
            if m is None:
                vmeta[v] = [hop + 1, float(best_s[k]), int(cnt[k])]
            else:
                m[2] += int(cnt[k])
                if best_s[k] > m[1]:
                    m[1] = float(best_s[k])
        new_mask = np.fromiter((int(v) not in scope for v in uv), bool, len(uv))
        if not new_mask.any():
            break
        cv, cs, ck, cp = uv[new_mask], best_s[new_mask], best_k[new_mask], bpar[new_mask]
        st["frontier"].append(len(F)); st["cands"].append(len(cv)); st["cand_total"] += len(cv)

        fut = None
        if policy in LOOK and hop + 1 < TA.MAX_HOPS:
            sc = np.fromiter(scope, np.int64, len(scope))
            fut, raw = future_scores(cv, rq, adjp, adji, deg, Xn, sc,
                                     top2=(policy == "L2_TOP2_FUTURE"))
            st["look_edges"] += raw

        if fut is not None and policy in ("L1_MAX_FUTURE", "L2_TOP2_FUTURE"):
            ordr = np.lexsort((cv, -fut))
        elif fut is not None and policy == "L3_FUTURE_THEN_CURRENT":
            ordr = np.lexsort((cv, -cs, -fut))
        elif fut is not None and policy == "L4_CURRENT_THEN_FUTURE":
            ordr = np.lexsort((cv, -fut, -cs))
        elif policy in ("B1_PARENT_DIVERSE", "B2_SEED_DIVERSE"):
            grp = cp if policy == "B1_PARENT_DIVERSE" else F_seed[cp]
            o = np.argsort(-ck, kind="stable")
            seen, first, rest = set(), [], []
            for i in o:
                g = int(grp[i])
                if g in seen:
                    rest.append(i)
                else:
                    first.append(i); seen.add(g)
            ordr = np.array(first + rest, np.int64)
        else:
            ordr = np.argsort(-ck, kind="stable")
        # D0 is the ONLY policy that does not cut here: the prune is deferred to the final `added`.
        keep = ordr if (policy == "D0_DELAYED_PRUNE" and hop >= BIND_HOP) else ordr[:M]
        st["kept"].append(len(keep))
        if keep_hops:                   # per-hop survivors: the transition-table provenance
            st.setdefault("kept_nodes", []).append(cv[keep].copy())

        if hop == trace_hop:
            trace = {"cand": cv.copy(), "score": cs.copy(), "key": ck.copy(), "parent": cp.copy(),
                     "future": (fut.copy() if fut is not None else None),
                     "kept": cv[keep].copy(), "keep_idx": np.asarray(keep).copy(),
                     "order": np.asarray(ordr).copy(), "M": M,
                     "frontier": F.copy(), "fseed": F_seed.copy(),
                     "scope": np.fromiter(scope, np.int64, len(scope))}
        for idx in keep:
            v = int(cv[idx])
            if v not in scope:
                scope.add(v); order.append((float(cs[idx]), v))
        if C2:
            Dk = Xn[cv[keep]] - Xn[F[cp[keep]]]
            nk = np.maximum(np.linalg.norm(Dk, axis=1), 1e-9)
            F_acc = F_acc[cp[keep]] + (Dk / nk[:, None]).astype(np.float32)
        F_seed = F_seed[cp[keep]]
        F = cv[keep]
        if policy != "D0_DELAYED_PRUNE" and len(scope) >= TA.M_MAX * 4:
            break
        if st["edges"] + st["look_edges"] >= budget:
            break
    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:TA.M_MAX]
    st["scope"] = len(scope)
    if keep_scope:                     # STEP 4 needs DISCOVERY (everything visited), not just `added`
        st["scope_set"] = scope
    return (added, vmeta, st, trace) if trace_hop is not None else (added, vmeta, st)


def arrays_of(added, vmeta, slots=SMAX):
    """the four frozen per-query arrays that struct_aggregate_full reads."""
    sn = np.full(slots, -1, np.int32); sh = np.zeros(slots, np.int8)
    sd = np.zeros(slots, np.float32); sc = np.zeros(slots, np.int32)
    for jj, v in enumerate(added[:slots]):
        m = vmeta[int(v)]
        sn[jj] = int(v); sh[jj] = m[0]; sd[jj] = m[1]; sc[jj] = m[2]
    return sn, sh, sd, sc
