"""L1 CONVERSION PHASE -- open the selector, minimal change only.

U_PC5 is frozen as the proposer.  The only thing that changes here is how a partition is SCORED at
the boundary, and the change is to collapse the proposal mechanism into exactly ONE vote.

    C = canonical Dense+SPLADE partition rank   (cpos, L = 200)
    S = S4 structural rank                      (spos)
    U = U_PC5 universal proposal rank           (u_rank, L <= M_TOTAL)

ret_rrf is removed as a co-equal SCORING signal because U already carries the node-level retrieval
continuation (C_NODE_DENSE_CONT + D_NODE_SPLADE_CONT are two of U's five sources).  It is kept as
candidate MEMBERSHIP only, so that the candidate set is exactly the one the +0.6351 U_PC5 pool
oracle was measured over and ACTUAL is comparable with that bookend.

SYMMETRY (STEP 1).  u_rank is built BEFORE excluding base50, so a boundary incumbent that U also
supports keeps its proposal vote and can defend its slot.  Giving proposal evidence only to
challengers is exactly the asymmetry that made earlier rounds churn ~100% of B.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1kb_core as KB
import _l1cg_core as CG

K0, P = PP.K0, PP.P
ROOT = PP.ROOT
CVD = f"{ROOT}/L1_CONVERSION"
DSETS = PP.DSETS
B = 6
M_TOTAL = 256
U_SRC = ["B_SPLADE_PARTITION", "I_CANON_CONT", "C_NODE_DENSE_CONT", "D_NODE_SPLADE_CONT",
         "G_GRAPH_NBR_RAW"]
RULES = ["S0_SAFE", "S1_CSU_RAW", "S2_CSU_LIFT", "S3_CU_ABLATION"]


# ------------------------------------------------------------------ STEP 1: symmetric U_RANK
def _rr(lists, M):
    """the frozen U_PC5 merge: round-robin over the sources, dedup, cut at M."""
    out, seen, i = [], set(), 0
    while len(out) < M:
        prog = False
        for l in lists:
            if i < len(l):
                prog = True
                p = int(l[i])
                if p not in seen:
                    seen.add(p); out.append(p)
                    if len(out) >= M:
                        return out
        if not prog:
            return out
        i += 1
    return out


def _dedup_all(seq):
    out, seen = [], set()
    for p in seq:
        p = int(p)
        if p >= 0 and p not in seen:
            seen.add(p); out.append(p)
    return out


def u_lists(ds, S, log=lambda *a: None, cache=True):
    """ONE symmetric ranking over the five U_PC5 sources, built WITHOUT the base50 exclusion.

    Same five sources, same interleave order, same dedup as the frozen proposer; only the drop-set
    differs, which is what makes u_rank defined for incumbents as well as challengers.

    BUDGET.  The frozen proposer charges M_TOTAL to REAL proposals (families are filtered to
    partitions outside base50 before the top-M cut), so cutting the symmetric merge at M_TOTAL
    would silently shrink the proposal set -- measured: 256 -> 172.6 proposals/query on MetaQA and
    the U_PC5 pool oracle 0.6351 -> 0.5871 on hop3, i.e. it would ALTER U_PC5.  The merge is
    therefore built to M_TOTAL + P so that u_rank is defined over the whole competition, and the
    M_TOTAL budget is applied in candidates() to the non-incumbent entries only.  At most P of the
    first M_TOTAL+P entries can be incumbents, so >= M_TOTAL real proposals are always available."""
    os.makedirs(f"{CVD}/u", exist_ok=True)
    fp = f"{CVD}/u/u_{ds}.pkl"
    if cache and os.path.exists(fp):
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    z, meta, nq = S["z"], S["meta"], S["nq"]
    # the cache is a lazy NpzFile and the loop below indexes it per query; decompress once.
    z = {k: z[k] for k in z.files} if hasattr(z, "files") else z
    hard = z["hard"]
    ch = PP.channels(ds, z, meta)
    PG = CG.part_graph(ds)
    seeds = z["seeds"]
    U = []
    for qi in range(nq):
        seedp = sorted({int(hard[int(v)]) for v in seeds[qi] if v >= 0})
        src = [_dedup_all(ch["PR_s"][qi]),                       # B_SPLADE_PARTITION
               _dedup_all(ch["base_rank_replay"][qi]),           # I_CANON_CONT
               _dedup_all(hard[z["ret_dense"][qi]]),             # C_NODE_DENSE_CONT
               _dedup_all(hard[z["ret_splade"][qi]]),            # D_NODE_SPLADE_CONT
               CG._merge_tables(PG["N1"], seedp, set())]         # G_GRAPH_NBR_RAW
        U.append(_rr(src, M_TOTAL + P))
    if cache:
        with open(fp, "wb") as fh:
            pickle.dump(U, fh, protocol=5)
    log(f"[u] {ds} mean |U| = {np.mean([len(u) for u in U]):.1f}")
    return U


def upos_of(U, qi):
    return {p: r for r, p in enumerate(U[qi])}


# ------------------------------------------------------------------ STEP 2: the four rules
def _lift(pos, L):
    """finite-list lift: evidence ABOVE merely appearing at the bottom of a list of length L."""
    fl = 1.0 / (K0 + L)
    return {p: 1.0 / (K0 + r) - fl for p, r in pos.items()}


def proposals(c, U, qi):
    """the frozen U_PC5 proposal set: the first M_TOTAL entries of U that are not incumbents."""
    return [p for p in U[qi] if p not in c["base50"]][:M_TOTAL]


def candidates(c, U, qi):
    """bnd + every challenger the SAFE pool has + every U_PC5 proposal.

    This is exactly the set the U_PC5 expanded-pool oracle (0.6351 on MetaQA hop3) was measured
    over, so ACTUAL and that oracle are directly comparable."""
    extra = proposals(c, U, qi)
    chal = list(c["chal"]) + extra
    return c["bnd"] + [p for p in dict.fromkeys(chal) if p not in c["bnd"]], extra


def select(rule, c, U, qi, Bv=B):
    """returns (chosen B partitions, candidate list). The selector is the ONLY thing that varies."""
    if rule == "S0_SAFE":
        X, _ = KB.f6_select(c["bnd"], list(c["chal"]), c["spos"], c["rpos"], c["cpos"], Bv)
        return X, c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
    cands, _ = candidates(c, U, qi)
    up = upos_of(U, qi)
    cp, sp = c["cpos"], c["spos"]
    if rule == "S2_CSU_LIFT":
        fc, fs, fu = _lift(cp, len(cp)), _lift(sp, len(sp)), _lift(up, len(up))
        sc = [(-(fc.get(p, 0.0) + fs.get(p, 0.0) + fu.get(p, 0.0)), cp.get(p, 10 ** 6), p)
              for p in cands]
    elif rule == "S1_CSU_RAW":
        sc = [(-((1.0 / (K0 + cp[p]) if p in cp else 0.0)
                 + (1.0 / (K0 + sp[p]) if p in sp else 0.0)
                 + (1.0 / (K0 + up[p]) if p in up else 0.0)), cp.get(p, 10 ** 6), p)
              for p in cands]
    elif rule == "S3_CU_ABLATION":
        sc = [(-((1.0 / (K0 + cp[p]) if p in cp else 0.0)
                 + (1.0 / (K0 + up[p]) if p in up else 0.0)), cp.get(p, 10 ** 6), p)
              for p in cands]
    else:
        raise ValueError(rule)
    sc.sort()
    return [p for _, _, p in sc[:Bv]], cands


# ------------------------------------------------------------------ scoring
def run(rule, S, U, Bv=B):
    """ALL coverage plus the conversion bookkeeping STEP 5 asks for."""
    ctxs, goldp, nq = S["ctxs"], S["goldp"], S["nq"]
    ind = np.zeros(nq, np.int8); churn = np.zeros(nq, np.int32)
    newsel = np.zeros(nq, np.int32); newgold = np.zeros(nq, np.int32)
    evgold = np.zeros(nq, np.int32); offered = np.zeros(nq, np.int32)
    final = []
    for qi, c in enumerate(ctxs):
        X, _ = select(rule, c, U, qi, Bv)
        fs = c["prot_set"] | set(X)
        assert len(fs) == P, f"{len(fs)} != {P}"
        ind[qi] = int(goldp[qi] <= fs)
        churn[qi] = len(fs - c["base50"])
        final.append(fs)
        # "truly new" = outside base50 AND outside the frozen challenger set (no spos/rpos at all)
        known = c["base50"] | set(c["chal"])
        new = {p for p in proposals(c, U, qi) if p not in known}
        offered[qi] = len(new)
        sx = set(X) & new
        newsel[qi] = len(sx)
        newgold[qi] = len(sx & goldp[qi])
        evgold[qi] = len((set(c["bnd"]) & goldp[qi]) - set(X))
    return {"ind": ind, "churn": churn, "new_selected": newsel, "new_gold_admitted": newgold,
            "gold_incumbents_evicted": evgold, "new_offered": offered, "final": final}


def substrate(ds, Bv=B):
    S = CG.substrate(ds) if Bv == B else None
    if S is None:
        import _l1cal_core as CC
        S = CC.substrate(ds, Bv)
        S["T"] = CG.targets(S)
    return S
