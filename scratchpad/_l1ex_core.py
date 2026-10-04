"""L1 EQUAL-EXPOSURE AUDIT -- coverage vs unique nodes exposed.

No new router, no new proposal family, no new scoring.  Existing methods only, re-measured against
the axis that was never controlled: how much of the corpus each configuration actually exposes.

EXPOSURE is computed from the real hard partition membership (np.bincount over z["hard"]), never
from partitions * a nominal size.  Every partition holds exactly one disjoint node set, so the node
union of a partition set is the sum of its sizes.

COVERAGE is one metric for every method: a query is covered iff ALL of its gold partitions are
inside the exposed set.  For SAFE that is exactly the frozen ACTUAL scoreboard; for a pool it is
that pool oracle.  This makes router output and candidate pool directly comparable on one axis.

EXPOSURE ORDER per method (used only to truncate -- never gold-based):
    CANON     the fused canonical partition ranking, base_rank_replay
    SAFE      the frozen router output: base_rank[:44] then the 6 selected, 50 partitions, no more
    SAFE_POOL SAFE order then the frozen F6 challengers (spos/rpos candidates)
    U_PC5     base50 in canonical order, then the U_PC5 proposals in u_rank order
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1cg_core as CG
import _l1cv_core as CV

K0, P = PP.K0, PP.P
ROOT = PP.ROOT
EXD = f"{ROOT}/L1_EXPOSURE_AUDIT"
DSETS = PP.DSETS
NAME = {"metaqa": "metaqa", "webqsp": "webqsp", "2wiki_clean": "2wiki",
        "musique_clean": "musique", "hotpotqa_clean": "hotpot", "squad_clean": "squad"}
PART_BUDGETS = [16, 25, 32, 50, 64, 75, 100, 128, 256]
NODE_BUDGETS = [2500, 5000, 10000, 20000]


def corpus_stats(S):
    """actual node -> partition membership.  Returns (sizes, n_nodes, n_parts)."""
    z = S["z"]
    hard = np.asarray(z["hard"])
    npart = int(hard.max()) + 1
    return np.bincount(hard, minlength=npart).astype(np.int64), int(hard.shape[0]), npart


def orders(ds, S, U):
    """per query, the exposure order of each method.  Truncation only; no gold anywhere."""
    ctxs, nq = S["ctxs"], S["nq"]
    ch = PP.channels(ds, {k: S["z"][k] for k in S["z"].files} if hasattr(S["z"], "files")
                     else S["z"], S["meta"])
    br = ch["base_rank_replay"]
    safe = CV.run("S0_SAFE", S, U)
    out = {}
    for qi, c in enumerate(ctxs):
        canon = [int(p) for p in br[qi]]
        prot = canon[:P - CV.B]
        chosen = [p for p in canon[P - CV.B:] if p in safe["final"][qi]]
        chosen += [p for p in sorted(safe["final"][qi]) if p not in set(prot) | set(chosen)]
        sord = prot + chosen
        assert len(sord) == P and set(sord) == safe["final"][qi]
        pool = sord + [p for p in c["chal"] if p not in safe["final"][qi]]
        base50 = canon[:P]
        upc5 = base50 + CV.proposals(c, U, qi)
        out[qi] = {"CANON": canon, "SAFE": sord, "SAFE_POOL": pool, "U_PC5": upc5}
    return out


def curve(order, gold, sizes):
    """(need_pos, cumulative node count) -- everything else derives from these two.

    need_pos = the earliest prefix length that contains every gold partition, or -1 if the order
    never contains them all.  cum[k-1] = unique nodes exposed by the first k partitions."""
    cum = np.cumsum(sizes[np.asarray(order, np.int64)]) if order else np.zeros(0, np.int64)
    pos = {}
    for i, p in enumerate(order):
        if p not in pos:
            pos[p] = i
    if any(g not in pos for g in gold):
        return -1, cum
    return (max(pos[g] for g in gold) + 1 if gold else 0), cum


def build(ds, log=lambda *a: None, cache=True):
    """per-query (need_pos, cum) for every method, plus corpus constants."""
    os.makedirs(f"{EXD}/cache", exist_ok=True)
    fp = f"{EXD}/cache/ex_{ds}.pkl"
    if cache and os.path.exists(fp):
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    S = CV.substrate(ds)
    U = CV.u_lists(ds, S)
    sizes, n_nodes, n_parts = corpus_stats(S)
    OR = orders(ds, S, U)
    goldp = S["goldp"]
    M = {}
    for m in ["CANON", "SAFE", "SAFE_POOL", "U_PC5"]:
        need = np.full(S["nq"], -1, np.int32)
        cums = []
        for qi in range(S["nq"]):
            n, c = curve(OR[qi][m], goldp[qi], sizes)
            need[qi] = n
            cums.append(c)
        M[m] = {"need": need, "cum": cums,
                "len": np.array([len(OR[qi][m]) for qi in range(S["nq"])], np.int32)}
    out = {"ds": ds, "nq": int(S["nq"]), "n_nodes": n_nodes, "n_parts": n_parts,
           "hops": S["hops"] if ds == "metaqa" else None,
           "mean_part_size": float(sizes.mean()), "M": M}
    if cache:
        with open(fp, "wb") as fh:
            pickle.dump(out, fh, protocol=5)
    log(f"[ex] {ds}: {n_nodes} nodes / {n_parts} partitions, mean partition {sizes.mean():.1f}")
    return out


def at_part_budget(E, m, k):
    """expose the first k partitions of the method order.  Returns (covered, parts, nodes)."""
    d = E["M"][m]
    nq = E["nq"]
    cov = np.zeros(nq, np.int8); pr = np.zeros(nq, np.int32); nd = np.zeros(nq, np.int64)
    for qi in range(nq):
        kk = min(k, int(d["len"][qi]))
        pr[qi] = kk
        nd[qi] = d["cum"][qi][kk - 1] if kk else 0
        n = d["need"][qi]
        cov[qi] = int(0 <= n <= kk)
    return cov, pr, nd


def at_node_budget(E, m, budget):
    """expose the longest prefix of the method order whose node union fits the budget."""
    d = E["M"][m]
    nq = E["nq"]
    cov = np.zeros(nq, np.int8); pr = np.zeros(nq, np.int32); nd = np.zeros(nq, np.int64)
    for qi in range(nq):
        c = d["cum"][qi]
        kk = int(np.searchsorted(c, budget, side="right"))
        pr[qi] = kk
        nd[qi] = c[kk - 1] if kk else 0
        n = d["need"][qi]
        cov[qi] = int(0 <= n <= kk)
    return cov, pr, nd


def stats(v):
    v = np.asarray(v, np.float64)
    return {"mean": round(float(v.mean()), 1), "median": round(float(np.median(v)), 1),
            "p90": round(float(np.percentile(v, 90)), 1),
            "p95": round(float(np.percentile(v, 95)), 1)}


def slices(E):
    out = {"ALL": np.ones(E["nq"], bool)}
    if E["hops"] is not None:
        for h in (1, 2, 3):
            out[f"hop{h}"] = E["hops"] == h
    return out
