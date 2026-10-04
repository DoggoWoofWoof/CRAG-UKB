"""Shared substrate for the KB MULTI-HOP / SET-AWARE phase.

Everything here is evaluation-time plumbing around the FROZEN L1 contract:

    final = base_rank[:50-B]  u  X,     X subset of (boundary u challengers),  |X| = B
    => exactly 50 canonical C partitions, always.  Keeping the whole boundary (X = bnd) is
       always available, so ABSTAIN is a representable action for every family.

The F6 reference selector is byte-identical to _l1ps_router.evaluate's F6 branch; every script in
this phase asserts its replay of the B6 checkpoint against _l1ps_router.evaluate itself, so the
numbers can never drift from the frozen scoreboard.

Gold is loaded ONLY for labelling/evaluation and is never visible to a selector.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
KBD = f"{ROOT}/KB_MULTI_HOP_SET_ROUTER"
DSETS = RT.DSETS
K0, P = RT.K0, RT.P
BASE_CFG = dict(B=6, M_struct=64, M_ret=32, agg="S4", fusion="F6", T=1, tag="F6")


def load(ds):
    z = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    return z, json.loads(str(z["meta_json"]))


def load_paths(ds):
    p = f"{KBD}/paths/path_{ds}.npz"
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def substrate(ds, z, meta, build_groups, M_struct=64, M_ret=32, agg="S4"):
    """RT.build_cache + path groups, memoised on disk.  Building them costs ~340s per corpus and
    every round needs exactly the same objects, so they are pickled once and reused."""
    import pickle
    d = f"{KBD}/ctx"
    os.makedirs(d, exist_ok=True)
    fp = f"{d}/sub_{ds}_Ms{M_struct}_Mr{M_ret}_{agg}.pkl"
    if os.path.exists(fp):
        with open(fp, "rb") as fh:
            o = pickle.load(fh)
        return o["C"], o["GRP"]
    C = RT.build_cache(ds, z, meta, [M_struct], [M_ret], [agg])
    GRP = build_groups(z, load_paths(ds), meta, z["hard"], M_struct)
    with open(fp, "wb") as fh:
        pickle.dump({"C": C, "GRP": GRP}, fh, protocol=5)
    return C, GRP


def goldparts(z, meta):
    gp, gptr = z["gold_part"], z["gold_ptr"]
    return [set(int(x) for x in gp[gptr[qi]:gptr[qi + 1]]) for qi in range(meta["n_dev_queries"])]


def f6_scores(C, qi, cfg):
    """(cands, score dict) exactly as the F6 branch of _l1ps_router.evaluate builds them."""
    SF = C["sfull"][(cfg["M_struct"], cfg["agg"])][qi]
    RF = C["rfull"][cfg["M_ret"]][qi]
    b50 = C["base50"][qi]
    chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
    spos = {p: r for r, p in enumerate(SF)}
    rpos = {p: r for r, p in enumerate(RF)}
    cpos = C["cpos"][qi]
    return chal, spos, rpos, cpos


def f6_select(bnd, chal, spos, rpos, cpos, B):
    """the frozen symmetric-competition selector: RRF over canonical+structural+retrieval,
    ties broken by canonical rank then partition id.  Returns the B chosen partitions."""
    cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
    sc = []
    for p in cands:
        s = 1.0 / (K0 + cpos[p]) if p in cpos else 0.0
        if p in spos:
            s += 1.0 / (K0 + spos[p])
        if p in rpos:
            s += 1.0 / (K0 + rpos[p])
        sc.append((-s, cpos.get(p, 10 ** 6), p))
    sc.sort()
    return [p for _, _, p in sc[:B]], sc


def contexts(z, meta, C, B, cfg=None):
    """per-query routing context.  No gold, no dataset identity, no fitted quantity."""
    cfg = dict(cfg or BASE_CFG); cfg["B"] = B
    nq = meta["n_dev_queries"]; base_rank = z["base_rank"]
    out = []
    for qi in range(nq):
        br = base_rank[qi]
        prot = [int(x) for x in br[:P - B]]
        bnd = [int(x) for x in br[P - B:P]]
        chal, spos, rpos, cpos = f6_scores(C, qi, cfg)
        out.append({"prot": prot, "prot_set": set(prot), "bnd": bnd, "chal": chal,
                    "spos": spos, "rpos": rpos, "cpos": cpos,
                    "sagg": C["saggf"][cfg["M_struct"]][qi],
                    "ragg": C["raggf"][cfg["M_ret"]][qi],
                    "base50": C["base50"][qi]})
    return out


def run_selector(ctxs, selector, goldp, ind_base, extra=None):
    """apply a selector to every query and score the resulting exactly-50 partition sets."""
    nq = len(ctxs)
    ind = np.zeros(nq, np.int8); churn = np.zeros(nq, np.int32)
    ind_any = np.zeros(nq, np.int8)
    adm = ev = 0
    finals = []
    for qi in range(nq):
        c = ctxs[qi]
        X = selector(c, qi, extra)
        fs = c["prot_set"] | set(X)
        assert len(fs) == P, f"selector produced {len(fs)} partitions, not {P}"
        finals.append(fs)
        ind[qi] = int(goldp[qi] <= fs)
        ind_any[qi] = int(bool(goldp[qi] & fs))
        newp = fs - c["base50"]
        churn[qi] = len(newp)
        adm += len(goldp[qi] & newp)
        ev += len(goldp[qi] & (c["base50"] - fs))
    m = RT.mcnemar(ind, ind_base)
    return {"ind": ind, "ind_any": ind_any, "churn": churn, "finals": finals,
            "ALL": float(ind.mean()), "ANY": float(ind_any.mean()),
            "gold_admitted": int(adm), "gold_evicted": int(ev), **m}


def sel_base(c, qi, extra):
    return list(c["bnd"])


def sel_f6(c, qi, extra):
    B = len(c["bnd"])
    X, _ = f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    return X


def per_hop_table(ind, hops):
    o = {}
    for h in sorted(set(int(x) for x in hops if x >= 0)):
        m = hops == h
        o[str(h)] = {"n": int(m.sum()), "ALL": round(float(ind[m].mean()), 4)}
    return o


def paired_hop(cur, base, hops):
    o = {}
    for h in sorted(set(int(x) for x in hops if x >= 0)):
        m = hops == h
        o[str(h)] = RT.mcnemar(cur[m], base[m])
    return o
