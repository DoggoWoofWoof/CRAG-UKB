"""L1 FINAL SCORING PHASE -- shared substrate for the R0..R6 boundary-scoring comparison.

Nothing here adds a feature.  Every rule below consumes exactly the three cached rankings the
frozen router already builds -- canonical partition rank (TOP200), S4 structural rank (M_struct=64),
node-level retrieval rank (M_ret=32) -- and differs only in the ARITHMETIC used to compare a
boundary incumbent against a challenger.  No gold at inference, no dataset identity, no learned
constant, no traversal, exactly P=50 out.

INDEXING CONTRACT.  The frozen router scores 1/(K0+r) with r ZERO-indexed (f6_select uses the dict
position directly), so every rule here keeps r zero-indexed.  The directive writes the finite-list
floor as 1/(K0+L+1) in one-indexed terms; the identical quantity zero-indexed is 1/(K0+L), which is
what `floor` returns -- the value one position past the end of a list of length L.
"""
import hashlib
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1kb_core as KB

K0, P = PP.K0, PP.P
ROOT = PP.ROOT
CALD = f"{ROOT}/L1_SCORING_CALIBRATION"
DSETS = PP.DSETS
RULES = ["R0_RAW_RRF", "R1_RRF_LIFT", "R2_RRF_UNIT", "R3_SIGNED_RET_RESIDUAL",
         "R4_POSITIVE_RET_RESIDUAL", "R5_PAIRWISE_RESIDUAL", "R6_PARETO_SWAP"]


# ------------------------------------------------------------------ channel calibration
def chan(pos, L):
    """(raw, lift, unit) lookup dicts for one channel of one query.

    raw(r)   = 1/(K0+r)                      -- the frozen contribution
    floor    = 1/(K0+L)                      -- one position past the end of this list
    lift(r)  = raw(r) - floor                -- evidence ABOVE merely appearing at the bottom
    unit(r)  = lift(r) / (raw(0) - floor)    -- the same, rescaled so each ranking spans [0,1]

    Absence contributes exactly 0.0 in all three, which is the frozen per-partition semantics.
    """
    if not pos:
        return {}, {}, {}
    fl = 1.0 / (K0 + L)
    top = 1.0 / K0 - fl
    raw = {p: 1.0 / (K0 + r) for p, r in pos.items()}
    lift = {p: raw[p] - fl for p in pos}
    unit = {p: (lift[p] / top if top > 0 else 0.0) for p in pos} if top > 0 else {p: 0.0 for p in pos}
    return raw, lift, unit


def calib(c):
    """all three scales for all three channels of one routing context."""
    cr, cl, cu = chan(c["cpos"], len(c["cpos"]))
    sr, sl, su = chan(c["spos"], len(c["spos"]))
    rr, rl, ru = chan(c["rpos"], len(c["rpos"]))
    return {"raw": (cr, sr, rr), "lift": (cl, sl, rl), "unit": (cu, su, ru),
            "L": (len(c["cpos"]), len(c["spos"]), len(c["rpos"]))}


def cands_of(c):
    """the frozen candidate set: boundary incumbents first, then de-duplicated challengers."""
    return c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]


def resid_pos(K, p):
    """POSITIVE retrieval residual: the part of the node view not explained by the partition view."""
    cu, su, ru = K["unit"]
    return max(0.0, ru.get(p, 0.0) - cu.get(p, 0.0))


def resid_signed(K, p):
    cu, su, ru = K["unit"]
    return ru.get(p, 0.0) - cu.get(p, 0.0)


# ------------------------------------------------------------------ the seven rules
def _topB(c, key, B):
    """additive rule: score every candidate independently, keep the B best.

    Ties break exactly as the frozen selector does -- canonical rank, then partition id -- so a
    rule that produces the frozen scores reproduces the frozen selection bit-for-bit.
    """
    sc = [(-key(p), c["cpos"].get(p, 10 ** 6), p) for p in cands_of(c)]
    sc.sort()
    return [p for _, _, p in sc[:B]]


def sel_R0(c, K, B):
    cr, sr, rr = K["raw"]
    return _topB(c, lambda p: cr.get(p, 0.0) + sr.get(p, 0.0) + rr.get(p, 0.0), B)


def sel_R1(c, K, B):
    cl, sl, rl = K["lift"]
    return _topB(c, lambda p: cl.get(p, 0.0) + sl.get(p, 0.0) + rl.get(p, 0.0), B)


def sel_R2(c, K, B):
    cu, su, ru = K["unit"]
    return _topB(c, lambda p: cu.get(p, 0.0) + su.get(p, 0.0) + ru.get(p, 0.0), B)


def sel_R3(c, K, B):
    """SIGNED residual, exactly as specified.  NOTE the algebra: canonical + S4 + (node - canonical)
    cancels the canonical term identically, so R3 is structure + node-retrieval with the canonical
    partition channel DELETED.  Run as specified; reported as the algebraic identity it is."""
    cu, su, ru = K["unit"]
    return _topB(c, lambda p: cu.get(p, 0.0) + su.get(p, 0.0)
                 + (ru.get(p, 0.0) - cu.get(p, 0.0)), B)


def sel_R4(c, K, B):
    """POSITIVE residual.  max(0,.) is non-linear, so the canonical term survives: a candidate whose
    node evidence is no stronger than its partition evidence scores canonical + S4 and nothing more."""
    cu, su, ru = K["unit"]
    return _topB(c, lambda p: cu.get(p, 0.0) + su.get(p, 0.0)
                 + max(0.0, ru.get(p, 0.0) - cu.get(p, 0.0)), B)


def _pairs(c, K, B):
    """deterministic incumbent/challenger pairing for the replacement rules.

    The incumbent most at risk is the WORST one, so the boundary is walked worst-first (the boundary
    is base_rank[P-B:P], already in improving order).  Challengers are ordered by their own R4 score,
    best first, ties by canonical rank then id.  Slot k contests challenger k against incumbent k;
    every slot yields exactly one partition, so the output is exactly B.
    """
    cu, su, ru = K["unit"]
    key = lambda p: cu.get(p, 0.0) + su.get(p, 0.0) + max(0.0, ru.get(p, 0.0) - cu.get(p, 0.0))
    ch = [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
    ch.sort(key=lambda p: (-key(p), c["cpos"].get(p, 10 ** 6), p))
    inc = list(c["bnd"])[::-1]
    return list(zip(inc, ch + [None] * (len(inc) - len(ch))))


def sel_R5(c, K, B):
    """pairwise replacement: swap only when the summed evidence delta is strictly positive."""
    cu, su, ru = K["unit"]
    out = []
    for i, ch in _pairs(c, K, B):
        if ch is None:
            out.append(i); continue
        d = ((cu.get(ch, 0.0) - cu.get(i, 0.0))
             + (su.get(ch, 0.0) - su.get(i, 0.0))
             + (resid_pos(K, ch) - resid_pos(K, i)))
        out.append(ch if d > 0 else i)
    return out


def sel_R6(c, K, B):
    """conservative dominance control: replace only on a Pareto-dominating evidence vector."""
    cu, su, ru = K["unit"]
    out = []
    for i, ch in _pairs(c, K, B):
        if ch is None:
            out.append(i); continue
        dc = cu.get(ch, 0.0) - cu.get(i, 0.0)
        ds = su.get(ch, 0.0) - su.get(i, 0.0)
        rc, ri = resid_pos(K, ch), resid_pos(K, i)
        a = dc >= 0 and ds >= 0 and (dc > 0 or ds > 0)
        b = rc > 0 and dc >= 0 and ds >= 0 and (rc - ri) >= 0 and (dc > 0 or ds > 0 or rc > ri)
        out.append(ch if (a or b) else i)
    return out


SEL = {"R0_RAW_RRF": sel_R0, "R1_RRF_LIFT": sel_R1, "R2_RRF_UNIT": sel_R2,
       "R3_SIGNED_RET_RESIDUAL": sel_R3, "R4_POSITIVE_RET_RESIDUAL": sel_R4,
       "R5_PAIRWISE_RESIDUAL": sel_R5, "R6_PARETO_SWAP": sel_R6}


# ------------------------------------------------------------------ evaluation
def apply_rule(ctxs, KS, name, B, goldp, base50s):
    """run one rule over every query.  Returns coverage indicator + swap bookkeeping.

    Gold is used ONLY here, post hoc, to label outcomes.  No selector sees it.
    """
    nq = len(ctxs)
    ind = np.zeros(nq, np.int8); churn = np.zeros(nq, np.int32)
    adm = ev = 0
    good = bad = neu = nswap = 0
    fn = SEL[name]
    for qi, c in enumerate(ctxs):
        X = fn(c, KS[qi], B)
        assert len(X) == B, f"{name} produced {len(X)} replacements, not {B}"
        fs = c["prot_set"] | set(X)
        assert len(fs) == P, f"{name} produced {len(fs)} partitions, not {P}"
        g = goldp[qi]
        ind[qi] = int(g <= fs)
        new = fs - base50s[qi]
        churn[qi] = len(new)
        adm += len(g & new)
        ev += len(g & (base50s[qi] - fs))
        out_ = set(c["bnd"]) - fs
        in_ = set(X) - set(c["bnd"])
        nswap += len(in_)
        gi, go = len(g & in_), len(g & out_)
        good += gi; bad += go
        neu += max(0, len(in_) - gi - go)
    return {"ind": ind, "churn": churn, "ALL": float(ind.mean()),
            "gold_admitted": int(adm), "gold_evicted": int(ev),
            "swaps": int(nswap), "GOOD": int(good), "BAD": int(bad), "NEUTRAL": int(neu),
            "churn_mean": round(float(churn.mean()), 3)}


def mc(cur, ref):
    return PP.mcnemar(cur, ref)


# ------------------------------------------------------------------ deterministic folds
def folds(ds, nq):
    """sha1(ds:qi) parity -> DISCOVERY / VALIDATION.  Fixed before any result is looked at."""
    h = np.array([int(hashlib.sha1(f"{ds}:{qi}".encode()).hexdigest(), 16) & 1
                  for qi in range(nq)], np.int8)
    return h == 0, h == 1


def substrate(ds, B=6):
    """everything a rule needs for one dataset, built once."""
    z, meta = PP.load(ds)
    nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta)
    C, _ = KB.substrate(ds, z, meta, __import__("_l1kb_router").build_groups)
    ctxs = KB.contexts(z, meta, C, B)
    KS = [calib(c) for c in ctxs]
    base50s = [c["base50"] for c in ctxs]
    ind_base = np.array([int(goldp[qi] <= (ctxs[qi]["prot_set"] | set(ctxs[qi]["bnd"])))
                         for qi in range(nq)], np.int8)
    return {"z": z, "meta": meta, "nq": nq, "goldp": goldp, "ctxs": ctxs, "KS": KS,
            "base50s": base50s, "ind_base": ind_base, "hops": z["hops"]}
