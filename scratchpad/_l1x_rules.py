"""L1X -- SELECT lane, step 2: OFFLINE rule search and information ceiling over the feature dump of _l1x_feat.py (development only).
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.  Everything below re-orders the
pool P_q = O[:5000] of the shipped node-level L1 with the features it already computes; nothing is walked.)

Three parts, each descriptive (no verdict, nothing frozen):
  A  single-feature views and every pairwise reciprocal-rank fusion (K0 = 60, equal weights, no constant fitted): ALL gold @ B_N for B_N in M_CURVE,
     the rule chosen on one half of the population (rows with even row id) and reported on the other half (the winner's-curse check).
  B  a greedy forward RRF selection (<= 4 views, K0 = 60) on half A, reported on half B; plus a handful of pre-declared parameter-free forms
     (ratios/products of the IR_L1 score with a rank view, lexicographic agreement orders).
  C  the INFORMATION CEILING of the feature set: a gradient-boosted classifier trained on the pool features, 4-fold cross-fitted by query.  This is
     a DIAGNOSTIC of how much ordering information the features carry; a learned scorer is NOT an L1 candidate (L1 is parameter-free).
Pool restriction: all views re-order only the 5,000 candidates of the shipped order; a gold outside the pool is lost to every rule (the oracle pool
ceiling is reported).  ALL@M = every gold of the query lies in the first M of the re-ordered pool.  Ties are broken by the shipped position.

Usage: python -u scratchpad/_l1x_rules.py <feat npz> <tag>   -> results/L1_X/rules_<ds>__<tag>.json (write-once)
"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

K0 = 60
POOL = 5000
M_CURVE = (100, 250, 500, 1000, 2000, 5000)
M_OBJ = (100, 250, 500, 1000)                      # the selection objective: the mean of ALL over these budgets
NAMES = ["fpos", "frank", "drank", "srank", "dsc", "ssc", "S_out", "S_in", "S_knn", "S_ner", "Ssum", "nseed", "nfam", "minhit", "maxc",
         "incs", "lrank", "top10", "gold"]
IX = {n: i for i, n in enumerate(NAMES)}


def sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


class Data(object):
    def __init__(self, path):
        z = np.load(path)
        self.rows, self.gptr = z["rows"], z["gptr"]
        self.nq = len(self.rows)
        feat = z["feat"]
        assert len(feat) == self.nq * POOL and (z["q"].reshape(self.nq, POOL)[:, 0] == np.arange(self.nq)).all()
        self.X = feat.reshape(self.nq, POOL, len(NAMES))
        self.qinfo = z["qinfo"]
        self.hops = z["hops"]
        self.ngold = np.diff(self.gptr)
        self.G = self.X[:, :, IX["gold"]] > 0
        self.n_in = self.G.sum(1)
        self.pos_O = z["pos_O"]
        self.half = (self.rows % 2).astype(np.int8)          # 0 = half A (selection), 1 = half B (report)
        assert (self.X[:, 0, IX["fpos"]] == 0).all()
        self._rank = {}

    def view(self, name):
        """higher = better score of one feature (a parameter-free orientation)."""
        X = self.X
        if name in ("fpos", "frank", "drank", "srank", "minhit"):
            return -X[:, :, IX[name]].astype(np.float64)
        if name == "lrank":                               # unscored nodes last
            v = X[:, :, IX["lrank"]].astype(np.float64)
            return np.where(v < 0, -1e18, -v)
        return X[:, :, IX[name]].astype(np.float64)

    def rank(self, name):
        """0-based rank of every candidate under the view (descending; ties -> the shipped position)."""
        if name not in self._rank:
            order = np.argsort(-self.view(name), axis=1, kind="stable")
            r = np.empty(order.shape, np.int32)
            np.put_along_axis(r, order, np.broadcast_to(np.arange(POOL, dtype=np.int32), order.shape), 1)
            self._rank[name] = r
        return self._rank[name]

    def allq(self, S):
        """{M: per-query bool ALL@M} of the re-ordering by descending S (ties -> shipped position)."""
        order = np.argsort(-S, axis=1, kind="stable")
        Gs = np.take_along_axis(self.G, order, 1)
        last = POOL - 1 - np.argmax(Gs[:, ::-1], axis=1)
        ok = self.n_in == self.ngold
        return {M: ok & (last < M) for M in M_CURVE}


def summ(aq, mask=None):
    return {str(M): round(float(a[mask].mean() if mask is not None else a.mean()), 4) for M, a in aq.items()}


def obj(aq, mask):
    return float(np.mean([aq[M][mask].mean() for M in M_OBJ]))


def paired(a, b):
    """gained = b serves ALL, a does not."""
    g, l = int((b & ~a).sum()), int((a & ~b).sum())
    from math import comb
    n = g + l
    p = 1.0
    if n:
        k = min(g, l)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"gained": g, "lost": l, "p_mcnemar_descriptive": round(p, 6)}


def main():
    lower_priority()
    path, tag = sys.argv[1], sys.argv[2]
    d = Data(path)
    ds = os.path.basename(path).split("__")[0].replace("feat_", "")
    out = os.path.join(REPO, "results", "L1_X", "rules_%s__%s.json" % (ds, tag))
    assert not os.path.exists(out), "write-once: %s" % out
    t0 = time.time()
    A, B = d.half == 0, d.half == 1
    base = d.allq(-d.X[:, :, IX["fpos"]].astype(np.float64))
    res = {"mode": "L1X_SELECT_OFFLINE_RULES (development; descriptive; no verdict)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "feat": {"path": os.path.basename(path), "sha256": sha_file(path)}, "n_rows": d.nq, "halves": {"A": int(A.sum()), "B": int(B.sum())},
           "baseline": {"all": summ(base), "A": summ(base, A), "B": summ(base, B)}}
    ok = d.n_in == d.ngold
    res["pool_oracle"] = {"share_of_queries_with_every_gold_in_the_pool": round(float(ok.mean()), 4)}
    # oracle ranker inside the pool: ALL@M iff all golds in pool and |gold| <= M (always true here); report the pool-oracle per M
    res["pool_oracle"]["ALL_if_perfect_ordering"] = {str(M): round(float((ok & (d.ngold <= M)).mean()), 4) for M in M_CURVE}
    print("baseline", res["baseline"]["all"], "pool oracle", res["pool_oracle"], flush=True)
    views = ["fpos", "frank", "drank", "srank", "Ssum", "S_out", "S_in", "S_knn", "S_ner", "nseed", "nfam", "maxc", "top10", "minhit", "lrank", "incs", "dsc", "ssc"]
    # ---------------------------------------------------------------- A: single views and all pairwise RRFs
    single = {}
    for v in views:
        aq = d.allq(d.view(v))
        single[v] = {"all": summ(aq), "A": summ(aq, A), "B": summ(aq, B), "obj_A": round(obj(aq, A), 4), "obj_B": round(obj(aq, B), 4)}
    res["A_single_views"] = single
    print("singles done %.0fs" % (time.time() - t0), flush=True)
    rvs = ["fpos", "frank", "drank", "srank", "Ssum", "S_out", "S_in", "S_knn", "S_ner", "nseed", "nfam", "maxc", "top10", "minhit", "lrank"]
    inv = {v: 1.0 / (K0 + d.rank(v)) for v in rvs}
    pairs = {}
    for a_, b_ in itertools.combinations(rvs, 2):
        aq = d.allq(inv[a_] + inv[b_])
        pairs["%s+%s" % (a_, b_)] = {"all": summ(aq), "A": summ(aq, A), "B": summ(aq, B), "obj_A": round(obj(aq, A), 4), "obj_B": round(obj(aq, B), 4)}
    res["A_pairwise_rrf"] = pairs
    top = sorted(pairs, key=lambda k: -pairs[k]["obj_A"])[:10]
    res["A_pairwise_top10_by_half_A"] = [{"rule": k, "obj_A": pairs[k]["obj_A"], "obj_B": pairs[k]["obj_B"]} for k in top]
    print("pairs done %.0fs; best by A: %s" % (time.time() - t0, res["A_pairwise_top10_by_half_A"][:3]), flush=True)
    # ---------------------------------------------------------------- B: greedy forward RRF and the pre-declared forms
    chosen, S, steps = [], np.zeros((d.nq, POOL)), []
    for step in range(4):
        best = None
        for v in rvs:
            if v in chosen:
                continue
            aq = d.allq(S + inv[v])
            o = obj(aq, A)
            if best is None or o > best[0]:
                best = (o, v, aq)
        chosen.append(best[1])
        S = S + inv[best[1]]
        steps.append({"views": list(chosen), "obj_A": round(best[0], 4), "obj_B": round(obj(best[2], B), 4), "A": summ(best[2], A), "B": summ(best[2], B),
                      "paired_vs_shipped_on_B": {str(M): paired(base[M][B], best[2][M][B]) for M in (100, 500, 1000)}})
        print("greedy", steps[-1]["views"], steps[-1]["obj_A"], steps[-1]["obj_B"], flush=True)
    res["B_greedy_rrf"] = steps
    forms = {}
    Ssum = d.X[:, :, IX["Ssum"]].astype(np.float64)
    fr = d.X[:, :, IX["frank"]].astype(np.float64)
    nfam = d.X[:, :, IX["nfam"]].astype(np.float64)
    nseed = d.X[:, :, IX["nseed"]].astype(np.float64)
    dsc = d.X[:, :, IX["dsc"]].astype(np.float64)
    ssc = d.X[:, :, IX["ssc"]].astype(np.float64)
    dmax, smax = d.qinfo[:, 0][:, None], d.qinfo[:, 1][:, None]
    cand = {"L/(K0+frank)": Ssum / (K0 + fr),
            "log(L)-log(K0+frank)": np.where(Ssum > 0, np.log(np.maximum(Ssum, 1e-300)) - np.log(K0 + fr), -1e18),
            "nfam_then_shipped": nfam * 1e6 - d.X[:, :, IX["fpos"]],
            "nseed_then_shipped": nseed * 1e6 - d.X[:, :, IX["fpos"]],
            "nfam_then_L": nfam * 1e6 + Ssum,
            "dense_norm+splade_norm": dsc / np.maximum(dmax, 1e-12) + ssc / np.maximum(smax, 1e-12),
            "L_x_(dense_norm+splade_norm)": Ssum * (dsc / np.maximum(dmax, 1e-12) + ssc / np.maximum(smax, 1e-12)),
            "shipped_rrf+rrf(nfam)": inv["fpos"] + 1.0 / (K0 + d.rank("nfam")),
            }
    for k, Sx in cand.items():
        aq = d.allq(Sx)
        forms[k] = {"all": summ(aq), "A": summ(aq, A), "B": summ(aq, B), "obj_A": round(obj(aq, A), 4), "obj_B": round(obj(aq, B), 4)}
    res["B_declared_forms"] = forms
    print("forms done %.0fs" % (time.time() - t0), flush=True)
    # ---------------------------------------------------------------- C: information ceiling (diagnostic; NOT an L1 candidate)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng = np.random.RandomState(0)
        feats = [n for n in NAMES if n != "gold"]
        d._rank.clear()                                   # memory only (the rank views are not used below); results unchanged
        Z = np.concatenate([d.X[:, :, [IX[n] for n in feats]].astype(np.float32),
                            (dsc / np.maximum(dmax, 1e-12))[:, :, None].astype(np.float32), (ssc / np.maximum(smax, 1e-12))[:, :, None].astype(np.float32)], axis=2)
        fold = (d.rows // 2 % 4).astype(np.int8)
        sc = np.zeros((d.nq, POOL))
        for k in range(4):
            tr = np.flatnonzero(fold != k)
            te = np.flatnonzero(fold == k)
            idx_q, idx_p = [], []
            for qi in tr:
                gi = np.flatnonzero(d.G[qi])
                pos = np.arange(300)
                rest = rng.choice(np.arange(300, POOL), 300, replace=False)
                sel = np.unique(np.concatenate([gi, pos, rest]))
                idx_q.append(np.full(len(sel), qi))
                idx_p.append(sel)
            iq, ip = np.concatenate(idx_q), np.concatenate(idx_p)
            clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08, early_stopping=False, random_state=0)
            clf.fit(Z[iq, ip], d.G[iq, ip].astype(np.int8))
            for qi in te:
                sc[qi] = clf.predict_proba(Z[qi])[:, 1]
            print("  gbm fold %d done %.0fs" % (k, time.time() - t0), flush=True)
        aq = d.allq(sc)
        res["C_information_ceiling_GBM_cross_fitted"] = {"all": summ(aq), "A": summ(aq, A), "B": summ(aq, B),
                                                         "paired_vs_shipped": {str(M): paired(base[M], aq[M]) for M in M_CURVE},
                                                         "note": "a learned scorer: a DIAGNOSTIC of the ordering information in the features, not an L1 rule"}
        if d.hops is not None and d.hops.max() > 0:
            res["C_by_hop"] = {str(h): {"n": int((d.hops == h).sum()), "shipped": summ(base, d.hops == h), "gbm": summ(aq, d.hops == h)} for h in sorted(set(d.hops.tolist()))}
    except ImportError as e:
        res["C_information_ceiling_GBM_cross_fitted"] = "sklearn missing: %s" % e
    res["seconds"] = round(time.time() - t0, 1)
    res["code"] = {"path": "scratchpad/_l1x_rules.py", "sha256": sha_file(os.path.abspath(__file__))}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
