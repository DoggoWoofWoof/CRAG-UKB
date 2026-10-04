"""L1X -- SELECT lane, step 6: the EXHAUSTIVE equal-weight RRF rule table over the relation-conditioned views of _l1x_relsel.py (development only).
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.)

Views (all computed by _l1x_relsel.py from ONE application of A^T and the static relation profile of the candidate; the question-to-relation weight is
w = 1 / (1 + rank of the relation by dense cosine), no constant fitted):
    fpos   the shipped order O                       frank  FLAT rank (ascending)              Ssum   the IR_L1 score L(u)
    S_out  / S_in   the family scores L_STRUCT_out / L_STRUCT_in
    Sro / Sri   the hit-to-candidate STRUCT_out / STRUCT_in mass weighted by w(q, relation of that edge)
    Tin / Tout  max over the relations on the edges INTO / OUT OF the candidate of w(q, relation)   (a static type profile of the candidate)
This evaluates EVERY equal-weight reciprocal-rank fusion (K0 = 60) of every subset of size 1..4 of those nine views, re-ordering the 5,000-node pool of the
shipped L1 (ties -> the shipped position): ALL-gold @ B_N in (100, 250, 500, 1000, 2000) on half A (rows with even row id: the SELECTION half) and half B
(odd: the REPORT half), plus the same on each hop class when the dataset has hops.  Nothing is chosen here.
  COMBINE: the objective is the mean over the datasets of (obj_A(rule) - obj_A(shipped)), obj = the mean ALL over B_N 100/250/500/1000, subject to no dataset
  regressing by more than TOL on A; the winner is reported on half B per dataset.

  python -u scratchpad/_l1x_relcross.py RUN <dataset> <tag>                       -> results/L1_X/relcross_<ds>__<tag>.json (write-once)
  python -u scratchpad/_l1x_relcross.py COMBINE <tag> <ds> <ds> ... [--tol=0.003]  (prints; no record)
"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
K0 = 60
MB = (100, 250, 500, 1000, 2000)
OUT = os.path.join(REPO, "results", "L1_X")
VIEWS = ["fpos", "frank", "Ssum", "S_out", "S_in", "Sro", "Sri", "Tin", "Tout"]


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


def run(ds, tag):
    lower_priority()
    out = os.path.join(OUT, "relcross_%s__%s.json" % (ds, tag))
    assert not os.path.exists(out), "write-once: %s" % out
    zp = os.path.join(OUT, "relsel_%s__v1.npz" % ds)
    z = np.load(zp)
    meta = json.load(open(os.path.join(OUT, "relsel_%s__v1.json" % ds), encoding="utf-8"))
    C, rows, ri = z["cols"], z["rows"], z["rowinfo"]
    COLN = meta["columns"]
    nq, POOL, _ = C.shape
    ngold = ri[:, 3].astype(int)
    G = C[:, :, COLN.index("gold")] > 0
    n_in = G.sum(1)
    hops = np.zeros(nq, int)
    fp = os.path.join(OUT, "feat_%s__v1.npz" % ds)
    if os.path.exists(fp):
        fz = np.load(fp)
        assert (fz["rows"] == rows).all()
        hops = fz["hops"]
    Hs = sorted(set(hops.tolist()))
    A, B = (rows % 2) == 0, (rows % 2) == 1
    ok = n_in == ngold

    def rank_of(S):
        order = np.argsort(-S, axis=1, kind="stable")
        r = np.empty(order.shape, np.int32)
        np.put_along_axis(r, order, np.broadcast_to(np.arange(POOL, dtype=np.int32), order.shape), 1)
        return r

    inv = {}
    for n in VIEWS:
        if n == "fpos":
            r = np.broadcast_to(np.arange(POOL, dtype=np.int32), (nq, POOL))
        else:
            v = C[:, :, COLN.index(n)].astype(np.float64)
            r = rank_of(-v if n == "frank" else v)
        inv[n] = (1.0 / (K0 + r)).astype(np.float32)

    def allq(S):
        order = np.argsort(-S, axis=1, kind="stable")
        Gs = np.take_along_axis(G, order, 1)
        last = POOL - 1 - np.argmax(Gs[:, ::-1], axis=1)
        return {M: ok & (last < M) for M in MB}

    def pack(aq):
        o = {"A": [round(float(aq[M][A].mean()), 4) for M in MB], "B": [round(float(aq[M][B].mean()), 4) for M in MB]}
        if len(Hs) > 1:
            o["hopA100"] = [round(float(aq[100][A & (hops == h)].mean()), 4) for h in Hs]
            o["hopB100"] = [round(float(aq[100][B & (hops == h)].mean()), 4) for h in Hs]
        return o

    t0 = time.time()
    base = allq(-np.broadcast_to(np.arange(POOL, dtype=np.float64), (nq, POOL)))
    res = {"mode": "L1X_RELATION_RULE_TABLE (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "relsel": {"path": os.path.basename(zp), "sha256": sha_file(zp)}, "budgets": list(MB), "hops": Hs, "halves": {"A": int(A.sum()), "B": int(B.sum())},
           "shipped": pack(base), "rules": {}}
    rl = []
    for k in (1, 2, 3, 4):
        rl += list(itertools.combinations(VIEWS, k))
    for i, c in enumerate(rl):
        S = inv[c[0]].astype(np.float64)
        for v in c[1:]:
            S = S + inv[v]
        res["rules"]["+".join(c)] = pack(allq(S))
        if i % 50 == 0:
            print("  rule %d / %d (%.0fs)" % (i, len(rl), time.time() - t0), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["code"] = {"path": "scratchpad/_l1x_relcross.py", "sha256": sha_file(os.path.abspath(__file__))}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f)
    print("done ->", out, flush=True)


def obj4(v):
    return float(np.mean(v[:4]))


def combine(tag, dss, tol):
    recs = {ds: json.load(open(os.path.join(OUT, "relcross_%s__%s.json" % (ds, tag)), encoding="utf-8")) for ds in dss}
    bA = {ds: obj4(recs[ds]["shipped"]["A"]) for ds in dss}
    bB = {ds: obj4(recs[ds]["shipped"]["B"]) for ds in dss}
    names = set.intersection(*[set(r["rules"]) for r in recs.values()])
    rows = []
    for n in names:
        dA = {ds: obj4(recs[ds]["rules"][n]["A"]) - bA[ds] for ds in dss}
        dB = {ds: obj4(recs[ds]["rules"][n]["B"]) - bB[ds] for ds in dss}
        rows.append((n, float(np.mean(list(dA.values()))), min(dA.values()), dA, dB))
    ok = sorted([r for r in rows if r[2] >= -tol], key=lambda r: -r[1])
    print("datasets %s ; rules %d ; admissible %d (tol %.4f)" % (dss, len(rows), len(ok), tol))
    print("shipped objA %s objB %s" % ({k: round(v, 4) for k, v in bA.items()}, {k: round(v, 4) for k, v in bB.items()}))
    for n, m, mn, dA, dB in ok[:15]:
        print("  %-30s meanDA %+.4f minDA %+.4f | dA %s | dB %s" % (n, m, mn, {k: round(v, 4) for k, v in dA.items()}, {k: round(v, 4) for k, v in dB.items()}))
        for ds in dss:
            r = recs[ds]["rules"][n]
            print("       %-8s B@100..1000 %s  (shipped %s)" % (ds, r["B"][:4], recs[ds]["shipped"]["B"][:4]))
    print("--- best per dataset by A:")
    for ds in dss:
        b = max(rows, key=lambda r: r[3][ds])
        print("  %-8s %-30s dA %+.4f dB %+.4f" % (ds, b[0], b[3][ds], b[4][ds]))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[0] == "RUN":
        run(a[1], a[2])
    elif a[0] == "COMBINE":
        tol = 0.003
        for x in sys.argv[1:]:
            if x.startswith("--tol="):
                tol = float(x[6:])
        combine(a[1], a[2:], tol)
    else:
        raise SystemExit(__doc__)
