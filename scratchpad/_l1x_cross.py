"""L1X -- SELECT lane, step 3: EXHAUSTIVE equal-weight RRF rule table per dataset, for a cross-dataset (maximin) choice (development only).
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.)

For one feature dump of _l1x_feat.py this evaluates EVERY equal-weight reciprocal-rank fusion (K0 = 60, no constant fitted) of
  - every subset of size 1..3 of the rank views RVS, and
  - every size-4 subset that contains the shipped order 'fpos',
re-ordering the 5,000-node pool of the shipped L1 (ties -> the shipped position).  Per rule: ALL-gold @ B_N in (100, 250, 500, 1000, 2000) on half A
(rows with even row id: the SELECTION half) and half B (odd: the REPORT half).  No rule is chosen here; _l1x_cross.py COMBINE picks across datasets:
  the objective is the mean over datasets of (obj_A(rule) - obj_A(shipped)), subject to every dataset's A-objective change >= -TOL (a rule may not
  regress any dataset); the winner is then reported on half B per dataset.  The shipped rule is the reference (delta 0).

  python -u scratchpad/_l1x_cross.py RUN <feat npz> <tag>      -> results/L1_X/cross_<ds>__<tag>.json (write-once)
  python -u scratchpad/_l1x_cross.py COMBINE <tag> <ds> <ds> ... [--tol=0.003]   -> prints the table (no record)
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
import _l1x_rules as R

RVS = ["fpos", "frank", "drank", "srank", "Ssum", "S_out", "S_in", "S_knn", "S_ner", "nseed", "nfam", "maxc", "top10", "minhit", "lrank"]
MB = (100, 250, 500, 1000, 2000)


def rules():
    out = []
    for k in (1, 2, 3):
        out += list(itertools.combinations(RVS, k))
    out += [("fpos",) + c for c in itertools.combinations([v for v in RVS if v != "fpos"], 3)]
    return out


def run(path, tag):
    R.lower_priority()
    d = R.Data(path)
    ds = os.path.basename(path).split("__")[0].replace("feat_", "")
    out = os.path.join(REPO, "results", "L1_X", "cross_%s__%s.json" % (ds, tag))
    assert not os.path.exists(out), "write-once: %s" % out
    A, B = d.half == 0, d.half == 1
    t0 = time.time()
    inv = {v: (1.0 / (R.K0 + d.rank(v))).astype(np.float32) for v in RVS}
    base = d.allq(-d.X[:, :, R.IX["fpos"]].astype(np.float64))
    res = {"mode": "L1X_CROSS_RULE_TABLE (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "feat": {"path": os.path.basename(path), "sha256": R.sha_file(path)}, "halves": {"A": int(A.sum()), "B": int(B.sum())},
           "budgets": list(MB), "shipped": {"A": [round(float(base[M][A].mean()), 4) for M in MB], "B": [round(float(base[M][B].mean()), 4) for M in MB]},
           "rules": {}}
    rl = rules()
    for i, c in enumerate(rl):
        S = inv[c[0]].astype(np.float64)
        for v in c[1:]:
            S = S + inv[v]
        aq = d.allq(S)
        res["rules"]["+".join(c)] = {"A": [round(float(aq[M][A].mean()), 4) for M in MB], "B": [round(float(aq[M][B].mean()), 4) for M in MB]}
        if i % 100 == 0:
            print("  rule %d / %d (%.0fs)" % (i, len(rl), time.time() - t0), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["code"] = {"path": "scratchpad/_l1x_cross.py", "sha256": R.sha_file(os.path.abspath(__file__))}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f)
    print("done ->", out, flush=True)


def obj4(v):
    return float(np.mean(v[:4]))


def combine(tag, dss, tol):
    recs = {ds: json.load(open(os.path.join(REPO, "results", "L1_X", "cross_%s__%s.json" % (ds, tag)))) for ds in dss}
    base_A = {ds: obj4(recs[ds]["shipped"]["A"]) for ds in dss}
    base_B = {ds: obj4(recs[ds]["shipped"]["B"]) for ds in dss}
    names = set.intersection(*[set(r["rules"]) for r in recs.values()])
    rows = []
    for n in names:
        dA = {ds: obj4(recs[ds]["rules"][n]["A"]) - base_A[ds] for ds in dss}
        dB = {ds: obj4(recs[ds]["rules"][n]["B"]) - base_B[ds] for ds in dss}
        rows.append((n, float(np.mean(list(dA.values()))), min(dA.values()), dA, dB))
    ok = [r for r in rows if r[2] >= -tol]
    ok.sort(key=lambda r: -r[1])
    print("datasets %s ; rules %d ; admissible (no dataset regresses by more than %.4f on A) %d" % (dss, len(rows), tol, len(ok)))
    print("shipped obj A %s  B %s" % ({k: round(v, 4) for k, v in base_A.items()}, {k: round(v, 4) for k, v in base_B.items()}))
    for n, m, mn, dA, dB in ok[:15]:
        print("  %-34s meanDA %+.4f minDA %+.4f | dA %s | dB %s" % (n, m, mn, {k: round(v, 4) for k, v in dA.items()}, {k: round(v, 4) for k, v in dB.items()}))
    print("--- best per dataset by A (inadmissible allowed):")
    for ds in dss:
        best = max(rows, key=lambda r: r[3][ds])
        print("  %-8s %-34s dA %+.4f dB %+.4f" % (ds, best[0], best[3][ds], best[4][ds]))


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
