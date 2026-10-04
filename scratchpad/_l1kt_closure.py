"""PART A closure proof: why the frozen F6 selector converts exactly zero of the capacity gain.

The SAFE pool is `bnd ++ chal`, where `bnd` is the six boundary partitions of the base ranking
(cpos 44..49) and `chal` is DEFINED as every partition carrying struct (spos) or ret (rpos)
evidence outside base50.  So any partition NOT in the SAFE pool has neither spos nor rpos, and its
F6 score is at most

        1 / (K0 + cpos)  <=  1 / (60 + 50)  =  0.009091

while every one of the six `bnd` members scores at least 1 / (60 + 49) = 0.009174.  The sixth-best
pool score therefore strictly dominates every appendable candidate, at every depth, on every corpus:
appending candidates cannot change F6's output.  This module checks that algebra empirically.

  python scratchpad/_l1kt_closure.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1ca_admit as AD

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
K0, MISS = BC.K0, BC.MISS
DMAX = 32
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]


def f6_score(p, spos, rpos, cpos):
    s = 0.0
    for d in (cpos, spos, rpos):
        if p in d:
            s += 1.0 / (K0 + d[p])
    return s


def run(ds):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    ctxs = S["ctxs"]
    n_app = n_ev = n_could = nq_app = nq_could = 0
    max_app, min_6th = -1.0, 1e9
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        pset = set(int(p) for p in c["prot"])
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        _, cands, _ = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        a1 = AD.a1_pool(r, cpos, len(cands) + DMAX, pset)
        app = [p for p in a1 if p not in set(cands)][:DMAX]
        if not app:
            continue
        nq_app += 1
        n_app += len(app)
        n_ev += sum(1 for p in app if p in spos or p in rpos)
        sc = sorted((f6_score(p, spos, rpos, cpos) for p in cands), reverse=True)
        s6 = sc[5] if len(sc) >= 6 else (sc[-1] if sc else 0.0)
        best = max(f6_score(p, spos, rpos, cpos) for p in app)
        max_app = max(max_app, best)
        min_6th = min(min_6th, s6)
        k = sum(1 for p in app if f6_score(p, spos, rpos, cpos) > s6)
        n_could += k
        nq_could += int(k > 0)
    return {"ds": ds, "nq": nq, "queries_with_appended": nq_app, "appended_candidates": n_app,
            "appended_with_struct_or_ret_evidence": n_ev,
            "appended_that_could_enter_top6": n_could,
            "queries_where_that_could_happen": nq_could,
            "max_appended_F6_score": round(float(max_app), 6),
            "min_6th_best_pool_score": round(float(min_6th), 6),
            "bound_1_over_110": round(1.0 / 110, 6), "bound_1_over_109": round(1.0 / 109, 6)}


if __name__ == "__main__":
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    out = {}
    for ds in DS:
        t = time.time()
        out[ds] = run(ds)
        print(f"[{time.time()-t:6.1f}s]", json.dumps(out[ds]), flush=True)
    json.dump(out, open(f"{KTD}/diag/closure.json", "w"), indent=1)
