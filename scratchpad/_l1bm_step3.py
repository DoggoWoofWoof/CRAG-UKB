"""STEP 3 support measurement: how much TIE MASS is there for a lexicographic rule to act on?

A lexicographic rule differs from its own primary key only where that primary key ties.  Both keys
here are continuous cosines, so the only structural tie class is the sentinel: candidates with no
legal, not-already-in-scope child all share future = NEG.  This measures that mass directly at the
binding hop, so the STEP 3 result is reported as a fact about the key rather than as a claim.

  python scratchpad/_l1bm_step3.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1bm_core as BM
import _l1bm_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main(ds="metaqa"):
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    n_c = n_dead = n_tie_f = n_tie_s = 0
    dead_in_top64 = 0
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        _, _, _, tr = BM.expand_beam(sd, rq, adjp, adji, deg, Xn, BM.BEAM,
                                     "L1_MAX_FUTURE", trace_hop=BM.BIND_HOP)
        if tr is None or not len(tr["cand"]):
            continue
        f = tr["future"]; c = tr["score"]; n = len(f)
        n_c += n
        dead = f <= BM.NEG / 2
        n_dead += int(dead.sum())
        # tie mass in each key, counting members of any group of size >= 2
        for arr, acc in ((f, "f"), (c, "s")):
            u, cnt = np.unique(arr, return_counts=True)
            t = int(cnt[cnt > 1].sum())
            if acc == "f":
                n_tie_f += t
            else:
                n_tie_s += t
        dead_in_top64 += int(dead[tr["order"][:BM.BEAM]].sum())
        if (qi + 1) % 500 == 0:
            log(f"   {qi+1}/{nq}")
    out = {"ds": ds, "nq": nq, "position2_candidates": n_c,
           "no_legal_child_sentinel": n_dead,
           "sentinel_frac": round(n_dead / max(1, n_c), 4),
           "tied_in_future_key": n_tie_f, "tied_future_frac": round(n_tie_f / max(1, n_c), 4),
           "tied_in_current_key": n_tie_s, "tied_current_frac": round(n_tie_s / max(1, n_c), 4),
           "sentinel_entering_beam64": dead_in_top64,
           "sentinel_per_query_in_beam": round(dead_in_top64 / nq, 2)}
    json.dump(out, open(f"{BM.BMD}/diag/step3_{ds}.json", "w"), indent=1)
    for k, v in out.items():
        log(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["metaqa"]))
