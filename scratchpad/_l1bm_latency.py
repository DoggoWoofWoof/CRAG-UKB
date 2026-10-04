"""Clean latency benchmark -- the sweep's ms/q column is contaminated and must not be reported.

During the ten-policy sweep other jobs (STEP 3, the parity gates) were sharing the CPU and finished
partway through, so policies that ran last look artificially fast.  This re-times every policy back
to back on the same fixed query slice with nothing else running, after a warm-up, and reports the
median of R repeats so the STEP 9 latency multiplier is a real number.

Only the beam is timed; the substrate load and the residual are outside the loop, identically for
every policy.

  python scratchpad/_l1bm_latency.py [ds] [n_queries] [repeats]
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


def main(ds="metaqa", n=400, reps=3):
    n, reps = int(n), int(reps)
    S = EV.substrate(ds); z = S["z"]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    idx = np.linspace(0, S["nq"] - 1, n).astype(int)          # spread across the hop blocks
    SD = [[int(s) for s in z["seeds"][qi] if s >= 0] for qi in idx]
    RQ = [TA.residual(Qm[qi].astype(np.float64), SD[j], Xn) for j, qi in enumerate(idx)]
    for j in range(min(30, n)):                               # warm-up, not timed
        BM.expand_beam(SD[j], RQ[j], adjp, adji, deg, Xn, BM.BEAM, "M0_BASELINE")
    OUT = {"ds": ds, "n_queries": n, "repeats": reps, "ms_per_query": {}}
    for pol in BM.POLICIES:
        ts = []
        for _ in range(reps):
            t = time.time()
            for j in range(n):
                BM.expand_beam(SD[j], RQ[j], adjp, adji, deg, Xn, BM.BEAM, pol)
            ts.append(1000.0 * (time.time() - t) / n)
        OUT["ms_per_query"][pol] = round(float(np.median(ts)), 2)
        log(f"  {pol:26s} {OUT['ms_per_query'][pol]:7.2f} ms/q   "
            f"(runs {' '.join(f'{x:.1f}' for x in ts)})")
    base = OUT["ms_per_query"]["M0_BASELINE"]
    OUT["multiplier_vs_SAFE"] = {p: round(v / base, 2) for p, v in OUT["ms_per_query"].items()}
    json.dump(OUT, open(f"{BM.BMD}/diag/latency_{ds}.json", "w"), indent=1)
    log(f"wrote {BM.BMD}/diag/latency_{ds}.json")
    return OUT


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
