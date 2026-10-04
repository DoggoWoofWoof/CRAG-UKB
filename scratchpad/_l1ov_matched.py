"""PHASE 15 -- matched-node-exposure control.

Overlap must not receive free credit for reading more nodes.  A halo at beta = 0.5 reads about
1.24x the nodes a hard P=50 selection reads, and the honest question is not "does it beat P=50"
but "does it beat simply selecting MORE hard core partitions until the node budget matches".

So for every overlap cell this finds the hard-core depth P' whose mean unique exposure matches
that cell's, and reports the hard baseline there.  P' comes from the SAME frozen BASE block
ranking (base_rank), so the control is the incumbent system read deeper -- nothing new is built
and nothing is tuned.

A useful overlap must beat wider hard-partition retrieval at matched node exposure.

  python scratchpad/_l1ov_matched.py <ds> [core_tag]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as EV
import _l1ep_pu as PU
import _l1ep_c as EC

OUT, ROOT, log = OV.OUT, EC.ROOT, OV.log


def hard_depth_curve(ds, hard, npart, log=log):
    """ALL_REQUIRED and mean unique exposure for hard-core selections at depth P' = 1..200.

    Uses base_rank, the frozen block ranking, so P' = 50 reproduces the incumbent exactly.
    """
    z0 = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}
    z.update(EC.rebuild(ds, hard, npart, z, meta, log))
    br = np.asarray(z["base_rank"])
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    sizes = np.bincount(np.asarray(hard, np.int64), minlength=npart).astype(np.int64)
    gp = [np.unique(np.asarray(hard)[np.asarray(sorted({int(x) for x in
          g[gptr[qi]:gptr[qi + 1]]}), np.int64)]) if gptr[qi + 1] > gptr[qi]
          else np.zeros(0, np.int64) for qi in range(nq)]
    depths = list(range(1, min(br.shape[1], 200) + 1))
    out = {}
    for P in depths:
        allf = np.zeros(nq, np.int8)
        expo = np.zeros(nq, np.int64)
        for qi in range(nq):
            S = br[qi, :P]
            expo[qi] = int(sizes[S].sum())
            need = gp[qi]
            allf[qi] = int(len(need) > 0 and np.isin(need, S).all())
        out[P] = {"ALL_REQUIRED_FETCHED": round(float(allf.mean()), 4),
                  "UNIQUE_EXPOSURE": round(float(expo.mean()), 1),
                  # kept so the overlap-vs-deeper-hard comparison can be tested pairwise
                  "_ind": allf.tolist()}
    return out, meta


def main(ds, tag="CURRENT"):
    hard, npart = PU.load_assignment(ds, tag) if tag == "CURRENT" else (
        np.load(f"scratchpad/_l1ep/parts/{ds}__{tag}.npy"), None)
    hard = np.asarray(hard, np.int64)
    if npart is None:
        npart = int(hard.max()) + 1
    curve, meta = hard_depth_curve(ds, hard, npart, log)
    base50 = curve[50]
    log(f"  {ds}: hard P=50 ALLREQ {base50['ALL_REQUIRED_FETCHED']} "
        f"exposure {base50['UNIQUE_EXPOSURE']}")

    cov = json.load(open(f"{OUT}/overlap/COVERAGE_{ds}.json")).get(tag, {})
    xs = np.array([curve[P]["UNIQUE_EXPOSURE"] for P in curve])
    ys = np.array([curve[P]["ALL_REQUIRED_FETCHED"] for P in curve])
    Ps = np.array(list(curve))
    rows = {}
    for name, rec in cov.items():
        if name.startswith("_") or "F6" not in rec:
            continue
        f = rec["F6"]
        e = f["UNIQUE_EXPOSURE_P50"]
        i = int(np.argmin(np.abs(xs - e)))
        rows[name] = {
            "overlap_ALL_REQUIRED": f["ALL_REQUIRED_FETCHED"],
            "overlap_exposure": e,
            "EXPOSURE_MULTIPLIER": f["EXPOSURE_MULTIPLIER"],
            "matched_hard_depth": int(Ps[i]),
            "matched_hard_exposure": float(xs[i]),
            "matched_hard_ALL_REQUIRED": float(ys[i]),
            "OVERLAP_MINUS_MATCHED_HARD": round(f["ALL_REQUIRED_FETCHED"] - float(ys[i]), 4),
            "BEATS_MATCHED_HARD": bool(f["ALL_REQUIRED_FETCHED"] > float(ys[i])),
            "matched_exposure_error": round(float(xs[i]) / max(e, 1e-9) - 1, 4),
            # base_rank carries only 200 blocks; when the match saturates there the hard control
            # reads FEWER nodes than the overlap cell and the delta flatters overlap.
            "EXPOSURE_MATCHED": bool(abs(float(xs[i]) / max(e, 1e-9) - 1) < 0.05),
            "matched_depth_saturated": bool(int(Ps[i]) == int(Ps[-1]) and float(xs[i]) < e),
            "_ind_matched_hard": curve[int(Ps[i])]["_ind"]}
        log(f"  {name:18s} overlap {f['ALL_REQUIRED_FETCHED']:.4f} @ {e:9.0f} nodes   vs   "
            f"hard P'={int(Ps[i]):3d} {float(ys[i]):.4f} @ {float(xs[i]):9.0f}   "
            f"delta {rows[name]['OVERLAP_MINUS_MATCHED_HARD']:+.4f}"
            + ("" if rows[name]["EXPOSURE_MATCHED"] else "  [NOT MATCHED"
               + (", hard depth saturated]" if rows[name]["matched_depth_saturated"] else "]")))
    fp = f"{OUT}/exposure/MATCHED_{ds}.json"
    os.makedirs(f"{OUT}/exposure", exist_ok=True)
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    rec[tag] = {"HARD_DEPTH_CURVE": {str(k): v for k, v in curve.items()},
                "MATCHED": rows, "npart": int(npart), "nq": int(meta["n_dev_queries"])}
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "CURRENT")
