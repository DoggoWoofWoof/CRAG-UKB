"""STEP 10 -- nested discovery/validation folds AND leave-one-dataset-out selection.

The selection criterion is fixed BEFORE looking at any held-out number, and it is the one the
directive states: among R0..R6, take the largest macro mean delta vs SAFE_F6, subject to NO corpus
regressing significantly (McNemar p < 0.05 with a negative delta).  If nothing clears the
constraint, the answer is SAFE_F6 -- keeping the incumbent is always admissible.

Folds are sha1(ds:qi) parity, fixed in _l1cal_core, so they are reproducible and query-disjoint.
Selection never sees dataset identity: it optimises one number pooled across corpora.

  python scratchpad/_l1cal_val.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1pp_core as PP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
ALPHA = 0.05
CANDS = CC.RULES                       # R0 is SAFE_F6 itself, so "keep the incumbent" is in the pool


def load():
    D = {}
    for ds in CC.DSETS:
        z = np.load(f"{CC.CALD}/diag/ind_{ds}.npz")
        D[ds] = {k: z[k].astype(np.int8) for k in z.files}
        D[ds]["disc"], D[ds]["val"] = CC.folds(ds, len(z["base"]))
    return D


def deltas(D, dss, mask_key, rule):
    """per-corpus (delta vs SAFE_F6, mcnemar p) on the requested half."""
    out = {}
    for ds in dss:
        m = D[ds][mask_key] if mask_key else np.ones(len(D[ds]["base"]), bool)
        cur, ref = D[ds][rule][m], D[ds]["R0_RAW_RRF"][m]
        out[ds] = (float(cur.mean() - ref.mean()), PP.mcnemar(cur, ref)["mcnemar_p"])
    return out


def select(D, dss, mask_key):
    """the pre-registered universal criterion.  Returns (winner, per-rule diagnostics)."""
    rows = {}
    for r in CANDS:
        d = deltas(D, dss, mask_key, r)
        macro = float(np.mean([v[0] for v in d.values()]))
        sigreg = [ds for ds, (dl, p) in d.items() if dl < 0 and p < ALPHA]
        rows[r] = {"macro": round(macro, 5), "worst": round(min(v[0] for v in d.values()), 5),
                   "sig_regressions": sigreg, "per_ds": {k: [round(v[0], 5), round(v[1], 6)]
                                                         for k, v in d.items()}}
    ok = [r for r in CANDS if not rows[r]["sig_regressions"]]
    win = max(ok, key=lambda r: (rows[r]["macro"], rows[r]["worst"])) if ok else "R0_RAW_RRF"
    return win, rows


def main():
    D = load()
    OUT = {"ALPHA": ALPHA, "criterion":
           "argmax macro mean dSAFE_F6 s.t. no corpus with negative delta at McNemar p<0.05; "
           "ties broken by worst-corpus delta; fallback = R0_RAW_RRF (= SAFE_F6)"}

    # ---------------- nested discovery -> validation
    win_d, rows_d = select(D, CC.DSETS, "disc")
    log(f"DISCOVERY winner = {win_d}")
    for r in CANDS:
        log("   %-26s macro %+.5f  worst %+.5f  sig-regressions %s"
            % (r, rows_d[r]["macro"], rows_d[r]["worst"], rows_d[r]["sig_regressions"] or "none"))
    val = deltas(D, CC.DSETS, "val", win_d)
    vmacro = float(np.mean([v[0] for v in val.values()]))
    vsig = [ds for ds, (dl, p) in val.items() if dl < 0 and p < ALPHA]
    survives = (vmacro >= 0) and not vsig
    OUT["NESTED"] = {"discovery_winner": win_d, "discovery": rows_d,
                     "validation_of_winner": {k: [round(v[0], 5), round(v[1], 6)]
                                              for k, v in val.items()},
                     "validation_macro": round(vmacro, 5),
                     "validation_sig_regressions": vsig,
                     "SURVIVES_VALIDATION": bool(survives)}
    log("VALIDATION of %s: macro %+.5f  sig-regressions %s  -> SURVIVES=%s"
        % (win_d, vmacro, vsig or "none", survives))
    for ds, (dl, p) in val.items():
        log("   %-15s %+.4f (p %.4f)" % (ds[:14], dl, p))

    # ---------------- leave-one-dataset-out
    lodo = {}
    for ho in CC.DSETS:
        tr = [d for d in CC.DSETS if d != ho]
        w, rws = select(D, tr, None)
        dl, p = deltas(D, [ho], None, w)[ho]
        lodo[ho] = {"selected_on_other_5": w, "heldout_delta_vs_SAFE_F6": round(dl, 5),
                    "heldout_mcnemar_p": round(p, 6),
                    "heldout_regression_significant": bool(dl < 0 and p < ALPHA),
                    "train_macro": rws[w]["macro"]}
        log("LODO hold-out %-15s -> %-26s  held-out %+.4f (p %.4f)%s"
            % (ho[:14], w, dl, p, "  SIG-REGRESSION" if dl < 0 and p < ALPHA else ""))
    sel = {v["selected_on_other_5"] for v in lodo.values()}
    OUT["LODO"] = {"folds": lodo, "distinct_selections": sorted(sel),
                   "SELECTION_STABLE": len(sel) == 1,
                   "any_heldout_sig_regression":
                       any(v["heldout_regression_significant"] for v in lodo.values()),
                   "heldout_macro": round(float(np.mean(
                       [v["heldout_delta_vs_SAFE_F6"] for v in lodo.values()])), 5)}
    log("LODO selections %s  stable=%s  held-out macro %+.5f  any sig regression=%s"
        % (sorted(sel), OUT["LODO"]["SELECTION_STABLE"], OUT["LODO"]["heldout_macro"],
           OUT["LODO"]["any_heldout_sig_regression"]))

    OUT["PROMOTABLE"] = bool(OUT["NESTED"]["SURVIVES_VALIDATION"]
                             and OUT["LODO"]["SELECTION_STABLE"]
                             and not OUT["LODO"]["any_heldout_sig_regression"]
                             and win_d != "R0_RAW_RRF")
    OUT["WINNER"] = win_d
    json.dump(OUT, open(f"{CC.CALD}/diag/step10_validation.json", "w"), indent=1)
    log("PROMOTABLE =", OUT["PROMOTABLE"], "| WINNER =", win_d)


if __name__ == "__main__":
    main()
