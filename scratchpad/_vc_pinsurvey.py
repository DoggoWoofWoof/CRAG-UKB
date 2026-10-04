"""FBX_SCALE Track B (addendum 16 basis) -- LABEL-FREE survey of the coarsest-level surrogate hypergraphs of the nested partition on WebQSP: pins, vertices and kept signal of
Q512L<l>N<c> for the net caps c and ladder levels l below.  No Zoltan, no partition, no gold label: only the net cap (scratchpad/_h2lt.cap_nets) and the exact quotient (scratchpad/_ml_coarsen.quotient)
of the stored top hypergraph H4_H2LTOP_k25 under the frozen M2_W512 cumulative cluster maps, exactly as _h2lt.cmd_top builds a surrogate.

  python -u scratchpad/_vc_pinsurvey.py RUN        -> results/FREEBASE_SCALE/vc/PIN_SURVEY__v1.json   (write-once)
"""
import io
import json
import os
import sys
import time

import numpy as np

import _h2lt as T

C, H = T.C, T.H
OUT = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "vc", "PIN_SURVEY__v1.json")
S = 25
CAPS = (None, 8, 7, 6, 5, 4, 3)
LEVELS = (3, 4, 5, 6)
FIT_PINS = 0.39e9                    # pins that fit the 73.4 GB budget at ~190 B/pin (addendum 14)
FB_PINS = {"STRUCT-only (4.25e9 pins)": 4.25e9, "SK (6e9 pins)": 6.0e9}


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    t0 = time.time()
    Hf = T.load_top_hg(S)
    N, eptr, eidx, ew = Hf["N"], Hf["eptr"], Hf["eidx"], Hf["ew"]
    P0 = Hf["P"]
    sz0 = np.diff(eptr)
    sig_all = float((ew * (sz0 - 1)).sum())
    rows = []
    for c in CAPS:
        if c is None:
            ep, ei, ewk, st = eptr, eidx, ew, {"cap": None, "nets_kept": int(len(ew)), "pins_kept": int(len(eidx)), "signal_share_kept": 1.0, "vertices_without_a_net": int((np.bincount(eidx, minlength=N) == 0).sum())}
        else:
            ep, ei, ewk, st = T.cap_nets(eptr, eidx, ew, N, c)
        row = {"cap": c, "after_cap": {"pins": int(len(ei)), "pin_reduction": round(P0 / max(len(ei), 1), 4), "signal_share_kept": st["signal_share_kept"], "vertices_without_a_net": st["vertices_without_a_net"], "nets": int(len(ewk))}, "levels": {}}
        for lv in LEVELS:
            cl, cm, _ = T.cluster_map(512, lv)
            Vc, qep, qei, qew, vw, qs = C.quotient(N, ep, ei, ewk, np.ones(N, np.int64), cl)
            red = P0 / max(len(qei), 1)
            row["levels"][str(lv)] = {"clusters": int(Vc), "nets": int(len(qew)), "pins": int(len(qei)), "pin_reduction_vs_top": round(red, 4),
                                       "freebase_pins_at_this_reduction": {k: round(v / red) for k, v in FB_PINS.items()},
                                       "fits_0.39e9": {k: bool(v / red <= FIT_PINS) for k, v in FB_PINS.items()}}
            H.log("cap %s level %d: V %d M %d P %d (%.3fx)" % (c, lv, Vc, len(qew), len(qei), red))
        rows.append(row)
    rec = {"RECORD": "VC_PIN_SURVEY", "dataset": "webqsp", "S": S, "label_free": True, "gold_labels_read": 0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "top_hypergraph": {"file": H.rel(Hf["npz"]), "sha256": Hf["npz_sha256"], "N": N, "M": Hf["M"], "P": P0}, "fit_pins": FIT_PINS, "freebase_pins": FB_PINS, "survey": rows, "seconds": round(time.time() - t0, 1),
           "code": {"scratchpad/_vc_pinsurvey.py": H.sha_file(os.path.abspath(__file__)), "scratchpad/_h2lt.py": H.sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_h2lt.py"))}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    H.wj(OUT, rec)
    H.log("wrote %s %s" % (OUT, H.sha_file(OUT)))


if __name__ == "__main__":
    if sys.argv[1:] == ["RUN"]:
        main()
    else:
        raise SystemExit(__doc__)
