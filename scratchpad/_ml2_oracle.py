"""FBX_SCALE stage 4C -- structure-only record: how far can ANY clustering of the frozen H4_SK vertices reduce the pins, as a function of the number of clusters?  (No gold label, no recall.)

  python -u scratchpad/_ml2_oracle.py            # write-once results/FREEBASE_SCALE/ml2/ORACLE_PIN_REDUCTION__v1.json

The clusterings measured are (i) the seven frozen PHG_SK maps read as clusterings (V = K clusters of ~N/K vertices, the best locality-aware clusterings the lane owns), (the maps of the COARSEN records of
_ml2_run.py already carry their own cluster counts and pin reductions).  For each: the exact quotient of the unsplit H4_SK nets (pins mapped to clusters, duplicate pins removed, single-pin nets dropped, identical nets merged),
its pin count and the pin reduction fine_pins / coarse_pins.  The point is structural: the reduction is governed by the number of clusters V, not by how the clustering was found.
"""
import io
import json
import os
import time

import numpy as np

import _ml2_coarsen as C2
import _ml2_run as R
import _ml_coarsen as C
import _l1h_host as H

OUT = os.path.join(R.OUT, "ORACLE_PIN_REDUCTION__v1.json")


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    N, fams, cnt = R.graph_SK()
    eptr, eidx, ew = C2.sk_nets(N, fams)
    fine = len(eidx)
    rows = []
    parts = os.path.join(H.REPO, "results", "L1_HOST", "parts")
    for K in (100, 250, 500, 1000, 2000, 5000, 25928):
        p = os.path.join(parts, "webqsp__H4_SK_k%d__PHG_con.npy" % K)
        cl = np.load(p).astype(np.int64)
        Vc, ep, ei, ew2, vw, st = C.quotient(N, eptr, eidx, ew, np.ones(N, np.int64), cl)
        rows.append({"clustering": "PHG_SK map K=%d (as clusters)" % K, "clusters": int(Vc), "mean_cluster_weight": round(N / float(Vc), 1), "coarse_nets": int(st["M"]), "coarse_pins": int(st["P"]),
                     "pin_reduction_unsplit": round(fine / float(st["P"]), 4), "file_sha256": H.sha_file(p)})
    rec = {"RECORD": "ML2_ORACLE_PIN_REDUCTION", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "N": int(N), "fine_unsplit_pins": int(fine), "fine_unsplit_nets": int(len(eptr) - 1),
           "definition": __doc__, "rows": rows}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    for r in rows:
        print("%-34s clusters %8d  pins %11d  %.3fx" % (r["clustering"], r["clusters"], r["coarse_pins"], r["pin_reduction_unsplit"]))


if __name__ == "__main__":
    main()
