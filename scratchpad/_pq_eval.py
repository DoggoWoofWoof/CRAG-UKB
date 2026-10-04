"""FBX_SCALE Track C (addendum 6) -- score the PQ-KNN arms against the frozen PHG_SK under the addendum-1 gate, K in {100, 250, 500}.

The per-row arithmetic, the population (786 rows), the matched-B_P comparison and the verdict rule are _ml2_eval.py's (itself _fbx_calib_cn.py's), imported and run unchanged; only the map source
changes.  Methods:  PHG = the frozen PHG_SK maps (H4_SK = STRUCT u canonical KNN, NP 4), the reference;  PQ<m> = the same partitioner, same rule, same K, on H4_{STRUCT u KNN over m-byte
product-quantised canonical vectors}.  Modes per (method, K): own (B_P from the method's own K100 map), matched (B_P = PHG's own B_P at that K: THE GATE).  The gate is addendum 1's: rows below
PHG at matched B_P over every B_N <= 7 PASS (RECALL_PRESERVING), >= 24 FAIL, else BORDERLINE.

  python scratchpad/_pq_eval.py EVAL <tag> <bundle_tag> <control_tag> PQ<m>[,PQ<m>...]     # control_tag = an earlier EVAL whose PHG cells must be reproduced bit for bit
"""
import os
import sys

import numpy as np

import _fbx_calib as C
import _l1d_lib as D
import _ml2_eval as E

PQP = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "pq", "parts")
ADD6 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_6.json")
ADD6A = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_6A.json")


def attempted(name, K, N):
    return True


def load_maps(names, N):
    maps, meta = _load_ref(N)
    for K in E.KGRID:
        for nm in names:
            base = os.path.join(PQP, "webqsp__H4_%s_k%d__PHG_con" % (nm, K))
            if os.path.exists(base + ".RUN.json"):
                Rc = C.jl(base + ".RUN.json")
                assert Rc["STATUS"] == "OK"
                p = os.path.join(D.REPO, Rc["output"]["file"])
                assert D.sha_file(p) == Rc["output"]["sha256"], "%s K %d map differs from its RUN.json" % (nm, K)
                maps[(nm, K)] = np.load(p).astype(np.int32)
                meta[(nm, K)] = {"STATUS": "OK", "sha256": Rc["output"]["sha256"], "file": Rc["output"]["file"], "run_json_sha256": D.sha_file(base + ".RUN.json")}
            else:
                fj = base + ".FAILED.json"
                Rf = C.jl(fj)                       # a missing record is an error: every arm has a RUN or a FAILED record at every gate K
                assert Rf["STATUS"] == "PARTITION_INVALID"
                meta[(nm, K)] = {"STATUS": "PARTITION_INVALID", "file": D.rel(fj), "sha256": D.sha_file(fj)}
    return maps, meta


_orig_load = E.load_maps


def _load_ref(N):
    return _orig_load([], N)


_orig_wj = C.wj


def _wj(p, rec):
    rec["stage"] = "FBX_SCALE Track C: compressed-KNN calibration on WebQSP (PQ-KNN arms vs the frozen PHG_SK)"
    rec["addendum_6_sha256"] = D.sha_file(ADD6)
    rec["addendum_6a_sha256"] = D.sha_file(ADD6A)
    rec["code"]["scratchpad/_pq_eval.py"] = D.sha_file(os.path.abspath(__file__))
    return _orig_wj(p, rec)


if __name__ == "__main__":
    E.load_maps, E.attempted, E.ADDENDUM5, E.__doc__ = load_maps, attempted, ADD6, __doc__
    C.wj = _wj
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "EVAL":
        E.run_eval(a[1], a[2], a[3], [x for x in a[4].split(",") if x])
    else:
        raise SystemExit(__doc__)
