"""FBX_SCALE Track C step 2 (addendum 10) -- score the IVF-KNN arms against the frozen PHG_SK under the addendum-1 gate, K in {100, 250, 500}.

_pq_eval.py with the maps of the IVF arms (results/FREEBASE_SCALE/ivf/parts/) and addendum 10 as the declaration; the per-row arithmetic, the population (786 rows), the matched-B_P comparison and
the verdict rule are _ml2_eval.py's, unchanged.  Arm names are the tags of `_ivf_knn.py FULL`.

  python scratchpad/_ivf_eval.py EVAL <tag> <bundle_tag> <control_tag> <arm>[,<arm>...]     # control_tag = an earlier EVAL whose PHG cells must be reproduced bit for bit
"""
import os
import sys

import _fbx_calib as C
import _l1d_lib as D
import _ml2_eval as E
import _pq_eval as Q

IVFP = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "ivf", "parts")
ADD10 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_10.json")
Q.PQP = IVFP
_orig_wj = C.wj


def _wj(p, rec):
    rec["stage"] = "FBX_SCALE Track C step 2: IVF search over PQ64 codes on WebQSP (IVF-KNN arms vs the frozen PHG_SK)"
    rec["addendum_10_sha256"] = D.sha_file(ADD10)
    rec["code"]["scratchpad/_ivf_eval.py"] = D.sha_file(os.path.abspath(__file__))
    return _orig_wj(p, rec)


if __name__ == "__main__":
    E.load_maps, E.attempted, E.ADDENDUM5, E.__doc__ = Q.load_maps, Q.attempted, ADD10, __doc__
    C.wj = _wj
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "EVAL":
        E.run_eval(a[1], a[2], a[3], [x for x in a[4].split(",") if x])
    else:
        raise SystemExit(__doc__)
