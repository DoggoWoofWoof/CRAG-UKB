"""FBX_SCALE Track B, addendum 8 -- score the noise-floor replicates of the flat PHG against the frozen flat PHG_SK under the addendum-1 gate, K in {100, 250, 500}.

The per-row arithmetic, the population (786 rows), the matched-B_P comparison and the verdict rule are _ml2_eval.py's, imported and run unchanged (as _h2l_eval.py did for addendum 7);
only the map source changes.  Methods: PHG = the frozen flat PHG_SK maps (the reference); SKR1 / SKN3 = the same recipe with one declared change (scratchpad/_h2lm_noise.py).

  python scratchpad/_h2lm_eval.py EVAL <tag> <bundle_tag> <control_tag> SKR1[,SKN3]          # control_tag = an earlier EVAL whose PHG cells must be reproduced bit for bit
"""
import os
import sys

import numpy as np

import _fbx_calib as C
import _l1d_lib as D
import _ml2_eval as E

H2P = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "h2l", "parts")
ADD8 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_8.json")
_orig_load = E.load_maps
_orig_wj = C.wj


def attempted(name, K, N):
    return True


def load_maps(names, N):
    maps, meta = _orig_load([], N)
    for K in E.KGRID:
        for nm in names:
            base = os.path.join(H2P, "webqsp__H4_%s_k%d__PHG_con" % (nm, K))
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


def _wj(p, rec):
    rec["stage"] = "FBX_SCALE Track B addendum 8: run-to-run noise floor of the addendum-1 gate (flat PHG recipe with one declared change vs the frozen flat PHG_SK)"
    rec["addendum_8_sha256"] = D.sha_file(ADD8)
    rec["code"]["scratchpad/_h2lm_eval.py"] = D.sha_file(os.path.abspath(__file__))
    rec["code"]["scratchpad/_h2lm_noise.py"] = D.sha_file(os.path.join(D.HERE, "_h2lm_noise.py"))
    return _orig_wj(p, rec)


if __name__ == "__main__":
    E.load_maps, E.attempted, E.ADDENDUM5, E.__doc__ = load_maps, attempted, ADD8, __doc__
    C.wj = _wj
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "EVAL":
        E.run_eval(a[1], a[2], a[3], [x for x in a[4].split(",") if x])
    else:
        raise SystemExit(__doc__)
