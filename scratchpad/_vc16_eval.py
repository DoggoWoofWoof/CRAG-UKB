"""FBX_SCALE Track B, addendum 16 -- score the nested (two-level) arms whose TOP level is a V-CYCLE (scratchpad/_vc.py TOP) at the INTERIOR surrogates of addendum 16 against the frozen flat PHG_SK under the addendum-1 gate, K in {100, 250, 500}.

Identical to scratchpad/_h2lt_eval.py (addendum 9) in every respect -- the per-row arithmetic, the 786-row population, the matched-B_P comparison, the verdict rule, the map loader -- except the declaration it
cites: this module imports _h2lt_eval and _ml2_eval and runs them unchanged; only the record's stage text and the addendum hash it embeds differ.  Methods: PHG = the frozen flat PHG_SK maps (the reference);
H2LT_<arm>_<S> = the nested map whose S-way top level is the V-cycle arm <arm> (VC<spec><REFINER>).

  python scratchpad/_vc16_eval.py EVAL <tag> <bundle_tag> <control_tag> H2LT_<arm>_<S>[,...]          # control_tag = an earlier EVAL whose PHG cells must be reproduced bit for bit
"""
import os
import sys

import _fbx_calib as C
import _l1d_lib as D
import _ml2_eval as E
import _h2lt_eval as X

ADD16 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_16.json")
_orig_wj = C.wj


def _wj(p, rec):
    rec["stage"] = "FBX_SCALE Track B addendum 16: nested partition with a V-cycle top level at interior surrogates vs the frozen flat PHG_SK (addendum-1 gate)"
    rec.pop("addendum_9_sha256", None)
    rec["addendum_16_sha256"] = D.sha_file(ADD16)
    rec["code"]["scratchpad/_vc16_eval.py"] = D.sha_file(os.path.abspath(__file__))
    rec["code"]["scratchpad/_h2lt_eval.py"] = D.sha_file(os.path.join(D.HERE, "_h2lt_eval.py"))
    rec["code"]["scratchpad/_vc.py"] = D.sha_file(os.path.join(D.HERE, "_vc.py"))
    rec["code"]["scratchpad/_vc_sub.py"] = D.sha_file(os.path.join(D.HERE, "_vc_sub.py"))
    return _orig_wj(p, rec)


if __name__ == "__main__":
    E.load_maps, E.attempted, E.ADDENDUM5, E.__doc__ = X.load_maps, X.attempted, ADD16, __doc__
    C.wj = _wj
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "EVAL":
        E.run_eval(a[1], a[2], a[3], [x for x in a[4].split(",") if x])
    else:
        raise SystemExit(__doc__)
