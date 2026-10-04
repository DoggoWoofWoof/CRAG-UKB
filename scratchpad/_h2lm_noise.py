"""FBX_SCALE Track B, addendum 8 -- the run-to-run NOISE FLOOR of the addendum-1 gate: the frozen flat PHG recipe, re-run with ONE declared change, scored against the frozen PHG_SK maps.

Addendum 7 left both nested arms at 8 rows below PHG_SK (pass line 7) and noted that no run-to-run noise floor of the flat PHG had been measured.  This module produces that floor: the
SAME algorithm on the SAME hypergraph (the frozen H4_SK_K, content digest checked) with one change that perturbs the partition without changing the algorithm:
  SKR1   PHG_RANDOMIZE_INPUT = 1 (Zoltan randomly permutes its input; DETERMINISTIC stays 1, so the run is repeatable)   -- everything else frozen, NP 4
  SKN3   NP = 3 (the documented rank-count dependence: 'partitions are not assumed identical across rank counts')          -- everything else frozen
The frozen driver, parameters, hypergraph, validity rule and empty-block repair are unchanged (scratchpad/_l1h_host.cmd_phg runs them); only the tag of the output differs.

  python -u scratchpad/_h2lm_noise.py <SKR1|SKN3> <K> [<K> ...]         # write-once maps + RUN.json under results/FREEBASE_SCALE/h2l/parts/webqsp__H4_<variant>_k<K>__PHG_con.*
"""
import os
import sys

import _h2l as L                      # sets H.PDIR to the h2l parts directory and re-points H.tag_of; both are overridden below
import _l1h_host as H

P = H.P
DS = "webqsp"
VARIANTS = {"SKR1": {"PHG_RANDOMIZE_INPUT": "1"}, "SKN3": {}}
NP = {"SKR1": None, "SKN3": 3}


def frozen_tag(K):
    return "H4_SK_k%d" % int(K)


def run(variant, Ks):
    assert variant in VARIANTS, variant
    H.PDIR = L.OUT
    # 1. the frozen H4_SK_K hypergraph (exists from addendum 7's CONTROL; rebuilt and recorded if not), under its own tag
    H.tag_of = frozen_tag
    H.cmd_hg(DS, Ks)
    # 2. the partition job under the variant's tag, reading the frozen hypergraph file
    H.tag_of = lambda K: "H4_%s_k%d" % (variant, int(K))
    H.hg_npz = lambda ds, K: os.path.join(H.PDIR, "%s__%s.npz" % (ds, frozen_tag(K)))
    base, base_np = list(P.PARAMS), P.NP
    P.PARAMS = [(k, VARIANTS[variant].get(k, v)) for k, v in base]
    if NP[variant] is not None:
        P.NP = NP[variant]
    try:
        H.log("variant %s: parameters %s, NP %d" % (variant, {k: v for k, v in P.PARAMS if k in ("PHG_RANDOMIZE_INPUT", "IMBALANCE_TOL", "DETERMINISTIC")}, P.NP))
        H.cmd_phg(DS, Ks)
    finally:
        P.PARAMS, P.NP = base, base_np


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) >= 2 and a[0] in VARIANTS:
        H.log("placement:", __import__("json").dumps(H.placement()))
        run(a[0], [int(x) for x in a[1:]])
    else:
        raise SystemExit(__doc__)
