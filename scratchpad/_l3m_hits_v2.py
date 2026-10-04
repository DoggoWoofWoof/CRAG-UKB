"""MuSiQue text-L3 M0a v2: the same cached L1 front end as _l3m_hits.py (unchanged, its v1 record pins its sha) but with the FLAT_RRF order kept to the top 5,000 (the L1_DEV budget curve reaches B_N = 5000), so ranked L3 arms can be compared
with FLAT / IR_L1 at every B_N of the curve.  Same arithmetic, same regression against results/L1_DEV/loc_musique__v1.npz.

  python -u scratchpad/_l3m_hits_v2.py RUN [--rows=K]
Output (write-once): data/_cache/l3m_hits_musique_v2.npz + results/L3_MUSIQUE/M0a_HITS__musique_v2.json
NOTE: the npz key is still called `top1000` (inherited from v1); in v2 it holds the top-5000 columns (the record's `topk` says so)."""
import os
import sys

import _l3m_hits as V1

V1.TOPK = 5000
V1.OUT_NPZ = os.path.join(V1.REPO, "data", "_cache", "l3m_hits_musique_v2.npz")
V1.OUT_JSON = os.path.join(V1.REPO, "results", "L3_MUSIQUE", "M0a_HITS__musique_v2.json")

if __name__ == "__main__":
    V1.main()
