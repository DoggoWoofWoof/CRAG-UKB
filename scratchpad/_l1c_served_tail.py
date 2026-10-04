"""Static diagnostic for section 14.3: how often does canonical L1's correct answer RELY on the served P50 tail?

For every DEV_A query that canonical L1 gets right (ALL gold blocks inside the served P50), take the served position of its worst
(largest-position) gold block and bucket it: 0-9 / 10-24 / 25-49.  Positions 25-49 are exactly the region an asymmetric HOLD
exchange may touch (section 14.3); positions >= 50 - B are the region CORE(B) may touch.  No beam, no arm, nothing written under data/.
    python -u _l1c_served_tail.py      -> results/L1_COVPART/served_tail_gold_share_A.json
"""
import json
import os

import numpy as np

import _l1g_core as G

X = G.X
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART", "served_tail_gold_share_A.json")
out = {"definition": "DEV_A queries with L1 ALL-gold@P50 = 1; served position (0-based, frozen RRF order) of the worst gold block; "
                     "share_25-49 = fraction of L1-correct queries whose worst gold sits at served positions 25-49 (the HOLD exchange region)"}
for name in ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]:
    D = G.Data(name, dense_fp32=False)
    C = D.C
    m = D.A
    base_rank = np.asarray(C.base_rank, np.int64)
    pos0 = X.rank_pos(base_rank, D.npart)
    gm = G.gold_mask(D)
    ok = np.asarray(D.base_all, bool)
    rows = np.nonzero(m & ok)[0]
    wp = np.array([pos0[i, np.nonzero(gm[i])[0]].max() for i in rows])
    hops = np.asarray(C.hops)
    d = {"L1_correct": int(len(rows)), "worst_gold_served_pos_0-9": int((wp < 10).sum()), "10-24": int(((wp >= 10) & (wp < 25)).sum()),
         "25-49": int((wp >= 25).sum()), "share_25-49": round(float((wp >= 25).mean()), 3), "share_10-49": round(float((wp >= 10).mean()), 3)}
    if (hops >= 0).any():
        for h in (1, 2, 3):
            s = hops[rows] == h
            d["hop%d_share_25-49" % h] = round(float((wp[s] >= 25).mean()), 3)
    out[name] = d
    print(name, json.dumps(d))
json.dump(out, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", OUT)
