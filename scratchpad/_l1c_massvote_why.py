"""Mechanism check for massvote A1 -> A3 losses (metaqa DEV_A): is the lost gold block the OWN block of a top-ranked hit whose
vote got split by the normalisation?  Reads nothing new; rebuilds the A1/A3 tables exactly as _l1c_massvote does."""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

import _l1g_core as G

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
cap = int(round(N / npart))
K0, K_LOCK = G.S.K0, G.K_LOCK
xo, ao = D.cd.struct_csr(directed=True)
xo, ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
xu, au = D.cd.struct_csr(directed=False)
deg_u = np.diff(np.asarray(xu, np.int64))
D.cd._csr.clear()
hub = (deg_u + 1) > cap
A_in = sp.csr_matrix((np.ones(len(ao), np.int8), ao, xo), shape=(N, N)).T.tocsr()
xi, ai = np.asarray(A_in.indptr, np.int64), np.asarray(A_in.indices, np.int64)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_l1c_massvote.py")).read().split("def weighted_table")[1].split("ADM = ")[0].replace("(neigh, rule):", "def weighted_table(neigh, rule):", 1))
ADM = [(xo, ao, None), (xi, ai, ~hub)]
T1, _ = weighted_table(ADM, "full")
T3, _ = weighted_table(ADM, "support")
dl = [D.d_ids[i, :K_LOCK] for i in range(nq)]
sl = [D.s_ids[i, :K_LOCK] for i in range(nq)]
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
res = {}
ranks = {}
for tag, tab in (("A1", T1), ("A3", T3)):
    Cd, Sd = weighted_partition_ranking(dl, tab)
    Cs, Ss = weighted_partition_ranking(sl, tab)
    base = G.F0([Cd, Cs], npart)
    pos = G.positions(base)
    ranks[tag] = (pos, Sd, Ss)
pos1, pos3 = ranks["A1"][0], ranks["A3"][0]
ok1 = (gm & (pos1 < G.P_MAIN)).sum(axis=1) == gs
ok3 = (gm & (pos3 < G.P_MAIN)).sum(axis=1) == gs
for h in (1, 2, 3):
    s = m & (hops == h)
    lost = np.nonzero(s & ok1 & ~ok3)[0]
    gained = np.nonzero(s & ~ok1 & ok3)[0]
    own_top = 0
    own_any10 = 0
    worst_pos = []
    for i in lost:
        gb = np.nonzero(gm[i] & (pos3[i] >= G.P_MAIN))[0]           # gold blocks that fell out of P50 under A3
        top = {int(hard[D.d_ids[i, 0]]), int(hard[D.s_ids[i, 0]])}
        top10 = set(int(x) for x in hard[np.concatenate([D.d_ids[i, :10], D.s_ids[i, :10]])])
        own_top += int(any(int(b) in top for b in gb))
        own_any10 += int(any(int(b) in top10 for b in gb))
        worst_pos.append(int(pos3[i, gb].max()))
    res["hop%d" % h] = {"n": int(s.sum()), "A1_ok": int((s & ok1).sum()), "A3_ok": int((s & ok3).sum()), "lost_A1_to_A3": int(len(lost)), "gained": int(len(gained)),
                        "lost_where_a_fallen_gold_block_is_the_own_block_of_the_rank1_hit": own_top,
                        "lost_where_a_fallen_gold_block_is_the_own_block_of_a_top10_hit": own_any10,
                        "fallen_gold_block_position_under_A3_median": int(np.median(worst_pos)) if worst_pos else None}
G.log(json.dumps(res))
G.S.wj(os.path.join(G.X.REPO, "results", "L1_COVPART", "massvote_why_A_%s.json" % name), res)
