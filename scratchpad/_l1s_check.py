"""Sanity: (1) served retrieval cache rows == replay cache lists, (2) BASE reproduced from hard membership +
the frozen numerics, (3) query/node embeddings reproduce the served dense scores, (4) DEV_A BASE number."""
import sys

import numpy as np

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name)
C = D.C
S.log("%s nq=%d N=%d npart=%d Q %s Qs %s E %s d_sc %s" % (name, D.nq, D.N, D.npart, D.Q.shape, D.Qs.shape, D.E.shape, D.d_sc.shape))
# (2) BASE from hard membership
dK = [D.d_ids[i, :S.K_LOCK] for i in range(D.nq)]
sK = [D.s_ids[i, :S.K_LOCK] for i in range(D.nq)]
PR_d = S.canon_channel_rank(dK, D.mem, D.npart)
PR_s = S.canon_channel_rank(sK, D.mem, D.npart)
base = S.rrf_ranks([PR_d, PR_s])
same = (base[:, :200] == D.base_rank[:, :200]).all(axis=1)
S.log("BASE reproduced from hard membership: rows identical (first 200) %d/%d; first-50 identical %d/%d" % (
    same.sum(), D.nq, (base[:, :50] == D.base_rank[:, :50]).all(axis=1).sum(), D.nq))
# (3) dense scores reproduce
i = 0
sc = D.E[D.d_ids[i, :10]] @ D.Q[i]
S.log("dense score check q0: served %s recomputed %s" % (np.round(D.d_sc[i, :10], 4), np.round(sc, 4)))
ss = np.asarray((D.Es[D.s_ids[i, :10]] @ D.Qs[i].T).todense()).ravel()
S.log("splade score check q0: served %s recomputed %s" % (np.round(D.s_sc[i, :10], 3), np.round(ss, 3)))
S.log("dense score range: top1 mean %.3f  rank100 mean %.3f  rank1000 mean %.3f | splade top1 %.2f r100 %.2f r1000 %.2f" % (
    D.d_sc[:, 0].mean(), D.d_sc[:, 99].mean(), D.d_sc[:, 999].mean(), D.s_sc[:, 0].mean(), D.s_sc[:, 99].mean(), D.s_sc[:, 999].mean()))
a = D.r_base["ALL_split"]
S.log("BASE DEV_A ALL %.4f (n=%d)  DEV_B %.4f  full %.4f  scope %s" % (a["A"]["ALL"], a["A"]["n"], a["B"]["ALL"], a["ALL"]["ALL"], D.r_base["scope_nodes"]))
ngb = np.array([len(g) for g in C.gb])
S.log("gold blocks/query mean %.2f  >49: %.3f ; block sizes mean %.0f min %d max %d" % (ngb.mean(), (ngb > 49).mean(), D.sizes.mean(), D.sizes.min(), D.sizes.max()))
