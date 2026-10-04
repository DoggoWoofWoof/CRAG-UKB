"""Where is the P50 headroom?  Decomposition of the DEV_A queries a cache's BASE fails, under the frozen evidence
(served Dense/SPLADE top-100 hits through the legacy node -> blocks table):
   INFEASIBLE   more than P50 gold blocks (no selector can cover)
   UNREACHED    at least one gold block receives no vote from any served hit   -> needs NEW evidence (structure or new points)
   REACHED_WEAK all gold blocks have evidence, but at least one is outside both channels' top-50   -> deep re-ranking only
   FUSION       all gold blocks are inside the union of the two channels' top-50s          -> a better fusion could fix it
Also: BASE success split by the query's channel agreement (is agreement a usable confidence signal?)."""
import os
import sys

import numpy as np

import _l1g_core as G

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = G.Data(name, dense_fp32=False)
C = D.C
m = D.A
gm = G.gold_mask(D)
rd, rs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
in50 = (G.positions(rd) < G.P_MAIN) | (G.positions(rs) < G.P_MAIN)
ngold = gm.sum(axis=1)
infeasible = ngold > G.P_MAIN
unreached = (gm & ~ev).sum(axis=1) > 0
weak = ~unreached & ((gm & ~in50).sum(axis=1) > 0)
fusion = ~unreached & ~weak
base = D.base_all.astype(bool)
fail = ~base
res = {"cache": name, "partition": G.PARTITION_OF[name], "n_DEV_A": int(m.sum()), "BASE_ALL": round(float(base[m].mean()), 4)}


def frac(x):
    return round(float(x[m].mean()), 4)


res["fail"] = frac(fail)
res["fail_infeasible"] = frac(fail & infeasible)
res["fail_unreached"] = frac(fail & ~infeasible & unreached)
res["fail_reached_weak"] = frac(fail & ~infeasible & weak)
res["fail_fusion_fixable"] = frac(fail & ~infeasible & fusion)
res["gold_blocks_per_query"] = round(float(ngold[m].mean()), 3)
res["unreached_gold_blocks_per_failed_query"] = round(float((gm & ~ev).sum(axis=1)[m & fail & unreached].mean()), 3) if (m & fail & unreached).any() else 0.0
hops = C.hops
if (hops >= 0).any():
    res["by_hop"] = {}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        mm = m & (hops == h)
        res["by_hop"]["hop%d" % h] = {"n": int(mm.sum()), "BASE": round(float(base[mm].mean()), 3), "fail_unreached": round(float((fail & unreached)[mm].mean()), 3),
                                      "fail_reached_weak": round(float((fail & weak)[mm].mean()), 3), "fail_fusion_fixable": round(float((fail & fusion)[mm].mean()), 3)}
A = G.agreement_set([rd, rs])
qs = np.quantile(A[m], [0.25, 0.5, 0.75])
b = np.digitize(A, qs)
res["BASE_by_agreement_quartile"] = [round(float(base[m & (b == k)].mean()), 3) for k in range(4)]
res["unreached_by_agreement_quartile"] = [round(float(unreached[m & (b == k)].mean()), 3) for k in range(4)]
G.log("%s [%s] BASE %.4f | fail %.3f = infeasible %.3f + UNREACHED %.3f + reached-weak %.3f + fusion-fixable %.3f | gold blocks/q %.2f, unreached gold blocks per failed q %.2f" % (
    name, res["partition"], res["BASE_ALL"], res["fail"], res["fail_infeasible"], res["fail_unreached"], res["fail_reached_weak"], res["fail_fusion_fixable"],
    res["gold_blocks_per_query"], res["unreached_gold_blocks_per_failed_query"]))
if "by_hop" in res:
    G.log("  by hop: %s" % res["by_hop"])
G.log("  BASE by agreement quartile %s | unreached by agreement quartile %s" % (res["BASE_by_agreement_quartile"], res["unreached_by_agreement_quartile"]))
G.S.wj(os.path.join(G.OUT, "headroom_A_%s.json" % name), res)
