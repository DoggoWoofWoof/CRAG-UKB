"""Gold-free input statistic for the ruled H2_PATCH1 arm (nothing evaluated; gold is not read): after canonical SAFE selects its
50 blocks and its weakest admitted block (the sixth of the F6 competition) is removed, how many of the pinned beam's visited nodes
are NOT exposed by the 49 kept blocks (novel), split by the hop at which they were reached; how often the novel set exceeds the
removed block's capacity (truncation -- the only case in which the patch order matters); and what the two candidate orders
(first-visit; deepest-hop first) would put into the patch.
    python -u _l1c_h2patch_stat.py <cache>      -> results/L1_COVPART/h2patch_stat_A_<cache>.json
"""
import hashlib
import json
import os
import sys

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED_SHA = "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5"
CFG = dict(KB.BASE_CFG)
B = CFG["B"]
name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)

z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
sizes = np.asarray(C.part_sizes, np.int64)
assert sizes.sum() == N and int(np.bincount(hard, minlength=npart)[0]) == int(sizes[0])

groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else []
groups = [("all", list(rowsA))] + groups
per = {}
for qi in rowsA:
    c = ctxs[qi]
    Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    weakest = Xs[-1]
    kept = (c["prot_set"] | set(Xs)) - {weakest}
    assert len(kept) == RT.P - 1
    b1, b2 = beams1[qi], beams2[qi]
    nov1 = [v for v in b1 if int(hard[v]) not in kept]
    nov2 = [v for v in b2 if int(hard[v]) not in kept]
    nb = int(sizes[weakest])
    k = len(nov1) + len(nov2)
    pa = (nov1 + nov2)[:nb]                       # (a) frozen first-visit order
    pb = (nov2 + nov1)[:nb]                       # (b) deepest hop first, frozen within-hop order
    s1 = set(nov1)
    per[qi] = dict(visited=len(b1) + len(b2), v1=len(b1), v2=len(b2), nov1=len(nov1), nov2=len(nov2), nb=nb, k=k,
                   trunc=int(k > nb), fill=max(nb - k, 0), in_removed=sum(1 for v in b1 + b2 if int(hard[v]) == weakest),
                   pa_h1=sum(1 for v in pa if v in s1), pb_h1=sum(1 for v in pb if v in s1),
                   nov2_lost_a=len(nov2) - sum(1 for v in pa if v not in s1),
                   nov1_lost_b=len(nov1) - sum(1 for v in pb if v in s1),
                   weakest_cpos=c["cpos"].get(weakest, -1), weakest_is_boundary=int(weakest in c["bnd"]),
                   seeds_exposed=sum(1 for s in C.seeds[qi] if s >= 0 and int(hard[int(s)]) in (kept | {weakest})),
                   blocks_of_novel=len(set(int(hard[v]) for v in nov1 + nov2)))


def agg(rows_h):
    P_ = [per[qi] for qi in rows_h]
    n = len(P_)
    if n == 0:
        return {}
    f = lambda key: float(np.mean([p[key] for p in P_]))
    o = {"n": n,
         "visited_per_query_mean": round(f("visited"), 1), "hop1_visited_mean": round(f("v1"), 1), "hop2_visited_mean": round(f("v2"), 1),
         "novel_hop1_per_query_mean": round(f("nov1"), 1), "novel_hop2_per_query_mean": round(f("nov2"), 1),
         "novel_total_per_query_mean": round(f("k"), 1),
         "novel_fraction_of_visited_hop1": round(sum(p["nov1"] for p in P_) / max(1, sum(p["v1"] for p in P_)), 3),
         "novel_fraction_of_visited_hop2": round(sum(p["nov2"] for p in P_) / max(1, sum(p["v2"] for p in P_)), 3),
         "distinct_blocks_of_novel_nodes_mean": round(f("blocks_of_novel"), 1),
         "removed_block_size_mean": round(f("nb"), 1),
         "removed_block_size_p10_p50_p90": [int(np.percentile([p["nb"] for p in P_], q)) for q in (10, 50, 90)],
         "visited_nodes_inside_removed_block_mean": round(f("in_removed"), 2),
         "queries_with_no_novel_node": int(sum(p["k"] == 0 for p in P_)),
         "queries_truncated (novel > capacity)": int(sum(p["trunc"] for p in P_)),
         "truncated_fraction": round(f("trunc"), 3),
         "fill_from_removed_block_mean": round(f("fill"), 1),
         "patch_share_hop1_order_a_first_visit": round(sum(p["pa_h1"] for p in P_) / max(1, sum(min(p["k"], p["nb"]) for p in P_)), 3),
         "patch_share_hop1_order_b_deepest_first": round(sum(p["pb_h1"] for p in P_) / max(1, sum(min(p["k"], p["nb"]) for p in P_)), 3),
         "novel_hop2_nodes_cut_by_order_a_mean": round(f("nov2_lost_a"), 2),
         "novel_hop1_nodes_cut_by_order_b_mean": round(f("nov1_lost_b"), 2),
         "weakest_block_is_a_boundary_block_fraction": round(f("weakest_is_boundary"), 3),
         "weakest_block_served_position_mean_if_ranked": round(float(np.mean([p["weakest_cpos"] for p in P_ if p["weakest_cpos"] >= 0])), 1) if any(p["weakest_cpos"] >= 0 for p in P_) else None,
         "seeds_exposed_by_SAFE50_mean_of_5": round(f("seeds_exposed"), 2)}
    return o


res = {"cache": name, "n_DEV_A": nA, "B": B, "gold_read": False,
       "definition": {"kept": "canonical SAFE final 50 minus its weakest admitted block (sixth of the F6 competition)",
                      "novel": "pinned-beam visited node (seeds excluded) whose own block is not among the 49 kept blocks",
                      "capacity": "size of the removed block", "order_a": "frozen first-visit order (hop-1 beam then hop-2 beam)",
                      "order_b": "deepest hop first (hop-2 novel then hop-1 novel), frozen within-hop order"},
       "groups": {g: agg(rows_h) for g, rows_h in groups}}
for g in res["groups"]:
    G.log("%s %s" % (g, json.dumps(res["groups"][g])))
G.S.wj(os.path.join(OUT, "h2patch_stat_A_%s.json" % name), res)
G.log("done")
