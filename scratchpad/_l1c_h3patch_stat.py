"""Gold-free input statistic for the ruled H3_PATCH1 arm (ninth ruling of 2026-09-15; nothing evaluated; gold is not read).

The pinned uL3 beam (scratchpad/_l1c_microl3.py, sha asserted) is re-executed with DEPTH = 3 instead of 2: hops 1 and 2 are
computed by the identical code and are therefore identical to the frozen H2 beam (asserted downstream in _l1c_h3patch.py against
the section 14.1 record); hop 3 is the same transition step applied once more from the hop-2 beam.  After canonical SAFE selects
its 50 blocks and its weakest admitted block (the sixth of the F6 competition) is removed, this script counts, per query, the
visited nodes NOT exposed by the 49 kept blocks (novel) by the hop at which they were reached, how often the novel set exceeds
the removed block's capacity, and what the two candidate orders -- (a) frozen first-visit (hop-1, hop-2, hop-3 beams) and
(b) deepest hop first (hop-3, hop-2, hop-1) -- would put into the one-block patch.  It also records the hop-3 beam's cost.
    python -u _l1c_h3patch_stat.py <cache>      -> results/L1_COVPART/h3patch_stat_A_<cache>.json
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
SUBS = [("name = sys.argv[1]", "name = %r" % name),
        ('LATENT = "--latent" in sys.argv', "LATENT = False"),
        ("DEPTH = 2", "DEPTH = 3"),
        ("beams1, beams2 = [[] for _ in range(nq)], [[] for _ in range(nq)]",
         "beams1, beams2, beams3 = [[] for _ in range(nq)], [[] for _ in range(nq)], [[] for _ in range(nq)]"),
        ("beams1[qi], beams2[qi] = beams[0], beams[1]", "beams1[qi], beams2[qi], beams3[qi] = beams[0], beams[1], beams[2]")]
for a, b in SUBS:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
exec(head)                                                   # D, C, beams1, beams2, beams3, edges_h (nA, 3), secs, hard, hops, rowsA ...
assert DEPTH == 3 and edges_h.shape[1] == 3

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
jrow = {int(qi): j for j, qi in enumerate(rowsA)}
for qi in rowsA:
    c = ctxs[qi]
    Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    weakest = Xs[-1]
    kept = (c["prot_set"] | set(Xs)) - {weakest}
    assert len(kept) == RT.P - 1
    b1, b2, b3 = beams1[qi], beams2[qi], beams3[qi]
    nov1 = [v for v in b1 if int(hard[v]) not in kept]
    nov2 = [v for v in b2 if int(hard[v]) not in kept]
    nov3 = [v for v in b3 if int(hard[v]) not in kept]
    nb = int(sizes[weakest])
    k = len(nov1) + len(nov2) + len(nov3)
    k2 = len(nov1) + len(nov2)                                # what the depth-2 arm had
    pa = (nov1 + nov2 + nov3)[:nb]                            # (a) frozen first-visit order
    pb = (nov3 + nov2 + nov1)[:nb]                            # (b) deepest hop first, frozen within-hop order
    s1, s2, s3 = set(nov1), set(nov2), set(nov3)
    per[qi] = dict(v1=len(b1), v2=len(b2), v3=len(b3), nov1=len(nov1), nov2=len(nov2), nov3=len(nov3), nb=nb, k=k, k2=k2,
                   trunc=int(k > nb), trunc2=int(k2 > nb), fill=max(nb - k, 0), fill2=max(nb - k2, 0),
                   pa_h1=sum(1 for v in pa if v in s1), pa_h2=sum(1 for v in pa if v in s2), pa_h3=sum(1 for v in pa if v in s3),
                   pb_h1=sum(1 for v in pb if v in s1), pb_h2=sum(1 for v in pb if v in s2), pb_h3=sum(1 for v in pb if v in s3),
                   nov3_cut_a=len(nov3) - sum(1 for v in pa if v in s3), nov2_cut_a=len(nov2) - sum(1 for v in pa if v in s2),
                   nov2_cut_b=len(nov2) - sum(1 for v in pb if v in s2), nov1_cut_b=len(nov1) - sum(1 for v in pb if v in s1),
                   h3_edges=int(edges_h[jrow[int(qi)], 2]), h3_slots_a=sum(1 for v in pa if v in s3),
                   blocks_of_novel=len(set(int(hard[v]) for v in nov1 + nov2 + nov3)),
                   h3_in_removed=sum(1 for v in b3 if int(hard[v]) == weakest))


def agg(rows_h):
    P_ = [per[qi] for qi in rows_h]
    n = len(P_)
    if n == 0:
        return {}
    f = lambda key: float(np.mean([p[key] for p in P_]))
    tot = lambda key: sum(p[key] for p in P_)
    used = max(1, sum(min(p["k"], p["nb"]) for p in P_))
    o = {"n": n,
         "visited_per_hop_mean (hop1 / hop2 / hop3)": [round(f("v1"), 1), round(f("v2"), 1), round(f("v3"), 1)],
         "novel_per_hop_mean (hop1 / hop2 / hop3)": [round(f("nov1"), 1), round(f("nov2"), 1), round(f("nov3"), 1)],
         "novel_total_per_query_mean_depth3": round(f("k"), 1), "novel_total_per_query_mean_depth2": round(f("k2"), 1),
         "novel_fraction_of_visited (hop1 / hop2 / hop3)": [round(tot("nov1") / max(1, tot("v1")), 3), round(tot("nov2") / max(1, tot("v2")), 3), round(tot("nov3") / max(1, tot("v3")), 3)],
         "distinct_blocks_of_novel_nodes_mean": round(f("blocks_of_novel"), 1),
         "removed_block_size_mean": round(f("nb"), 1),
         "queries_truncated_depth3 (novel > capacity)": int(tot("trunc")), "truncated_fraction_depth3": round(f("trunc"), 3),
         "queries_truncated_depth2": int(tot("trunc2")),
         "fill_from_removed_block_mean_depth3": round(f("fill"), 1), "fill_from_removed_block_mean_depth2": round(f("fill2"), 1),
         "patch_composition_order_a_first_visit (hop1 / hop2 / hop3 share of novel positions)": [round(tot("pa_h1") / used, 3), round(tot("pa_h2") / used, 3), round(tot("pa_h3") / used, 3)],
         "patch_composition_order_b_deepest_first (hop1 / hop2 / hop3 share)": [round(tot("pb_h1") / used, 3), round(tot("pb_h2") / used, 3), round(tot("pb_h3") / used, 3)],
         "hop3_novel_nodes_admitted_per_query_order_a_mean": round(f("h3_slots_a"), 1),
         "hop3_novel_nodes_cut_by_order_a_mean": round(f("nov3_cut_a"), 2), "hop2_novel_nodes_cut_by_order_a_mean": round(f("nov2_cut_a"), 2),
         "hop2_novel_nodes_cut_by_order_b_mean": round(f("nov2_cut_b"), 2), "hop1_novel_nodes_cut_by_order_b_mean": round(f("nov1_cut_b"), 2),
         "queries_where_order_b_cuts_a_hop2_novel_node": int(sum(p["nov2_cut_b"] > 0 for p in P_)),
         "queries_where_order_a_admits_no_hop3_node_although_some_exist": int(sum((p["h3_slots_a"] == 0) and (p["nov3"] > 0) for p in P_)),
         "hop3_visited_nodes_inside_removed_block_mean": round(f("h3_in_removed"), 2),
         "hop3_transitions_scored_mean": round(f("h3_edges"), 1)}
    return o


res = {"cache": name, "n_DEV_A": nA, "B": B, "gold_read": False, "depth": 3, "beam_module_sha256": PINNED_SHA,
       "head_substitutions": SUBS,
       "definition": {"kept": "canonical SAFE final 50 minus its weakest admitted block (sixth of the F6 competition)",
                      "novel": "pinned-beam visited node (seeds excluded) whose own block is not among the 49 kept blocks; the beam is the pinned code with DEPTH = 3",
                      "capacity": "size of the removed block", "order_a": "frozen first-visit order (hop-1, hop-2, hop-3 beams)",
                      "order_b": "deepest hop first (hop-3, hop-2, hop-1 novel), frozen within-hop order"},
       "cost": {"transitions_scored_per_query_mean (hop1 / hop2 / hop3)": [round(float(edges_h[:, i].mean()), 1) for i in range(3)],
                "transitions_scored_per_query_p95_total": int(np.percentile(edges_h.sum(1), 95)),
                "beam_wall_clock_ms_mean": round(1000 * float(secs.mean()), 1), "beam_wall_clock_ms_p95": round(1000 * float(np.percentile(secs, 95)), 1),
                "nodes_visited_per_query_mean_depth3": round(float(np.mean([len(beams1[qi]) + len(beams2[qi]) + len(beams3[qi]) for qi in rowsA])), 1)},
       "groups": {g: agg(rows_h) for g, rows_h in groups}}
G.log("cost %s" % json.dumps(res["cost"]))
for g in res["groups"]:
    G.log("%s %s" % (g, json.dumps(res["groups"][g])))
G.S.wj(os.path.join(OUT, "h3patch_stat_A_%s.json" % name), res)
G.log("done")
