"""H3_PATCH1 attribution (diagnostic only; no arm, no selection change): where do the missing gold nodes of the SAFE failures sit
relative to the depth-3 beam -- visited at hop h, scored as a hop-h candidate (in C_h = admissible neighbours of the hop-(h-1) beam)
but not selected by the top-100, or never scored at all?  This separates a beam-WIDTH limit from a beam-REACH limit.
    python -u _l1c_h3patch_why.py <cache>   -> results/L1_COVPART/h3patch_why_A_<cache>.json
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
CAND = {}                                                    # (qi, hop) -> unique scored candidate nodes of that hop
SUBS = [("name = sys.argv[1]", "name = %r" % name),
        ('LATENT = "--latent" in sys.argv', "LATENT = False"),
        ("DEPTH = 2", "DEPTH = 3"),
        ("beams1, beams2 = [[] for _ in range(nq)], [[] for _ in range(nq)]",
         "beams1, beams2, beams3 = [[] for _ in range(nq)], [[] for _ in range(nq)], [[] for _ in range(nq)]"),
        ("beams1[qi], beams2[qi] = beams[0], beams[1]", "beams1[qi], beams2[qi], beams3[qi] = beams[0], beams[1], beams[2]"),
        ("        beams.append(nb)\n        B = nb\n", "        beams.append(nb)\n        if m[qi]:\n            CAND[(qi, h)] = np.unique(u_idx)\n        B = nb\n")]
for a, b in SUBS:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
exec(head)
assert DEPTH == 3

z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
sizes = np.asarray(C.part_sizes, np.int64)
gold_nodes = [np.asarray(C.gold_nodes[qi], np.int64) for qi in range(nq)]
seeds_of = {qi: set(int(s) for s in seeds[qi] if s >= 0) for qi in rowsA}
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
res = {"cache": name, "n_DEV_A": nA, "diagnostic_only": True, "depth": 3, "beam_module_sha256": PINNED_SHA, "head_substitutions": SUBS, "groups": {}}
for gname, rows_h in groups:
    o = {"SAFE_failures": 0, "missing_gold_nodes": 0, "missing_is_seed": 0, "missing_is_hub": 0,
         "visited_at_hop (1 / 2 / 3)": [0, 0, 0],
         "scored_but_not_selected_first_at_hop (1 / 2 / 3)": [0, 0, 0],
         "never_scored (not adjacent to any beam node under the frozen admissibility)": 0,
         "candidates_scored_unique_per_query_mean (hop 1 / 2 / 3)": [[], [], []],
         "failures_where_all_missing_nodes_are_scored_within_depth3": 0,
         "failures_where_all_missing_nodes_are_scored_or_visited_within_depth2": 0,
         "missing_nodes_per_failure_mean": []}
    for qi in rows_h:
        c = ctxs[qi]
        Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        safe50 = c["prot_set"] | set(Xs)
        miss = [int(g) for g in gold_nodes[qi] if int(hard[g]) not in safe50]
        for h in range(3):
            o["candidates_scored_unique_per_query_mean (hop 1 / 2 / 3)"][h].append(len(CAND.get((qi, h), ())))
        if not miss:
            continue
        o["SAFE_failures"] += 1
        o["missing_gold_nodes"] += len(miss)
        o["missing_nodes_per_failure_mean"].append(len(miss))
        vis = {}
        for h, bl in enumerate((beams1[qi], beams2[qi], beams3[qi])):
            for v in bl:
                vis.setdefault(int(v), h + 1)
        cand = {}
        for h in range(3):
            for v in CAND.get((qi, h), ()):
                cand.setdefault(int(v), h + 1)
        all_scored3 = True
        all_scored2 = True
        for g in miss:
            o["missing_is_hub"] += int(bool(hub[g]))
            if g in seeds_of[qi]:
                o["missing_is_seed"] += 1
                continue
            if g in vis:
                o["visited_at_hop (1 / 2 / 3)"][vis[g] - 1] += 1
                all_scored2 &= vis[g] <= 2
            elif g in cand:
                o["scored_but_not_selected_first_at_hop (1 / 2 / 3)"][cand[g] - 1] += 1
                all_scored2 &= cand[g] <= 2
            else:
                o["never_scored (not adjacent to any beam node under the frozen admissibility)"] += 1
                all_scored3 = False
                all_scored2 = False
        o["failures_where_all_missing_nodes_are_scored_within_depth3"] += int(all_scored3)
        o["failures_where_all_missing_nodes_are_scored_or_visited_within_depth2"] += int(all_scored2)
    o["candidates_scored_unique_per_query_mean (hop 1 / 2 / 3)"] = [round(float(np.mean(x)), 1) if x else None for x in o["candidates_scored_unique_per_query_mean (hop 1 / 2 / 3)"]]
    o["missing_nodes_per_failure_mean"] = round(float(np.mean(o["missing_nodes_per_failure_mean"])), 2) if o["missing_nodes_per_failure_mean"] else None
    res["groups"][gname] = o
    G.log("%s %s" % (gname, json.dumps(o)))
G.S.wj(os.path.join(OUT, "h3patch_why_A_%s.json" % name), res)
G.log("done")
