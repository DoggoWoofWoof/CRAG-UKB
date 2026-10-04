"""Attribution for H2_AS_SAFE_CANDIDATE_GENERATOR (no new arm, nothing selected): on the metaqa caches, for the hop-2 DEV_A queries
whose missing gold blocks the H2 generator DOES propose, where do those blocks land in the unchanged F6 competition for the six slots,
what occupies the slots instead, and where in the beam the gold blocks were first reached.  Same pinned beam, same frozen SAFE
machinery, same min-rank candidate rule as _l1c_h2safe.py.
    python -u _l1c_h2safe_why.py <cache>      -> results/L1_COVPART/h2safe_why_A_<cache>.json
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
goldp = Cc["goldp"]
K0 = RT.K0


def bucket(r):
    return "0-5" if r <= 5 else "6-24" if r <= 24 else "25-49" if r <= 49 else "50+"


def sbucket(r):
    return "1-6" if r <= 6 else "7-12" if r <= 12 else "13-24" if r <= 24 else "25+"


res = {"cache": name, "n_DEV_A": nA, "groups": {}}
for hname, hsel in [("hop2", 2), ("hop3", 3), ("hop1", 1)]:
    rows_h = [qi for qi in rowsA if hops[qi] == hsel]
    o = {"n": len(rows_h), "n_gold_outside_P50": 0, "generator_proposes_all_missing": 0,
         "worst_missing_gold__slot_rank_in_F6_competition": {"1-6": 0, "7-12": 0, "13-24": 0, "25+": 0},
         "missing_gold_blocks_per_query_mean": 0.0,
         "worst_missing_gold__channels": {"canonical_top200": 0, "static_structural": 0, "retrieval": 0, "H2_only": 0},
         "worst_missing_gold__first_reached_at_hop": {"hop1": 0, "hop2": 0},
         "worst_missing_gold__within_hop_beam_rank_of_best_node": {"0-5": 0, "6-24": 0, "25-49": 0, "50+": 0},
         "worst_missing_gold__H2_first_visit_position": {"0-5": 0, "6-24": 0, "25-49": 0, "50+": 0},
         "worst_missing_gold__hop1_blocks_preceding_it": [],
         "slots_composition_when_generator_proposes_all_missing": {"incumbent_kept": 0, "static_structural_challenger": 0, "retrieval_challenger": 0,
                                                                   "H2_credited_hop1_block": 0, "H2_credited_hop2_block": 0, "gold": 0},
         "score_gap__sixth_slot_minus_worst_missing_gold_mean": []}
    nmiss = []
    for qi in rows_h:
        c = ctxs[qi]
        miss = [b for b in goldp[qi] if b not in c["base50"]]
        if not miss:
            continue
        o["n_gold_outside_P50"] += 1
        nmiss.append(len(miss))
        nodes = beams1[qi] + beams2[qi]
        bl = list(dict.fromkeys(int(hard[v]) for v in nodes))
        h2pos = {b: r for r, b in enumerate(bl)}
        if not all(b in h2pos for b in miss):
            continue
        o["generator_proposes_all_missing"] += 1
        # first-visit hop and within-hop rank of the best node of each block
        first_hop, within = {}, {}
        for r, v in enumerate(beams1[qi]):
            b = int(hard[v])
            if b not in first_hop:
                first_hop[b] = 1
                within[b] = r
        for r, v in enumerate(beams2[qi]):
            b = int(hard[v])
            if b not in first_hop:
                first_hop[b] = 2
                within[b] = r
        # the unchanged F6 competition with the H2 source
        chal = list(c["chal"]) + [b for b in bl if b not in c["base50"]]
        spos = dict(c["spos"])
        for b, r in h2pos.items():
            spos[b] = min(spos.get(b, 10 ** 9), r)
        Xs, sc = KB.f6_select(c["bnd"], chal, spos, c["rpos"], c["cpos"], B)
        order = [p for _, _, p in sc]
        score = {p: -s for s, _, p in sc}
        slot = {p: i + 1 for i, p in enumerate(order)}
        w = max(miss, key=lambda b: (slot.get(b, 10 ** 9), b))
        o["worst_missing_gold__slot_rank_in_F6_competition"][sbucket(slot[w])] += 1
        o["worst_missing_gold__channels"]["canonical_top200"] += int(w in c["cpos"])
        o["worst_missing_gold__channels"]["static_structural"] += int(w in c["spos"])
        o["worst_missing_gold__channels"]["retrieval"] += int(w in c["rpos"])
        o["worst_missing_gold__channels"]["H2_only"] += int(w not in c["cpos"] and w not in c["spos"] and w not in c["rpos"])
        o["worst_missing_gold__first_reached_at_hop"]["hop%d" % first_hop[w]] += 1
        o["worst_missing_gold__within_hop_beam_rank_of_best_node"][bucket(within[w])] += 1
        o["worst_missing_gold__H2_first_visit_position"][bucket(h2pos[w])] += 1
        o["worst_missing_gold__hop1_blocks_preceding_it"].append(sum(1 for b in bl[:h2pos[w]] if first_hop[b] == 1))
        o["score_gap__sixth_slot_minus_worst_missing_gold_mean"].append(score[order[B - 1]] - score[w])
        comp = o["slots_composition_when_generator_proposes_all_missing"]
        for p in Xs:
            if p in goldp[qi]:
                comp["gold"] += 1
            if p in c["bnd"]:
                comp["incumbent_kept"] += 1
            elif h2pos.get(p, 10 ** 9) < c["spos"].get(p, 10 ** 9):
                comp["H2_credited_hop%d_block" % first_hop[p]] += 1
            elif p in c["spos"]:
                comp["static_structural_challenger"] += 1
            elif p in c["rpos"]:
                comp["retrieval_challenger"] += 1
    o["missing_gold_blocks_per_query_mean"] = round(float(np.mean(nmiss)), 2) if nmiss else None
    o["worst_missing_gold__hop1_blocks_preceding_it"] = (round(float(np.mean(o["worst_missing_gold__hop1_blocks_preceding_it"])), 1)
                                                         if o["worst_missing_gold__hop1_blocks_preceding_it"] else None)
    o["score_gap__sixth_slot_minus_worst_missing_gold_mean"] = (round(float(np.mean(o["score_gap__sixth_slot_minus_worst_missing_gold_mean"])), 5)
                                                                if o["score_gap__sixth_slot_minus_worst_missing_gold_mean"] else None)
    res["groups"][hname] = o
    G.log("%s %s" % (hname, json.dumps(o)))
G.S.wj(os.path.join(OUT, "h2safe_why_A_%s.json" % name), res)
G.log("done")
