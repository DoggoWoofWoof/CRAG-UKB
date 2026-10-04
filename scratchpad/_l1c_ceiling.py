"""L1_COVPART ceiling probe (DEV_A, query-AWARE, gold-AWARE -- a diagnostic, never a candidate): how much of the UNREACHED mass
could ANY balanced partition with the frozen block capacity convert?

Start from a valid partition; visit the DEV_A queries that fail UNREACHED; for every unreached gold node g pick a target
t among the query's voting nodes (served hits and their directed out-neighbours -- the only nodes whose blocks receive
votes) and MOVE g into block(t); if block(t) is at the balance bound, SWAP g with a node of block(t) that is neither a gold
nor a voting node of any DEV_A query.  Moves are greedy in query order and never undone, so the result is one concrete
balanced partition and its reach is a LOWER bound on the query-aware optimum -- and, since a query-independent partition
cannot exploit which nodes are gold for which query, a fair indication of the ceiling of the partition lever.
Records: results/L1_COVPART/ceiling_A_<cache>.json.
"""
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G
from _l1c_path2_build import path2_keys  # noqa: E402  (the lane's PATH2 adjacency: non-hub closed 2-hop STRUCT neighbourhoods)

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa_phg"
start_tag = sys.argv[2] if len(sys.argv) > 2 else "FROZEN"
policy = sys.argv[3] if len(sys.argv) > 3 else "capacity"      # capacity: voting block with most room | rank: best-ranked voting block with room
#                                                              | rank_ball2: best-ranked block among voting nodes within g's PATH2 ball (structure-constrained)
scope = sys.argv[4] if len(sys.argv) > 4 else "unreached"      # unreached: golds in unvoted blocks | missed: every gold whose block is outside P50
t0 = time.time()
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N = D.nq, D.N
m = D.A
hops = np.asarray(C.hops)
ds = name.replace("_phg", "")
if start_tag == "FROZEN":
    hard = D.hard.astype(np.int64).copy()
else:
    hard = np.load(os.path.join(PDIR, "%s__%s.npy" % (ds, start_tag))).astype(np.int64)
    D.override_partition(hard.copy(), start_tag)
k = D.npart
bound = int(np.ceil(1.03 * N / k))                                  # the PHG contract bound (phg.validity), the stricter of the two lane bounds
xo, ao = D.cd.struct_csr(directed=True); ao = np.asarray(ao, np.int64); xo = np.asarray(xo, np.int64)
D.cd._csr.clear()
if policy == "rank_ball2":
    import scipy.sparse as sp
    _N, ST, _KN, _NX = D.cd.keysets()
    cap = int(round(N / k))
    keys2, hub2, _deg, _d2 = path2_keys(N, ST, cap)
    u2, v2 = keys2 // N, keys2 % N
    B2 = sp.coo_matrix((np.ones(2 * len(u2), np.int8), (np.concatenate([u2, v2]), np.concatenate([v2, u2]))), shape=(N, N)).tocsr()
    x2, a2 = np.asarray(B2.indptr, np.int64), np.asarray(B2.indices, np.int64)
    G.log("PATH2 adjacency: %d keys, %d hubs, cap %d" % (len(keys2), int(hub2.sum()), cap))


def profile(tag):
    ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
    Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
    base = G.F0([Cd, Cs], D.npart)
    pos = G.positions(base)
    gm = G.gold_mask(D)
    gs = gm.sum(axis=1)
    feas = gs <= G.P_MAIN
    ball = (gm & (pos < G.P_MAIN)).sum(axis=1) == gs
    reach = (gm & ~ev).sum(axis=1) == 0
    out = {"BASE_ALL": round(float(ball[m].mean()), 4), "reach_all": round(float(reach[m].mean()), 4), "feasible": round(float(feas[m].mean()), 4),
           "UNREACHED_pts": round(float((~ball & feas & ~reach)[m].mean()) * 100, 1),
           "per_hop": {"hop%d" % h: {"BASE_ALL": round(float(ball[m & (hops == h)].mean()), 4), "reach_all": round(float(reach[m & (hops == h)].mean()), 4)}
                       for h in sorted(set(int(x) for x in hops[m] if x >= 0))},
           "size_min": int(D.sizes.min()), "size_max": int(D.sizes.max())}
    G.log("%-10s %s" % (tag, json.dumps(out)))
    return out, ev, reach, feas, ball, pos


res = {"cache": name, "start": start_tag, "policy": policy, "scope": scope, "swap_partner_protection": (sys.argv[5] if len(sys.argv) > 5 else "golds"),
       "blocks": k, "balance_bound": bound, "n_DEV_A": int(m.sum())}
res["before"], ev, reach, feas, ball, pos = profile("start")
# voting nodes of every DEV_A query (hits + directed out-neighbours) and the gold nodes: protected from being swapped out
voters = np.zeros(N, bool)
golds = np.zeros(N, bool)
vlist = {}
for i in np.nonzero(m)[0]:
    hits = np.unique(np.concatenate([D.d_ids[i, :G.K_LOCK], D.s_ids[i, :G.K_LOCK]]))
    cnt = xo[hits + 1] - xo[hits]
    idx = np.repeat(xo[hits], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    vs = np.unique(np.concatenate([hits, ao[idx]]))
    vlist[int(i)] = vs
    voters[vs] = True
    golds[np.asarray(C.gold_nodes[i], np.int64)] = True
protect = sys.argv[5] if len(sys.argv) > 5 else "golds"         # swap partners: golds -> any non-gold node | voters -> neither gold nor voting node
free = ~golds if protect == "golds" else ~(voters | golds)
sizes = np.bincount(hard, minlength=k)
moves = swaps = conflicts = stuck = no_ball_voter = 0
targets_of = {}
order = [int(i) for i in np.nonzero(m & feas & (~reach if scope == "unreached" else ~ball))[0]]
for i in order:
    vs = vlist[i]
    gold = np.asarray(C.gold_nodes[i], np.int64)
    vblocks = np.unique(hard[vs])
    for g in gold:
        if (hard[g] in vblocks) if scope == "unreached" else (pos[i, hard[g]] < G.P_MAIN):
            continue
        if g in targets_of:                                       # already placed for an earlier query: leave it (conflict)
            conflicts += 1
            continue
        # target block: capacity -> the voting block with the most room; rank -> the best-ranked voting block (frozen fused order)
        #               rank_ball2 -> best-ranked block of a voting node inside g's PATH2 ball (what a structure-preserving partition could do)
        if policy == "rank_ball2":
            nb = a2[x2[g]:x2[g + 1]]
            vb = np.unique(hard[np.intersect1d(nb, vs, assume_unique=True)]) if len(nb) else np.zeros(0, np.int64)
            if len(vb) == 0:
                no_ball_voter += 1
                continue
            cand = vb[np.argsort(pos[i, vb], kind="stable")]
        else:
            cand = vblocks[np.argsort(sizes[vblocks] if policy == "capacity" else pos[i, vblocks], kind="stable")]
        placed = False
        for b in cand:
            if sizes[b] < bound:
                sizes[hard[g]] -= 1; hard[g] = b; sizes[b] += 1
                moves += 1; placed = True
                break
        if not placed:
            for b in cand:                                           # swap with a free node of that block
                fr = np.nonzero((hard == b) & free)[0]
                if len(fr):
                    x = int(fr[0])
                    hard[x] = hard[g]; hard[g] = b
                    swaps += 1; placed = True
                    break
        if placed:
            targets_of[int(g)] = int(hard[g])
        else:
            stuck += 1
G.log("greedy: %d queries, moves %d swaps %d conflicts %d stuck %d no_voter_in_ball2 %d (%.0fs)" % (len(order), moves, swaps, conflicts, stuck, no_ball_voter, time.time() - t0))
assert (np.bincount(hard, minlength=k) <= bound).all() and (np.bincount(hard, minlength=k) > 0).all()
D.override_partition(hard.copy(), "ORACLE_PACKED")
res["after_query_aware_packing"], ev2, reach2, feas2, ball2, pos2 = profile("packed")
g_, l_, p_ = G.X.mcnemar(reach[m], reach2[m])
res["reach_all_paired"] = {"gained": g_, "lost": l_, "p": p_}
g_, l_, p_ = G.X.mcnemar(ball[m], ball2[m])
res["BASE_ALL_paired"] = {"gained": g_, "lost": l_, "p": p_}
res["greedy"] = {"queries_visited": len(order), "moves": moves, "swaps": swaps, "conflicts_node_already_placed": conflicts, "stuck": stuck, "no_voter_in_ball2": no_ball_voter,
                 "voting_nodes": int(voters.sum()), "gold_nodes_DEV_A": int(golds.sum()), "free_nodes": int(free.sum())}
G.S.wj(os.path.join(OUT, "ceiling_A_%s__%s_%s_%s%s.json" % (name, start_tag, policy, scope, "" if protect == "golds" else "__protect_voters")), res)
G.log("done %.0fs" % (time.time() - t0))
