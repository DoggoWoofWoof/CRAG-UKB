"""QMAX_BALANCED_H2_PATCH1 (fourteenth ruling, item 1; PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json written first): ONE composition,
    QMAX_F0 block ranking  ->  SAFE (frozen six-slot F6 swap)  ->  BALANCED_H2_PATCH1 (section 21 balanced beam, section 16 one-block patch)
on the five DEV_A caches.  Same algorithm everywhere; no training; no relation dictionary; no embedding modification; no new constant.

The ONLY change relative to the pinned section-21 chain is the block ranking that enters the SAFE machinery: the served BASE ranking
(frozen RRF of the served dense / SPLADE block channels) is replaced by the L1_GEOM AGGREGATION arm QMAX_F0 = the same frozen RRF over the
same two channels plus the exhaustive dense block max (score(B) = max_{v in B} q . e_v over all actual nodes; _l1g_candidate.dense_blockmax_rank,
pinned).  The first TOPP = 200 blocks of that ranking replace base_rank in the replay-cache view consumed by _l1ps_router.build_cache /
_l1kb_core.contexts: protected = its first 44 blocks, boundary = its blocks 45-50, canonical position cpos = its positions; the structural
(s_node: frozen seed-based expansion, independent of the block ranking) and retrieval (ret_rrf) challengers, the F6 admission, the weakest
admitted block, the pinned beam (seeds = the frozen top-5 RRF nodes, independent of the block ranking), the balanced compression and the
one-block patch are byte-identical in logic to sections 16 / 21 -- the functions are copied verbatim from _l1c_h2balanced.py (pinned) and the
reference chain L1 / SAFE / H2_PATCH1 / BALANCED_H2_PATCH1 is recomputed here and asserted equal to the frozen section-16 / 21 records;
QMAX_F0 is asserted equal to the L1_GEOM record of its DEV_A numbers.
Arms: reference chain L1, SAFE, H2_PATCH1, BALANCED_H2_PATCH1 (reproduced); composed chain QMAX_F0 (P50 of the new ranking), QMAX_SAFE,
QMAX_BALANCED_H2_PATCH1 (PRIMARY -- the only decided-on arm; the other two are its attribution stages).  Not run: any other composition.
    python -u scratchpad/_l1c_qmaxbal.py <cache>            -> results/L1_COVPART/qmaxbal_A_<cache>.json
    python -u scratchpad/_l1c_qmaxbal.py metaqa --dry       -> no record; identities only (BASE, QMAX_F0, SAFE, beams reproduced; |QMAX_SAFE| = 50),
                                                               no coverage number of any composed arm
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G
import _l1g_candidate as CAND
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json")
PRE_GEOM = os.path.join(G.X.REPO, "results", "L1_GEOM", "PREREGISTRATION_DEV_B.json")
NEW_MODULES = ("_l1c_qmaxbal.py", "_l1c_qmaxbal_summary.py")
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d",
          "_l1c_h2balanced.py": "7aed6b1d4399a1b432d0e6be48013cf73c92d4e47d2e5d2dd7b669b508d84bbe",
          "_l1g_candidate.py": "b01c27a6e64b18756db1ca1d8910e8db4bb54fa019d9809698cb1369206169e0",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327"}
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
ARM_H2, ARM_BAL = "H2_PATCH1", "BALANCED_H2_PATCH1"
ARM_Q50, ARM_QSAFE, PRIMARY = "QMAX_F0", "QMAX_SAFE", "QMAX_BALANCED_H2_PATCH1"
ARMS = ("L1", "SAFE", ARM_H2, ARM_BAL, ARM_Q50, ARM_QSAFE, PRIMARY)
CHAIN_OF = {"L1": "BASE", "SAFE": "BASE", ARM_H2: "BASE", ARM_BAL: "BASE", ARM_Q50: "QMAX", ARM_QSAFE: "QMAX", PRIMARY: "QMAX"}
P50_OF = {"BASE": "L1", "QMAX": ARM_Q50}
SAFE_OF = {"BASE": "SAFE", "QMAX": ARM_QSAFE}
ALPHA = 0.01


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def label(o, alpha=ALPHA):
    if o["gained"] > o["lost"] and o["p"] < alpha:
        return "GAIN"
    if o["lost"] > o["gained"] and o["p"] < alpha:
        return "LOSS"
    return "NEUTRAL"


name = sys.argv[1]
DRY = "--dry" in sys.argv
assert name in CACHES, name
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
pre_geom = json.load(open(PRE_GEOM, encoding="utf-8"))
for fn in ("_l1g_candidate.py", "_l1g_core.py", "_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py"):
    assert sha_file(os.path.join(HERE, fn)) == pre_geom["code_sha256"][fn], "%s differs from the L1_GEOM pre-registration" % fn
assert sha_file(G.X.CACHES[name]) == pre_geom["populations"]["caches"][name]["sha256"], "cache %s differs from the L1_GEOM pre-registration" % name
seen_q = pre_geom["recorded_DEV_A"][name]["AGGREGATION"]
assert seen_q["key"] == ARM_Q50
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    for mod in NEW_MODULES:
        assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    fp_out = os.path.join(OUT, "qmaxbal_A_%s.json" % name)
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
else:
    assert name == "metaqa", "the dry run is metaqa only"
    fp_out = None
rec16 = json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % name), encoding="utf-8"))
rec21 = json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % name), encoding="utf-8"))

# ---------------------------------------------------------------- the pinned uL3 head: data, STRUCT adjacency, frozen beam on DEV_A  [as in sections 14-21]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
T_ALL = time.time()
exec(head)                                                   # D, C, beams1, beams2 (the pinned beam on DEV_A), expand_real, fused_order, seeds, hub ...
secs_pinned = secs.copy()
edges_pinned = edges_h.copy()

# ---------------------------------------------------------------- the QMAX_F0 block ranking (L1_GEOM AGGREGATION arm; pinned code path), asserted against its record
t0q = time.time()
D.E = D._E16[:].astype(np.float32)                           # the fp32 node matrix the pinned dense_blockmax_rank reads (what Data(dense_fp32=True) builds)
Cd = G.block_channel(D, D.d_ids)
Cs = G.block_channel(D, D.s_ids)
base_full = G.F0([Cd, Cs], npart)
TOPP = int(D.base_rank.shape[1])
assert (base_full[:, :TOPP] == D.base_rank).all(), "F0(Cd, Cs) does not reproduce the served base_rank"
Cmax = CAND.dense_blockmax_rank(D)
rank_q = G.F0([Cd, Cs, Cmax], npart)
D.E = None
out_q, allv_q = D.eval_rank(rank_q, ARM_Q50, quiet=True, splits=("A",))
oqa = out_q["A"]
assert abs(oqa["ALL"] - seen_q["ALL"]) < 1e-9 and oqa["gained"] == seen_q["gained"] and oqa["lost"] == seen_q["lost"], (oqa, seen_q)
for h, v in seen_q["per_hop"].items():
    assert abs(oqa["per_hop"][h] - v) < 1e-9, h
allv_q = np.asarray(allv_q, bool)
G.log("QMAX_F0 reproduced on DEV_A (ALL %.4f +%d/-%d vs BASE %.4f) in %.0fs; P50 overlap with BASE P50 mean %.1f blocks" % (
    oqa["ALL"], oqa["gained"], oqa["lost"], D.r_base["ALL_split"]["A"]["ALL"], time.time() - t0q,
    float(np.mean([len(set(rank_q[qi, :P_MAIN].tolist()) & set(base_full[qi, :P_MAIN].tolist())) for qi in rowsA]))))


# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording (must equal beams1 / beams2)  [section 21 verbatim]
def beam_pinned_with_candidates(qi):
    q = D.Q[qi].astype(np.float32)
    visited = np.zeros(N, bool)
    Bf = [int(s) for s in seeds[qi] if s >= 0]
    visited[Bf] = True
    beams, cands, parents = [], [], []
    for h in range(DEPTH):
        cand = expand_real(q, Bf, visited)
        if cand is None:
            beams.append([])
            cands.append(np.zeros(0, np.int64))
            parents.append({})
            Bf = []
            continue
        v_idx, u_idx, (dst, dr, mid) = cand
        order = fused_order(dst, dr, mid)
        nb, par = [], {}
        for k in order:
            u = int(u_idx[k])
            if not visited[u]:
                visited[u] = True
                nb.append(u)
                par[u] = int(v_idx[k])
                if len(nb) >= BEAM:
                    break
        beams.append(nb)
        cands.append(np.unique(u_idx))
        parents.append(par)
        Bf = nb
    return beams, cands, parents


# ---------------------------------------------------------------- the balanced compression  [section 21 verbatim]
def rr_compress(v_idx, u_idx, order, frontier, visited, budget):
    """parameter-free per-parent rank round-robin over the frozen fused transition order; marks visited exactly like the pinned beam.
    returns (kept targets in sequence order, parent of each kept target, stats)."""
    pos = np.full(N, -1, np.int64)
    pos[np.asarray(frontier, np.int64)] = np.arange(len(frontier))
    v_o, u_o = v_idx[order], u_idx[order]
    pp = pos[v_o]
    assert (pp >= 0).all()
    key = pp.astype(np.int64) * N + u_o
    _, first = np.unique(key, return_index=True)             # (parent, target) pairs: keep the first occurrence in fused order
    first.sort()
    pp, uu = pp[first], u_o[first]
    o = np.argsort(pp, kind="stable")                        # within-parent rank = position among the parent's pairs in fused order
    ps = pp[o]
    start = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(ps)]))
    r = np.empty(len(pp), np.int64)
    r[o] = np.arange(len(ps)) - start[grp]
    seq = np.lexsort((pp, r))                                # A1 B1 C1 ... A2 B2 C2 ...
    nb, par, last_r = [], {}, -1
    for j in seq:
        u = int(uu[j])
        if not visited[u]:
            visited[u] = True
            nb.append(u)
            par[u] = int(v_o[first[j]])
            last_r = int(r[j])
            if len(nb) >= budget:
                break
    st = {"parents": int(len(frontier)), "parents_with_candidates": int(len(start)), "candidates_unique": int(len(np.unique(uu))), "pairs": int(len(uu)),
          "kept": len(nb), "truncated": int(len(np.unique(uu)) > budget), "rounds_used": last_r + 1,
          "largest_parent_candidate_list": int(np.bincount(pp).max()) if len(pp) else 0}
    return nb, par, st


def beam_balanced(qi):
    q = D.Q[qi].astype(np.float32)
    visited = np.zeros(N, bool)
    Bf = [int(s) for s in seeds[qi] if s >= 0]
    visited[Bf] = True
    beams, cands, parents, stats, n_edges = [], [], [], [], []
    t = time.perf_counter()
    for h in range(DEPTH):
        cand = expand_real(q, Bf, visited)
        if cand is None:
            beams.append([])
            cands.append(np.zeros(0, np.int64))
            parents.append({})
            stats.append(None)
            n_edges.append(0)
            Bf = []
            continue
        v_idx, u_idx, (dst, dr, mid) = cand
        n_edges.append(int(len(u_idx)))
        order = fused_order(dst, dr, mid)
        nb, par, st = rr_compress(v_idx, u_idx, order, Bf, visited, BEAM)
        beams.append(nb)
        cands.append(np.unique(u_idx))
        parents.append(par)
        stats.append(st)
        Bf = nb
    return beams, cands, parents, stats, n_edges, time.perf_counter() - t


def diversity(beam, par):
    """distinct parents represented in a kept beam and the largest single-parent share."""
    if not beam:
        return 0, 0.0
    cnt = {}
    for u in beam:
        p = par[u]
        cnt[p] = cnt.get(p, 0) + 1
    return len(cnt), max(cnt.values()) / float(len(beam))


bal1, bal2 = [[] for _ in range(nq)], [[] for _ in range(nq)]
CAND_P, CAND_B = {}, {}
div = {"pinned": {"hop1": [], "hop2": []}, "balanced": {"hop1": [], "hop2": []}}
stat_b = {"hop1": [], "hop2": []}
secs_bal = np.zeros(nA, np.float64)
edges_bal = np.zeros((nA, DEPTH), np.int64)
same_set = np.zeros((nA, DEPTH), bool)
jacc = np.zeros((nA, DEPTH), np.float64)
n_pinned_cands = np.zeros((nA, DEPTH), np.int64)
t0b = time.time()
for j, qi in enumerate(rowsA):
    bp, cp, pp_ = beam_pinned_with_candidates(qi)
    assert bp[0] == beams1[qi] and bp[1] == beams2[qi], "pinned beam with candidate recording differs from the exec'd pinned beam"
    bb, cb, pb, st, ne, dt = beam_balanced(qi)
    bal1[qi], bal2[qi] = bb[0], bb[1]
    secs_bal[j] = dt
    edges_bal[j] = ne
    for h in range(DEPTH):
        CAND_P[(qi, h)] = cp[h]
        CAND_B[(qi, h)] = cb[h]
        n_pinned_cands[j, h] = len(cp[h])
        a, b_ = set(bp[h]), set(bb[h])
        same_set[j, h] = a == b_
        jacc[j, h] = len(a & b_) / float(len(a | b_)) if (a | b_) else 1.0
        hk = "hop%d" % (h + 1)
        div["pinned"][hk].append(diversity(bp[h], pp_[h]))
        div["balanced"][hk].append(diversity(bb[h], pb[h]))
        stat_b[hk].append(st[h])
    if j % 200 == 0:
        G.log("  balanced %d / %d (%.0fs) hop sizes %d/%d same-set %s jaccard %s" % (j, nA, time.time() - t0b, len(bb[0]), len(bb[1]), same_set[j].tolist(), np.round(jacc[j], 3).tolist()))
G.log("balanced beam done (%.0fs): wall-clock / query mean %.1f ms (pinned %.1f ms)" % (time.time() - t0b, 1000 * secs_bal.mean(), 1000 * secs_pinned.mean()))


def agg_stats(lst):
    lst = [s for s in lst if s is not None]
    if not lst:
        return None
    keys = lst[0].keys()
    return {k: round(float(np.mean([s[k] for s in lst])), 2) for k in keys}


beam_div = {"n_DEV_A": nA, "wall_clock_ms_per_query": {"pinned": round(1000 * float(secs_pinned.mean()), 1), "balanced": round(1000 * float(secs_bal.mean()), 1)},
            "transitions_scored_per_query_mean (hop1 / hop2)": {"pinned": [round(float(x), 1) for x in edges_pinned.mean(0)], "balanced": [round(float(x), 1) for x in edges_bal.mean(0)]},
            "candidates_unique_per_query_mean (hop1 / hop2)": {"pinned": [round(float(x), 1) for x in n_pinned_cands.mean(0)],
                                                               "balanced": [round(float(np.mean([len(CAND_B[(qi, h)]) for qi in rowsA])), 1) for h in range(DEPTH)]},
            "beam_size_mean (hop1 / hop2)": {"pinned": [round(float(np.mean([len(beams1[qi]) for qi in rowsA])), 1), round(float(np.mean([len(beams2[qi]) for qi in rowsA])), 1)],
                                             "balanced": [round(float(np.mean([len(bal1[qi]) for qi in rowsA])), 1), round(float(np.mean([len(bal2[qi]) for qi in rowsA])), 1)]},
            "queries_truncated_at_hop (hop1 / hop2) balanced": [int(sum(1 for s in stat_b[h] if s and s["truncated"])) for h in ("hop1", "hop2")],
            "same_kept_set_as_pinned (queries; hop1 / hop2)": [int(same_set[:, h].sum()) for h in range(DEPTH)],
            "jaccard_kept_vs_pinned_mean (hop1 / hop2)": [round(float(jacc[:, h].mean()), 3) for h in range(DEPTH)],
            "distinct_parents_in_kept_beam_mean (hop1 / hop2)": {a: [round(float(np.mean([x[0] for x in div[a][h]])), 1) for h in ("hop1", "hop2")] for a in ("pinned", "balanced")},
            "largest_single_parent_share_mean (hop1 / hop2)": {a: [round(float(np.mean([x[1] for x in div[a][h]])), 3) for h in ("hop1", "hop2")] for a in ("pinned", "balanced")},
            "balanced_compression_stats_mean": {h: agg_stats(stat_b[h]) for h in ("hop1", "hop2")}}
bd21 = rec21["beam_diversity_DEV_A"]
for k in bd21:
    if k != "wall_clock_ms_per_query":
        assert beam_div[k] == bd21[k], "balanced beam figure %s differs from the section-21 record" % k
G.log("balanced beam reproduced: every deterministic figure of h2balanced_A_%s.json beam_diversity_DEV_A matches" % name)

# ---------------------------------------------------------------- canonical SAFE machinery (frozen code path) on the raw cache arrays, for BOTH block rankings
z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
assert zz["base_rank"].shape == (nq, TOPP) and (zz["base_rank"] == D.base_rank).all()
zq = dict(zz)
zq["base_rank"] = rank_q[:, :TOPP].astype(np.int32)         # THE composition: the QMAX_F0 ranking enters the frozen SAFE machinery as the block ranking
RAW = {"BASE": zz, "QMAX": zq}
Cc = {ch: RT.build_cache(name, RAW[ch], meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]]) for ch in ("BASE", "QMAX")}
ctxs = {ch: KB.contexts(RAW[ch], meta, Cc[ch], B, CFG) for ch in ("BASE", "QMAX")}
goldp = Cc["BASE"]["goldp"]
assert Cc["QMAX"]["goldp"] == goldp
ind_base = np.asarray(Cc["BASE"]["ind_base"], bool)
a0 = np.asarray(D.base_all, bool)
assert (ind_base[m] == a0[m]).all(), "SAFE machinery BASE differs from the lane's served BASE"
assert (np.asarray(Cc["QMAX"]["ind_base"], bool)[m] == allv_q[m]).all(), "SAFE machinery QMAX P50 differs from the lane's QMAX_F0 evaluation"
for qi in rowsA:                                             # the challengers do not depend on the block ranking (seed-based structural expansion; ret_rrf continuation)
    assert ctxs["BASE"][qi]["spos"] == ctxs["QMAX"][qi]["spos"] and ctxs["BASE"][qi]["rpos"] == ctxs["QMAX"][qi]["rpos"]
sizes = np.asarray(C.part_sizes, np.int64)
assert int(sizes.sum()) == N and (np.bincount(hard, minlength=npart) == sizes).all()
order_nodes = np.argsort(hard, kind="stable")
ptr = np.zeros(npart + 1, np.int64)
ptr[1:] = np.cumsum(sizes)


def nodes_of(b):
    return order_nodes[ptr[b]:ptr[b + 1]]


gold_nodes = [np.asarray(C.gold_nodes[qi], np.int64) for qi in range(nq)]
for qi in rowsA:
    assert set(int(hard[g]) for g in gold_nodes[qi]) == goldp[qi]

finals = {arm: [set() for _ in range(nq)] for arm in ("L1", "SAFE", ARM_Q50, ARM_QSAFE)}
weakest_of = {ch: np.full(nq, -1, np.int64) for ch in ("BASE", "QMAX")}
admitted = {ch: [None] * nq for ch in ("BASE", "QMAX")}
for ch in ("BASE", "QMAX"):
    for qi in rowsA:
        c = ctxs[ch][qi]
        finals[P50_OF[ch]][qi] = set(c["base50"])
        Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        safe50 = c["prot_set"] | set(Xs)
        assert len(safe50) == P_MAIN
        finals[SAFE_OF[ch]][qi] = safe50
        weakest_of[ch][qi] = int(Xs[-1])                     # the sixth block of the F6 competition = SAFE's weakest admission
        admitted[ch][qi] = [int(x) for x in Xs]
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path)
ind_L1 = np.array([int(goldp[qi] <= finals["L1"][qi]) for qi in range(nq)], bool)
ind_S = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
assert (ind_L1[m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind_S[m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced (SAFE %.4f, BASE %.4f)" % (ind_S[m].mean(), ind_L1[m].mean()))

# identities of the composed chain that involve no gold (reported in the dry run as well)
same_safe = np.array([finals[ARM_QSAFE][qi] == finals["SAFE"][qi] for qi in rowsA])
inter_safe = np.array([len(finals[ARM_QSAFE][qi] & finals["SAFE"][qi]) for qi in rowsA])
inter_p50 = np.array([len(finals[ARM_Q50][qi] & finals["L1"][qi]) for qi in rowsA])
same_weak = np.array([weakest_of["QMAX"][qi] == weakest_of["BASE"][qi] for qi in rowsA])
adm_over = np.array([len(set(admitted["QMAX"][qi]) & set(admitted["BASE"][qi])) for qi in rowsA])
chal_adm = {ch: np.array([sum(1 for x in admitted[ch][qi] if x not in finals[P50_OF[ch]][qi]) for qi in rowsA]) for ch in ("BASE", "QMAX")}
exposure = {arm: np.array([int(sizes[list(finals[arm][qi])].sum()) for qi in rowsA]) for arm in finals}
ident = {"QMAX_P50 == BASE_P50 blocks per query (mean of 50)": round(float(inter_p50.mean()), 2),
         "QMAX_SAFE == SAFE (queries)": int(same_safe.sum()), "QMAX_SAFE & SAFE blocks per query (mean of 50)": round(float(inter_safe.mean()), 2),
         "same weakest admitted block (queries)": int(same_weak.sum()), "admitted six overlap (mean of 6)": round(float(adm_over.mean()), 2),
         "challengers admitted from outside the chain's own P50 (mean of 6): BASE / QMAX": [round(float(chal_adm["BASE"].mean()), 2), round(float(chal_adm["QMAX"].mean()), 2)],
         "nominal exposure mean nodes: L1 / SAFE / QMAX_F0 / QMAX_SAFE": [round(float(exposure[a].mean()), 1) for a in ("L1", "SAFE", ARM_Q50, ARM_QSAFE)]}
G.log("composed-chain identities (no gold): %s" % json.dumps(ident))
if DRY:
    G.log("DRY RUN: BASE, QMAX_F0, SAFE and both beams reproduced; |QMAX_SAFE| = 50 on every DEV_A row; no coverage number of any composed arm computed; no record (%.0fs)" % (time.time() - T_ALL))
    sys.exit(0)


# ---------------------------------------------------------------- the patch rule of section 16 (verbatim logic), applied to a given beam on a given chain's SAFE
def build_patches(b1s, b2s, ch):
    patch = [None] * nq
    for qi in rowsA:
        safe50 = finals[SAFE_OF[ch]][qi]
        weakest = int(weakest_of[ch][qi])
        kept = safe50 - {weakest}
        nb = int(sizes[weakest])
        b1, b2 = b1s[qi], b2s[qi]
        nov1 = [int(v) for v in b1 if int(hard[v]) not in kept]
        nov2 = [int(v) for v in b2 if int(hard[v]) not in kept]
        novel = nov1 + nov2                                  # first-visit order of the beam (the pre-registered order)
        assert len(set(novel)) == len(novel)
        nodes = novel[:nb]
        n_novel_used = len(nodes)
        n_fill = 0
        if len(nodes) < nb:
            rpos_node = {int(v): r for r, v in enumerate(C.ret_rrf[qi]) if v >= 0}
            used = set(nodes)
            fill = sorted((int(v) for v in nodes_of(weakest) if int(v) not in used), key=lambda v: (rpos_node.get(v, 10 ** 9), v))
            n_fill = nb - len(nodes)
            nodes = nodes + fill[:n_fill]
        assert len(nodes) == nb and len(set(nodes)) == nb
        s1 = set(nov1)
        patch[qi] = {"kept": kept, "weakest": weakest, "nb": nb, "nodes": set(nodes), "k": len(novel), "nov1": len(nov1), "nov2": len(nov2),
                     "truncated": int(len(novel) > nb), "n_novel_used": n_novel_used, "n_fill": n_fill,
                     "used_hop1": sum(1 for v in nodes[:n_novel_used] if v in s1), "used_hop2": sum(1 for v in nodes[:n_novel_used] if v not in s1),
                     "novel_set": set(novel), "novel_pos": {v: r for r, v in enumerate(novel)}}
        assert int(sizes[list(kept)].sum()) + nb == int(sizes[list(safe50)].sum())
    return patch


patches = {ARM_H2: build_patches(beams1, beams2, "BASE"), ARM_BAL: build_patches(bal1, bal2, "BASE"), PRIMARY: build_patches(bal1, bal2, "QMAX")}


def served_by(arm, qi, g):
    b = int(hard[g])
    if arm in finals:
        return b in finals[arm][qi]
    p = patches[arm][qi]
    return b in p["kept"] or g in p["nodes"]


ind = {arm: np.zeros(nq, bool) for arm in ARMS}
ind_any = {arm: np.zeros(nq, bool) for arm in ARMS}
for arm in ARMS:
    for qi in rowsA:
        flags = [served_by(arm, qi, int(g)) for g in gold_nodes[qi]]
        ind[arm][qi] = bool(flags) and all(flags)
        ind_any[arm][qi] = any(flags)
assert (ind["L1"][m] == ind_base[m]).all() and (ind["SAFE"][m] == ind_S[m]).all()
assert (ind[ARM_Q50][m] == allv_q[m]).all()
for arm in patches:
    ch = CHAIN_OF[arm]
    for qi in rowsA:
        assert int(sizes[list(patches[arm][qi]["kept"])].sum()) + patches[arm][qi]["nb"] == exposure[SAFE_OF[ch]][np.searchsorted(rowsA, qi)]


def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


def mcl(a, b, s):
    o = mc(a, b, s)
    o["label"] = label(o)
    return o


# ---------------------------------------------------------------- gold accounting per patch arm (vs its own chain's SAFE)
def gold_accounting(arm):
    ch = CHAIN_OF[arm]
    gain_nodes = np.zeros(nq, np.int64)
    gain_hop1 = np.zeros(nq, np.int64)
    loss_nodes = np.zeros(nq, np.int64)
    for qi in rowsA:
        p = patches[arm][qi]
        for g in gold_nodes[qi]:
            g = int(g)
            sS, sP = served_by(SAFE_OF[ch], qi, g), served_by(arm, qi, g)
            if sP and not sS:
                gain_nodes[qi] += 1
                gain_hop1[qi] += int(p["novel_pos"].get(g, 10 ** 9) < p["nov1"])
            elif sS and not sP:
                loss_nodes[qi] += 1
                assert int(hard[g]) == p["weakest"]
    return gain_nodes, gain_hop1, loss_nodes


acct = {arm: gold_accounting(arm) for arm in patches}


def evaluate(arm):
    iv, ia = ind[arm], ind_any[arm]
    ch = CHAIN_OF[arm]
    o = {"arm": arm, "chain": ch, "ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4), "per_hop": {}}
    if arm in finals:
        o["exposure"] = "50 blocks; nominal node exposure = sum of the selected block sizes"
        o["exposure_nodes_mean"] = round(float(exposure[arm].mean()), 1)
    if arm in patches:
        patch = patches[arm]
        gain_nodes, gain_hop1, loss_nodes = acct[arm]
        o["exposure"] = "matched: |served nodes| equal to %s per query (asserted)" % SAFE_OF[ch]
        o["exposure_nodes_mean"] = round(float(exposure[SAFE_OF[ch]].mean()), 1)
        o["novel_nodes_per_query_mean"] = round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1)
        o["novel_hop1_per_query_mean"] = round(float(np.mean([patch[qi]["nov1"] for qi in rowsA])), 1)
        o["novel_hop2_per_query_mean"] = round(float(np.mean([patch[qi]["nov2"] for qi in rowsA])), 1)
        o["patch_size_mean"] = round(float(np.mean([patch[qi]["nb"] for qi in rowsA])), 1)
        o["patch_composition_mean (hop-1 novel / hop-2 novel / fill from removed block)"] = [round(float(np.mean([patch[qi][k] for qi in rowsA])), 1) for k in ("used_hop1", "used_hop2", "n_fill")]
        o["queries_truncated"] = int(sum(patch[qi]["truncated"] for qi in rowsA))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(patch[qi]["k"] == 0 for qi in rowsA))
        o["gold_nodes_gained_vs_%s" % SAFE_OF[ch]] = int(gain_nodes[rowsA].sum())
        o["gold_nodes_gained_vs_%s_from_hop1_novel" % SAFE_OF[ch]] = int(gain_hop1[rowsA].sum())
        o["queries_with_gold_node_gained_vs_%s" % SAFE_OF[ch]] = int((gain_nodes[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss_nodes[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss_nodes[rowsA] > 0).sum())
    comps = []                                               # (key, reference arm) paired comparisons reported for this arm
    if arm != "L1":
        comps.append(("vs_L1", "L1"))
    if arm in patches:
        comps.append(("vs_%s" % SAFE_OF[ch], SAFE_OF[ch]))
    if arm == ARM_BAL:
        comps.append(("vs_H2_PATCH1", ARM_H2))
    if ch == "QMAX" and arm != ARM_Q50:
        comps.append(("vs_QMAX_F0", ARM_Q50))
    if arm == ARM_QSAFE:
        comps.append(("vs_SAFE", "SAFE"))
    if arm == PRIMARY:
        comps.append(("vs_BALANCED_H2_PATCH1", ARM_BAL))
        comps.append(("vs_H2_PATCH1", ARM_H2))
        comps.append(("vs_SAFE", "SAFE"))
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        e = {"n": int(s.sum()), "ALL": round(float(iv[s].mean()), 4), "ANY": round(float(ia[s].mean()), 4)}
        for key, ref in comps:
            e[key] = mc(ind[ref], iv, s)
        if arm in patches:
            e["gold_nodes_gained"] = int(acct[arm][0][s].sum())
            e["gold_nodes_lost_with_removed_block"] = int(acct[arm][2][s].sum())
        o["per_hop"]["hop%d" % h] = e
    for key, ref in comps:
        o["ALL_" + key] = mcl(ind[ref], iv, m)
        o["ANY_" + key] = mc(ind_any[ref], ia, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_" + key] = {"n": int(s.sum()), **mc(ind[ref], iv, s), "pooled_ref": round(float(ind[ref][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm in patches:
        o["significant_loss_vs_%s" % SAFE_OF[ch]] = bool(o["ALL_vs_%s" % SAFE_OF[ch]]["lost"] > o["ALL_vs_%s" % SAFE_OF[ch]]["gained"] and o["ALL_vs_%s" % SAFE_OF[ch]]["p"] < 0.05)
    if arm == ARM_BAL:
        o["significant_loss_vs_H2_PATCH1"] = bool(o["ALL_vs_H2_PATCH1"]["lost"] > o["ALL_vs_H2_PATCH1"]["gained"] and o["ALL_vs_H2_PATCH1"]["p"] < 0.05)
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "PREREGISTERED_COMPOSITION_TEST_DEV_A", "arm": PRIMARY,
       "preregistration": {"path": os.path.relpath(PRE, G.X.REPO).replace("\\", "/"), "sha256": sha_file(PRE)},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__))},
       "pinned": PINNED, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name], "L1_GEOM_record": {"path": os.path.relpath(PRE_GEOM, G.X.REPO).replace("\\", "/"), "recorded_DEV_A_AGGREGATION": seen_q},
       "definition": {"ranking_QMAX_F0": "_l1g_core.F0 (the frozen RRF, K0 = 60, dense tie-break) of [Cd, Cs, Cmax]: Cd / Cs = _l1g_core.block_channel of the served dense / SPLADE "
                                         "top-100 ids through the legacy own + directed-out-neighbour table; Cmax = _l1g_candidate.dense_blockmax_rank (score(B) = max_{v in B} q . e_v over "
                                         "all actual nodes, stable descending order); asserted to reproduce the L1_GEOM recorded DEV_A numbers; F0([Cd, Cs]) asserted == the served base_rank",
                      "composition": "the first TOPP = 200 blocks of the QMAX_F0 ranking replace base_rank in the replay-cache view read by _l1ps_router.build_cache / _l1kb_core.contexts "
                                     "(protected = first 44, boundary = 45-50, cpos = QMAX positions); challengers (structural s_node, retrieval ret_rrf), F6 admission, weakest = the "
                                     "sixth admitted, beam (seeds = frozen top-5 RRF nodes), balanced compression and patch unchanged",
                      "beam": "pinned uL3 beam (frozen admissibility, transition geometry, RRF fused order, BEAM = 100, DEPTH = 2, seeds = frozen top-5); identical for both chains",
                      "compression_balanced": "section 21 verbatim: within-parent rank = position of (parent, target) among the parent's pairs in the frozen fused order (first occurrence per pair); "
                                              "sequence sorted by (rank, frontier position of the parent) = A1 B1 C1 ... A2 B2 C2 ...; first 100 distinct unvisited targets; no other constant",
                      "patch": "section 16 verbatim: the chain's SAFE minus its weakest admitted block + one query-local block of the same size = the beam's visited nodes not exposed by the "
                               "49 kept blocks in first-visit order (hop 1 then hop 2, each in beam order), topped up from the removed block (ret_rrf position, then node id)",
                      "metric": "ALL-gold on DEV_A; patch arms at matched unique-node exposure to their chain's SAFE; node-level == block-level for the four block arms (asserted)"},
       "beam_diversity_DEV_A": beam_div, "composed_chain_identities": ident, "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-24s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm", "chain")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- the reference chain must reproduce the section-16 and section-21 records exactly
o16, r16 = res["arms"][ARM_H2], rec16["arms"][ARM_H2]
assert o16["ALL"] == r16["ALL"] and o16["ANY"] == r16["ANY"], (o16["ALL"], r16["ALL"])
assert (o16["ALL_vs_SAFE"]["gained"], o16["ALL_vs_SAFE"]["lost"]) == (r16["ALL_vs_SAFE"]["gained"], r16["ALL_vs_SAFE"]["lost"])
for h in o16["per_hop"]:
    assert o16["per_hop"][h]["ALL"] == r16["per_hop"][h]["ALL"], h
    assert (o16["per_hop"][h]["vs_SAFE"]["gained"], o16["per_hop"][h]["vs_SAFE"]["lost"]) == (r16["per_hop"][h]["vs_SAFE"]["gained"], r16["per_hop"][h]["vs_SAFE"]["lost"]), h
assert o16["novel_nodes_per_query_mean"] == r16["novel_nodes_per_query_mean"] and o16["queries_truncated"] == r16["queries_truncated"]
o21, r21 = res["arms"][ARM_BAL], rec21["arms"][ARM_BAL]
assert o21["ALL"] == r21["ALL"] and o21["ANY"] == r21["ANY"], (o21["ALL"], r21["ALL"])
for key in ("ALL_vs_SAFE", "ALL_vs_H2_PATCH1", "ALL_vs_L1"):
    assert (o21[key]["gained"], o21[key]["lost"]) == (r21[key]["gained"], r21[key]["lost"]), key
for h in o21["per_hop"]:
    assert o21["per_hop"][h]["ALL"] == r21["per_hop"][h]["ALL"], h
    for key in ("vs_SAFE", "vs_H2_PATCH1"):
        assert (o21["per_hop"][h][key]["gained"], o21["per_hop"][h][key]["lost"]) == (r21["per_hop"][h][key]["gained"], r21["per_hop"][h][key]["lost"]), (h, key)
for key in ("novel_nodes_per_query_mean", "queries_truncated", "gold_nodes_gained_vs_SAFE", "gold_nodes_lost_with_removed_block", "patch_size_mean"):
    assert o21[key] == r21[key], key
assert res["arms"]["SAFE"]["ALL"] == rec21["arms"]["SAFE"]["ALL"] and res["arms"]["L1"]["ALL"] == rec21["arms"]["L1"]["ALL"]
res["reference_chain_reproduced"] = "L1 / SAFE / H2_PATCH1 == h2patch_A_%s.json and BALANCED_H2_PATCH1 == h2balanced_A_%s.json (ALL, ANY, vs SAFE / H2_PATCH1 / L1, per hop, novel counts, patch size)" % (name, name)
G.log("reference chain reproduced: %s" % res["reference_chain_reproduced"])

# ---------------------------------------------------------------- stage increments of the two chains (does each stage's gain survive under the other ranking?)
stages = {}
for ch, seq in (("BASE", ("L1", "SAFE", ARM_BAL)), ("QMAX", (ARM_Q50, ARM_QSAFE, PRIMARY))):
    st = {}
    for a_, b_ in zip(seq[:-1], seq[1:]):
        st["%s -> %s" % (a_, b_)] = {**mcl(ind[a_], ind[b_], m), "from": res["arms"][a_]["ALL"], "to": res["arms"][b_]["ALL"]}
    st["%s -> %s (chain)" % (seq[0], seq[-1])] = {**mcl(ind[seq[0]], ind[seq[-1]], m), "from": res["arms"][seq[0]]["ALL"], "to": res["arms"][seq[-1]]["ALL"]}
    stages[ch] = st
res["stage_increments_DEV_A"] = stages
G.log("stage increments %s" % json.dumps(stages))

# ---------------------------------------------------------------- flips PRIMARY vs BALANCED_H2_PATCH1 (and vs QMAX_F0) with their cause
def flip_table(ref):
    f = {"newly_covered": {"n": 0, "gold_nodes_newly_served": 0, "of_which_served_by_a_kept_block_of_QMAX_SAFE": 0, "of_which_served_by_the_patch_nodes": 0},
         "newly_lost": {"n": 0, "gold_nodes_dropped": 0, "of_which_block_no_longer_kept_(QMAX_SAFE_kept_49_differ)": 0, "of_which_dropped_with_the_weakest_block": 0,
                        "of_which_visited_but_beyond_the_patch_capacity": 0, "of_which_not_visited_by_the_balanced_beam": 0}}
    for qi in rowsA:
        pP = patches[PRIMARY][qi]
        if ind[PRIMARY][qi] and not ind[ref][qi]:
            e = f["newly_covered"]
            e["n"] += 1
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(PRIMARY, qi, g) and not served_by(ref, qi, g):
                    e["gold_nodes_newly_served"] += 1
                    e["of_which_served_by_a_kept_block_of_QMAX_SAFE"] += int(int(hard[g]) in pP["kept"])
                    e["of_which_served_by_the_patch_nodes"] += int(int(hard[g]) not in pP["kept"] and g in pP["nodes"])
        elif ind[ref][qi] and not ind[PRIMARY][qi]:
            e = f["newly_lost"]
            e["n"] += 1
            vis = pP["novel_set"] | set(bal1[qi]) | set(bal2[qi])
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(ref, qi, g) and not served_by(PRIMARY, qi, g):
                    e["gold_nodes_dropped"] += 1
                    b = int(hard[g])
                    if b == pP["weakest"]:
                        e["of_which_dropped_with_the_weakest_block"] += 1
                    elif b not in finals[ARM_QSAFE][qi]:
                        e["of_which_block_no_longer_kept_(QMAX_SAFE_kept_49_differ)"] += 1
                    if g in vis and pP["novel_pos"].get(g, -1) >= pP["nb"]:
                        e["of_which_visited_but_beyond_the_patch_capacity"] += 1
                    if g not in vis:
                        e["of_which_not_visited_by_the_balanced_beam"] += 1
    return f


res["attribution_flips"] = {"PRIMARY_vs_BALANCED_H2_PATCH1": flip_table(ARM_BAL), "PRIMARY_vs_QMAX_F0": flip_table(ARM_Q50)}
G.log("flips %s" % json.dumps(res["attribution_flips"]))

# ---------------------------------------------------------------- the pre-registered per-cache outcome (the summary applies the cross-cache rule)
vB = res["arms"][PRIMARY]["ALL_vs_BALANCED_H2_PATCH1"]["label"]
vQ = res["arms"][PRIMARY]["ALL_vs_QMAX_F0"]["label"]
if "LOSS" in (vB, vQ):
    cell = "CONFLICT"
elif vB == "GAIN" and vQ == "GAIN":
    cell = "STACKS"
elif "GAIN" in (vB, vQ):
    cell = "CARRIES"
else:
    cell = "NEUTRAL"
res["per_cache_outcome"] = {"PRIMARY_vs_BALANCED_H2_PATCH1": vB, "PRIMARY_vs_QMAX_F0": vQ, "cell": cell,
                            "cell_rule": "CONFLICT if LOSS vs either parent; STACKS if GAIN vs both; CARRIES if GAIN vs one and NEUTRAL vs the other; NEUTRAL otherwise (alpha 0.01, exact McNemar)"}
G.log("per-cache outcome %s" % json.dumps(res["per_cache_outcome"]))
res["seconds"] = round(time.time() - T_ALL, 1)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
