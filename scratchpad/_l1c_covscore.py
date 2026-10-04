"""COVSCORE_H2_PATCH1 (fourteenth ruling, item 2; PREREGISTRATION_COVSCORE_H2_PATCH1.json written first): the frozen H2_PATCH1 of section 16
with ONE change -- the beam's frontier compression -- exactly as section 21 did, with the ruling's coverage-then-score rule in place of the
section-21 round-robin:
    (1) (parent, target) pairs in the frozen fused transition order, a pair reached twice keeping its FIRST occurrence;
    (2) within-parent rank r(m, u) = position of the pair among parent m's pairs in that order (0-based; section 21's definition);
    (3) phase 1 = the pairs with r = 0 in frontier order (A1 B1 C1 D1 ...: the best child of every active parent);
        phase 2 = ALL remaining pairs in the frozen global fused order (A2 A3 B2 A4 ...: the original frozen global candidate ranking);
    (4) B(h) = the first BEAM distinct unvisited targets of phase 1 followed by phase 2, a target already taken being skipped; visited marks exactly
        as the pinned beam.  B = {best child of each parent} u TopK(remaining candidates) until capacity.  No k_local, no lambda, no weight, no
        dataset branch: the only constants are the frozen BEAM / DEPTH / K0 and the frozen patch budget.
Everything else is byte-identical in logic to sections 16 / 21 (functions copied verbatim from the pinned _l1c_h2balanced.py): the pinned beam and
the balanced beam are recomputed and asserted against the section-16 / 21 records; arms L1, SAFE, H2_PATCH1, BALANCED_H2_PATCH1 (the incumbent
successor), COVSCORE_H2_PATCH1 (the new arm).  Metric: ALL-gold at matched exposure; primary comparison COVSCORE vs BALANCED_H2_PATCH1.
    python -u scratchpad/_l1c_covscore.py <cache>          -> results/L1_COVPART/covscore_A_<cache>.json
    python -u scratchpad/_l1c_covscore.py metaqa --dry     -> no record; identities + gold-free beam statistics only
    python -u scratchpad/_l1c_covscore.py --unit           -> the synthetic unit test of the compression (no cache, no graph, no gold)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def cs_compress_core(N, v_idx, u_idx, order, frontier, visited, budget):
    """coverage-then-score compression (pure function of the frozen fused order; N = node count for the pair key).
    returns (kept targets in sequence order, parent of each kept target, stats)."""
    pos = np.full(N, -1, np.int64)
    pos[np.asarray(frontier, np.int64)] = np.arange(len(frontier))
    v_o, u_o = v_idx[order], u_idx[order]
    pp = pos[v_o]
    assert (pp >= 0).all()
    key = pp.astype(np.int64) * N + u_o
    _, first = np.unique(key, return_index=True)             # (parent, target) pairs: keep the first occurrence in fused order
    first.sort()
    pp, uu = pp[first], u_o[first]                           # pairs in the frozen fused order (index j ascending = better)
    o = np.argsort(pp, kind="stable")                        # within-parent rank = position among the parent's pairs in fused order
    ps = pp[o]
    start = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(ps)]))
    r = np.empty(len(pp), np.int64)
    r[o] = np.arange(len(ps)) - start[grp]
    best = np.flatnonzero(r == 0)                            # the best child of every active parent
    phase1 = best[np.argsort(pp[best], kind="stable")]       # A1 B1 C1 D1 ... in frontier order
    phase2 = np.flatnonzero(r > 0)                           # A2 A3 B2 A4 ... in the frozen global fused order
    is1 = np.zeros(len(pp), bool)
    is1[phase1] = True
    nb, par, n1 = [], {}, 0
    for j in np.concatenate([phase1, phase2]):
        u = int(uu[j])
        if not visited[u]:
            visited[u] = True
            nb.append(u)
            par[u] = int(v_o[first[j]])
            n1 += int(is1[j])
            if len(nb) >= budget:
                break
    st = {"parents": int(len(frontier)), "parents_with_candidates": int(len(start)), "candidates_unique": int(len(np.unique(uu))), "pairs": int(len(uu)),
          "kept": len(nb), "truncated": int(len(np.unique(uu)) > budget), "kept_phase1_coverage": n1, "kept_phase2_score": len(nb) - n1,
          "parents_represented": len(set(par.values())), "largest_parent_candidate_list": int(np.bincount(pp).max()) if len(pp) else 0}
    return nb, par, st


def unit_test():
    """synthetic check of the compression against the hand-derived sequences (no cache, no graph, no gold)."""
    N = 100
    A, B_, C_ = 1, 2, 3
    a1, a2, a3, b1, b2, c1, c2 = 11, 12, 13, 21, 22, 31, 32
    # example 1 (section 21's): pairs in fused order (A,a1) (B,b1=a1) (A,a2) (C,c1) (B,b2) (A,a3) (A,a2 dup)
    v = np.array([A, B_, A, C_, B_, A, A]); u = np.array([a1, a1, a2, c1, b2, a3, a2]); order = np.arange(7)
    nb, par, st = cs_compress_core(N, v, u, order, [A, B_, C_], np.zeros(N, bool), 4)
    assert nb == [a1, c1, a2, b2] and st["kept_phase1_coverage"] == 2 and par[c1] == C_ and par[a1] == A, (nb, st)
    nb, par, st = cs_compress_core(N, v, u, order, [A, B_, C_], np.zeros(N, bool), 100)
    assert nb == [a1, c1, a2, b2, a3] and st["kept"] == 5, nb
    # example 2 (distinguishes the rule from the round-robin): (A,a1) (A,a2) (A,a3) (B,b1) (C,c1) (B,b2) (C,c2), budget 5
    v = np.array([A, A, A, B_, C_, B_, C_]); u = np.array([a1, a2, a3, b1, c1, b2, c2]); order = np.arange(7)
    nb, par, st = cs_compress_core(N, v, u, order, [A, B_, C_], np.zeros(N, bool), 5)
    assert nb == [a1, b1, c1, a2, a3] and st["kept_phase1_coverage"] == 3 and st["kept_phase2_score"] == 2, nb   # round-robin would give [a1, b1, c1, a2, b2]
    # example 3: frontier order [B, A, C], c1 already visited, budget 3 -> phase 1 [b1, a1] (C's best child visited), phase 2 in fused order (A,a2)
    vis = np.zeros(N, bool); vis[c1] = True
    nb, par, st = cs_compress_core(N, v, u, order, [B_, A, C_], vis, 3)
    assert nb == [b1, a1, a2] and st["kept_phase1_coverage"] == 2, nb
    # example 4: a parent whose best child was taken by an earlier parent gets nothing in phase 1; its later children compete by global order
    v = np.array([A, B_, C_, B_, A]); u = np.array([a1, a1, c1, b2, a2]); order = np.arange(5)
    nb, par, st = cs_compress_core(N, v, u, order, [A, B_, C_], np.zeros(N, bool), 3)
    assert nb == [a1, c1, b2] and par[b2] == B_ and st["kept_phase1_coverage"] == 2, nb
    # example 5: the fused order given as a permutation (order != identity)
    v = np.array([B_, A, A, C_]); u = np.array([b2, a1, a2, c1]); order = np.array([1, 3, 0, 2])     # fused order: (A,a1) (C,c1) (B,b2) (A,a2)
    nb, par, st = cs_compress_core(N, v, u, order, [A, B_, C_], np.zeros(N, bool), 100)
    assert nb == [a1, b2, c1, a2], nb
    print("unit test OK: 5 synthetic examples")


if "--unit" in sys.argv:
    unit_test()
    sys.exit(0)

import _l1g_core as G  # noqa: E402
import _l1kb_core as KB  # noqa: E402
import _l1ps_router as RT  # noqa: E402

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_COVSCORE_H2_PATCH1.json")
NEW_MODULES = ("_l1c_covscore.py", "_l1c_covscore_summary.py")
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d",
          "_l1c_h2balanced.py": "7aed6b1d4399a1b432d0e6be48013cf73c92d4e47d2e5d2dd7b669b508d84bbe"}
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
ARM_H2, ARM_BAL, ARM = "H2_PATCH1", "BALANCED_H2_PATCH1", "COVSCORE_H2_PATCH1"
ARMS = ("L1", "SAFE", ARM_H2, ARM_BAL, ARM)
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
unit_test()
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    for mod in NEW_MODULES:
        assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    fp_out = os.path.join(OUT, "covscore_A_%s.json" % name)
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
exec(head)                                                   # D, C, beams1, beams2 (the pinned beam on DEV_A), expand_real, fused_order, seeds, hub, admissible ...
secs_pinned = secs.copy()
edges_pinned = edges_h.copy()


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


# ---------------------------------------------------------------- the balanced compression  [section 21 verbatim; the incumbent successor]
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


def cs_compress(v_idx, u_idx, order, frontier, visited, budget):
    return cs_compress_core(N, v_idx, u_idx, order, frontier, visited, budget)


def beam_with(compress, qi):
    """the pinned beam with the frontier compression replaced by `compress` (section 21's beam_balanced, parametrised)."""
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
        nb, par, st = compress(v_idx, u_idx, order, Bf, visited, BEAM)
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
cov1, cov2 = [[] for _ in range(nq)], [[] for _ in range(nq)]
CAND_P, CAND_B, CAND_C = {}, {}, {}
PAR_P, PAR_B, PAR_C = {}, {}, {}
div = {k: {"hop1": [], "hop2": []} for k in ("pinned", "balanced", "covscore")}
stat_b = {"hop1": [], "hop2": []}
stat_c = {"hop1": [], "hop2": []}
secs_bal = np.zeros(nA, np.float64)
secs_cov = np.zeros(nA, np.float64)
edges_bal = np.zeros((nA, DEPTH), np.int64)
edges_cov = np.zeros((nA, DEPTH), np.int64)
same_set = np.zeros((nA, DEPTH), bool)
jacc = np.zeros((nA, DEPTH), np.float64)
same_set_c = {"pinned": np.zeros((nA, DEPTH), bool), "balanced": np.zeros((nA, DEPTH), bool)}
jacc_c = {"pinned": np.zeros((nA, DEPTH), np.float64), "balanced": np.zeros((nA, DEPTH), np.float64)}
n_pinned_cands = np.zeros((nA, DEPTH), np.int64)
t0b = time.time()
for j, qi in enumerate(rowsA):
    bp, cp, pp_ = beam_pinned_with_candidates(qi)
    assert bp[0] == beams1[qi] and bp[1] == beams2[qi], "pinned beam with candidate recording differs from the exec'd pinned beam"
    bb, cb, pb, st, ne, dt = beam_with(rr_compress, qi)
    bal1[qi], bal2[qi] = bb[0], bb[1]
    secs_bal[j] = dt
    edges_bal[j] = ne
    bc, cc, pc, stc, nec, dtc = beam_with(cs_compress, qi)
    cov1[qi], cov2[qi] = bc[0], bc[1]
    secs_cov[j] = dtc
    edges_cov[j] = nec
    for h in range(DEPTH):
        CAND_P[(qi, h)], CAND_B[(qi, h)], CAND_C[(qi, h)] = cp[h], cb[h], cc[h]
        PAR_P[(qi, h)], PAR_B[(qi, h)], PAR_C[(qi, h)] = pp_[h], pb[h], pc[h]
        n_pinned_cands[j, h] = len(cp[h])
        a, b_, c_ = set(bp[h]), set(bb[h]), set(bc[h])
        same_set[j, h] = a == b_
        jacc[j, h] = len(a & b_) / float(len(a | b_)) if (a | b_) else 1.0
        same_set_c["pinned"][j, h] = c_ == a
        same_set_c["balanced"][j, h] = c_ == b_
        jacc_c["pinned"][j, h] = len(a & c_) / float(len(a | c_)) if (a | c_) else 1.0
        jacc_c["balanced"][j, h] = len(b_ & c_) / float(len(b_ | c_)) if (b_ | c_) else 1.0
        hk = "hop%d" % (h + 1)
        div["pinned"][hk].append(diversity(bp[h], pp_[h]))
        div["balanced"][hk].append(diversity(bb[h], pb[h]))
        div["covscore"][hk].append(diversity(bc[h], pc[h]))
        stat_b[hk].append(st[h])
        stat_c[hk].append(stc[h])
    if j % 200 == 0:
        G.log("  beams %d / %d (%.0fs) sizes bal %d/%d cov %d/%d | cov vs pinned jaccard %s | cov vs balanced jaccard %s" % (
            j, nA, time.time() - t0b, len(bb[0]), len(bb[1]), len(bc[0]), len(bc[1]), np.round(jacc_c["pinned"][j], 3).tolist(), np.round(jacc_c["balanced"][j], 3).tolist()))
G.log("beams done (%.0fs): wall-clock / query mean balanced %.1f ms, covscore %.1f ms (pinned %.1f ms)" % (time.time() - t0b, 1000 * secs_bal.mean(), 1000 * secs_cov.mean(), 1000 * secs_pinned.mean()))


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
cov_div = {"wall_clock_ms_per_query": round(1000 * float(secs_cov.mean()), 1),
           "transitions_scored_per_query_mean (hop1 / hop2)": [round(float(x), 1) for x in edges_cov.mean(0)],
           "candidates_unique_per_query_mean (hop1 / hop2)": [round(float(np.mean([len(CAND_C[(qi, h)]) for qi in rowsA])), 1) for h in range(DEPTH)],
           "beam_size_mean (hop1 / hop2)": [round(float(np.mean([len(cov1[qi]) for qi in rowsA])), 1), round(float(np.mean([len(cov2[qi]) for qi in rowsA])), 1)],
           "queries_truncated_at_hop (hop1 / hop2)": [int(sum(1 for s in stat_c[h] if s and s["truncated"])) for h in ("hop1", "hop2")],
           "same_kept_set_as_pinned (queries; hop1 / hop2)": [int(same_set_c["pinned"][:, h].sum()) for h in range(DEPTH)],
           "same_kept_set_as_balanced (queries; hop1 / hop2)": [int(same_set_c["balanced"][:, h].sum()) for h in range(DEPTH)],
           "jaccard_kept_vs_pinned_mean (hop1 / hop2)": [round(float(jacc_c["pinned"][:, h].mean()), 3) for h in range(DEPTH)],
           "jaccard_kept_vs_balanced_mean (hop1 / hop2)": [round(float(jacc_c["balanced"][:, h].mean()), 3) for h in range(DEPTH)],
           "distinct_parents_in_kept_beam_mean (hop1 / hop2)": [round(float(np.mean([x[0] for x in div["covscore"][h]])), 1) for h in ("hop1", "hop2")],
           "largest_single_parent_share_mean (hop1 / hop2)": [round(float(np.mean([x[1] for x in div["covscore"][h]])), 3) for h in ("hop1", "hop2")],
           "covscore_compression_stats_mean": {h: agg_stats(stat_c[h]) for h in ("hop1", "hop2")}}
G.log("covscore beam (gold-free) %s" % json.dumps(cov_div))

# ---------------------------------------------------------------- canonical SAFE machinery (frozen code path), on the raw cache arrays  [section 16, unchanged]
z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
goldp = Cc["goldp"]
ind_base = np.asarray(Cc["ind_base"], bool)
a0 = np.asarray(D.base_all, bool)
assert (ind_base[m] == a0[m]).all(), "SAFE machinery BASE differs from the lane's served BASE"
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

# ---------------------------------------------------------------- the patch rule of section 16 (verbatim logic), applied to a given beam
finals = {"L1": [set() for _ in range(nq)], "SAFE": [set() for _ in range(nq)]}
weakest_of = np.full(nq, -1, np.int64)
for qi in rowsA:
    c = ctxs[qi]
    finals["L1"][qi] = set(c["base50"])
    Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    safe50 = c["prot_set"] | set(Xs)
    assert len(safe50) == P_MAIN
    finals["SAFE"][qi] = safe50
    weakest_of[qi] = int(Xs[-1])                             # the sixth block of the F6 competition = SAFE's weakest admission
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path)
ind_L1 = np.array([int(goldp[qi] <= finals["L1"][qi]) for qi in range(nq)], bool)
ind_S = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
assert (ind_L1[m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind_S[m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced (SAFE %.4f, BASE %.4f)" % (ind_S[m].mean(), ind_L1[m].mean()))
if DRY:
    G.log("DRY RUN: BASE, SAFE and the pinned / balanced beams reproduced; the covscore beam built (gold-free statistics above); no coverage number of the new arm; no record (%.0fs)" % (time.time() - T_ALL))
    sys.exit(0)


def build_patches(b1s, b2s):
    patch = [None] * nq
    for qi in rowsA:
        safe50 = finals["SAFE"][qi]
        weakest = int(weakest_of[qi])
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


patches = {ARM_H2: build_patches(beams1, beams2), ARM_BAL: build_patches(bal1, bal2), ARM: build_patches(cov1, cov2)}


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


def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


def mcl(a, b, s):
    o = mc(a, b, s)
    o["label"] = label(o)
    return o


# ---------------------------------------------------------------- gold accounting per patch arm
def gold_accounting(arm):
    gain_nodes = np.zeros(nq, np.int64)
    gain_hop1 = np.zeros(nq, np.int64)
    loss_nodes = np.zeros(nq, np.int64)
    for qi in rowsA:
        p = patches[arm][qi]
        for g in gold_nodes[qi]:
            g = int(g)
            sS, sP = served_by("SAFE", qi, g), served_by(arm, qi, g)
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
    o = {"arm": arm, "ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4), "per_hop": {}}
    if arm in patches:
        patch = patches[arm]
        gain_nodes, gain_hop1, loss_nodes = acct[arm]
        o["exposure"] = "matched: |served nodes| equal to canonical SAFE per query (asserted)"
        o["novel_nodes_per_query_mean"] = round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1)
        o["novel_hop1_per_query_mean"] = round(float(np.mean([patch[qi]["nov1"] for qi in rowsA])), 1)
        o["novel_hop2_per_query_mean"] = round(float(np.mean([patch[qi]["nov2"] for qi in rowsA])), 1)
        o["patch_size_mean"] = round(float(np.mean([patch[qi]["nb"] for qi in rowsA])), 1)
        o["patch_composition_mean (hop-1 novel / hop-2 novel / fill from removed block)"] = [round(float(np.mean([patch[qi][k] for qi in rowsA])), 1) for k in ("used_hop1", "used_hop2", "n_fill")]
        o["queries_truncated"] = int(sum(patch[qi]["truncated"] for qi in rowsA))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(patch[qi]["k"] == 0 for qi in rowsA))
        o["gold_nodes_gained_vs_SAFE"] = int(gain_nodes[rowsA].sum())
        o["gold_nodes_gained_vs_SAFE_from_hop1_novel"] = int(gain_hop1[rowsA].sum())
        o["queries_with_gold_node_gained_vs_SAFE"] = int((gain_nodes[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss_nodes[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss_nodes[rowsA] > 0).sum())
    comps = []
    if arm != "L1":
        comps.append(("vs_L1", "L1"))
    if arm in patches:
        comps.append(("vs_SAFE", "SAFE"))
    if arm in (ARM_BAL, ARM):
        comps.append(("vs_H2_PATCH1", ARM_H2))
    if arm == ARM:
        comps.append(("vs_BALANCED_H2_PATCH1", ARM_BAL))
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
        o["significant_loss_vs_SAFE"] = bool(o["ALL_vs_SAFE"]["lost"] > o["ALL_vs_SAFE"]["gained"] and o["ALL_vs_SAFE"]["p"] < 0.05)
    if arm in (ARM_BAL, ARM):
        o["significant_loss_vs_H2_PATCH1"] = bool(o["ALL_vs_H2_PATCH1"]["lost"] > o["ALL_vs_H2_PATCH1"]["gained"] and o["ALL_vs_H2_PATCH1"]["p"] < 0.05)
    if arm == ARM:
        o["significant_loss_vs_BALANCED_H2_PATCH1"] = bool(o["ALL_vs_BALANCED_H2_PATCH1"]["lost"] > o["ALL_vs_BALANCED_H2_PATCH1"]["gained"] and o["ALL_vs_BALANCED_H2_PATCH1"]["p"] < 0.05)
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "PREREGISTERED_SUCCESSOR_TEST_DEV_A", "arm": ARM,
       "preregistration": {"path": os.path.relpath(PRE, G.X.REPO).replace("\\", "/"), "sha256": sha_file(PRE)},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__))},
       "pinned": PINNED, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name],
       "definition": {"beam": "pinned uL3 beam (frozen admissibility, transition geometry, RRF fused order, BEAM = 100, DEPTH = 2, seeds = frozen top-5)",
                      "compression_pinned": "first 100 distinct unvisited targets in the global fused order",
                      "compression_balanced": "section 21: (parent, target) pairs in fused order (first occurrence), within-parent rank r, sequence sorted by (r, frontier position) = A1 B1 C1 ... A2 B2 C2 ...",
                      "compression_covscore": "phase 1 = the pairs with r = 0 in frontier order (the best child of every active parent: A1 B1 C1 D1 ...); phase 2 = all remaining pairs in the frozen "
                                              "global fused order (A2 A3 B2 A4 ...); the first 100 distinct unvisited targets of phase 1 then phase 2; a parent whose best child is already taken "
                                              "contributes nothing in phase 1; no other constant",
                      "patch": "section 16 verbatim: SAFE minus its weakest admitted block + one query-local block of the same size = the beam's visited nodes not exposed by the 49 kept "
                               "blocks in first-visit order (hop 1 then hop 2, each in beam order), topped up from the removed block (ret_rrf position, then node id)",
                      "metric": "ALL-gold under matched unique-node exposure; node-level == block-level for L1 / SAFE (asserted)"},
       "beam_diversity_DEV_A": beam_div, "covscore_beam_DEV_A": cov_div, "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-20s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- H2_PATCH1 and BALANCED_H2_PATCH1 must reproduce the section-16 / 21 records exactly
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
res["reference_arms_reproduced"] = "H2_PATCH1 == h2patch_A_%s.json and BALANCED_H2_PATCH1 == h2balanced_A_%s.json (ALL, ANY, vs SAFE / H2_PATCH1 / L1, per hop, novel counts, patch size)" % (name, name)
G.log("reference arms reproduced: %s" % res["reference_arms_reproduced"])


# ---------------------------------------------------------------- flips COVSCORE vs BALANCED (and vs H2_PATCH1) with their cause
def flip_table(ref, b1_ref, b2_ref):
    f = {"newly_covered": {"n": 0, "gold_nodes_newly_served": 0, "of_which_newly_visited_by_the_covscore_beam": 0, "of_which_visited_by_both_beams_but_cut_by_the_reference_patch": 0},
         "newly_lost": {"n": 0, "gold_nodes_dropped": 0, "of_which_no_longer_visited_by_the_covscore_beam": 0, "of_which_visited_but_beyond_the_patch_capacity": 0}}
    for qi in rowsA:
        pC, pR = patches[ARM][qi], patches[ref][qi]
        visC = pC["novel_set"] | set(cov1[qi]) | set(cov2[qi])
        visR = pR["novel_set"] | set(b1_ref[qi]) | set(b2_ref[qi])
        if ind[ARM][qi] and not ind[ref][qi]:
            e = f["newly_covered"]
            e["n"] += 1
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(ARM, qi, g) and not served_by(ref, qi, g):
                    e["gold_nodes_newly_served"] += 1
                    e["of_which_newly_visited_by_the_covscore_beam"] += int(g in visC and g not in visR)
                    e["of_which_visited_by_both_beams_but_cut_by_the_reference_patch"] += int(g in visC and g in visR)
        elif ind[ref][qi] and not ind[ARM][qi]:
            e = f["newly_lost"]
            e["n"] += 1
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(ref, qi, g) and not served_by(ARM, qi, g):
                    e["gold_nodes_dropped"] += 1
                    e["of_which_no_longer_visited_by_the_covscore_beam"] += int(g not in visC)
                    e["of_which_visited_but_beyond_the_patch_capacity"] += int(g in visC)
    return f


res["attribution_flips"] = {"COVSCORE_vs_BALANCED_H2_PATCH1": flip_table(ARM_BAL, bal1, bal2), "COVSCORE_vs_H2_PATCH1": flip_table(ARM_H2, beams1, beams2)}
G.log("flips %s" % json.dumps(res["attribution_flips"]))

# ---------------------------------------------------------------- THE DIAGNOSTIC of the ruling: does the compression keep more correct H2 parents / gold children?
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
BEAMS = {"pinned": (beams1, beams2, CAND_P, PAR_P), "balanced": (bal1, bal2, CAND_B, PAR_B), "covscore": (cov1, cov2, CAND_C, PAR_C)}
diag = {}
for gname, rows_h in groups:
    o = {"SAFE_failures": 0, "missing_gold_nodes": 0}
    for bn in BEAMS:
        o[bn] = {"missing_gold_visited (hop1 / hop2)": [0, 0], "missing_gold_scored_at_hop2_but_not_kept": 0, "missing_gold_never_scored_within_depth2": 0,
                 "gold_bearing_hop1_parents (parents with a missing gold node among their hop-2 candidates)": 0, "gold_bearing_hop1_parents_with_a_gold_child_kept": 0,
                 "failures_with_all_missing_nodes_visited": 0, "failures_covered_by_the_arm": 0}
    for qi in rows_h:
        if ind["SAFE"][qi] or not len(gold_nodes[qi]):
            continue
        miss = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
        seeds_q = set(int(x) for x in seeds[qi] if x >= 0)
        miss = [g for g in miss if g not in seeds_q]
        o["SAFE_failures"] += 1
        o["missing_gold_nodes"] += len(miss)
        for bn, (b1s, b2s, CANDS, _) in BEAMS.items():
            e = o[bn]
            s1, s2 = set(b1s[qi]), set(b2s[qi])
            c2 = set(CANDS[(qi, 1)].tolist())
            all_vis = True
            for g in miss:
                if g in s1:
                    e["missing_gold_visited (hop1 / hop2)"][0] += 1
                elif g in s2:
                    e["missing_gold_visited (hop1 / hop2)"][1] += 1
                elif g in c2:
                    e["missing_gold_scored_at_hop2_but_not_kept"] += 1
                    all_vis = False
                else:
                    e["missing_gold_never_scored_within_depth2"] += 1
                    all_vis = False
            ms = set(miss) - s1 - seeds_q                        # missing gold nodes still open at hop 2 (not a seed, not kept at hop 1)
            ms_arr = np.fromiter(ms, np.int64, len(ms))
            for v in b1s[qi]:
                nb_ = admissible(int(v))
                nbrs = set(nb_[np.isin(nb_, ms_arr)].tolist()) if len(ms_arr) else set()
                if nbrs:
                    e["gold_bearing_hop1_parents (parents with a missing gold node among their hop-2 candidates)"] += 1
                    e["gold_bearing_hop1_parents_with_a_gold_child_kept"] += int(bool(nbrs & s2))
            e["failures_with_all_missing_nodes_visited"] += int(all_vis)
        o["pinned"]["failures_covered_by_the_arm"] += int(ind[ARM_H2][qi])
        o["balanced"]["failures_covered_by_the_arm"] += int(ind[ARM_BAL][qi])
        o["covscore"]["failures_covered_by_the_arm"] += int(ind[ARM][qi])
    diag[gname] = o
    G.log("diag %s %s" % (gname, json.dumps(o)))
res["diagnostic_frontier_compression_vs_gold"] = diag

# ---------------------------------------------------------------- the pre-registered per-cache labels (the summary applies the cross-cache rule)
oC = res["arms"][ARM]
res["per_cache_labels"] = {"ALL_vs_BALANCED_H2_PATCH1": oC["ALL_vs_BALANCED_H2_PATCH1"]["label"], "ALL_vs_H2_PATCH1": oC["ALL_vs_H2_PATCH1"]["label"], "ALL_vs_SAFE": oC["ALL_vs_SAFE"]["label"],
                           "text_safe_vs_BALANCED (not (lost > gained and p < 0.05))": not oC["significant_loss_vs_BALANCED_H2_PATCH1"],
                           "text_safe_vs_H2_PATCH1": not oC["significant_loss_vs_H2_PATCH1"], "text_safe_vs_SAFE": not oC["significant_loss_vs_SAFE"]}
if oC["per_hop"]:
    res["per_cache_labels"]["hop_regressions_vs_BALANCED (lost > gained and p < 0.05)"] = {h: bool(e["vs_BALANCED_H2_PATCH1"]["lost"] > e["vs_BALANCED_H2_PATCH1"]["gained"] and e["vs_BALANCED_H2_PATCH1"]["p"] < 0.05)
                                                                                          for h, e in oC["per_hop"].items()}
G.log("labels %s" % json.dumps(res["per_cache_labels"]))
res["seconds"] = round(time.time() - T_ALL, 1)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
