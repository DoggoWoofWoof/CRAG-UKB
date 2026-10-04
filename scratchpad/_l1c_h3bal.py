"""BALANCED_H3_PATCH1 (fourteenth ruling, item 3; PREREGISTRATION_BALANCED_H3_PATCH1.json written first): the ONE H3_PATCH1 successor on the
balanced frontier -- the section-17 arm (the frozen real-edge beam one hop deeper under the identical section-16 one-block patch) with the
section-21 round-robin frontier compression at every hop instead of the pinned global-order compression.  Nothing else changes:
    beam       = the pinned uL3 beam exec'd with DEPTH = 3 exactly as section 17 (same head substitutions, asserted), frozen admissibility, transition
                 geometry, RRF fused order, BEAM = 100, seeds = the frozen top-5; the pinned depth-3 beam is recomputed and H3_PATCH1 asserted equal to
                 h3patch_A_<cache>.json; hops 1-2 of the balanced depth-3 beam are the section-21 balanced beam (asserted: BALANCED_H2_PATCH1 rebuilt from
                 them equals h2balanced_A_<cache>.json; H2_PATCH1 rebuilt from the pinned hops 1-2 equals h2patch_A_<cache>.json)
    compression= section 21 verbatim (rr_compress: per-parent rank round-robin over the frozen fused order) at hops 1, 2 AND 3 -- one rule, no hop branch
    patch      = section 16 / 17 verbatim: SAFE minus its weakest admitted block + one query-local block of the same size holding the beam's visited nodes
                 not exposed by the 49 kept blocks, in first-visit order (hop 1, hop 2, hop 3 -- the order section 17 pre-registered, which preserves the
                 depth-2 patch), topped up from the removed block (ret_rrf position, then node id); |served nodes| = canonical SAFE per query (asserted)
    no wider patch, no H3-specific scoring, no new constant, no dataset branch ("No wider patch initially. No H3-specific scoring. Same real-edge maths.")
Arms: L1, SAFE, H2_PATCH1 (s16), H3_PATCH1 (s17), BALANCED_H2_PATCH1 (s21) -- all reference, recomputed and asserted -- and BALANCED_H3_PATCH1 (new).
Metric: ALL-gold at matched exposure; primary comparison BALANCED_H3_PATCH1 vs BALANCED_H2_PATCH1 at hop 3 on the two metaqa caches.
    python -u scratchpad/_l1c_h3bal.py <cache>          -> results/L1_COVPART/h3bal_A_<cache>.json
    python -u scratchpad/_l1c_h3bal.py metaqa --dry     -> no record; identities + gold-free depth-3 beam statistics only
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1g_core as G  # noqa: E402
import _l1kb_core as KB  # noqa: E402
import _l1ps_router as RT  # noqa: E402

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_BALANCED_H3_PATCH1.json")
NEW_MODULES = ("_l1c_h3bal.py", "_l1c_h3bal_summary.py")
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
ARM_H2, ARM_H3, ARM_BAL, ARM = "H2_PATCH1", "H3_PATCH1", "BALANCED_H2_PATCH1", "BALANCED_H3_PATCH1"
ARMS = ("L1", "SAFE", ARM_H2, ARM_H3, ARM_BAL, ARM)
DEPTHS = {ARM_H2: 2, ARM_H3: 3, ARM_BAL: 2, ARM: 3}
ALPHA = 0.01
D2 = 2                                                       # the depth of the section-16 / 21 records (their figures are per hop 1 / 2)


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
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    for mod in NEW_MODULES:
        assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    fp_out = os.path.join(OUT, "h3bal_A_%s.json" % name)
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
else:
    assert name == "metaqa", "the dry run is metaqa only"
    fp_out = None
rec16 = json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % name), encoding="utf-8"))
rec17 = json.load(open(os.path.join(OUT, "h3patch_A_%s.json" % name), encoding="utf-8"))
rec21 = json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % name), encoding="utf-8"))
pre17 = json.load(open(os.path.join(OUT, "PREREGISTRATION_H3_PATCH1_DEV_A.json"), encoding="utf-8"))

# ---------------------------------------------------------------- the pinned uL3 head exec'd at DEPTH = 3 exactly as section 17 (same substitutions, asserted)
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
SUBS = [("name = sys.argv[1]", "name = %r" % name),
        ('LATENT = "--latent" in sys.argv', "LATENT = False"),
        ("DEPTH = 2", "DEPTH = 3"),
        ("beams1, beams2 = [[] for _ in range(nq)], [[] for _ in range(nq)]",
         "beams1, beams2, beams3 = [[] for _ in range(nq)], [[] for _ in range(nq)], [[] for _ in range(nq)]"),
        ("beams1[qi], beams2[qi] = beams[0], beams[1]", "beams1[qi], beams2[qi], beams3[qi] = beams[0], beams[1], beams[2]")]
assert [list(x) for x in SUBS[1:]] == [list(x) for x in pre17["frozen_and_reused_verbatim"]["head_substitutions"][1:]], "substitutions differ from section 17"
assert [list(x) for x in SUBS] == [list(x) for x in rec17["head_substitutions"]], "substitutions differ from the section-17 record"
for a, b in SUBS:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
T_ALL = time.time()
exec(head)                                                   # D, C, beams1..3 (the pinned depth-3 beam on DEV_A), expand_real, fused_order, seeds, hub, admissible ...
assert DEPTH == 3 and edges_h.shape[1] == 3
secs_pinned = secs.copy()
edges_pinned = edges_h.copy()


# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording (must equal beams1 / beams2 / beams3)  [section 21 verbatim]
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


# ---------------------------------------------------------------- the balanced compression  [section 21 verbatim; applied at every hop]
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


def beam_with(compress, qi):
    """the pinned beam with the frontier compression replaced by `compress` (section 21's beam_balanced, parametrised); DEPTH = 3 here."""
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


HOPS = ("hop1", "hop2", "hop3")
bal = [[[] for _ in range(nq)] for _ in range(DEPTH)]
PIN = (beams1, beams2, beams3)
CAND_P, CAND_B = {}, {}
PAR_P, PAR_B = {}, {}
div = {k: {h: [] for h in HOPS} for k in ("pinned", "balanced")}
stat_b = {h: [] for h in HOPS}
secs_bal = np.zeros(nA, np.float64)
edges_bal = np.zeros((nA, DEPTH), np.int64)
same_set = np.zeros((nA, DEPTH), bool)
jacc = np.zeros((nA, DEPTH), np.float64)
n_pinned_cands = np.zeros((nA, DEPTH), np.int64)
t0b = time.time()
for j, qi in enumerate(rowsA):
    bp, cp, pp_ = beam_pinned_with_candidates(qi)
    assert bp[0] == beams1[qi] and bp[1] == beams2[qi] and bp[2] == beams3[qi], "pinned beam with candidate recording differs from the exec'd pinned beam"
    bb, cb, pb, st, ne, dt = beam_with(rr_compress, qi)
    for h in range(DEPTH):
        bal[h][qi] = bb[h]
    secs_bal[j] = dt
    edges_bal[j] = ne
    for h in range(DEPTH):
        CAND_P[(qi, h)], CAND_B[(qi, h)] = cp[h], cb[h]
        PAR_P[(qi, h)], PAR_B[(qi, h)] = pp_[h], pb[h]
        n_pinned_cands[j, h] = len(cp[h])
        a, b_ = set(bp[h]), set(bb[h])
        same_set[j, h] = a == b_
        jacc[j, h] = len(a & b_) / float(len(a | b_)) if (a | b_) else 1.0
        hk = HOPS[h]
        div["pinned"][hk].append(diversity(bp[h], pp_[h]))
        div["balanced"][hk].append(diversity(bb[h], pb[h]))
        stat_b[hk].append(st[h])
    if j % 200 == 0:
        G.log("  beams %d / %d (%.0fs) sizes pinned %s balanced %s | jaccard %s" % (j, nA, time.time() - t0b, [len(x) for x in bp], [len(x) for x in bb], np.round(jacc[j], 3).tolist()))
bal1, bal2, bal3 = bal
G.log("beams done (%.0fs): wall-clock / query mean balanced depth-3 %.1f ms (pinned depth-3 %.1f ms)" % (time.time() - t0b, 1000 * secs_bal.mean(), 1000 * secs_pinned.mean()))


def agg_stats(lst):
    lst = [s for s in lst if s is not None]
    if not lst:
        return None
    keys = lst[0].keys()
    return {k: round(float(np.mean([s[k] for s in lst])), 2) for k in keys}


# hops 1-2: the section-21 figures (asserted against h2balanced_A_<cache>.json; the balanced beam's hops 1-2 do not depend on the depth)
beam_div = {"n_DEV_A": nA, "wall_clock_ms_per_query": {"pinned": round(1000 * float(secs_pinned.mean()), 1), "balanced": round(1000 * float(secs_bal.mean()), 1)},
            "transitions_scored_per_query_mean (hop1 / hop2)": {"pinned": [round(float(x), 1) for x in edges_pinned.mean(0)[:D2]], "balanced": [round(float(x), 1) for x in edges_bal.mean(0)[:D2]]},
            "candidates_unique_per_query_mean (hop1 / hop2)": {"pinned": [round(float(x), 1) for x in n_pinned_cands.mean(0)[:D2]],
                                                               "balanced": [round(float(np.mean([len(CAND_B[(qi, h)]) for qi in rowsA])), 1) for h in range(D2)]},
            "beam_size_mean (hop1 / hop2)": {"pinned": [round(float(np.mean([len(beams1[qi]) for qi in rowsA])), 1), round(float(np.mean([len(beams2[qi]) for qi in rowsA])), 1)],
                                             "balanced": [round(float(np.mean([len(bal1[qi]) for qi in rowsA])), 1), round(float(np.mean([len(bal2[qi]) for qi in rowsA])), 1)]},
            "queries_truncated_at_hop (hop1 / hop2) balanced": [int(sum(1 for s in stat_b[h] if s and s["truncated"])) for h in HOPS[:D2]],
            "same_kept_set_as_pinned (queries; hop1 / hop2)": [int(same_set[:, h].sum()) for h in range(D2)],
            "jaccard_kept_vs_pinned_mean (hop1 / hop2)": [round(float(jacc[:, h].mean()), 3) for h in range(D2)],
            "distinct_parents_in_kept_beam_mean (hop1 / hop2)": {a: [round(float(np.mean([x[0] for x in div[a][h]])), 1) for h in HOPS[:D2]] for a in ("pinned", "balanced")},
            "largest_single_parent_share_mean (hop1 / hop2)": {a: [round(float(np.mean([x[1] for x in div[a][h]])), 3) for h in HOPS[:D2]] for a in ("pinned", "balanced")},
            "balanced_compression_stats_mean": {h: agg_stats(stat_b[h]) for h in HOPS[:D2]}}
bd21 = rec21["beam_diversity_DEV_A"]
for k in bd21:
    if k != "wall_clock_ms_per_query":
        assert beam_div[k] == bd21[k], "balanced beam figure %s differs from the section-21 record" % k
G.log("balanced beam hops 1-2 reproduced: every deterministic figure of h2balanced_A_%s.json beam_diversity_DEV_A matches" % name)
# depth 3: the new gold-free figures (hop 3 of both beams; the section-17 cost figures for the pinned beam are asserted below)
d3 = {"wall_clock_ms_per_query": {"pinned": round(1000 * float(secs_pinned.mean()), 1), "balanced": round(1000 * float(secs_bal.mean()), 1)},
      "transitions_scored_per_query_mean (hop1 / hop2 / hop3)": {"pinned": [round(float(x), 1) for x in edges_pinned.mean(0)], "balanced": [round(float(x), 1) for x in edges_bal.mean(0)]},
      "candidates_unique_per_query_mean (hop1 / hop2 / hop3)": {"pinned": [round(float(x), 1) for x in n_pinned_cands.mean(0)],
                                                                "balanced": [round(float(np.mean([len(CAND_B[(qi, h)]) for qi in rowsA])), 1) for h in range(DEPTH)]},
      "beam_size_mean (hop1 / hop2 / hop3)": {"pinned": [round(float(np.mean([len(PIN[h][qi]) for qi in rowsA])), 1) for h in range(DEPTH)],
                                              "balanced": [round(float(np.mean([len(bal[h][qi]) for qi in rowsA])), 1) for h in range(DEPTH)]},
      "queries_truncated_at_hop (hop1 / hop2 / hop3) balanced": [int(sum(1 for s in stat_b[h] if s and s["truncated"])) for h in HOPS],
      "same_kept_set_as_pinned (queries; hop1 / hop2 / hop3)": [int(same_set[:, h].sum()) for h in range(DEPTH)],
      "jaccard_kept_vs_pinned_mean (hop1 / hop2 / hop3)": [round(float(jacc[:, h].mean()), 3) for h in range(DEPTH)],
      "distinct_parents_in_kept_beam_mean (hop1 / hop2 / hop3)": {a: [round(float(np.mean([x[0] for x in div[a][h]])), 1) for h in HOPS] for a in ("pinned", "balanced")},
      "largest_single_parent_share_mean (hop1 / hop2 / hop3)": {a: [round(float(np.mean([x[1] for x in div[a][h]])), 3) for h in HOPS] for a in ("pinned", "balanced")},
      "balanced_compression_stats_mean": {h: agg_stats(stat_b[h]) for h in HOPS},
      "hop3_rounds_used_histogram_balanced": {}}
rounds3 = [s["rounds_used"] for s in stat_b["hop3"] if s is not None]
if rounds3:
    u_, c_ = np.unique(np.asarray(rounds3), return_counts=True)
    d3["hop3_rounds_used_histogram_balanced"] = {str(int(k)): int(v) for k, v in zip(u_, c_)}
assert d3["transitions_scored_per_query_mean (hop1 / hop2 / hop3)"]["pinned"] == rec17["cost_depth3"]["transitions_scored_per_query_mean (hop1 / hop2 / hop3)"], "pinned depth-3 cost differs from section 17"
G.log("depth-3 beams (gold-free) %s" % json.dumps(d3))

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
# gold-free patch-capacity arithmetic of the new arm (no gold read): novel counts per hop and how many hop-3 novel nodes the one-block budget admits
cap_stat = {"novel_per_hop_mean (1 / 2 / 3)": [0.0, 0.0, 0.0], "novel_total_depth3_mean": 0.0, "queries_truncated_depth3": 0, "queries_truncated_depth2": 0,
            "hop3_novel_admitted_per_query_mean": 0.0, "hop3_novel_cut_per_query_mean": 0.0, "hop2_novel_cut_per_query_mean": 0.0, "queries_admitting_no_hop3_node": 0,
            "patch_size_mean": 0.0}
acc_ = np.zeros(9, np.float64)
for qi in rowsA:
    kept = finals["SAFE"][qi] - {int(weakest_of[qi])}
    nb = int(sizes[int(weakest_of[qi])])
    nov = [[int(v) for v in bl[qi] if int(hard[v]) not in kept] for bl in (bal1, bal2, bal3)]
    n1, n2, n3 = len(nov[0]), len(nov[1]), len(nov[2])
    adm3 = max(0, min(n3, nb - n1 - n2))
    cut2 = max(0, n1 + n2 - nb) if n1 + n2 > nb else 0
    cut2 = min(cut2, n2)
    acc_ += [n1, n2, n3, n1 + n2 + n3, int(n1 + n2 + n3 > nb), int(n1 + n2 > nb), adm3, n3 - adm3, cut2]
    cap_stat["queries_admitting_no_hop3_node"] += int(adm3 == 0)
    cap_stat["patch_size_mean"] += nb
acc_ /= nA
cap_stat["novel_per_hop_mean (1 / 2 / 3)"] = [round(float(x), 1) for x in acc_[:3]]
cap_stat["novel_total_depth3_mean"] = round(float(acc_[3]), 1)
cap_stat["queries_truncated_depth3"] = int(round(acc_[4] * nA))
cap_stat["queries_truncated_depth2"] = int(round(acc_[5] * nA))
cap_stat["hop3_novel_admitted_per_query_mean"] = round(float(acc_[6]), 1)
cap_stat["hop3_novel_cut_per_query_mean"] = round(float(acc_[7]), 1)
cap_stat["hop2_novel_cut_per_query_mean"] = round(float(acc_[8]), 2)
cap_stat["patch_size_mean"] = round(cap_stat["patch_size_mean"] / nA, 1)
G.log("balanced depth-3 patch capacity (gold-free) %s" % json.dumps(cap_stat))
if DRY:
    G.log("DRY RUN: BASE, SAFE, the pinned depth-3 beam and the balanced hops 1-2 reproduced; the balanced depth-3 beam built (gold-free statistics above); no coverage number of the new arm; no record (%.0fs)" % (time.time() - T_ALL))
    sys.exit(0)


# ---------------------------------------------------------------- the patch rule of section 16 / 17 (verbatim logic), applied to a given beam of depth 2 or 3
def build_patches(beam_lists):
    depth = len(beam_lists)
    patch = [None] * nq
    for qi in rowsA:
        safe50 = finals["SAFE"][qi]
        weakest = int(weakest_of[qi])
        kept = safe50 - {weakest}
        nb = int(sizes[weakest])
        nov = [[int(v) for v in bl[qi] if int(hard[v]) not in kept] for bl in beam_lists]
        novel = sum(nov, [])                                 # first-visit order of the beam: hop 1, hop 2 (, hop 3) -- the pre-registered order of sections 16 / 17
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
        hop_of = {}
        for h, lst in enumerate(nov):
            for v in lst:
                hop_of[v] = h + 1
        comp = [sum(1 for v in nodes[:n_novel_used] if hop_of[v] == h) for h in (1, 2, 3)]
        patch[qi] = {"kept": kept, "weakest": weakest, "nb": nb, "nodes": set(nodes), "k": len(novel), "nov": [len(x) for x in nov] + [0] * (3 - depth),
                     "nov1": len(nov[0]), "nov2": len(nov[1]), "truncated": int(len(novel) > nb), "n_novel_used": n_novel_used, "n_fill": n_fill, "comp": comp,
                     "used_hop1": comp[0], "used_hop2": comp[1], "used_hop3": comp[2],
                     "novel_set": set(novel), "novel_pos": {v: r for r, v in enumerate(novel)}, "hop_of": hop_of, "fill_set": set(nodes[n_novel_used:])}
        assert int(sizes[list(kept)].sum()) + nb == int(sizes[list(safe50)].sum())
    return patch


patches = {ARM_H2: build_patches((beams1, beams2)), ARM_H3: build_patches((beams1, beams2, beams3)),
           ARM_BAL: build_patches((bal1, bal2)), ARM: build_patches((bal1, bal2, bal3))}
for qi in rowsA:
    assert patches[ARM_H2][qi]["kept"] == patches[ARM][qi]["kept"] == patches[ARM_H3][qi]["kept"] == patches[ARM_BAL][qi]["kept"]


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
    gain_hop = np.zeros((nq, 3), np.int64)
    loss_nodes = np.zeros(nq, np.int64)
    for qi in rowsA:
        p = patches[arm][qi]
        for g in gold_nodes[qi]:
            g = int(g)
            sS, sP = served_by("SAFE", qi, g), served_by(arm, qi, g)
            if sP and not sS:
                gain_nodes[qi] += 1
                gain_hop[qi, p["hop_of"][g] - 1] += 1
            elif sS and not sP:
                loss_nodes[qi] += 1
                assert int(hard[g]) == p["weakest"]
    return gain_nodes, gain_hop, loss_nodes


acct = {arm: gold_accounting(arm) for arm in patches}
COMPS = {ARM_H2: [("vs_L1", "L1"), ("vs_SAFE", "SAFE")],
         ARM_H3: [("vs_L1", "L1"), ("vs_SAFE", "SAFE"), ("vs_H2_PATCH1", ARM_H2)],
         ARM_BAL: [("vs_L1", "L1"), ("vs_SAFE", "SAFE"), ("vs_H2_PATCH1", ARM_H2)],
         ARM: [("vs_L1", "L1"), ("vs_SAFE", "SAFE"), ("vs_H2_PATCH1", ARM_H2), ("vs_H3_PATCH1", ARM_H3), ("vs_BALANCED_H2_PATCH1", ARM_BAL)],
         "SAFE": [("vs_L1", "L1")], "L1": []}


def evaluate(arm):
    iv, ia = ind[arm], ind_any[arm]
    o = {"arm": arm, "ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4), "per_hop": {}}
    if arm in patches:
        patch = patches[arm]
        gain_nodes, gain_hop, loss_nodes = acct[arm]
        o["exposure"] = "matched: |served nodes| equal to canonical SAFE per query (asserted)"
        o["depth"] = DEPTHS[arm]
        o["novel_nodes_per_query_mean"] = round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1)
        o["novel_per_hop_mean"] = [round(float(np.mean([patch[qi]["nov"][h] for qi in rowsA])), 1) for h in range(3)]
        o["patch_size_mean"] = round(float(np.mean([patch[qi]["nb"] for qi in rowsA])), 1)
        o["patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)"] = [round(float(np.mean([patch[qi]["comp"][h] for qi in rowsA])), 1) for h in range(3)] + \
            [round(float(np.mean([patch[qi]["n_fill"] for qi in rowsA])), 1)]
        o["queries_truncated"] = int(sum(patch[qi]["truncated"] for qi in rowsA))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(patch[qi]["k"] == 0 for qi in rowsA))
        o["gold_nodes_gained_vs_SAFE"] = int(gain_nodes[rowsA].sum())
        o["gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)"] = [int(gain_hop[rowsA, h].sum()) for h in range(3)]
        o["queries_with_gold_node_gained_vs_SAFE"] = int((gain_nodes[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss_nodes[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss_nodes[rowsA] > 0).sum())
    comps = COMPS[arm]
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        e = {"n": int(s.sum()), "ALL": round(float(iv[s].mean()), 4), "ANY": round(float(ia[s].mean()), 4)}
        for key, ref in comps:
            e[key] = mc(ind[ref], iv, s)
        if arm in patches:
            e["gold_nodes_gained"] = int(acct[arm][0][s].sum())
            e["gold_nodes_gained_by_hop (1 / 2 / 3)"] = [int(acct[arm][1][s, hh].sum()) for hh in range(3)]
            e["gold_nodes_lost_with_removed_block"] = int(acct[arm][2][s].sum())
        o["per_hop"]["hop%d" % h] = e
    for key, ref in comps:
        o["ALL_" + key] = mcl(ind[ref], iv, m)
        o["ANY_" + key] = mc(ind_any[ref], ia, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_" + key] = {"n": int(s.sum()), **mc(ind[ref], iv, s), "pooled_ref": round(float(ind[ref][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    for key, ref in comps:
        if key != "vs_L1":
            o["significant_loss_" + key] = bool(o["ALL_" + key]["lost"] > o["ALL_" + key]["gained"] and o["ALL_" + key]["p"] < 0.05)
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "PREREGISTERED_SUCCESSOR_TEST_DEV_A", "arm": ARM,
       "preregistration": {"path": os.path.relpath(PRE, G.X.REPO).replace("\\", "/"), "sha256": sha_file(PRE)},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__))},
       "pinned": PINNED, "head_substitutions": SUBS, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name],
       "definition": {"beam": "pinned uL3 beam exec'd at DEPTH = 3 exactly as section 17 (frozen admissibility, transition geometry, RRF fused order, BEAM = 100, seeds = frozen top-5)",
                      "compression_pinned": "first 100 distinct unvisited targets in the global fused order (H2_PATCH1 / H3_PATCH1)",
                      "compression_balanced": "section 21 verbatim at every hop: (parent, target) pairs in fused order (first occurrence), within-parent rank r, sequence sorted by "
                                              "(r, frontier position) = A1 B1 C1 ... A2 B2 C2 ... (BALANCED_H2_PATCH1 / BALANCED_H3_PATCH1); one rule, no hop branch",
                      "patch": "section 16 / 17 verbatim: SAFE minus its weakest admitted block + one query-local block of the same size = the beam's visited nodes not exposed by the 49 "
                               "kept blocks in first-visit order (hop 1, hop 2, hop 3, each in beam order), topped up from the removed block (ret_rrf position, then node id)",
                      "metric": "ALL-gold under matched unique-node exposure; node-level == block-level for L1 / SAFE (asserted)"},
       "beam_diversity_DEV_A (hops 1-2; == section 21)": beam_div, "depth3_beams_DEV_A (gold-free)": d3, "balanced_depth3_patch_capacity (gold-free)": cap_stat, "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-20s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- H2_PATCH1, H3_PATCH1 and BALANCED_H2_PATCH1 must reproduce the section-16 / 17 / 21 records exactly
o16, r16 = res["arms"][ARM_H2], rec16["arms"][ARM_H2]
assert o16["ALL"] == r16["ALL"] and o16["ANY"] == r16["ANY"], (o16["ALL"], r16["ALL"])
assert (o16["ALL_vs_SAFE"]["gained"], o16["ALL_vs_SAFE"]["lost"]) == (r16["ALL_vs_SAFE"]["gained"], r16["ALL_vs_SAFE"]["lost"])
for h in o16["per_hop"]:
    assert o16["per_hop"][h]["ALL"] == r16["per_hop"][h]["ALL"], h
    assert (o16["per_hop"][h]["vs_SAFE"]["gained"], o16["per_hop"][h]["vs_SAFE"]["lost"]) == (r16["per_hop"][h]["vs_SAFE"]["gained"], r16["per_hop"][h]["vs_SAFE"]["lost"]), h
assert o16["novel_nodes_per_query_mean"] == r16["novel_nodes_per_query_mean"] and o16["queries_truncated"] == r16["queries_truncated"]
COMP4 = "patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)"
COMP3 = "patch_composition_mean (hop-1 novel / hop-2 novel / fill from removed block)"
for o_, r_ in ((o16, r16),):
    c = o_[COMP4]
    assert c[2] == 0.0 and [c[0], c[1], c[3]] == r_[COMP3] and o_["novel_per_hop_mean"][:2] == [r_["novel_hop1_per_query_mean"], r_["novel_hop2_per_query_mean"]], "depth-2 composition differs"
o17, r17 = res["arms"][ARM_H3], rec17["arms"][ARM_H3]
assert o17["ALL"] == r17["ALL"] and o17["ANY"] == r17["ANY"], (o17["ALL"], r17["ALL"])
for key in ("ALL_vs_SAFE", "ALL_vs_H2_PATCH1", "ALL_vs_L1"):
    assert (o17[key]["gained"], o17[key]["lost"]) == (r17[key]["gained"], r17[key]["lost"]), key
for h in o17["per_hop"]:
    assert o17["per_hop"][h]["ALL"] == r17["per_hop"][h]["ALL"], h
    for key in ("vs_SAFE", "vs_H2_PATCH1"):
        assert (o17["per_hop"][h][key]["gained"], o17["per_hop"][h][key]["lost"]) == (r17["per_hop"][h][key]["gained"], r17["per_hop"][h][key]["lost"]), (h, key)
for key in ("novel_nodes_per_query_mean", "novel_per_hop_mean", "queries_truncated", "gold_nodes_gained_vs_SAFE", "gold_nodes_lost_with_removed_block", "patch_size_mean",
            "gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)", COMP4, "queries_with_no_novel_node (patch == removed block)"):
    assert o17[key] == r17[key], key
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
c = o21[COMP4]
assert c[2] == 0.0 and [c[0], c[1], c[3]] == r21[COMP3] and o21["novel_per_hop_mean"][:2] == [r21["novel_hop1_per_query_mean"], r21["novel_hop2_per_query_mean"]], "depth-2 composition differs"
res["reference_arms_reproduced"] = ("H2_PATCH1 == h2patch_A_%s.json, H3_PATCH1 == h3patch_A_%s.json and BALANCED_H2_PATCH1 == h2balanced_A_%s.json "
                                    "(ALL, ANY, vs SAFE / H2_PATCH1 / L1, per hop, novel counts, gold gained / lost, patch size and composition)" % (name, name, name))
G.log("reference arms reproduced: %s" % res["reference_arms_reproduced"])


# ---------------------------------------------------------------- flips of the new arm vs BALANCED_H2_PATCH1 (the depth) and vs H3_PATCH1 (the frontier) with their cause
def flip_table(ref, ref_beams):
    f = {"newly_covered": {"n": 0, "gold_nodes_newly_served": 0, "of_which_entering_via_hop3_positions": 0, "of_which_entering_via_hop1_or_hop2_positions": 0,
                           "of_which_newly_visited_by_the_new_beam": 0, "of_which_visited_by_both_beams_but_cut_by_the_reference_patch": 0, "queries_needing_more_than_one_new_node": 0},
         "newly_lost": {"n": 0, "gold_nodes_dropped": 0, "of_which_were_fill_under_the_reference": 0, "of_which_were_novel_positions_under_the_reference": 0,
                        "of_which_no_longer_visited_by_the_new_beam": 0, "of_which_visited_but_beyond_the_patch_capacity": 0}}
    for qi in rowsA:
        pN, pR = patches[ARM][qi], patches[ref][qi]
        visN = pN["novel_set"] | set(bal1[qi]) | set(bal2[qi]) | set(bal3[qi])
        visR = pR["novel_set"] | set().union(*[set(bl[qi]) for bl in ref_beams])
        if ind[ARM][qi] and not ind[ref][qi]:
            e = f["newly_covered"]
            e["n"] += 1
            new = [int(g) for g in gold_nodes[qi] if served_by(ARM, qi, int(g)) and not served_by(ref, qi, int(g))]
            e["gold_nodes_newly_served"] += len(new)
            e["of_which_entering_via_hop3_positions"] += sum(1 for g in new if pN["hop_of"].get(g) == 3)
            e["of_which_entering_via_hop1_or_hop2_positions"] += sum(1 for g in new if pN["hop_of"].get(g) in (1, 2))
            e["of_which_newly_visited_by_the_new_beam"] += sum(1 for g in new if g in visN and g not in visR)
            e["of_which_visited_by_both_beams_but_cut_by_the_reference_patch"] += sum(1 for g in new if g in visN and g in visR)
            e["queries_needing_more_than_one_new_node"] += int(len(new) > 1)
        elif ind[ref][qi] and not ind[ARM][qi]:
            e = f["newly_lost"]
            e["n"] += 1
            lost = [int(g) for g in gold_nodes[qi] if served_by(ref, qi, int(g)) and not served_by(ARM, qi, int(g))]
            e["gold_nodes_dropped"] += len(lost)
            e["of_which_were_fill_under_the_reference"] += sum(1 for g in lost if g in pR["fill_set"])
            e["of_which_were_novel_positions_under_the_reference"] += sum(1 for g in lost if g in pR["nodes"] and g not in pR["fill_set"])
            e["of_which_no_longer_visited_by_the_new_beam"] += sum(1 for g in lost if g not in visN)
            e["of_which_visited_but_beyond_the_patch_capacity"] += sum(1 for g in lost if g in visN)
    return f


res["attribution_flips"] = {"BALANCED_H3_vs_BALANCED_H2_PATCH1": flip_table(ARM_BAL, (bal1, bal2)), "BALANCED_H3_vs_H3_PATCH1": flip_table(ARM_H3, (beams1, beams2, beams3)),
                            "BALANCED_H3_vs_H2_PATCH1": flip_table(ARM_H2, (beams1, beams2))}
G.log("flips %s" % json.dumps(res["attribution_flips"]))

# ---------------------------------------------------------------- THE DIAGNOSTIC of the ruling at depth 3: reach (visited / scored-but-cut / never scored) under the two frontiers
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
BEAMS = {"pinned_depth3": ((beams1, beams2, beams3), CAND_P, ARM_H3), "balanced_depth3": ((bal1, bal2, bal3), CAND_B, ARM)}
diag = {}
for gname, rows_h in groups:
    o = {"SAFE_failures": 0, "missing_gold_nodes": 0}
    for bn in BEAMS:
        o[bn] = {"missing_gold_visited (first at hop 1 / 2 / 3)": [0, 0, 0], "missing_gold_scored_but_cut (first at hop 1 / 2 / 3)": [0, 0, 0],
                 "missing_gold_never_scored_within_depth3": 0, "missing_gold_visited_fraction_mean": [],
                 "gold_bearing_hop1_parents (parents with a missing gold node among their hop-2 candidates)": 0, "gold_bearing_hop1_parents_with_a_gold_child_kept": 0,
                 "gold_bearing_hop2_parents (parents with a missing gold node among their hop-3 candidates)": 0, "gold_bearing_hop2_parents_with_a_gold_child_kept": 0,
                 "failures_with_no_missing_gold_node_visited": 0, "failures_with_every_missing_gold_node_scored_within_depth3": 0,
                 "failures_with_all_missing_nodes_visited": 0, "of_which_covered_by_the_arm": 0,
                 "of_which_not_covered__beyond_the_patch_capacity": 0, "of_which_not_covered__dropped_with_removed_block": 0, "of_which_not_covered__other": 0,
                 "failures_covered_by_the_arm": 0}
    for qi in rows_h:
        if ind["SAFE"][qi] or not len(gold_nodes[qi]):
            continue
        miss = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
        seeds_q = set(int(x) for x in seeds[qi] if x >= 0)
        miss = [g for g in miss if g not in seeds_q]
        o["SAFE_failures"] += 1
        o["missing_gold_nodes"] += len(miss)
        for bn, (bls, CANDS, arm) in BEAMS.items():
            e = o[bn]
            vis_hop = {}
            for h, lst in enumerate(bls):
                for v in lst[qi]:
                    vis_hop.setdefault(int(v), h + 1)
            cand_sets = [set(CANDS[(qi, h)].tolist()) for h in range(DEPTH)]
            nv, all_scored = 0, True
            for g in miss:
                if g in vis_hop:
                    nv += 1
                    e["missing_gold_visited (first at hop 1 / 2 / 3)"][vis_hop[g] - 1] += 1
                else:
                    hc = next((h for h in range(DEPTH) if g in cand_sets[h]), None)
                    if hc is None:
                        e["missing_gold_never_scored_within_depth3"] += 1
                        all_scored = False
                    else:
                        e["missing_gold_scored_but_cut (first at hop 1 / 2 / 3)"][hc] += 1
            e["missing_gold_visited_fraction_mean"].append(nv / max(1, len(miss)))
            s1, s2, s3 = set(bls[0][qi]), set(bls[1][qi]), set(bls[2][qi])
            for hp, (par_list, open_set, kept_next) in enumerate(((bls[0][qi], set(miss) - s1 - seeds_q, s2), (bls[1][qi], set(miss) - s1 - s2 - seeds_q, s3))):
                ms_arr = np.fromiter(open_set, np.int64, len(open_set))
                kp, kk = ("gold_bearing_hop1_parents (parents with a missing gold node among their hop-2 candidates)", "gold_bearing_hop1_parents_with_a_gold_child_kept") if hp == 0 else \
                    ("gold_bearing_hop2_parents (parents with a missing gold node among their hop-3 candidates)", "gold_bearing_hop2_parents_with_a_gold_child_kept")
                for v in par_list:
                    nb_ = admissible(int(v))
                    nbrs = set(nb_[np.isin(nb_, ms_arr)].tolist()) if len(ms_arr) else set()
                    if nbrs:
                        e[kp] += 1
                        e[kk] += int(bool(nbrs & kept_next))
            e["failures_with_no_missing_gold_node_visited"] += int(nv == 0)
            e["failures_with_every_missing_gold_node_scored_within_depth3"] += int(all_scored)
            e["failures_covered_by_the_arm"] += int(ind[arm][qi])
            if nv == len(miss):
                e["failures_with_all_missing_nodes_visited"] += 1
                p = patches[arm][qi]
                if ind[arm][qi]:
                    e["of_which_covered_by_the_arm"] += 1
                elif any(p["novel_pos"].get(g, -1) >= p["nb"] for g in miss):
                    e["of_which_not_covered__beyond_the_patch_capacity"] += 1
                elif acct[arm][2][qi] > 0:
                    e["of_which_not_covered__dropped_with_removed_block"] += 1
                else:
                    e["of_which_not_covered__other"] += 1
    for bn in BEAMS:
        fr = o[bn]["missing_gold_visited_fraction_mean"]
        o[bn]["missing_gold_visited_fraction_mean"] = round(float(np.mean(fr)), 3) if fr else None
    diag[gname] = o
    G.log("diag %s %s" % (gname, json.dumps(o)))
res["diagnostic_depth3_reach_vs_gold"] = diag

# ---------------------------------------------------------------- the pre-registered per-cache labels (the summary applies the cross-cache rule)
oN = res["arms"][ARM]
res["per_cache_labels"] = {"ALL_vs_BALANCED_H2_PATCH1": oN["ALL_vs_BALANCED_H2_PATCH1"]["label"], "ALL_vs_H3_PATCH1": oN["ALL_vs_H3_PATCH1"]["label"],
                           "ALL_vs_H2_PATCH1": oN["ALL_vs_H2_PATCH1"]["label"], "ALL_vs_SAFE": oN["ALL_vs_SAFE"]["label"],
                           "text_safe_vs_BALANCED_H2_PATCH1 (not (lost > gained and p < 0.05))": not oN["significant_loss_vs_BALANCED_H2_PATCH1"],
                           "text_safe_vs_H2_PATCH1": not oN["significant_loss_vs_H2_PATCH1"], "text_safe_vs_SAFE": not oN["significant_loss_vs_SAFE"]}
if oN["per_hop"]:
    res["per_cache_labels"]["hop_regressions_vs_BALANCED_H2_PATCH1 (lost > gained and p < 0.05)"] = {
        h: bool(e["vs_BALANCED_H2_PATCH1"]["lost"] > e["vs_BALANCED_H2_PATCH1"]["gained"] and e["vs_BALANCED_H2_PATCH1"]["p"] < 0.05) for h, e in oN["per_hop"].items()}
    res["per_cache_labels"]["hop3_vs_BALANCED_H2_PATCH1"] = mcl(ind[ARM_BAL], ind[ARM], m & (hops == 3))
    res["per_cache_labels"]["hop3_vs_H3_PATCH1"] = mcl(ind[ARM_H3], ind[ARM], m & (hops == 3))
G.log("labels %s" % json.dumps(res["per_cache_labels"]))
res["seconds"] = round(time.time() - T_ALL, 1)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
