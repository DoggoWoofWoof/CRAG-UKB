"""The frozen composite QMAX_BALANCED_H2_PATCH1 (FREEZE_QMAX_BALANCED_H2_PATCH1.json) run on ONE replay cache with
stage 2 streamed (_l1c_transfer_blockmax.py) -- the fifteenth ruling's transfer runner (2Wiki / HotpotQA) and its
identity test on the five DEV_A caches.

Chains (the section-24 logic, copied verbatim from the pinned _l1c_qmaxbal.py; no new constant, no dataset branch):
    BASE chain   L1 (served P50) -> SAFE (frozen F6 swap) -> H2_PATCH1 (section 16) / BALANCED_H2_PATCH1 (section 21)
    QMAX chain   QMAX_F0 (frozen RRF of [Cd, Cs, Cmax]; Cmax STREAMED) -> QMAX_SAFE -> QMAX_BALANCED_H2_PATCH1 (PRIMARY)
Modes
    python -u scratchpad/_l1c_transfer_composite.py <cache> --reproduce
        DEV_A cache (metaqa | metaqa_phg | squad | squad_phg | musique): the QMAX chain is run TWICE, with the pinned
        stage 2 (fp32 whole-corpus matrix; asserted to reproduce qmaxbal_A_<cache>.json exactly, as the BASE chain is
        asserted against h2patch_A / h2balanced_A) and with the streamed stage 2; the record reports the divergence
        of the streamed path: bitwise identity of the stage-2 ranking, the max score difference in fp32 ulps (exact),
        rows whose first-200 / first-50 fused blocks differ, and the coverage-level flips of every QMAX-chain arm --
        descriptively (no threshold, no decision).            -> results/L1_COVPART/transfer_repro_<cache>.json
    python -u scratchpad/_l1c_transfer_composite.py <name> --cache <replay_cache.npz> [--safe-record <L1_REPLAY json>]
        transfer mode on a NEW cache (dataset from the cache meta): split A (sha1 parity of the query id, the frozen
        rule) is the one-shot population, split B is never evaluated or written; arms = L1, SAFE, H2_PATCH1,
        BALANCED_H2_PATCH1 (BASE chain, descriptive stages), QMAX_F0, QMAX_SAFE (attribution stages), PRIMARY; the
        decision rule is the pre-registration's.                -> results/L1_COVPART/transfer_A_<name>.json
    ... --dry   identities only (BASE reproduced, |QMAX_SAFE| = 50, beams), no coverage number of any composed arm, no record
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
import _l1c_transfer_blockmax as TB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d",
          "_l1c_h2balanced.py": "7aed6b1d4399a1b432d0e6be48013cf73c92d4e47d2e5d2dd7b669b508d84bbe",
          "_l1c_qmaxbal.py": "cae88fa64a0516a4f55c99a4a3b464afd39e467dca42e4380a546ebbd545b1c8",
          "_l1g_candidate.py": "b01c27a6e64b18756db1ca1d8910e8db4bb54fa019d9809698cb1369206169e0",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327"}
DEV_A_CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
ARM_H2, ARM_BAL = "H2_PATCH1", "BALANCED_H2_PATCH1"
ARM_Q50, ARM_QSAFE, PRIMARY = "QMAX_F0", "QMAX_SAFE", "QMAX_BALANCED_H2_PATCH1"
ARMS_BASE = ("L1", "SAFE", ARM_H2, ARM_BAL)
ARMS_QMAX = (ARM_Q50, ARM_QSAFE, PRIMARY)
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


def argval(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _ws_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


PIN_SUFFIX = "__pinned_stage2"


def kname(arm):
    """record key name of an arm: the pinned-stage-2 twin chain (reproduce mode) uses the frozen arm names in its keys."""
    return arm.replace(PIN_SUFFIX, "")


name = sys.argv[1]
REPRO = "--reproduce" in sys.argv
DRY = "--dry" in sys.argv
CACHE_PATH = argval("--cache")
SAFE_REC_PATH = argval("--safe-record")
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
if REPRO:
    assert name in DEV_A_CACHES and CACHE_PATH is None, "reproduce mode runs on the five DEV_A caches only"
    SAFE_REC_PATH = os.path.join(G.X.REPO, SAFE_RECORD[name])
    fp_out = os.path.join(OUT, "transfer_repro_%s.json" % name)
else:
    assert CACHE_PATH is not None and name not in DEV_A_CACHES, "transfer mode needs --cache <replay_cache.npz> and a new name"
    CACHE_PATH = os.path.abspath(CACHE_PATH)
    assert os.path.exists(CACHE_PATH), CACHE_PATH
    z_meta = json.loads(str(np.load(CACHE_PATH, allow_pickle=True)["meta_json"]))
    G.X.CACHES[name] = CACHE_PATH                              # registration at run time; the pinned module file is not edited
    G.X.DS_OF[name] = z_meta["dataset"]
    fp_out = os.path.join(OUT, "transfer_A_%s.json" % name)
PRE = os.path.join(OUT, "PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json")
if not DRY:
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    pre = json.load(open(PRE, encoding="utf-8"))            # the pre-registration pins this runner, the stage-2 module, every import, cache and record it reads
    for mod, pn in pre["code"]["new_modules"].items():
        assert sha_file(os.path.join(HERE, mod)) == pn["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    if REPRO:
        assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
    else:
        assert z_meta["row_query_ids_sha256"] == pre["population"][z_meta["dataset"]]["row_query_ids_sha256 (the cache builder must reproduce this)"], "the transfer cache is not the pre-registered population"
rec16 = rec21 = rec24 = None
if REPRO:
    rec16 = json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % name), encoding="utf-8"))
    rec21 = json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % name), encoding="utf-8"))
    rec24 = json.load(open(os.path.join(OUT, "qmaxbal_A_%s.json" % name), encoding="utf-8"))
    assert rec24["code"]["sha256"] == PINNED["_l1c_qmaxbal.py"]

# ---------------------------------------------------------------- the pinned uL3 head: data, STRUCT adjacency, frozen beam on split A  [as in sections 14-24]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
T_ALL = time.time()
exec(head)                                                   # D, C, beams1, beams2 (the pinned beam on split A), expand_real, fused_order, seeds, hub ...
secs_pinned = secs.copy()
edges_pinned = edges_h.copy()
TOPP = int(D.base_rank.shape[1])

# ---------------------------------------------------------------- the two served channels + the streamed stage 2  ->  the QMAX_F0 ranking
t0q = time.time()
Cd = G.block_channel(D, D.d_ids)
Cs = G.block_channel(D, D.s_ids)
base_full = G.F0([Cd, Cs], npart)
assert (base_full[:, :TOPP] == D.base_rank).all(), "F0(Cd, Cs) does not reproduce the served base_rank"
qb_s = TB.blockmax_scores_streamed(D, log=G.log)
Cmax_s = np.argsort(-qb_s, axis=1, kind="stable").astype(np.int64)
rank_s = G.F0([Cd, Cs, Cmax_s], npart)
G.log("streamed stage 2 + QMAX_F0 ranking in %.0fs (process RSS %.0f MB)" % (time.time() - t0q, _ws_mb()))
stage2 = {"streamed": {"seconds": round(time.time() - t0q, 1), "block_rows": TB.BLOCK, "cq": TB.CQ,
                       "sha256_ranking": hashlib.sha256(Cmax_s.tobytes()).hexdigest(), "sha256_scores": hashlib.sha256(qb_s.tobytes()).hexdigest()}}
RANK = {"QMAX": rank_s}
if REPRO:
    t0p = time.time()
    D.E = D._E16[:].astype(np.float32)                       # the fp32 node matrix the pinned dense_blockmax_rank reads (what Data(dense_fp32=True) builds)
    Cmax_p = CAND.dense_blockmax_rank(D)
    qb_p = TB.blockmax_scores_pinned_path(D)
    D.E = None
    assert (np.argsort(-qb_p, axis=1, kind="stable").astype(np.int64) == Cmax_p).all(), "the inline pinned-path scores do not sort to the pinned function's ranking"
    rank_p = G.F0([Cd, Cs, Cmax_p], npart)
    RANK["QMAXP"] = rank_p
    seen_q = rec24["L1_GEOM_record"]["recorded_DEV_A_AGGREGATION"]
    out_q, allv_qp = D.eval_rank(rank_p, ARM_Q50, quiet=True, splits=("A",))
    oqa = out_q["A"]
    assert abs(oqa["ALL"] - seen_q["ALL"]) < 1e-9 and oqa["gained"] == seen_q["gained"] and oqa["lost"] == seen_q["lost"], (oqa, seen_q)
    ulp = TB.ulp_distance_max(qb_s, qb_p)
    absd = np.abs(qb_s.astype(np.float64) - qb_p.astype(np.float64))
    max_abs = float(absd.max())
    n_gt_1em6 = int((absd > 1e-6).sum())
    diff_entries = int((qb_s != qb_p).sum())
    same_rank = bool(np.array_equal(Cmax_s, Cmax_p))
    rows_top200 = int((rank_s[:, :TOPP] != rank_p[:, :TOPP]).any(axis=1).sum())
    rows_p50set = int(sum(1 for qi in range(nq) if set(rank_s[qi, :P_MAIN].tolist()) != set(rank_p[qi, :P_MAIN].tolist())))
    rows_p50setA = int(sum(1 for qi in rowsA if set(rank_s[qi, :P_MAIN].tolist()) != set(rank_p[qi, :P_MAIN].tolist())))
    rows_cmax = int((Cmax_s != Cmax_p).any(axis=1).sum())
    stage2["pinned"] = {"seconds": round(time.time() - t0p, 1), "sha256_ranking": hashlib.sha256(Cmax_p.tobytes()).hexdigest(),
                        "sha256_scores": hashlib.sha256(qb_p.tobytes()).hexdigest(), "fp32_matrix_bytes": int(N) * int(D.Q.shape[1]) * 4,
                        "QMAX_F0_reproduces_L1_GEOM_record": True}
    stage2["divergence_streamed_vs_pinned"] = {
        "stage2_ranking_bitwise_identical": same_rank,
        "score_entries_differing": diff_entries, "score_entries_total": int(qb_s.size),
        "max_abs_score_difference": max_abs, "score_entries_differing_by_more_than_1e-6": n_gt_1em6,
        "max_score_difference_in_fp32_ulps": ulp,
        "rows_with_a_different_stage2_order": rows_cmax,
        "rows_whose_first_%d_fused_QMAX_F0_blocks_differ" % TOPP: rows_top200,
        "rows_whose_fused_P50_SET_differs (all rows / split A)": [rows_p50set, rows_p50setA],
        "openblas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS")}
    G.log("stage-2 divergence: identical %s; %d / %d score entries differ (max |delta| %.3g, %d entries > 1e-6, max %.1f ulp); rows differing: stage-2 order %d, first-%d fused %d, P50 set %d (A %d)" % (
        same_rank, diff_entries, qb_s.size, max_abs, n_gt_1em6, ulp, rows_cmax, TOPP, rows_top200, rows_p50set, rows_p50setA))
    del qb_p
del qb_s
out_s, allv_qs = D.eval_rank(rank_s, ARM_Q50, quiet=True, splits=("A",))
allv_q = {"QMAX": np.asarray(allv_qs, bool)}
if REPRO:
    allv_q["QMAXP"] = np.asarray(allv_qp, bool)
G.log("QMAX_F0 (streamed) split A ALL %.4f +%d/-%d vs BASE %.4f; P50 overlap with BASE P50 mean %.1f blocks" % (
    out_s["A"]["ALL"], out_s["A"]["gained"], out_s["A"]["lost"], D.r_base["ALL_split"]["A"]["ALL"],
    float(np.mean([len(set(rank_s[qi, :P_MAIN].tolist()) & set(base_full[qi, :P_MAIN].tolist())) for qi in rowsA]))))


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
CAND_B = {}
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


beam_div = {"n_split_A": nA, "wall_clock_ms_per_query": {"pinned": round(1000 * float(secs_pinned.mean()), 1), "balanced": round(1000 * float(secs_bal.mean()), 1)},
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
if REPRO:
    bd21 = rec21["beam_diversity_DEV_A"]
    for k in bd21:
        if k not in ("wall_clock_ms_per_query", "n_DEV_A"):
            assert beam_div[k] == bd21[k], "balanced beam figure %s differs from the section-21 record" % k
    assert beam_div["n_split_A"] == bd21["n_DEV_A"]
    G.log("balanced beam reproduced: every deterministic figure of h2balanced_A_%s.json beam_diversity_DEV_A matches" % name)

# ---------------------------------------------------------------- canonical SAFE machinery (frozen code path) on the raw cache arrays, for every block ranking
z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
assert zz["base_rank"].shape == (nq, TOPP) and (zz["base_rank"] == D.base_rank).all()
RAW = {"BASE": zz}
for ch, rk in RANK.items():
    zq = dict(zz)
    zq["base_rank"] = rk[:, :TOPP].astype(np.int32)          # THE composition: the QMAX_F0 ranking enters the frozen SAFE machinery as the block ranking
    RAW[ch] = zq
CHAINS = list(RAW.keys())                                    # BASE, QMAX (streamed) [, QMAXP (pinned stage 2; reproduce mode)]
Cc = {ch: RT.build_cache(name, RAW[ch], meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]]) for ch in CHAINS}
ctxs = {ch: KB.contexts(RAW[ch], meta, Cc[ch], B, CFG) for ch in CHAINS}
goldp = Cc["BASE"]["goldp"]
for ch in CHAINS:
    assert Cc[ch]["goldp"] == goldp
ind_base = np.asarray(Cc["BASE"]["ind_base"], bool)
a0 = np.asarray(D.base_all, bool)
assert (ind_base[m] == a0[m]).all(), "SAFE machinery BASE differs from the lane's served BASE"
for ch in RANK:
    assert (np.asarray(Cc[ch]["ind_base"], bool)[m] == allv_q[ch][m]).all(), "SAFE machinery %s P50 differs from the lane's QMAX_F0 evaluation" % ch
for qi in rowsA:                                             # the challengers do not depend on the block ranking (seed-based structural expansion; ret_rrf continuation)
    for ch in RANK:
        assert ctxs["BASE"][qi]["spos"] == ctxs[ch][qi]["spos"] and ctxs["BASE"][qi]["rpos"] == ctxs[ch][qi]["rpos"]
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

P50_OF = {"BASE": "L1", "QMAX": ARM_Q50, "QMAXP": ARM_Q50 + "__pinned_stage2"}
SAFE_OF = {"BASE": "SAFE", "QMAX": ARM_QSAFE, "QMAXP": ARM_QSAFE + "__pinned_stage2"}
PRIMARY_OF = {"QMAX": PRIMARY, "QMAXP": PRIMARY + "__pinned_stage2"}
finals = {}
weakest_of = {ch: np.full(nq, -1, np.int64) for ch in CHAINS}
admitted = {ch: [None] * nq for ch in CHAINS}
for ch in CHAINS:
    finals[P50_OF[ch]] = [set() for _ in range(nq)]
    finals[SAFE_OF[ch]] = [set() for _ in range(nq)]
    for qi in rowsA:
        c = ctxs[ch][qi]
        finals[P50_OF[ch]][qi] = set(c["base50"])
        Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        safe50 = c["prot_set"] | set(Xs)
        assert len(safe50) == P_MAIN
        finals[SAFE_OF[ch]][qi] = safe50
        weakest_of[ch][qi] = int(Xs[-1])                     # the sixth block of the F6 competition = SAFE's weakest admission
        admitted[ch][qi] = [int(x) for x in Xs]
ind_L1 = np.array([int(goldp[qi] <= finals["L1"][qi]) for qi in range(nq)], bool)
ind_S = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
safe_check = "SAFE computed here by the frozen code path (no lane replay record given)"
if SAFE_REC_PATH:
    recS = json.load(open(SAFE_REC_PATH, encoding="utf-8"))
    assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path)
    assert (ind_L1[m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
    assert (ind_S[m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on split A rows"
    safe_check = "BASE and SAFE asserted equal (split A rows) to %s" % os.path.relpath(SAFE_REC_PATH, G.X.REPO).replace("\\", "/")
G.log("canonical SAFE %.4f, BASE %.4f on split A; %s" % (ind_S[m].mean(), ind_L1[m].mean(), safe_check))

# identities of the composed chain that involve no gold (reported in the dry run as well)
def identities(ch):
    same_safe = np.array([finals[SAFE_OF[ch]][qi] == finals["SAFE"][qi] for qi in rowsA])
    inter_safe = np.array([len(finals[SAFE_OF[ch]][qi] & finals["SAFE"][qi]) for qi in rowsA])
    inter_p50 = np.array([len(finals[P50_OF[ch]][qi] & finals["L1"][qi]) for qi in rowsA])
    same_weak = np.array([weakest_of[ch][qi] == weakest_of["BASE"][qi] for qi in rowsA])
    adm_over = np.array([len(set(admitted[ch][qi]) & set(admitted["BASE"][qi])) for qi in rowsA])
    chal_adm = {c_: np.array([sum(1 for x in admitted[c_][qi] if x not in finals[P50_OF[c_]][qi]) for qi in rowsA]) for c_ in ("BASE", ch)}
    expo = {arm: np.array([int(sizes[list(finals[arm][qi])].sum()) for qi in rowsA]) for arm in ("L1", "SAFE", P50_OF[ch], SAFE_OF[ch])}
    return {"QMAX_P50 == BASE_P50 blocks per query (mean of 50)": round(float(inter_p50.mean()), 2),
            "QMAX_SAFE == SAFE (queries)": int(same_safe.sum()), "QMAX_SAFE & SAFE blocks per query (mean of 50)": round(float(inter_safe.mean()), 2),
            "same weakest admitted block (queries)": int(same_weak.sum()), "admitted six overlap (mean of 6)": round(float(adm_over.mean()), 2),
            "challengers admitted from outside the chain's own P50 (mean of 6): BASE / QMAX": [round(float(chal_adm["BASE"].mean()), 2), round(float(chal_adm[ch].mean()), 2)],
            "nominal exposure mean nodes: L1 / SAFE / QMAX_F0 / QMAX_SAFE": [round(float(expo[a].mean()), 1) for a in ("L1", "SAFE", P50_OF[ch], SAFE_OF[ch])]}


exposure = {arm: np.array([int(sizes[list(finals[arm][qi])].sum()) for qi in rowsA]) for arm in finals}
ident = {ch: identities(ch) for ch in RANK}
for ch in RANK:
    G.log("composed-chain identities [%s] (no gold): %s" % (ch, json.dumps(ident[ch])))
if REPRO:
    for k, v in rec24["composed_chain_identities"].items():
        assert ident["QMAXP"][k] == v, "identity %s differs from the section-24 record under the pinned stage 2" % k
if DRY:
    G.log("DRY RUN: BASE, QMAX_F0, SAFE and both beams computed; |QMAX_SAFE| = 50 on every split-A row; no coverage number of any composed arm computed; no record (%.0fs)" % (time.time() - T_ALL))
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


patches = {ARM_H2: build_patches(beams1, beams2, "BASE"), ARM_BAL: build_patches(bal1, bal2, "BASE")}
CHAIN_OF = {"L1": "BASE", "SAFE": "BASE", ARM_H2: "BASE", ARM_BAL: "BASE"}
for ch in RANK:
    patches[PRIMARY_OF[ch]] = build_patches(bal1, bal2, ch)
    CHAIN_OF[P50_OF[ch]] = CHAIN_OF[SAFE_OF[ch]] = CHAIN_OF[PRIMARY_OF[ch]] = ch
ARMS = list(ARMS_BASE) + [a for ch in RANK for a in (P50_OF[ch], SAFE_OF[ch], PRIMARY_OF[ch])]


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
for ch in RANK:
    assert (ind[P50_OF[ch]][m] == allv_q[ch][m]).all()
for arm in patches:
    ch = CHAIN_OF[arm]
    for qi in rowsA:
        assert int(sizes[list(patches[arm][qi]["kept"])].sum()) + patches[arm][qi]["nb"] == exposure[SAFE_OF[ch]][np.searchsorted(rowsA, qi)]


def mc(a, b, s):
    g_, l_, p_ = G.X.mcnemar(a[s], b[s])
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
        safe_k = kname(SAFE_OF[ch])
        o["exposure"] = "matched: |served nodes| equal to %s per query (asserted)" % safe_k
        o["exposure_nodes_mean"] = round(float(exposure[SAFE_OF[ch]].mean()), 1)
        o["novel_nodes_per_query_mean"] = round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1)
        o["novel_hop1_per_query_mean"] = round(float(np.mean([patch[qi]["nov1"] for qi in rowsA])), 1)
        o["novel_hop2_per_query_mean"] = round(float(np.mean([patch[qi]["nov2"] for qi in rowsA])), 1)
        o["patch_size_mean"] = round(float(np.mean([patch[qi]["nb"] for qi in rowsA])), 1)
        o["patch_composition_mean (hop-1 novel / hop-2 novel / fill from removed block)"] = [round(float(np.mean([patch[qi][k] for qi in rowsA])), 1) for k in ("used_hop1", "used_hop2", "n_fill")]
        o["queries_truncated"] = int(sum(patch[qi]["truncated"] for qi in rowsA))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(patch[qi]["k"] == 0 for qi in rowsA))
        o["gold_nodes_gained_vs_%s" % safe_k] = int(gain_nodes[rowsA].sum())
        o["gold_nodes_gained_vs_%s_from_hop1_novel" % safe_k] = int(gain_hop1[rowsA].sum())
        o["queries_with_gold_node_gained_vs_%s" % safe_k] = int((gain_nodes[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss_nodes[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss_nodes[rowsA] > 0).sum())
    comps = []                                               # (key, reference arm) paired comparisons reported for this arm  [the section-24 set]
    if arm != "L1":
        comps.append(("vs_L1", "L1"))
    if arm in patches:
        comps.append(("vs_%s" % kname(SAFE_OF[ch]), SAFE_OF[ch]))
    if arm == ARM_BAL:
        comps.append(("vs_H2_PATCH1", ARM_H2))
    if ch != "BASE" and arm != P50_OF[ch]:
        comps.append(("vs_QMAX_F0", P50_OF[ch]))
    if ch != "BASE" and arm == SAFE_OF[ch]:
        comps.append(("vs_SAFE", "SAFE"))
    if ch != "BASE" and arm == PRIMARY_OF[ch]:
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
        sk = "ALL_vs_%s" % kname(SAFE_OF[ch])
        o["significant_loss_vs_%s" % kname(SAFE_OF[ch])] = bool(o[sk]["lost"] > o[sk]["gained"] and o[sk]["p"] < 0.05)
    if arm == ARM_BAL:
        o["significant_loss_vs_H2_PATCH1"] = bool(o["ALL_vs_H2_PATCH1"]["lost"] > o["ALL_vs_H2_PATCH1"]["gained"] and o["ALL_vs_H2_PATCH1"]["p"] < 0.05)
    if ch != "BASE" and arm == PRIMARY_OF[ch]:
        o["significant_loss_vs_SAFE"] = bool(o["ALL_vs_SAFE"]["lost"] > o["ALL_vs_SAFE"]["gained"] and o["ALL_vs_SAFE"]["p"] < 0.05)
    return o


# ---------------------------------------------------------------- per-query type (descriptive only; from the dataset's own eval split file, never TEST)
def query_types():
    try:
        d = D.cd
        sp = d.eval_split
        if sp not in d.split_ranges:
            return None
        path = os.path.join(d.dir, "queries", "%s.jsonl" % sp)
        if not os.path.exists(path):
            return None
        want = set(C.qids)
        ty = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if j.get("query_id") in want:
                    ty[j["query_id"]] = {k: str(j[k]) for k in ("type", "level") if k in j}
        keys = sorted(set(k for v in ty.values() for k in v))
        return {k: np.array([ty.get(q, {}).get(k, "?") for q in C.qids]) for k in keys} or None
    except Exception as e:      # descriptive only: never fail the run for it
        G.log("query types unavailable: %r" % (e,))
        return None


res = {"cache": name, "dataset": G.X.DS_OF[name], "partition": G.PARTITION_OF.get(name, meta.get("input_provider")), "n_split_A": nA, "n_cache_rows": nq,
       "mode": "REPRODUCE_DEV_A" if REPRO else "TRANSFER_ONE_SHOT_SPLIT_A",
       "status": "IDENTITY_TEST_STREAMED_STAGE2_DEV_A" if REPRO else "PREREGISTERED_TRANSFER_SPLIT_A",
       "arm": PRIMARY,
       "cache_file": {"path": os.path.relpath(C.path, G.X.REPO).replace("\\", "/"), "sha256": sha_file(C.path), "row_query_ids_sha256": meta.get("row_query_ids_sha256")},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__)),
                "stage2_module": {"path": "scratchpad/_l1c_transfer_blockmax.py", "sha256": sha_file(os.path.join(HERE, "_l1c_transfer_blockmax.py"))}},
       "pinned": PINNED, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_check": safe_check, "stage2": stage2,
       "definition": "FREEZE_QMAX_BALANCED_H2_PATCH1.json what_is_frozen (stages 1-7 verbatim; stage 2 by the streamed path of _l1c_transfer_blockmax.py)",
       "beam_diversity_split_A": beam_div, "composed_chain_identities": ident, "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-40s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm", "chain")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- stage increments of the chains
stages = {}
seqs = [("BASE", ("L1", "SAFE", ARM_BAL))] + [(ch, (P50_OF[ch], SAFE_OF[ch], PRIMARY_OF[ch])) for ch in RANK]
for ch, seq in seqs:
    st = {}
    for a_, b_ in zip(seq[:-1], seq[1:]):
        st["%s -> %s" % (a_, b_)] = {**mcl(ind[a_], ind[b_], m), "from": res["arms"][a_]["ALL"], "to": res["arms"][b_]["ALL"]}
    st["%s -> %s (chain)" % (seq[0], seq[-1])] = {**mcl(ind[seq[0]], ind[seq[-1]], m), "from": res["arms"][seq[0]]["ALL"], "to": res["arms"][seq[-1]]["ALL"]}
    stages[ch] = st
res["stage_increments_split_A"] = stages


# ---------------------------------------------------------------- flips PRIMARY vs BALANCED_H2_PATCH1 (and vs QMAX_F0) with their cause  [section 24 verbatim]
def flip_table(prim, ref, ch):
    f = {"newly_covered": {"n": 0, "gold_nodes_newly_served": 0, "of_which_served_by_a_kept_block_of_QMAX_SAFE": 0, "of_which_served_by_the_patch_nodes": 0},
         "newly_lost": {"n": 0, "gold_nodes_dropped": 0, "of_which_block_no_longer_kept_(QMAX_SAFE_kept_49_differ)": 0, "of_which_dropped_with_the_weakest_block": 0,
                        "of_which_visited_but_beyond_the_patch_capacity": 0, "of_which_not_visited_by_the_balanced_beam": 0}}
    for qi in rowsA:
        pP = patches[prim][qi]
        if ind[prim][qi] and not ind[ref][qi]:
            e = f["newly_covered"]
            e["n"] += 1
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(prim, qi, g) and not served_by(ref, qi, g):
                    e["gold_nodes_newly_served"] += 1
                    e["of_which_served_by_a_kept_block_of_QMAX_SAFE"] += int(int(hard[g]) in pP["kept"])
                    e["of_which_served_by_the_patch_nodes"] += int(int(hard[g]) not in pP["kept"] and g in pP["nodes"])
        elif ind[ref][qi] and not ind[prim][qi]:
            e = f["newly_lost"]
            e["n"] += 1
            vis = pP["novel_set"] | set(bal1[qi]) | set(bal2[qi])
            for g in gold_nodes[qi]:
                g = int(g)
                if served_by(ref, qi, g) and not served_by(prim, qi, g):
                    e["gold_nodes_dropped"] += 1
                    b = int(hard[g])
                    if b == pP["weakest"]:
                        e["of_which_dropped_with_the_weakest_block"] += 1
                    elif b not in finals[SAFE_OF[ch]][qi]:
                        e["of_which_block_no_longer_kept_(QMAX_SAFE_kept_49_differ)"] += 1
                    if g in vis and pP["novel_pos"].get(g, -1) >= pP["nb"]:
                        e["of_which_visited_but_beyond_the_patch_capacity"] += 1
                    if g not in vis:
                        e["of_which_not_visited_by_the_balanced_beam"] += 1
    return f


res["attribution_flips"] = {"PRIMARY_vs_BALANCED_H2_PATCH1": flip_table(PRIMARY, ARM_BAL, "QMAX"), "PRIMARY_vs_QMAX_F0": flip_table(PRIMARY, ARM_Q50, "QMAX")}


def cell_of(o):
    vB = o["ALL_vs_BALANCED_H2_PATCH1"]["label"]
    vQ = o["ALL_vs_QMAX_F0"]["label"]
    if "LOSS" in (vB, vQ):
        cell = "CONFLICT"
    elif vB == "GAIN" and vQ == "GAIN":
        cell = "STACKS"
    elif "GAIN" in (vB, vQ):
        cell = "CARRIES"
    else:
        cell = "NEUTRAL"
    return {"PRIMARY_vs_BALANCED_H2_PATCH1": vB, "PRIMARY_vs_QMAX_F0": vQ, "cell": cell,
            "cell_rule": "section 24: CONFLICT if LOSS vs either parent; STACKS if GAIN vs both; CARRIES if GAIN vs one and NEUTRAL vs the other; NEUTRAL otherwise (alpha 0.01, exact McNemar)"}


res["section24_cell_split_A"] = cell_of(res["arms"][PRIMARY])

types = query_types()
if types is not None:
    bt = {}
    for field, tarr in types.items():
        bt[field] = {}
        for t in sorted(set(tarr[m].tolist())):
            s = m & (tarr == t)
            bt[field][t] = {"n": int(s.sum()), **{arm: round(float(ind[arm][s].mean()), 4) for arm in ("L1", "SAFE", ARM_BAL, ARM_Q50, ARM_QSAFE, PRIMARY)},
                            "PRIMARY_vs_SAFE": mc(ind["SAFE"], ind[PRIMARY], s), "PRIMARY_vs_L1": mc(ind["L1"], ind[PRIMARY], s)}
    res["by_query_type_split_A (descriptive only)"] = bt

if REPRO:
    # ------------------------------------------------------------ the BASE chain and the pinned-stage-2 QMAX chain must reproduce the section-16 / 21 / 24 records exactly
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
    assert res["arms"]["SAFE"]["ALL"] == rec24["arms"]["SAFE"]["ALL"] and res["arms"]["L1"]["ALL"] == rec24["arms"]["L1"]["ALL"]
    for arm_rec, arm_here in ((ARM_Q50, P50_OF["QMAXP"]), (ARM_QSAFE, SAFE_OF["QMAXP"]), (PRIMARY, PRIMARY_OF["QMAXP"])):
        o, r = res["arms"][arm_here], rec24["arms"][arm_rec]
        assert o["ALL"] == r["ALL"] and o["ANY"] == r["ANY"], (arm_rec, o["ALL"], r["ALL"])
        for key in r:
            if key.startswith("ALL_vs_") or key.startswith("ANY_vs_"):
                assert (o[key]["gained"], o[key]["lost"]) == (r[key]["gained"], r[key]["lost"]), (arm_rec, key)
        for h in r["per_hop"]:
            assert o["per_hop"][h]["ALL"] == r["per_hop"][h]["ALL"], (arm_rec, h)
        for key in ("novel_nodes_per_query_mean", "queries_truncated", "patch_size_mean", "gold_nodes_gained_vs_QMAX_SAFE", "gold_nodes_lost_with_removed_block"):
            if key in r:
                assert o[key] == r[key], (arm_rec, key)
    assert cell_of(res["arms"][PRIMARY_OF["QMAXP"]])["cell"] == rec24["per_cache_outcome"]["cell"]
    res["reference_reproduced"] = ("BASE chain L1 / SAFE / H2_PATCH1 / BALANCED_H2_PATCH1 == h2patch_A / h2balanced_A / qmaxbal_A records; "
                                   "QMAX chain with the PINNED stage 2 == qmaxbal_A_%s.json (QMAX_F0, QMAX_SAFE, PRIMARY: ALL, ANY, every paired comparison, per hop, novel counts, patch size, gold accounting, the section-24 cell)" % name)
    G.log("reference reproduced: %s" % res["reference_reproduced"])
    # ------------------------------------------------------------ the streamed chain vs the pinned chain (descriptive; the identity test's coverage-level figures)
    cmp_ = {}
    for arm_s, arm_p in ((ARM_Q50, P50_OF["QMAXP"]), (ARM_QSAFE, SAFE_OF["QMAXP"]), (PRIMARY, PRIMARY_OF["QMAXP"])):
        g_, l_, p_ = G.X.mcnemar(ind[arm_p][m], ind[arm_s][m])
        cmp_[arm_s] = {"ALL_pinned_stage2": res["arms"][arm_p]["ALL"], "ALL_streamed_stage2": res["arms"][arm_s]["ALL"],
                       "rows_flipped (streamed covers, pinned not / pinned covers, streamed not)": [g_, l_], "p": p_,
                       "identical_indicator_vectors_split_A": bool((ind[arm_p][m] == ind[arm_s][m]).all())}
    same_final = {SAFE_OF["QMAX"]: int(sum(1 for qi in rowsA if finals[SAFE_OF["QMAX"]][qi] == finals[SAFE_OF["QMAXP"]][qi])),
                  P50_OF["QMAX"]: int(sum(1 for qi in rowsA if finals[P50_OF["QMAX"]][qi] == finals[P50_OF["QMAXP"]][qi]))}
    same_patch = int(sum(1 for qi in rowsA if patches[PRIMARY][qi]["nodes"] == patches[PRIMARY_OF["QMAXP"]][qi]["nodes"] and patches[PRIMARY][qi]["kept"] == patches[PRIMARY_OF["QMAXP"]][qi]["kept"]))
    res["streamed_vs_pinned_stage2"] = {"arms": cmp_, "rows_with_identical_served_set (QMAX_F0 P50 / QMAX_SAFE) of %d" % nA: same_final,
                                        "rows_with_identical_PRIMARY_served_set": same_patch,
                                        "section24_cell_streamed": res["section24_cell_split_A"]["cell"], "section24_cell_pinned_record": rec24["per_cache_outcome"]["cell"]}
    G.log("streamed vs pinned stage 2: %s" % json.dumps(res["streamed_vs_pinned_stage2"]))
    # ------------------------------------------------------------ the pre-registered acceptance gate of the streamed stage 2 (PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json)
    flips_max = max(2, int(round(0.005 * nA)))
    fl = {a: sum(cmp_[a]["rows_flipped (streamed covers, pinned not / pinned covers, streamed not)"]) for a in cmp_}
    gate = {"g1_max_abs_score_difference_le_1e-5": bool(max_abs <= 1e-5), "max_abs_score_difference": max_abs,
            "g2_split_A_rows_flipped_le_%d (0.5 %% of %d, min 2): QMAX_F0 / QMAX_SAFE / PRIMARY" % (flips_max, nA): [fl[ARM_Q50], fl[ARM_QSAFE], fl[PRIMARY]],
            "g2_pass": bool(fl[ARM_Q50] <= flips_max and fl[PRIMARY] <= flips_max),
            "g3_pinned_chain_reproduces_the_records": True}
    gate["STREAMED_STAGE2_ACCEPTED_ON_THIS_CACHE"] = bool(gate["g1_max_abs_score_difference_le_1e-5"] and gate["g2_pass"] and gate["g3_pinned_chain_reproduces_the_records"])
    res["streamed_stage2_gate"] = gate
    G.log("streamed stage-2 gate: %s" % json.dumps(gate))
else:
    # ------------------------------------------------------------ the pre-registered transfer outcome of this cache (split A, one shot)
    o = res["arms"][PRIMARY]
    vS, vL = o["ALL_vs_SAFE"]["label"], o["ALL_vs_L1"]["label"]
    sig_loss = bool(o["significant_loss_vs_SAFE"])
    if vS == "LOSS" or sig_loss:
        outcome = "TRANSFER_FAIL"
    elif vS == "GAIN" and vL == "GAIN":
        outcome = "TRANSFER_PASS"
    else:
        outcome = "TRANSFER_NEUTRAL"
    res["per_cache_outcome"] = {"PRIMARY_vs_SAFE": vS, "PRIMARY_vs_L1": vL, "significant_loss_vs_SAFE_0.05": sig_loss, "outcome": outcome,
                                "rule": "PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json: PASS iff PRIMARY vs canonical SAFE is GAIN (alpha 0.01, exact McNemar, gained > lost) AND vs L1 is GAIN AND no significant loss vs SAFE (0.05); FAIL iff LOSS vs SAFE or a significant loss; NEUTRAL otherwise"}
    G.log("per-cache outcome %s" % json.dumps(res["per_cache_outcome"]))
res["seconds"] = round(time.time() - T_ALL, 1)
res["process_rss_mb_at_end"] = round(_ws_mb(), 1)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
