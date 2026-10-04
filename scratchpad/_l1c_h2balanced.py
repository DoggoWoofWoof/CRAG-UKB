"""BALANCED_H2_PATCH1 (twelfth ruling of 2026-09-15; pre-registered in PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json, STATUS
POSTHOC_MECHANISM_TEST): the frozen H2_PATCH1 of section 16 with ONE change -- the beam's frontier compression.

The pinned uL3 beam (_l1c_microl3.py, sha 19774617...) keeps, at every hop, the first BEAM = K_LOCK = 100 distinct unvisited targets of
the frozen fused transition order (a GLOBAL top-100 over the union of all parents' candidates).  The balanced beam keeps the same
candidate generation (frozen admissibility, frozen transition geometry S_dst / S_dir / S_mid, frozen RRF fused order), the same width
and depth, and replaces only the truncation by a parameter-free per-parent rank round-robin:
    within-parent rank r(m, u) = position of the transition m -> u among parent m's transitions in the frozen fused order (a target reached
                                 twice from the same parent keeps its first occurrence);
    sequence            = all (parent, target) pairs sorted by (r, position of the parent in the frontier order)  == A1 B1 C1 ... A2 B2 C2 ...
    B(h)                = the first BEAM distinct unvisited targets of that sequence (a target already taken by an earlier parent is skipped).
No local K, no weight, no dataset constant: the only constants are the frozen BEAM / DEPTH / K0 of the pinned beam and the frozen patch
budget.  Everything downstream is byte-identical in logic to section 16: SAFE (frozen F6), the weakest admitted block replaced by one
query-local virtual block of the same size filled with the beam's visited nodes that the 49 kept blocks do not expose (frozen first-visit
order = the beam order), topped up from the removed block; matched unique-node exposure.  Arms: L1, SAFE, H2_PATCH1 (pinned beam; must
reproduce the section-16 record exactly), BALANCED_H2_PATCH1 (balanced beam).  Metric: ALL-gold at matched exposure.
    python -u _l1c_h2balanced.py <cache>      -> results/L1_COVPART/h2balanced_A_<cache>.json
    python -u _l1c_h2balanced.py --summary    -> results/L1_COVPART/h2balanced_A_SUMMARY.json  (applies the pre-registered rule)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d"}
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
ARM0 = "H2_PATCH1"
ARM = "BALANCED_H2_PATCH1"
ALPHA = 0.01


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def summary():
    recs = {c: json.load(open(os.path.join(OUT, "h2balanced_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
    text = {}
    for c in ("squad", "squad_phg", "musique"):
        a = recs[c]["arms"][ARM]
        text[c] = {"loss_vs_H2_PATCH1": a["significant_loss_vs_H2_PATCH1"], "loss_vs_SAFE": a["significant_loss_vs_SAFE"],
                   "vs_H2_PATCH1": a["ALL_vs_H2_PATCH1"], "vs_SAFE": a["ALL_vs_SAFE"], "ok": not (a["significant_loss_vs_H2_PATCH1"] or a["significant_loss_vs_SAFE"])}
    mq = {}
    for c in ("metaqa", "metaqa_phg"):
        a = recs[c]["arms"][ARM]
        h2 = a["per_hop"]["hop2"]["vs_H2_PATCH1"]
        reg = {}
        for h in ("hop1", "hop3"):
            x = a["per_hop"][h]["vs_H2_PATCH1"]
            reg[h] = {"gained": x["gained"], "lost": x["lost"], "p": x["p"], "regression": bool(x["lost"] > x["gained"] and x["p"] < 0.05)}
        mq[c] = {"hop2_vs_H2_PATCH1": h2, "hop2_gain": bool(h2["gained"] > h2["lost"] and h2["p"] < ALPHA), "hop1_hop3": reg,
                 "no_regression": not any(v["regression"] for v in reg.values()), "ALL_vs_H2_PATCH1": a["ALL_vs_H2_PATCH1"]}
    gain_ok = all(v["hop2_gain"] for v in mq.values())
    reg_ok = all(v["no_regression"] for v in mq.values())
    text_ok = all(v["ok"] for v in text.values())
    verdict = "PASS" if (gain_ok and reg_ok and text_ok) else "FAIL"
    if verdict == "PASS":
        label = "MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; not promoted)"
    elif not gain_ok:
        label = "NOT_CONFIRMED: no hop-2 gain over the frozen H2_PATCH1 on both metaqa caches"
    elif not text_ok:
        label = "NOT_CONFIRMED: text safety violated"
    else:
        label = "NOT_CONFIRMED: hop-1 / hop-3 regression on a metaqa cache"
    S = {"RECORD": "BALANCED_H2_PATCH1_DEV_A_SUMMARY", "STATUS": "POSTHOC_MECHANISM_TEST",
         "rule": "hop-2 ALL gain vs the frozen H2_PATCH1 with gained > lost and p < 0.01 on BOTH metaqa caches AND text safety on squad / squad_phg / "
                 "musique (no significant loss vs H2_PATCH1 and vs SAFE: not (lost > gained and p < 0.05)) AND no hop-1 / hop-3 regression vs H2_PATCH1 "
                 "on either metaqa cache (not (lost > gained and p < 0.05))",
         "metaqa": mq, "text_safety": text, "components": {"hop2_gain_both_metaqa": gain_ok, "no_hop1_hop3_regression": reg_ok, "text_safe": text_ok},
         "verdict": verdict, "label": label,
         "per_cache": {c: {"L1": recs[c]["arms"]["L1"]["ALL"], "SAFE": recs[c]["arms"]["SAFE"]["ALL"], ARM0: recs[c]["arms"][ARM0]["ALL"], ARM: recs[c]["arms"][ARM]["ALL"],
                           ARM + "_vs_" + ARM0: recs[c]["arms"][ARM]["ALL_vs_H2_PATCH1"], ARM + "_vs_SAFE": recs[c]["arms"][ARM]["ALL_vs_SAFE"],
                           "beam_diversity": recs[c]["beam_diversity_DEV_A"], "availability": recs[c].get("diagnostic_node_level_availability")}
                       for c in CACHES}}
    G.S.wj(os.path.join(OUT, "h2balanced_A_SUMMARY.json"), S)
    G.log("SUMMARY verdict %s (%s) gain_ok %s reg_ok %s text_ok %s" % (verdict, label, gain_ok, reg_ok, text_ok))


if "--summary" in sys.argv:
    summary()
    sys.exit(0)

name = sys.argv[1]
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
PREREG = os.path.join(OUT, "PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json")
_pre = json.load(open(PREREG, encoding="utf-8"))
assert _pre["code"]["this_module"]["sha256"] == sha_file(os.path.abspath(__file__)), "this module changed since the pre-registration"
for _mod, _pn in _pre["code"]["pinned_modules"].items():
    assert sha_file(os.path.join(G.X.REPO, _pn["path"])) == _pn["sha256"], "%s changed since the pre-registration" % _mod
assert not os.path.exists(os.path.join(OUT, "h2balanced_A_%s.json" % name)), "write-once"
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
T_ALL = time.time()
exec(head)                                                   # D, C, beams1, beams2 (the pinned beam on DEV_A), expand_real, fused_order, seeds, hub ...
secs_pinned = secs.copy()
edges_pinned = edges_h.copy()


# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording (must equal beams1 / beams2)
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


# ---------------------------------------------------------------- the balanced compression (the ONLY change)
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
G.log("beam diversity %s" % json.dumps(beam_div))

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


patches = {ARM0: build_patches(beams1, beams2), ARM: build_patches(bal1, bal2)}


def served_by(arm, qi, g):
    b = int(hard[g])
    if arm == "L1":
        return b in finals["L1"][qi]
    if arm == "SAFE":
        return b in finals["SAFE"][qi]
    p = patches[arm][qi]
    return b in p["kept"] or g in p["nodes"]


ARMS = ("L1", "SAFE", ARM0, ARM)
ind = {arm: np.zeros(nq, bool) for arm in ARMS}
ind_any = {arm: np.zeros(nq, bool) for arm in ARMS}
for arm in ARMS:
    for qi in rowsA:
        flags = [served_by(arm, qi, int(g)) for g in gold_nodes[qi]]
        ind[arm][qi] = bool(flags) and all(flags)
        ind_any[arm][qi] = any(flags)
assert (ind["L1"][m] == ind_base[m]).all()
ind_safe_block = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
assert (ind["SAFE"][m] == ind_safe_block[m]).all()
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path)
assert (ind["L1"][m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind["SAFE"][m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced (SAFE %.4f, BASE %.4f)" % (ind["SAFE"][m].mean(), ind["L1"][m].mean()))


def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


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


acct = {arm: gold_accounting(arm) for arm in (ARM0, ARM)}


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
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        o["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "ALL": round(float(iv[s].mean()), 4), "ANY": round(float(ia[s].mean()), 4)}
        if arm != "L1":
            o["per_hop"]["hop%d" % h]["vs_L1"] = mc(ind["L1"], iv, s)
        if arm in patches:
            o["per_hop"]["hop%d" % h]["vs_SAFE"] = mc(ind["SAFE"], iv, s)
            o["per_hop"]["hop%d" % h]["gold_nodes_gained"] = int(acct[arm][0][s].sum())
            o["per_hop"]["hop%d" % h]["gold_nodes_lost_with_removed_block"] = int(acct[arm][2][s].sum())
        if arm == ARM:
            o["per_hop"]["hop%d" % h]["vs_H2_PATCH1"] = mc(ind[ARM0], iv, s)
    if arm != "L1":
        o["ALL_vs_L1"] = mc(ind["L1"], iv, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_L1"] = {"n": int(s.sum()), **mc(ind["L1"], iv, s), "pooled_L1": round(float(ind["L1"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm in patches:
        o["ALL_vs_SAFE"] = mc(ind["SAFE"], iv, m)
        o["ANY_vs_SAFE"] = mc(ind_any["SAFE"], ia, m)
        o["significant_loss_vs_SAFE"] = bool(o["ALL_vs_SAFE"]["lost"] > o["ALL_vs_SAFE"]["gained"] and o["ALL_vs_SAFE"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_SAFE"] = {"n": int(s.sum()), **mc(ind["SAFE"], iv, s), "pooled_SAFE": round(float(ind["SAFE"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm == ARM:
        o["ALL_vs_H2_PATCH1"] = mc(ind[ARM0], iv, m)
        o["ANY_vs_H2_PATCH1"] = mc(ind_any[ARM0], ia, m)
        o["significant_loss_vs_H2_PATCH1"] = bool(o["ALL_vs_H2_PATCH1"]["lost"] > o["ALL_vs_H2_PATCH1"]["gained"] and o["ALL_vs_H2_PATCH1"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_H2_PATCH1"] = {"n": int(s.sum()), **mc(ind[ARM0], iv, s), "pooled_H2_PATCH1": round(float(ind[ARM0][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "POSTHOC_MECHANISM_TEST", "arm": ARM,
       "preregistration": {"path": os.path.relpath(PREREG, G.X.REPO).replace("\\", "/"), "sha256": sha_file(PREREG)},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__))},
       "pinned": PINNED, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name],
       "definition": {"beam": "pinned uL3 beam (frozen admissibility, transition geometry, RRF fused order, BEAM = 100, DEPTH = 2, seeds = frozen top-5)",
                      "compression_pinned": "first 100 distinct unvisited targets in the global fused order",
                      "compression_balanced": "within-parent rank = position of (parent, target) among the parent's pairs in the frozen fused order (first occurrence per pair); "
                                              "sequence sorted by (rank, frontier position of the parent) = A1 B1 C1 ... A2 B2 C2 ...; first 100 distinct unvisited targets; no other constant",
                      "patch": "section 16 verbatim: SAFE minus its weakest admitted block + one query-local block of the same size = the beam's visited nodes not exposed by the 49 kept "
                               "blocks in first-visit order (hop 1 then hop 2, each in beam order), topped up from the removed block (ret_rrf position, then node id)",
                      "metric": "ALL-gold under matched unique-node exposure; node-level == block-level for L1 / SAFE (asserted)"},
       "beam_diversity_DEV_A": beam_div, "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-19s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- H2_PATCH1 must reproduce the section-16 record exactly
rec16 = json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % name), encoding="utf-8"))["arms"][ARM0]
o16 = res["arms"][ARM0]
assert o16["ALL"] == rec16["ALL"] and o16["ANY"] == rec16["ANY"], (o16["ALL"], rec16["ALL"])
assert (o16["ALL_vs_SAFE"]["gained"], o16["ALL_vs_SAFE"]["lost"]) == (rec16["ALL_vs_SAFE"]["gained"], rec16["ALL_vs_SAFE"]["lost"])
for h in o16["per_hop"]:
    assert o16["per_hop"][h]["ALL"] == rec16["per_hop"][h]["ALL"], h
    assert (o16["per_hop"][h]["vs_SAFE"]["gained"], o16["per_hop"][h]["vs_SAFE"]["lost"]) == (rec16["per_hop"][h]["vs_SAFE"]["gained"], rec16["per_hop"][h]["vs_SAFE"]["lost"]), h
assert o16["novel_nodes_per_query_mean"] == rec16["novel_nodes_per_query_mean"] and o16["queries_truncated"] == rec16["queries_truncated"]
res["section16_reproduced"] = "H2_PATCH1 rebuilt == h2patch_A_%s.json (ALL, ANY, vs SAFE, per hop, novel counts)" % name
G.log("section 16 reproduced: %s" % res["section16_reproduced"])

# ---------------------------------------------------------------- flips BALANCED vs H2_PATCH1 with their cause
flips = {"newly_covered_vs_H2_PATCH1": {"n": 0, "gold_nodes_newly_served": 0, "of_which_newly_visited_by_the_balanced_beam": 0, "of_which_visited_by_both_beams_but_cut_by_the_pinned_patch": 0},
         "newly_lost_vs_H2_PATCH1": {"n": 0, "gold_nodes_dropped": 0, "of_which_no_longer_visited_by_the_balanced_beam": 0, "of_which_visited_but_beyond_the_patch_capacity": 0}}
for qi in rowsA:
    pP, pB = patches[ARM0][qi], patches[ARM][qi]
    visP = pP["novel_set"] | set(beams1[qi]) | set(beams2[qi])
    visB = pB["novel_set"] | set(bal1[qi]) | set(bal2[qi])
    if ind[ARM][qi] and not ind[ARM0][qi]:
        f = flips["newly_covered_vs_H2_PATCH1"]
        f["n"] += 1
        for g in gold_nodes[qi]:
            g = int(g)
            if served_by(ARM, qi, g) and not served_by(ARM0, qi, g):
                f["gold_nodes_newly_served"] += 1
                f["of_which_newly_visited_by_the_balanced_beam"] += int(g in visB and g not in visP)
                f["of_which_visited_by_both_beams_but_cut_by_the_pinned_patch"] += int(g in visB and g in visP)
    elif ind[ARM0][qi] and not ind[ARM][qi]:
        f = flips["newly_lost_vs_H2_PATCH1"]
        f["n"] += 1
        for g in gold_nodes[qi]:
            g = int(g)
            if served_by(ARM0, qi, g) and not served_by(ARM, qi, g):
                f["gold_nodes_dropped"] += 1
                f["of_which_no_longer_visited_by_the_balanced_beam"] += int(g not in visB)
                f["of_which_visited_but_beyond_the_patch_capacity"] += int(g in visB)
res["attribution_flips_vs_H2_PATCH1"] = flips
G.log("flips %s" % json.dumps(flips))

# ---------------------------------------------------------------- THE DIAGNOSTIC: the section-17 population (missing gold nodes scored within depth 2) and node-level availability
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
diag = {}
for gname, rows_h in groups:
    o = {"SAFE_failures": 0, "missing_gold_nodes": 0, "missing_is_seed (skipped exactly as in section 17)": 0,
         "pinned: visited (hop1 / hop2)": [0, 0], "pinned: scored candidate but cut by the width (first at hop1 / hop2)": [0, 0], "pinned: never scored": 0,
         "balanced: visited (hop1 / hop2)": [0, 0], "balanced: scored candidate but not kept (first at hop1 / hop2)": [0, 0], "balanced: never scored": 0,
         "failures_with_all_missing_nodes_scored_or_visited_within_depth2 (pinned; section 17 population)": 0,
         "  of_which_all_missing_nodes_VISITED_by_pinned_beam": 0, "  of_which_all_missing_nodes_VISITED_by_balanced_beam": 0,
         "  of_which_covered_by_H2_PATCH1": 0, "  of_which_covered_by_BALANCED_H2_PATCH1": 0,
         "failures_with_all_missing_nodes_visited_by_balanced_beam_and_NOT_covered_by_BALANCED: cut_by_patch_capacity": 0,
         "failures_with_all_missing_nodes_visited_by_balanced_beam_and_NOT_covered_by_BALANCED: gold_dropped_with_removed_block": 0,
         "candidates_unique_per_query_mean (hop1 / hop2) pinned": [0.0, 0.0], "candidates_unique_per_query_mean (hop1 / hop2) balanced": [0.0, 0.0]}
    cnt_p, cnt_b = np.zeros(2), np.zeros(2)
    for qi in rows_h:
        for h in range(2):
            cnt_p[h] += len(CAND_P[(qi, h)])
            cnt_b[h] += len(CAND_B[(qi, h)])
        if ind["SAFE"][qi] or not len(gold_nodes[qi]):
            continue
        miss = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
        o["SAFE_failures"] += 1
        o["missing_gold_nodes"] += len(miss)
        s1p, s2p = set(beams1[qi]), set(beams2[qi])
        s1b, s2b = set(bal1[qi]), set(bal2[qi])
        c1p, c2p = set(CAND_P[(qi, 0)].tolist()), set(CAND_P[(qi, 1)].tolist())
        c1b, c2b = set(CAND_B[(qi, 0)].tolist()), set(CAND_B[(qi, 1)].tolist())
        all_scored_p = True
        all_vis_p = True
        all_vis_b = True
        seeds_q = set(int(x) for x in seeds[qi] if x >= 0)
        for g in miss:
            if g in seeds_q:                                     # a missing gold node that is a frozen seed: section 17 skips it (neither scored nor visited by a hop)
                o["missing_is_seed (skipped exactly as in section 17)"] += 1
                continue
            if g in s1p:
                o["pinned: visited (hop1 / hop2)"][0] += 1
            elif g in s2p:
                o["pinned: visited (hop1 / hop2)"][1] += 1
            elif g in c1p:
                o["pinned: scored candidate but cut by the width (first at hop1 / hop2)"][0] += 1
                all_vis_p = False
            elif g in c2p:
                o["pinned: scored candidate but cut by the width (first at hop1 / hop2)"][1] += 1
                all_vis_p = False
            else:
                o["pinned: never scored"] += 1
                all_scored_p = False
                all_vis_p = False
            if g in s1b:
                o["balanced: visited (hop1 / hop2)"][0] += 1
            elif g in s2b:
                o["balanced: visited (hop1 / hop2)"][1] += 1
            elif g in c1b:
                o["balanced: scored candidate but not kept (first at hop1 / hop2)"][0] += 1
                all_vis_b = False
            elif g in c2b:
                o["balanced: scored candidate but not kept (first at hop1 / hop2)"][1] += 1
                all_vis_b = False
            else:
                o["balanced: never scored"] += 1
                all_vis_b = False
        if all_scored_p:
            o["failures_with_all_missing_nodes_scored_or_visited_within_depth2 (pinned; section 17 population)"] += 1
            o["  of_which_all_missing_nodes_VISITED_by_pinned_beam"] += int(all_vis_p)
            o["  of_which_all_missing_nodes_VISITED_by_balanced_beam"] += int(all_vis_b)
            o["  of_which_covered_by_H2_PATCH1"] += int(ind[ARM0][qi])
            o["  of_which_covered_by_BALANCED_H2_PATCH1"] += int(ind[ARM][qi])
        if all_vis_b and not ind[ARM][qi]:
            pB = patches[ARM][qi]
            if any(pB["novel_pos"].get(g, -1) >= pB["nb"] for g in miss):
                o["failures_with_all_missing_nodes_visited_by_balanced_beam_and_NOT_covered_by_BALANCED: cut_by_patch_capacity"] += 1
            elif acct[ARM][2][qi] > 0:
                o["failures_with_all_missing_nodes_visited_by_balanced_beam_and_NOT_covered_by_BALANCED: gold_dropped_with_removed_block"] += 1
    nh = max(1, len(rows_h))
    o["candidates_unique_per_query_mean (hop1 / hop2) pinned"] = [round(float(x / nh), 1) for x in cnt_p]
    o["candidates_unique_per_query_mean (hop1 / hop2) balanced"] = [round(float(x / nh), 1) for x in cnt_b]
    diag[gname] = o
    G.log("diag %s %s" % (gname, json.dumps(o)))
res["diagnostic_node_level_availability"] = diag
# cross-check the section-17 attribution record (depth-3 beam; hops 1-2 identical to the pinned depth-2 beam) where it exists
cp = os.path.join(OUT, "h3patch_why_A_%s.json" % name)
if os.path.exists(cp):
    g17 = json.load(open(cp, encoding="utf-8"))["groups"]
    for gname in diag:
        if gname in g17:
            assert diag[gname]["SAFE_failures"] == g17[gname]["SAFE_failures"] and diag[gname]["missing_gold_nodes"] == g17[gname]["missing_gold_nodes"], gname
            assert diag[gname]["failures_with_all_missing_nodes_scored_or_visited_within_depth2 (pinned; section 17 population)"] == g17[gname]["failures_where_all_missing_nodes_are_scored_or_visited_within_depth2"], gname
    res["section17_crosscheck"] = "SAFE failures, missing gold nodes and the within-depth-2 population reproduced against h3patch_why_A_%s.json" % name
    G.log(res["section17_crosscheck"])

res["seconds"] = round(time.time() - T_ALL, 1)
G.S.wj(os.path.join(OUT, "h2balanced_A_%s.json" % name), res)
G.log("done (%.0fs)" % res["seconds"])
