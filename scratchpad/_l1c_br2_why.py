"""BOUNDED_VERTEXCUT_R2 post-hoc DIAGNOSTIC (metaqa DEV_A; written AFTER the pre-registered cells were seen; NOT a cell, no coverage
number for any new serving rule, nothing promoted): why does the R = 2 substrate collapse on MetaQA (R2_P50 0.333 vs the served hard
BASE 0.640 and V1 0.678) while it is neutral-and-cheaper on the text graphs?

Hypothesis read off the records: the served hard BASE routes through the LEGACY table (own block + blocks of the directed STRUCT
out-neighbours = a static 1-hop vote spread; own-block-only gives 0.18), and the unbounded vertex-cut V1 reproduces that spread by
construction (a node's memberships are the blocks of ALL its incident edges), whereas R = 2 caps the spread at two blocks.  So the
quantity to measure is REACH: for each membership table, how many blocks a top-100 hit votes for, and whether every gold node of a
query lies in at least one block that receives any vote (evidence).  Coverage cannot exceed reach.

Two different spreads are in play: the VOTE spread (a hit votes for the blocks of its out-neighbours: the legacy table) and the
MEMBERSHIP spread (a node is present in, and served through, several blocks: the vertex-cut, up to deg(v) blocks for a hub under V1,
two under R2).  38 % of MetaQA's gold nodes are hubs (degree > 100), so the second one matters there.

Tables: HARD_LEGACY (the served BASE's table), HARD_OWN (own block only), V1_k1004 (section 19), R2_k864 (this ruling), OWNERS_ONLY_k864
(R = 1 control), and -- reach only -- R2_k864 + the legacy out-neighbour spread (each hit also votes for the R2 memberships of its
directed out-neighbours), the natural follow-up that is NOT evaluated as a coverage cell here.

    python -u scratchpad/_l1c_br2_why.py   -> results/L1_COVPART/br2_why_A_metaqa.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core as G  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_vcut_replay as R  # noqa: E402
import _l1c_br2_lib as L  # noqa: E402

X = S.X
K_LOCK = S.K_LOCK
OUT = V.OUT
log = S.log
sha_file, pin = L.sha_file, L.pin
K_V1 = 1004
T0 = time.time()


def spread_table(mem, xo, ao, N, npart):
    """own memberships + memberships of the directed STRUCT out-neighbours (the legacy rule applied to a multi-membership table)."""
    ptr, flat = mem
    lam = np.diff(ptr)
    deg_out = np.diff(xo)
    ao = np.asarray(ao, np.int64)
    src = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
    tgt = np.concatenate([np.arange(N, dtype=np.int64), ao])
    cnt = lam[tgt]
    rows = np.repeat(src, cnt)
    starts = ptr[tgt]
    idx = np.repeat(starts - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    keys = np.unique(rows * np.int64(npart) + flat[idx])
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    p2 = np.zeros(N + 1, np.int64)
    p2[1:] = np.cumsum(mlen)
    return (p2, (keys % npart).astype(np.int64))


def reach(D, vote, memb, npart, mA, masks, hub):
    """vote = the table the HITS vote through (evidence side); memb = the table a GOLD node is served through (a node is served when
    any of its member blocks is selected -- the section-19 serving rule, mem_all); the two differ for hubs under V1 (hub -> home
    only for voting, all incident-edge blocks for serving) and for the spread tables."""
    vptr, vflat = vote
    mptr, mflat = memb
    ev = R.evidence(D, vote, npart)
    lam_v, lam_m = np.diff(vptr), np.diff(mptr)
    votes = {}
    for ch, ids in (("dense_top100", D.d_ids[:, :K_LOCK]), ("splade_top100", D.s_ids[:, :K_LOCK])):
        h = ids[mA].ravel()
        h = h[h >= 0]
        votes[ch] = round(float(lam_v[h].mean()), 3)
    allr = np.zeros(D.nq, bool)
    frac = np.zeros(D.nq, np.float64)
    hub_ok, hub_n, nonhub_ok, nonhub_n = 0, 0, 0, 0
    for i, g in enumerate(D.C.gold_nodes):
        if len(g) == 0:
            continue
        ok = [bool(ev[i, mflat[mptr[x]:mptr[x + 1]]].any()) for x in g]
        allr[i] = all(ok)
        frac[i] = float(np.mean(ok))
        if mA[i]:
            for x, o in zip(g, ok):
                if hub[x]:
                    hub_n += 1
                    hub_ok += int(o)
                else:
                    nonhub_n += 1
                    nonhub_ok += int(o)
    gold_lam = float(np.mean(np.concatenate([lam_m[np.asarray(g, np.int64)] for g, m in zip(D.C.gold_nodes, mA) if m and len(g)])))
    out = {"votes_per_hit_mean (distinct blocks)": votes, "vote_table_entries": int(len(vflat)), "membership_entries": int(len(mflat)),
           "lambda_mean_all_nodes (membership)": round(float(lam_m.mean()), 3), "lambda_mean_gold_nodes_A (membership)": round(gold_lam, 3),
           "blocks_with_evidence_per_query_mean": round(float(ev[mA].sum(1).mean()), 1),
           "reached_all_gold_rate": {kk: round(float(allr[vv].mean()), 4) for kk, vv in masks.items()},
           "gold_nodes_reached_fraction_mean": {kk: round(float(frac[vv].mean()), 4) for kk, vv in masks.items()},
           "gold_node_reached_rate_A_by_hubness": {"hub": round(hub_ok / max(hub_n, 1), 4), "non_hub": round(nonhub_ok / max(nonhub_n, 1), 4), "hub_gold_nodes": hub_n, "non_hub_gold_nodes": nonhub_n}}
    return out, allr


def main():
    fp_out = os.path.join(OUT, "br2_why_A_metaqa.json")
    assert not os.path.exists(fp_out), "write-once"
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
    D = G.Data("metaqa", dense_fp32=False)
    mA = D.C.split == "A"
    masks = R.hop_masks(D, mA)
    Gr = V.Graph("metaqa")
    assert Gr.N == D.N and Gr.keys_sha == pre["graphs_frozen"]["metaqa"]["struct_keys_sha256"]
    N, hub = Gr.N, Gr.hub
    xo, ao = D.cd.struct_csr(directed=True)
    D.cd._csr.clear()

    r2_p, home, alt, k = L.load_r2("metaqa", Gr)
    mem_all, Y, blocks_csc, sizes = L.membership(home, alt, N, k)
    mem_vc = R.vote_table(mem_all, home, hub)
    mem_home = R.home_table(home)
    cap_p = os.path.join(X.REPO, pre["v1_substrate_metaqa"]["path"])
    assert sha_file(cap_p) == pre["v1_substrate_metaqa"]["sha256"]
    z = np.load(cap_p).astype(np.int64)
    mem1, lam1, home1, ties1, sizes1, blocks1 = R.memberships(Gr, z, K_V1)
    mem_vc1 = R.vote_table(mem1, home1, hub)

    # gold-node 1-hop ceiling: every gold node is a top-100 hit of either channel or an undirected STRUCT neighbour of one
    xu, au = Gr.xu, Gr.au
    ceil = np.zeros(D.nq, bool)
    for i, g in enumerate(D.C.gold_nodes):
        if len(g) == 0:
            continue
        hits = set(int(x) for x in D.d_ids[i, :K_LOCK] if x >= 0) | set(int(x) for x in D.s_ids[i, :K_LOCK] if x >= 0)
        ok = True
        for x in g:
            if int(x) in hits:
                continue
            nb = au[xu[x]:xu[x + 1]]
            if not any(int(y) in hits for y in nb):
                ok = False
                break
        ceil[i] = ok
    sp_r2 = spread_table(mem_vc, xo, ao, N, k)
    sp_own = spread_table(mem_home, xo, ao, N, k)
    tables = {
        "HARD_LEGACY (served BASE: own block + directed out-neighbour blocks; k 432)": (D.mem, D.mem_hard, D.npart),
        "HARD_OWN (own block only; k 432)": (D.mem_hard, D.mem_hard, D.npart),
        "V1_k1004 (section 19 vertex-cut: votes hub -> home only; served through ALL incident-edge blocks)": (mem_vc1, mem1, K_V1),
        "V1_k1004 served through the vote table only (hub gold served by its home block only; NOT the section-19 rule)": (mem_vc1, mem_vc1, K_V1),
        "R2_k864 (this ruling: home + one alternate; votes hub -> home only; served through both blocks)": (mem_vc, mem_all, k),
        "OWNERS_ONLY_k864 (R = 1 control)": (mem_home, mem_home, k),
        "R2_k864 + legacy out-neighbour spread (REACH ONLY; not a cell; the natural follow-up, not evaluated)": (sp_r2, mem_all, k),
        "OWNERS_ONLY_k864 + legacy out-neighbour spread (REACH ONLY; not a cell)": (sp_own, mem_home, k),
    }
    res = {}
    for name, (vote, memb, npart) in tables.items():
        t = time.time()
        res[name], _ = reach(D, vote, memb, npart, mA, masks, hub)
        log("%-95s votes/hit %s reached_all %s (%.0fs)" % (name[:95], res[name]["votes_per_hit_mean (distinct blocks)"], res[name]["reached_all_gold_rate"], time.time() - t))
    rec = {"RECORD": "BOUNDED_VERTEXCUT_R2_WHY_METAQA", "STATUS": "POSTHOC_DIAGNOSTIC (written after the pre-registered cells were seen; reach only; no new coverage cell; nothing promoted)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A", "cache": "metaqa", "n_DEV_A": int(mA.sum()),
           "preregistration": pin(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")),
           "supersedes": {"path": "results/L1_COVPART/_history/br2_why_A_metaqa__v1_gold_side_via_vote_table.json", "why": "v1 measured the gold side through the VOTE table, which under-counts V1's hub gold nodes (served through all their incident-edge blocks); v2 uses the serving membership table on the gold side, exactly as the replay serves"},
           "records_read": {"br2_replay_A_metaqa.json": pin(os.path.join(OUT, "br2_replay_A_metaqa.json")), "br2_SUMMARY.json": pin(os.path.join(OUT, "br2_SUMMARY.json")),
                            "r2_substrate": pin(r2_p), "v1_substrate_CAP": pin(cap_p)},
           "code": pin(os.path.abspath(__file__)),
           "definition": {"votes_per_hit": "mean number of distinct blocks a top-100 hit node of the channel votes for under the table (DEV_A)",
                          "reached_all_gold_rate": "share of queries whose gold nodes ALL lie in at least one block that received at least one vote from the top-100 hits of either channel (the section-19 evidence); coverage at any budget cannot exceed it",
                          "gold_within_1_struct_hop_of_a_hit": "share of queries whose gold nodes are all either a top-100 hit or an undirected STRUCT neighbour of one; NOT a bound on reach (a block of 100 nodes also receives votes from unrelated hits), stated to size the direct 1-hop route",
                          "reach_vs_coverage": "the replay serves a node when ANY of its member blocks is selected (mem_all); the hits vote through the vote table (hub -> home only); the gold side here uses the membership table, so this reach is the quantity coverage is bounded by"},
           "gold_within_1_struct_hop_of_a_hit": {kk: round(float(ceil[vv].mean()), 4) for kk, vv in masks.items()},
           "tables": res,
           "coverage_seen (from the pre-registered records, for the ladder; ALL DEV_A)": {"HARD_OWN_BLOCK_VOTE": 0.1786, "OWNERS_ONLY_P50": 0.2265, "R2_P50": 0.3333, "R2_MATCHED_LE": 0.3473, "HARD_MTK_BASE (legacy spread)": 0.6397, "V1_P50": 0.6776, "V1_MATCHED_LE": 0.6876},
           "seconds": round(time.time() - T0, 1)}
    S.wj(fp_out, rec)
    log("gold within 1 STRUCT hop of a hit %s -> %s" % (rec["gold_within_1_struct_hop_of_a_hit"], os.path.relpath(fp_out, X.REPO)))


if __name__ == "__main__":
    main()
