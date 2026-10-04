"""Flat (partition-free) dense + SPLADE node rankings vs the served partition routes at MATCHED per-query exposure.

The user's question (2026-09-25, verbatim): "when we come back to our question do the partitions even help than purely
having dense and splade rankings? have we measured that with our hypergraph approach?"  It had not been measured on the
canonical hypergraph stack: every canonical-era comparison was partition vs partition (the G2 exposure audit's CANON arm is
itself a vote-ranked partition list; the 2026-08-10 l1_candgen test used the legacy partitions, the legacy encoder, a
dense-only flat list and mean recall).  This module measures it on the five DEV_A caches (split A), read-only.

Flat rankings (no partition, exhaustive over ALL N corpus nodes, from the frozen stored vectors; no tuned constant):
    FLAT_DENSE    q . e_v from the SAME fp32 products as the accepted streamed stage 2 (same query batches of TB.CQ cache
                  rows, same TB.BLOCK position chunks; their block maxima must equal the accepted stage 2 bit for bit);
                  descending, ties -> lower position
    FLAT_SPLADE   the sparse dot of the canonical SPLADE query and document vectors (CSR shards, fp32); descending, ties ->
                  lower position; only nodes with a positive score are ranked (no lexical evidence -> never served)
    FLAT_RRF      the frozen node-level RRF (_ta_prepartition.node_rrf) extended to full depth: 1/(K0 + p_dense) +
                  1/(K0 + p_splade) with 0-based positions, K0 = 60, float64; a node absent from a channel's list takes
                  that list's depth as its position (node_rrf: TOP200; here the dense list is all N nodes and the SPLADE
                  list is every positive-score node); descending, ties broken by the dense order (node_rrf's insertion
                  order).  Checked: node_rrf over this module's own top-200 lists == the served ret_rrf wherever both of
                  its top-200 lists equal the served ones.
Routes (the served partition machinery, reproduced and asserted against transfer_repro_<cache>.json and the replay record):
    L1@P          the top P blocks of the served node-vote ranking F0(Cd, Cs), P in 1, 2, 5, 10, 20, 50 (L1 = L1@50)
    DENSE_VOTE    the top 50 blocks of the dense node-vote channel Cd alone; SPLADE_VOTE likewise with Cs
    PRIMARY       the frozen composite QMAX_BALANCED_H2_PATCH1 (the 49 kept SAFE blocks + the patch's nodes)
Matched exposure: for a route R and a split-A query q, B_q(R) = the number of distinct nodes R serves for q (the sum of
its blocks' sizes; PRIMARY: kept blocks + patch nodes, == its SAFE exposure, asserted).  The flat arm paired with R serves
the top B_q(R) nodes of its ranking (FLAT_SPLADE: at most its positive-score nodes).  ALL-gold = every gold node of q
served.  Paired exact McNemar per pair (route = reference, flat = arm).
Also (descriptive, budget-free): the exposure each ranking needs to cover ALL gold nodes of q -- FLAT_*: 1 + the deepest
gold position; L1: the nodes of the served block ranking F0(Cd, Cs) down to the deepest gold block.

    python -u scratchpad/_l1c_flat.py <cache> --dry   identities and validation: the served routes are reproduced on the
                                                      real gold nodes (known reference numbers, none printed), then every
                                                      flat, pairing and record path runs on STAND-IN gold nodes (seeded,
                                                      same count per row); no flat coverage number exists; no record
    python -u scratchpad/_l1c_flat.py <cache>         the pre-registered run -> results/L1_COVPART/flat_A_<cache>.json
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT
import _l1c_transfer_blockmax as TB
import _ta_prepartition as TA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d",
          "_l1c_h2balanced.py": "7aed6b1d4399a1b432d0e6be48013cf73c92d4e47d2e5d2dd7b669b508d84bbe",
          "_l1c_qmaxbal.py": "cae88fa64a0516a4f55c99a4a3b464afd39e467dca42e4380a546ebbd545b1c8",
          "_l1g_candidate.py": "b01c27a6e64b18756db1ca1d8910e8db4bb54fa019d9809698cb1369206169e0",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327",
          "_l1c_transfer_composite.py": "1241ec58e759ab99e77ff17936b6edc6b072344ea1d47a76a79f940a1047cc5b",
          "_l1c_transfer_blockmax.py": "f9232ef95b5f317bef85cb1f7304569ea4b00bf6d283a99fc8abe153ad239ee9"}
DEV_A_CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
K0 = TA.K0                                                   # 60, the frozen RRF constant
P_CURVE = (1, 2, 5, 10, 20, 50)
FIXED_BUDGETS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
FLATS = ("FLAT_DENSE", "FLAT_SPLADE", "FLAT_RRF")
ALPHA = 0.01                                                 # GAIN / LOSS labels (as every section-24 record)
N_AGREE = 100                                                # served-list agreement depth (= K_LOCK, the node-vote depth)
ARM_BAL, ARM_Q50, ARM_QSAFE, ARM_QPRIM = "BALANCED_H2_PATCH1", "QMAX_F0", "QMAX_SAFE", "QMAX_BALANCED_H2_PATCH1"
MARK_BEAM = "# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording"
MARK_SAFE = "# ---------------------------------------------------------------- canonical SAFE machinery"
MARK_PATCH = "# ---------------------------------------------------------------- the patch rule of section 16"
MARK_PATCH_END = "\n\n\npatches = {ARM_H2"
PAIR_WORDS = {"GAIN": "FLAT_BEATS_PARTITIONS", "LOSS": "PARTITIONS_HELP", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def label(o, alpha=ALPHA):
    if o["gained"] > o["lost"] and o["p"] < alpha:
        return "GAIN"
    if o["lost"] > o["gained"] and o["p"] < alpha:
        return "LOSS"
    return "NEUTRAL"


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def host_state():
    o = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        o.update({"host_total_gib": round(vm.total / 2 ** 30, 2), "host_available_gib": round(vm.available / 2 ** 30, 2),
                  "swap_used_gib": round(sw.used / 2 ** 30, 2)})
        me, fo = os.getpid(), []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                if p.info["pid"] != me and (p.info["name"] or "").lower().startswith("python") and p.info["memory_info"].rss > 500 * 2 ** 20:
                    fo.append({"pid": p.info["pid"], "rss_mb": int(p.info["memory_info"].rss / 2 ** 20)})
            except Exception:
                pass
        o["other_python_processes_over_500_MB (untouched)"] = fo
    except Exception as e:
        o["error"] = repr(e)
    return o


def stats(x):
    x = np.asarray(x, np.float64)
    return {"mean": round(float(x.mean()), 1), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


# ---------------------------------------------------------------- arguments, pins, pre-registration
CODE_SHA = sha_file(os.path.abspath(__file__))                 # the code this process runs: hashed at start, re-checked before the record
name = sys.argv[1]
DRY = "--dry" in sys.argv
DRY_OUT = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-out=")), None)   # DRY only: the assembled stand-in record, outside results/
assert DRY_OUT is None or (DRY and not os.path.abspath(DRY_OUT).startswith(os.path.abspath(G.X.REPO))), "--dry-out is a dry-run file outside the repository"
assert name in DEV_A_CACHES, "the flat baseline runs on the five DEV_A caches (split A) only"
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
fp_out = os.path.join(OUT, "flat_A_%s.json" % name)
PRE = os.path.join(OUT, "PREREGISTRATION_FLAT_VS_PARTITION.json")
pre_sha = None
if not DRY:
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    pre = json.load(open(PRE, encoding="utf-8"))
    pre_sha = sha_file(PRE)
    for mod, pn in pre["code"]["new_modules"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(G.X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
rec = json.load(open(os.path.join(OUT, "transfer_repro_%s.json" % name), encoding="utf-8"))   # the accepted streamed-stage-2 record (reference)
assert rec["streamed_stage2_gate"]["STREAMED_STAGE2_ACCEPTED_ON_THIS_CACHE"] is True
HOST0 = host_state()
G.log("host at start: %s" % json.dumps(HOST0))

# ---------------------------------------------------------------- the pinned uL3 head: data, STRUCT adjacency, frozen beam on split A  [as in sections 14-24]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
T_ALL = time.time()
exec(head)                                                   # D, C, beams1, beams2 (the pinned beam on split A), expand_real, fused_order, seeds, hops ...
secs_pinned = secs.copy()                                    # read by the exec'd balanced-beam slice (as in the pinned runner)
edges_pinned = edges_h.copy()
TOPP = int(D.base_rank.shape[1])
TIMES = {"uL3_head_and_pinned_beam": round(time.time() - T_ALL, 1)}

# ---------------------------------------------------------------- the section-21 balanced beam and the section-16 patch rule, verbatim from the pinned runner
csrc = open(os.path.join(HERE, "_l1c_transfer_composite.py"), "rb").read().decode("utf-8")
REPRO, rec21 = False, None
t_ = time.time()
exec(csrc[csrc.index(MARK_BEAM):csrc.index(MARK_SAFE)])      # beam_pinned_with_candidates (== beams1 / beams2, asserted), rr_compress, beam_balanced -> bal1, bal2
exec(csrc[csrc.index(MARK_PATCH):csrc.index(MARK_PATCH_END)])   # build_patches(b1s, b2s, ch)
del CAND_B
TIMES["balanced_beam"] = round(time.time() - t_, 1)

# ---------------------------------------------------------------- the served node-vote channels and the accepted stage 2 (reference)
t_ = time.time()
Cd = G.block_channel(D, D.d_ids)
Cs = G.block_channel(D, D.s_ids)
base_full = G.F0([Cd, Cs], npart)
assert (base_full[:, :TOPP] == D.base_rank).all(), "F0(Cd, Cs) does not reproduce the served base_rank"
qb_ref = TB.blockmax_scores_streamed(D, log=G.log)
assert hashlib.sha256(qb_ref.tobytes()).hexdigest() == rec["stage2"]["streamed"]["sha256_scores"], "stage 2 differs from the accepted streamed record"
Cmax = np.argsort(-qb_ref, axis=1, kind="stable").astype(np.int64)
assert hashlib.sha256(Cmax.tobytes()).hexdigest() == rec["stage2"]["streamed"]["sha256_ranking"]
TIMES["stage2_reference_streamed"] = round(time.time() - t_, 1)
G.log("stage 2 == the accepted streamed stage 2 of transfer_repro_%s.json (scores and ranking sha256) in %.0fs" % (name, TIMES["stage2_reference_streamed"]))

# ---------------------------------------------------------------- blocks
sizes = np.asarray(C.part_sizes, np.int64)
assert int(sizes.sum()) == N and (np.bincount(hard, minlength=npart) == sizes).all() and (sizes >= 1).all()
order_nodes = np.argsort(hard, kind="stable")                # node ids ascending inside every block
ptr = np.zeros(npart + 1, np.int64)
ptr[1:] = np.cumsum(sizes)


def nodes_of(b):
    return order_nodes[ptr[b]:ptr[b + 1]]


# ---------------------------------------------------------------- the frozen SAFE machinery + patch (QMAX and BASE chains only)
z0 = np.load(C.path, allow_pickle=True)
zz = {k_: z0[k_] for k_ in z0.files if k_ != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
assert zz["base_rank"].shape == (nq, TOPP) and (zz["base_rank"] == D.base_rank).all()
finals, SAFE_OF, weakest_of = {}, {}, {}                     # read by the exec'd build_patches
CHAL, GOLDP, SERVED, CHAIN_SECS = {}, [], {}, {}


def run_chain(ch, top):
    """frozen SAFE (RT.build_cache + KB.contexts + KB.f6_select) and the section-16 patch with the balanced beam on the
    block ranking `top` (None = the served BASE ranking).  Returns the served P50 / SAFE sets and the patch per row."""
    t = time.time()
    zq = zz if top is None else dict(zz, base_rank=np.ascontiguousarray(top[:, :TOPP]).astype(np.int32))
    Cc = RT.build_cache(name, zq, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    cx = KB.contexts(zq, meta, Cc, B, CFG)
    first = not GOLDP
    if first:
        GOLDP.append(Cc["goldp"])
    else:
        assert Cc["goldp"] == GOLDP[0]
    p50, safe = {}, {}
    weak = np.full(nq, -1, np.int64)
    for qi in rowsA:
        c = cx[qi]
        p50[qi] = set(int(x) for x in c["base50"])
        assert p50[qi] == set(int(x) for x in zq["base_rank"][qi, :P_MAIN])
        Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        s50 = c["prot_set"] | set(Xs)
        assert len(s50) == P_MAIN
        safe[qi] = s50
        weak[qi] = int(Xs[-1])                               # the sixth block of the F6 competition = SAFE's weakest admission
        if first:
            CHAL[qi] = (c["spos"], c["rpos"])
        else:
            assert (c["spos"], c["rpos"]) == CHAL[qi], "the challengers depend on the block ranking"
    del Cc, cx
    finals["SAFE:" + ch], SAFE_OF[ch], weakest_of[ch] = safe, "SAFE:" + ch, weak
    patch = build_patches(bal1, bal2, ch)
    del finals["SAFE:" + ch]
    for qi in rowsA:                                         # the served node set of the patched arm is disjoint and matched to SAFE
        pt = patch[qi]
        assert len(pt["nodes"]) == pt["nb"] and all(int(hard[v]) not in pt["kept"] for v in pt["nodes"])
        assert int(sizes[list(pt["kept"])].sum()) + len(pt["nodes"]) == int(sizes[list(safe[qi])].sum())
    SERVED[ch] = (p50, safe, patch)
    CHAIN_SECS[ch] = round(time.time() - t, 1)
    G.log("chain %-5s SAFE + patch in %5.1fs (RSS %.0f MB)" % (ch, CHAIN_SECS[ch], _rss_mb()))


t_ = time.time()
top_q = G.F0([Cd, Cs, Cmax], npart)[:, :TOPP]
run_chain("QMAX", top_q)
run_chain("BASE", None)
TIMES["chains"] = round(time.time() - t_, 1)

# ---------------------------------------------------------------- per-row budgets of every route (the nodes it serves; no gold)
rowsA = np.asarray(rowsA, np.int64)
BUDGET = {}
for P in P_CURVE:
    BUDGET["L1@%d" % P] = np.array([int(sizes[D.base_rank[qi, :P]].sum()) for qi in rowsA], np.int64)
BUDGET["DENSE_VOTE"] = np.array([int(sizes[Cd[qi, :P_MAIN]].sum()) for qi in rowsA], np.int64)
BUDGET["SPLADE_VOTE"] = np.array([int(sizes[Cs[qi, :P_MAIN]].sum()) for qi in rowsA], np.int64)
qp50, qsafe, qpatch = SERVED["QMAX"]
BUDGET["PRIMARY"] = np.array([int(sizes[list(qpatch[qi]["kept"])].sum()) + len(qpatch[qi]["nodes"]) for qi in rowsA], np.int64)
assert (BUDGET["PRIMARY"] == np.array([int(sizes[list(qsafe[qi])].sum()) for qi in rowsA])).all()
G.log("budgets (nodes served per query): %s" % json.dumps({k: stats(v) for k, v in BUDGET.items()}))

# ---------------------------------------------------------------- gold pairs (split A), route coverage, the reference reproduction (real gold nodes)
nA = len(rowsA)
jA = {int(q): j for j, q in enumerate(rowsA)}


def build_pairs(gnodes):
    pg, pp = [], [0]
    for qi in rowsA:
        pg.extend(int(g) for g in gnodes[qi])
        pp.append(len(pg))
    pg, pp = np.asarray(pg, np.int64), np.asarray(pp, np.int64)
    return pg, pp, rowsA[np.repeat(np.arange(nA), np.diff(pp))], hard[pg]


def ind_from_flags(f):
    a = np.array([pair_ptr[j + 1] > pair_ptr[j] and bool(f[pair_ptr[j]:pair_ptr[j + 1]].all()) for j in range(nA)], bool)
    y = np.array([bool(f[pair_ptr[j]:pair_ptr[j + 1]].any()) for j in range(nA)], bool)
    return a, y


def mask_top(rank, k):
    M = np.zeros((nq, npart), bool)
    np.put_along_axis(M, rank[:, :k], True, axis=1)
    return M


def route_coverage():
    """ALL-gold / any-gold per split-A row of every route for the current gold pairs"""
    I, Y = {}, {}
    for P in P_CURVE:
        I["L1@%d" % P], Y["L1@%d" % P] = ind_from_flags(mask_top(D.base_rank, P)[pair_q, pair_b])
    I["DENSE_VOTE"], Y["DENSE_VOTE"] = ind_from_flags(mask_top(Cd, P_MAIN)[pair_q, pair_b])
    I["SPLADE_VOTE"], Y["SPLADE_VOTE"] = ind_from_flags(mask_top(Cs, P_MAIN)[pair_q, pair_b])
    I["PRIMARY"], Y["PRIMARY"] = ind_from_flags(np.array([int(b_) in qpatch[int(q_)]["kept"] or int(g_) in qpatch[int(q_)]["nodes"]
                                                         for q_, b_, g_ in zip(pair_q, pair_b, pair_g)], bool))
    bp50, bsafe, bpatch = SERVED["BASE"]
    I["SAFE"], Y["SAFE"] = ind_from_flags(np.array([int(b_) in bsafe[int(q_)] for q_, b_ in zip(pair_q, pair_b)], bool))
    I[ARM_BAL], Y[ARM_BAL] = ind_from_flags(np.array([int(b_) in bpatch[int(q_)]["kept"] or int(g_) in bpatch[int(q_)]["nodes"]
                                                     for q_, b_, g_ in zip(pair_q, pair_b, pair_g)], bool))
    I[ARM_Q50], Y[ARM_Q50] = ind_from_flags(np.array([int(b_) in qp50[int(q_)] for q_, b_ in zip(pair_q, pair_b)], bool))
    I[ARM_QSAFE], Y[ARM_QSAFE] = ind_from_flags(np.array([int(b_) in qsafe[int(q_)] for q_, b_ in zip(pair_q, pair_b)], bool))
    return I, Y


gold_real = [np.asarray(C.gold_nodes[qi], np.int64) for qi in range(nq)]
pair_g, pair_ptr, pair_q, pair_b = build_pairs(gold_real)
IND, ANYV = route_coverage()
recA = rec["arms"]
for a, r in (("L1@50", "L1"), ("SAFE", "SAFE"), (ARM_BAL, ARM_BAL), (ARM_Q50, ARM_Q50), (ARM_QSAFE, ARM_QSAFE), ("PRIMARY", ARM_QPRIM)):
    assert round(float(IND[a].mean()), 4) == recA[r]["ALL"] and round(float(ANYV[a].mean()), 4) == recA[r]["ANY"], (a, float(IND[a].mean()), recA[r]["ALL"])
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert (IND["L1@50"] == np.asarray(recS["_ind_BASE"], bool)[rowsA]).all(), "BASE differs from the frozen replay record"
assert (IND["SAFE"] == np.asarray(recS["_ind_SAFE"], bool)[rowsA]).all(), "canonical SAFE not reproduced"
REPRODUCED = ("L1 / SAFE / BALANCED_H2_PATCH1 / QMAX_F0 / QMAX_SAFE / QMAX_BALANCED_H2_PATCH1 == transfer_repro_%s.json (ALL, ANY); "
              "L1 and SAFE == %s (split A, per query)" % (name, SAFE_RECORD[name]))
G.log("reference reproduced (no number printed): %s" % REPRODUCED)
if DRY:                                                      # from here on the dry run uses STAND-IN gold nodes: the same count per row,
    rng_dry = np.random.default_rng(12345)                   # drawn uniformly from the corpus -- no flat coverage number of the real gold exists
    gold_dry = [np.sort(rng_dry.choice(N, size=len(g), replace=False)).astype(np.int64) for g in gold_real]
    pair_g, pair_ptr, pair_q, pair_b = build_pairs(gold_dry)
    IND, ANYV = route_coverage()
    REPRODUCED += "; DRY RUN: every flat, pairing and record path below ran on STAND-IN gold nodes"
POS = {f: np.full(len(pair_g), -1, np.int64) for f in FLATS}          # 0-based flat position of every gold node (-1 = never served)

# ---------------------------------------------------------------- the flat pass: every cache row's exhaustive dense products (stage-2 identity), split-A rows ranked
t_ = time.time()
Q = np.ascontiguousarray(D.Q, dtype=np.float32)
Es = D.cd.embeddings("splade", "docs")
SPL = [Es.shard(k) for k in range(Es.n_shards)]              # every CSR shard held once (the reader caches one shard only)
assert sum(int(M.shape[0]) for M in SPL) == N and all(int(M.shape[1]) == int(D.Qs.shape[1]) for M in SPL)
isA = np.zeros(nq, bool)
isA[rowsA] = True
perm_blocks = order_nodes                                    # columns grouped by block, for the stage-2 identity
starts = ptr[:-1]
NPOS = np.zeros(nA, np.int64)                                # SPLADE positive-score nodes per query (the SPLADE list depth)
AGREE = {"dense_top100_set": np.zeros(nA), "splade_top100_set": np.zeros(nA),
         "dense_top200_exact": np.zeros(nA, bool), "splade_top200_exact": np.zeros(nA, bool), "ret_rrf_reproduced": np.zeros(nA, bool)}
ranked = 0
RANKED = np.zeros(nA, bool)
rrf_ident_checked = 0
for qa in range(0, nq, TB.CQ):
    qz = min(nq, qa + TB.CQ)
    SD = np.empty((qz - qa, N), np.float32)
    for a, b_, blk in D._E16.blocks(TB.BLOCK):
        Ec = np.ascontiguousarray(np.asarray(blk, np.float32))            # the same fp16 -> fp32 chunk as the accepted stage 2
        SD[:, a:b_] = Q[qa:qz] @ Ec.T                                     # the same (m, c) product
    qb_here = np.maximum.reduceat(SD[:, perm_blocks], starts, axis=1)
    assert qb_here.tobytes() == qb_ref[qa:qz].tobytes(), "flat dense products do not reproduce the accepted stage 2 (rows %d-%d)" % (qa, qz)
    del qb_here
    rows_here = [qi for qi in range(qa, qz) if isA[qi]]
    if not rows_here:
        continue
    Qd = np.asarray(D.Qs[rows_here].todense(), np.float32)                # (m, V) dense SPLADE queries of these rows
    SS = np.empty((len(rows_here), N), np.float32)
    a = 0
    for M in SPL:
        SS[:, a:a + M.shape[0]] = np.asarray(M @ Qd.T, np.float32).T
        a += M.shape[0]
    for i, qi in enumerate(rows_here):
        j = jA[qi]
        sd, ss = SD[qi - qa], SS[i]
        assert np.isfinite(sd).all() and np.isfinite(ss).all() and (ss >= 0).all()
        od = np.argsort(-sd, kind="stable")                               # dense order, ties -> lower position
        pd = np.empty(N, np.int64)
        pd[od] = np.arange(N)
        npos = int((ss > 0).sum())
        os_ = np.argsort(-ss, kind="stable")[:npos]                       # positive-score nodes, descending, ties -> lower position
        ps = np.full(N, npos, np.int64)                                   # absent from the SPLADE list -> its depth (node_rrf's convention)
        ps[os_] = np.arange(npos)
        fv = 1.0 / (K0 + pd) + 1.0 / (K0 + ps)                            # float64
        of = od[np.argsort(-fv[od], kind="stable")]                       # ties broken by the dense order
        pf = np.empty(N, np.int64)
        pf[of] = np.arange(N)
        NPOS[j] = npos
        dsv, ssv = D.d_ids[qi], D.s_ids[qi]
        AGREE["dense_top100_set"][j] = len(set(od[:N_AGREE].tolist()) & set(dsv[:N_AGREE].tolist())) / float(N_AGREE)
        ka = min(N_AGREE, npos)                                           # beyond the positive-score nodes a served list is arbitrary
        AGREE["splade_top100_set"][j] = (len(set(os_[:ka].tolist()) & set(ssv[:ka].tolist())) / float(ka)) if ka else 1.0
        de, se = bool((od[:TA.TOP200] == dsv[:TA.TOP200]).all()), bool(npos >= TA.TOP200 and (os_[:TA.TOP200] == ssv[:TA.TOP200]).all())
        AGREE["dense_top200_exact"][j], AGREE["splade_top200_exact"][j] = de, se
        if de and se:                                                     # node_rrf over these top-200 lists == the served ret_rrf
            f200 = TA.node_rrf(od[:TA.TOP200], os_[:TA.TOP200])
            served = [int(x) for x in C.ret_rrf[qi] if x >= 0]
            assert [int(x) for x in f200[:len(served)]] == served, "node_rrf over identical top-200 lists != served ret_rrf (row %d)" % qi
            AGREE["ret_rrf_reproduced"][j] = True
            rrf_ident_checked += 1
        sl = slice(pair_ptr[j], pair_ptr[j + 1])
        g = pair_g[sl]
        POS["FLAT_DENSE"][sl] = pd[g]
        POS["FLAT_SPLADE"][sl] = np.where(ss[g] > 0, ps[g], -1)
        POS["FLAT_RRF"][sl] = pf[g]
        RANKED[j] = True
        ranked += 1
    del SD, SS, Qd
TIMES["flat_pass"] = round(time.time() - t_, 1)
assert ranked == nA and RANKED.all()
VALID = {"flat_dense_block_max == accepted stage 2 (bitwise, every cache row)": True,
         "rows_ranked": ranked,
         "dense_top100_set_overlap_with_served_mean": round(float(AGREE["dense_top100_set"][RANKED].mean()), 5),
         "splade_top100_set_overlap_with_served_mean": round(float(AGREE["splade_top100_set"][RANKED].mean()), 5),
         "rows_dense_top200_order_identical_to_served": int(AGREE["dense_top200_exact"][RANKED].sum()),
         "rows_splade_top200_order_identical_to_served": int(AGREE["splade_top200_exact"][RANKED].sum()),
         "rows_where_node_rrf_over_own_top200 == served ret_rrf (checked wherever both top-200 lists are identical; asserted)": rrf_ident_checked,
         "splade_positive_nodes_per_query": stats(NPOS[RANKED]) if ranked else None}
G.log("flat pass in %.0fs (RSS %.0f MB): %s" % (TIMES["flat_pass"], _rss_mb(), json.dumps(VALID)))
assert VALID["dense_top100_set_overlap_with_served_mean"] >= 0.99 and VALID["splade_top100_set_overlap_with_served_mean"] >= 0.99, \
    "the exhaustive rankings disagree with the served retrieval caches beyond float noise (wrong vectors or row mapping)"

# ---------------------------------------------------------------- flat coverage at a per-row budget
has_gold = np.diff(pair_ptr) > 0
GMAX, GMIN = {}, {}
for f in FLATS:
    p_ = POS[f]
    never = p_ < 0
    big = np.where(never, np.iinfo(np.int64).max // 4, p_)
    GMAX[f] = np.array([big[pair_ptr[j]:pair_ptr[j + 1]].max() if has_gold[j] else -1 for j in range(nA)], np.int64)
    GMIN[f] = np.array([big[pair_ptr[j]:pair_ptr[j + 1]].min() if has_gold[j] else -1 for j in range(nA)], np.int64)


def flat_cov(f, budget):
    budget = np.asarray(budget, np.int64)
    return has_gold & (GMAX[f] < budget), has_gold & (GMIN[f] < budget)


hopsA = np.asarray(hops)[rowsA]
HOPS = sorted(set(int(x) for x in hopsA if x >= 0))
ngold = np.diff(pair_ptr)
NGOLD = sorted(set(int(x) for x in ngold if x > 0))


def paired(ref_ind, arm_ind, s=None):
    s = np.ones(nA, bool) if s is None else s
    g_, l_, p_ = G.X.mcnemar(ref_ind[s], arm_ind[s])
    o = {"gained (flat covers, route does not)": g_, "lost (route covers, flat does not)": l_, "p": p_}
    return o


def pair_out(route, f, budget_key):
    fa, fy = flat_cov(f, BUDGET[budget_key])
    o = paired(IND[route], fa)
    lab = label({"gained": o["gained (flat covers, route does not)"], "lost": o["lost (route covers, flat does not)"], "p": o["p"]})
    out = {"route": route, "flat": f, "budget": budget_key, "budget_nodes": stats(BUDGET[budget_key]),
           "ALL_route": round(float(IND[route].mean()), 4), "ALL_flat": round(float(fa.mean()), 4),
           "ANY_route": round(float(ANYV[route].mean()), 4), "ANY_flat": round(float(fy.mean()), 4),
           "points_flat_minus_route": round(100.0 * (float(fa.mean()) - float(IND[route].mean())), 2),
           "paired": o, "label": lab, "verdict": PAIR_WORDS[lab], "per_hop": {}, "per_n_gold": {}}
    for h in HOPS:
        s = hopsA == h
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "ALL_route": round(float(IND[route][s].mean()), 4), "ALL_flat": round(float(fa[s].mean()), 4),
                                       "paired": paired(IND[route], fa, s)}
    for k in NGOLD:
        s = ngold == k
        out["per_n_gold"][str(k)] = {"n": int(s.sum()), "ALL_route": round(float(IND[route][s].mean()), 4), "ALL_flat": round(float(fa[s].mean()), 4),
                                     "paired": paired(IND[route], fa, s)}
    lost = IND[route] & ~fa
    gained = fa & ~IND[route]
    lost_rk = lost & (GMAX[f] < N)                                        # the flat list ranks every gold node of the query
    out["where_they_differ"] = {
        "lost_queries_where_flat_never_ranks_a_gold_node (no positive score; FLAT_SPLADE only)": int((lost & ~(GMAX[f] < N)).sum()),
        "flat_depth_needed_in_lost_queries (1 + deepest gold position, over the lost queries it ranks; median / p90)":
            [float(np.median(GMAX[f][lost_rk] + 1)), float(np.percentile(GMAX[f][lost_rk] + 1, 90))] if lost_rk.any() else None,
        "route_budget_in_lost_queries (median)": float(np.median(BUDGET[budget_key][lost])) if lost.any() else None,
        "flat_depth_needed_in_gained_queries (median)": float(np.median(GMAX[f][gained] + 1)) if gained.any() else None}
    return out


PAIRS = {}
for P in P_CURVE:
    PAIRS["L1@%d vs FLAT_RRF" % P] = pair_out("L1@%d" % P, "FLAT_RRF", "L1@%d" % P)
PAIRS["PRIMARY vs FLAT_RRF"] = pair_out("PRIMARY", "FLAT_RRF", "PRIMARY")
PAIRS["DENSE_VOTE vs FLAT_DENSE"] = pair_out("DENSE_VOTE", "FLAT_DENSE", "DENSE_VOTE")
PAIRS["SPLADE_VOTE vs FLAT_SPLADE"] = pair_out("SPLADE_VOTE", "FLAT_SPLADE", "SPLADE_VOTE")
DESCRIPTIVE = {"L1@50 vs FLAT_DENSE (descriptive)": pair_out("L1@50", "FLAT_DENSE", "L1@50"),
               "L1@50 vs FLAT_SPLADE (descriptive)": pair_out("L1@50", "FLAT_SPLADE", "L1@50")}
for k_, v_ in PAIRS.items():
    o = v_["paired"]
    if not DRY:
        G.log("%-28s route %.4f flat %.4f (%+.2f pts) +%d/-%d p=%.3g %s | budget mean %.0f" % (
            k_, v_["ALL_route"], v_["ALL_flat"], v_["points_flat_minus_route"], o["gained (flat covers, route does not)"],
            o["lost (route covers, flat does not)"], o["p"], v_["verdict"], v_["budget_nodes"]["mean"]))

# ---------------------------------------------------------------- fixed budgets (flat only) and the budget-free exposure needed to cover ALL gold nodes
FIXED = {f: {str(b_): {"ALL": round(float(flat_cov(f, np.full(nA, b_))[0].mean()), 4), "ANY": round(float(flat_cov(f, np.full(nA, b_))[1].mean()), 4)}
             for b_ in FIXED_BUDGETS} for f in FLATS}
pos_full = G.positions(base_full)                            # 0-based position of every block in the served node-vote ranking
cum = np.cumsum(sizes[base_full], axis=1)                    # nodes served down to each position
E_L1 = np.full(nA, -1, np.int64)
for j, qi in enumerate(rowsA):
    if has_gold[j]:
        d = int(pos_full[qi, pair_b[pair_ptr[j]:pair_ptr[j + 1]]].max())
        E_L1[j] = int(cum[qi, d])
EXPOSURE_TO_COVER = {"L1 block ranking F0(Cd, Cs) (nodes down to the deepest gold block)": {}}


def q_(x):
    x = np.asarray(x, np.float64)
    return {"p25": float(np.percentile(x, 25)), "median": float(np.median(x)), "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90))}


EXPOSURE_TO_COVER["L1 block ranking F0(Cd, Cs) (nodes down to the deepest gold block)"] = q_(E_L1[has_gold])
for f in FLATS:
    ok = has_gold & (GMAX[f] < N)
    e_f = GMAX[f] + 1
    both = ok & has_gold
    EXPOSURE_TO_COVER[f] = {"rows_coverable (every gold node ranked)": int(ok.sum()), **(q_(e_f[ok]) if ok.any() else {}),
                            "rows_flat_needs_fewer_nodes_than_L1": int((both & (e_f < E_L1)).sum()),
                            "rows_flat_needs_more_nodes_than_L1": int((both & (e_f > E_L1)).sum()) + int((has_gold & ~ok).sum()),
                            "rows_equal": int((both & (e_f == E_L1)).sum())}
    if HOPS:
        EXPOSURE_TO_COVER[f]["per_hop median (flat / L1)"] = {"hop%d" % h: [float(np.median(e_f[ok & (hopsA == h)])) if (ok & (hopsA == h)).any() else None,
                                                                             float(np.median(E_L1[has_gold & (hopsA == h)]))] for h in HOPS}
if not DRY:
    G.log("fixed budgets: %s" % json.dumps({f: {b_: FIXED[f][b_]["ALL"] for b_ in ("100", "1000", "5000")} for f in FLATS}))
    G.log("exposure to cover ALL: %s" % json.dumps(EXPOSURE_TO_COVER))

# ---------------------------------------------------------------- record
res = {"cache": name, "dataset": G.X.DS_OF[name], "partition": G.PARTITION_OF.get(name), "n_split_A": nA, "n_cache_rows": nq, "N": int(N), "npart": int(npart),
       "block_size_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
       "mode": "PREREGISTERED_FLAT_VS_PARTITION_SPLIT_A", "status": "DEV_A_DIAGNOSTIC (no selection, no adoption)",
       "preregistration": {"path": os.path.relpath(PRE, G.X.REPO).replace("\\", "/"), "sha256": pre_sha},
       "cache_file": {"path": os.path.relpath(C.path, G.X.REPO).replace("\\", "/"), "sha256": sha_file(C.path), "row_query_ids_sha256": meta.get("row_query_ids_sha256")},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": CODE_SHA},
       "pinned": PINNED, "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0,
       "reference_reproduced": REPRODUCED, "flat_validation": VALID,
       "routes_ALL": {a: round(float(IND[a].mean()), 4) for a in IND}, "routes_ANY": {a: round(float(ANYV[a].mean()), 4) for a in ANYV},
       "budgets": {k: stats(v) for k, v in BUDGET.items()},
       "pairs (pre-registered; route = reference, flat = arm at the route's per-query budget)": PAIRS,
       "descriptive_pairs": DESCRIPTIVE,
       "flat_fixed_budgets": FIXED, "exposure_to_cover_ALL (budget-free, descriptive)": EXPOSURE_TO_COVER,
       "hops_available": HOPS, "n_gold_strata": NGOLD,
       "_gmax_FLAT_RRF (0-based deepest gold position per split-A row)": GMAX["FLAT_RRF"].tolist(),
       "_gmax_FLAT_DENSE": GMAX["FLAT_DENSE"].tolist(), "_gmax_FLAT_SPLADE": GMAX["FLAT_SPLADE"].tolist(),
       "_gmax_never_ranked_sentinel (a gold node with no positive FLAT_SPLADE score; -1 = a row without gold nodes)": int(np.iinfo(np.int64).max // 4),
       "_ind_L1": IND["L1@50"].astype(int).tolist(), "_ind_PRIMARY": IND["PRIMARY"].astype(int).tolist(),
       "_budget_L1": BUDGET["L1@50"].tolist(), "_budget_PRIMARY": BUDGET["PRIMARY"].tolist(),
       "seconds_by_stage": TIMES, "chain_seconds": CHAIN_SECS}
res["seconds"] = round(time.time() - T_ALL, 1)
res["process_rss_mb_at_end"] = round(_rss_mb(), 1)
assert sha_file(os.path.abspath(__file__)) == CODE_SHA, "the module file changed during the run: the record would misstate the code that ran"
if DRY:
    blob = json.dumps(res)
    if DRY_OUT:
        with open(DRY_OUT, "w", encoding="utf-8") as f_:
            f_.write(blob)
    G.log("DRY RUN (%.0fs, RSS %.0f MB): pins; F0(Cd, Cs) == served base_rank; stage 2 == the accepted streamed record; flat dense "
          "block maxima == stage 2 bitwise on all %d cache rows; QMAX and BASE chains (|SAFE| = 50, patch disjoint and matched to SAFE); "
          "every split-A row ranked; every coverage, pairing and record path exercised on STAND-IN gold nodes (no real coverage number "
          "computed); record assembled (%d bytes) and NOT written.  times %s" % (time.time() - T_ALL, _rss_mb(), nq, len(blob), json.dumps(TIMES)))
    sys.exit(0)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
