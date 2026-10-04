"""Partition-prototype efficiency ladder for stage 2 of the frozen composite QMAX_BALANCED_H2_PATCH1
(FREEZE_QMAX_BALANCED_H2_PATCH1.json) -- the fifteenth ruling's third priority (efficiency), requested 2026-09-25
(verbatim request text sha256 ae0921a8..., pinned by PREREGISTRATION_PROTOTYPE_LADDER.json):
    "How few fixed representatives per graph partition are sufficient to preserve the ALL-gold recall of our
     current node-level routing?"

Stage 2 of the composite (Cmax) scores every block by QMAX(B, q) = max_{v in B} q . e_v over ALL N nodes.  Here Cmax
is replaced by a score over a fixed, query-free set of representatives per block, built once from the frozen fp16
vectors of that block's own nodes (no fitting across blocks or queries, no gold, no training, no tuned constant):
    CENT_MEAN   q . c_B,  c_B = mean_{v in B} e_v
    CENT_NORM   q . norm(sum_{v in B} norm(e_v))
    FPS_m       max_{j <= m} q . e_{p_j}: p_1 = the node nearest the block mean (the MEDOID rung is FPS_1),
                p_k = the node maximising min_{j < k} ||e_v - e_{p_j}|| (deterministic farthest-point sampling on the
                stored vectors, float64, ties -> lowest node id); m in 1, 2, 4, 8, 16, 32; |B| <= m -> every node
    RAND_m      the same max over the first m nodes of a seeded per-block permutation (control, same ladder)
A node-subset score is QMAX restricted to the subset and is computed from the SAME fp32 dot products as the accepted
streamed stage 2 (one pass over the fp16 shards with the chunking and query batches of _l1c_transfer_blockmax.py): the
all-nodes subset must equal the accepted stage 2 bit for bit, and FPS_m / RAND_m must equal QMAX on every block with
|B| <= m and never exceed it (asserted).

Levels (each variant against the QMAX chain at the same level; exact McNemar, split A only):
    ALONE     the top 50 blocks of the representative channel alone (the IVF-like endpoint "prototypes -> P50")
    F0        the frozen RRF of [Cd, Cs, C_variant]  (node votes + stage 2: the composite's routing; P50)
    SAFE      the frozen F6 swap on that ranking
    PRIMARY   the frozen BALANCED_H2_PATCH1 patch on that SAFE (the composite end to end; the adoption level)
Everything but stage 2 is the frozen chain: the pinned uL3 head, the section-21 balanced beam and the section-16 patch
rule are exec'd verbatim from the pinned runner _l1c_transfer_composite.py; RT.build_cache / KB.contexts /
KB.f6_select are called unchanged.  The QMAX and BASE chains must reproduce transfer_repro_<cache>.json (asserted).

    python -u scratchpad/_l1c_proto.py <cache> --dry   identities only: no coverage number, overlap or diagnostic of
                                                       any representative arm; no record
    python -u scratchpad/_l1c_proto.py <cache>         the pre-registered run -> results/L1_COVPART/proto_A_<cache>.json
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
LADDER = (1, 2, 4, 8, 16, 32)
M_MAX = max(LADDER)
RAND_SEED = 20260925                                         # RAND_m: one numpy Generator (PCG64) stream, blocks in id order
COH_SEED = 20260926                                          # the size-matched random partition of the cohesion diagnostic
NODE_VARIANTS = ["FPS_%d" % k for k in LADDER] + ["RAND_%d" % k for k in LADDER]
VARIANTS = ["CENT_MEAN", "CENT_NORM"] + NODE_VARIANTS
LEVELS = ("ALONE", "F0", "SAFE", "PRIMARY")
ARM_BAL, ARM_Q50, ARM_QSAFE, ARM_QPRIM = "BALANCED_H2_PATCH1", "QMAX_F0", "QMAX_SAFE", "QMAX_BALANCED_H2_PATCH1"
REF = {"ALONE": "QMAX_ALONE", "F0": ARM_Q50, "SAFE": ARM_QSAFE, "PRIMARY": ARM_QPRIM}
REF0 = {"F0": "L1", "SAFE": "SAFE", "PRIMARY": ARM_BAL}      # the same level without stage 2 (node votes only)
ALPHA = 0.01                                                 # GAIN / LOSS labels (as every section-24 record)
ALPHA_LOSS = 0.05                                            # significant loss
TOL_FRAC = 0.005                                             # PRESERVES: net loss <= max(2, round(0.5 % of n_A))
REQUEST_SHA256 = "ae0921a89dd3120ddd853ffcc1ba7c5eb7c7584426f13c7c4a5d7063228fae6a"
MARK_BEAM = "# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording"
MARK_SAFE = "# ---------------------------------------------------------------- canonical SAFE machinery"
MARK_PATCH = "# ---------------------------------------------------------------- the patch rule of section 16"
MARK_PATCH_END = "\n\n\npatches = {ARM_H2"


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


# ---------------------------------------------------------------- arguments, pins, pre-registration
name = sys.argv[1]
DRY = "--dry" in sys.argv
assert name in DEV_A_CACHES, "the ladder runs on the five DEV_A caches (split A) only"
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
fp_out = os.path.join(OUT, "proto_A_%s.json" % name)
PRE = os.path.join(OUT, "PREREGISTRATION_PROTOTYPE_LADDER.json")
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
secs_pinned = secs.copy()
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
assert int(sizes.sum()) == N and (np.bincount(hard, minlength=npart) == sizes).all()
order_nodes = np.argsort(hard, kind="stable")                # node ids ascending inside every block
ptr = np.zeros(npart + 1, np.int64)
ptr[1:] = np.cumsum(sizes)


def nodes_of(b):
    return order_nodes[ptr[b]:ptr[b + 1]]


# ---------------------------------------------------------------- query-free representatives (offline, per block, from the block's own frozen vectors)
def fps_block(X):
    """medoid-seeded farthest-point order of one block.  X: (n, dim) float64 stored vectors, rows in node-id order.
    returns (order <= M_MAX local indices, covering radius per ladder m, mean distance to the nearest representative per ladder m)."""
    n = X.shape[0]
    sq = (X * X).sum(axis=1)
    mu = X.mean(axis=0)
    d_mu = sq - 2.0 * (X @ mu) + float(mu @ mu)
    p = int(np.argmin(d_mu))                                 # p_1: the node nearest the block mean (first occurrence = lowest node id)
    sel = np.zeros(n, bool)
    sel[p] = True
    order = [p]
    dmin = np.maximum(sq + sq[p] - 2.0 * (X @ X[p]), 0.0)
    dmin[p] = 0.0
    rad, qe = {1: float(np.sqrt(dmin.max()))}, {1: float(np.sqrt(dmin).mean())}
    for k in range(2, min(M_MAX, n) + 1):
        p = int(np.argmax(np.where(sel, -np.inf, dmin)))     # the unselected node farthest from the selected set (ties -> lowest node id)
        sel[p] = True
        order.append(p)
        dmin = np.minimum(dmin, np.maximum(sq + sq[p] - 2.0 * (X @ X[p]), 0.0))
        dmin[p] = 0.0
        if k in LADDER:
            rad[k], qe[k] = float(np.sqrt(dmin.max())), float(np.sqrt(dmin).mean())
    for k in LADDER:
        if k not in rad:                                     # k > n: every node is a representative
            rad[k], qe[k] = 0.0, 0.0
    return order, [rad[k] for k in LADDER], [qe[k] for k in LADDER]


def represent(nodes, full):
    X = np.asarray(D._E16[nodes], np.float32).astype(np.float64)
    n = X.shape[0]
    nrm = np.sqrt((X * X).sum(axis=1))
    assert (nrm > 0).all(), "a zero stored vector"
    U = X / nrm[:, None]
    s = U.sum(axis=0)
    ns = float(np.sqrt(s @ s))
    order, rad, qe = fps_block(X)
    if not full:
        return ns / n, s, rad, qe
    cn = s / ns
    return ns / n, s, rad, qe, X.mean(axis=0), cn, U @ cn, order


t_ = time.time()
dim = int(D.Q.shape[1])
cent_mean = np.zeros((npart, dim), np.float32)
cent_norm = np.zeros((npart, dim), np.float32)
fps_order = np.full((npart, M_MAX), -1, np.int64)
R_act = np.zeros(npart)
rad_act = np.zeros((npart, len(LADDER)))
qe_act = np.zeros((npart, len(LADDER)))
cos_c = np.zeros(N, np.float32)                              # cos(e_v, CENT_NORM of v's block)
pct_c = np.zeros(N, np.float32)                              # v's centroid-distance percentile inside its block: 0 = most central, 1 = most peripheral
S_sum = np.zeros(dim)
for bi in range(npart):
    nodes = nodes_of(bi)
    R, s, rad, qe, mu, cn, cs, order = represent(nodes, True)
    cent_mean[bi], cent_norm[bi] = mu.astype(np.float32), cn.astype(np.float32)
    R_act[bi], rad_act[bi], qe_act[bi] = R, rad, qe
    S_sum += s
    fps_order[bi, :len(order)] = nodes[np.asarray(order, np.int64)]
    cos_c[nodes] = cs
    o = np.argsort(-cs, kind="stable")
    r = np.empty(len(nodes), np.float64)
    r[o] = np.arange(len(nodes))
    pct_c[nodes] = r / max(len(nodes) - 1, 1)
rng = np.random.default_rng(RAND_SEED)
rand_order = np.full((npart, M_MAX), -1, np.int64)
for bi in range(npart):
    nodes = nodes_of(bi)
    perm = rng.permutation(len(nodes))[:M_MAX]
    rand_order[bi, :len(perm)] = nodes[perm]
members = {"ALLNODES": np.ones(N, bool)}
for k in LADDER:
    for tag, od in (("FPS", fps_order), ("RAND", rand_order)):
        mk = np.zeros(N, bool)
        sel_ = od[:, :k]
        mk[sel_[sel_ >= 0]] = True
        assert (np.bincount(hard[mk], minlength=npart) == np.minimum(k, sizes)).all(), (tag, k)
        members["%s_%d" % (tag, k)] = mk
for tag in ("FPS", "RAND"):
    for k1, k2 in zip(LADDER[:-1], LADDER[1:]):
        assert not (members["%s_%d" % (tag, k1)] & ~members["%s_%d" % (tag, k2)]).any(), "representative sets are not nested"
VECTORS = {"QMAX": int(N), "CENT_MEAN": int(npart), "CENT_NORM": int(npart)}
for k in LADDER:
    VECTORS["FPS_%d" % k] = VECTORS["RAND_%d" % k] = int(np.minimum(k, sizes).sum())
TIMES["representatives_offline"] = round(time.time() - t_, 1)
G.log("representatives built offline in %.0fs: vectors scored per query %s (QMAX %d)" % (TIMES["representatives_offline"], json.dumps(VECTORS), N))


# ---------------------------------------------------------------- one pass over the fp16 shards: every node subset from the accepted stage-2 products
def representative_pass(subsets):
    """(nq, npart) float32 max_{v in B, v in subset} q . e_v for every subset mask, from the SAME fp32 products
    Q[qa:qz] @ E[a:b].T (same chunk, same query batch, same reduction) as _l1c_transfer_blockmax.blockmax_scores_streamed."""
    Q = np.ascontiguousarray(D.Q, dtype=np.float32)
    out = {v: np.full((nq, npart), -np.inf, np.float32) for v in subsets}
    for a, b_, blk in D._E16.blocks(TB.BLOCK):
        Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
        hc = hard[a:b_]
        lperm = np.argsort(hc, kind="stable")
        hs = hc[lperm]
        segs = {}
        for v, mk_ in subsets.items():
            keep = mk_[a + lperm]
            if keep.any():
                ub, st = np.unique(hs[keep], return_index=True)
                segs[v] = (lperm[keep], ub, st)
        for qa in range(0, nq, TB.CQ):
            qz = min(nq, qa + TB.CQ)
            QE = Q[qa:qz] @ Ec.T
            for v, (sl, ub, st) in segs.items():
                loc = np.maximum.reduceat(QE[:, sl], st, axis=1)
                out[v][qa:qz][:, ub] = np.maximum(out[v][qa:qz][:, ub], loc)
    for v in out:
        assert np.isfinite(out[v]).all(), "a block received no representative (%s)" % v
    return out


t_ = time.time()
QB = representative_pass(members)
assert QB["ALLNODES"].tobytes() == qb_ref.tobytes(), "the all-nodes subset does not reproduce the accepted stage 2 bit for bit"
del QB["ALLNODES"]
IDENT = {"all_nodes_subset == accepted stage 2 (bitwise)": True, "blocks_with_|B|<=m (FPS_m / RAND_m == QMAX exactly there)": {},
         "node-subset scores <= QMAX everywhere": True, "FPS_m and RAND_m monotone in m (nested sets)": True}
for v in NODE_VARIANTS:
    k = int(v.split("_")[1])
    small = sizes <= k
    assert (QB[v] <= qb_ref).all(), v
    assert (QB[v][:, small] == qb_ref[:, small]).all(), v
    IDENT["blocks_with_|B|<=m (FPS_m / RAND_m == QMAX exactly there)"][str(k)] = int(small.sum())
for tag in ("FPS", "RAND"):
    for k1, k2 in zip(LADDER[:-1], LADDER[1:]):
        assert (QB["%s_%d" % (tag, k1)] <= QB["%s_%d" % (tag, k2)]).all()
Qf = np.ascontiguousarray(D.Q, dtype=np.float32)
QB["CENT_MEAN"] = Qf @ cent_mean.T
QB["CENT_NORM"] = Qf @ cent_norm.T
TIMES["representative_scores_pass"] = round(time.time() - t_, 1)
G.log("representative scores in %.0fs; identities: %s" % (TIMES["representative_scores_pass"], json.dumps(IDENT)))

# ---------------------------------------------------------------- the frozen SAFE machinery + patch, one chain at a time
z0 = np.load(C.path, allow_pickle=True)
zz = {k_: z0[k_] for k_ in z0.files if k_ != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
assert zz["base_rank"].shape == (nq, TOPP) and (zz["base_rank"] == D.base_rank).all()
finals, SAFE_OF, weakest_of = {}, {}, {}                     # read by the exec'd build_patches
CHAL, GOLDP, QREF = {}, [], {}
FLAGS, OVL, EXPO, CHAIN_SECS = {}, {}, {}, {}

if not DRY:
    gold_nodes = [np.asarray(C.gold_nodes[qi], np.int64) for qi in range(nq)]
    pair_q, pair_g, pair_ptr = [], [], [0]
    for qi in rowsA:
        for g in gold_nodes[qi]:
            pair_q.append(int(qi))
            pair_g.append(int(g))
        pair_ptr.append(len(pair_q))
    pair_q, pair_g, pair_ptr = np.asarray(pair_q, np.int64), np.asarray(pair_g, np.int64), np.asarray(pair_ptr, np.int64)
    pair_b = hard[pair_g]
    pair_j = np.repeat(np.arange(nA), np.diff(pair_ptr))


def mask_top(rank, k=P_MAIN):
    M = np.zeros((nq, npart), bool)
    np.put_along_axis(M, rank[:, :k], True, axis=1)
    return M


def flags_from_sets(sets):
    return np.array([int(b_) in sets[int(q_)] for q_, b_ in zip(pair_q, pair_b)], bool)


def flags_from_patch(patch):
    return np.array([int(b_) in patch[int(q_)]["kept"] or int(g_) in patch[int(q_)]["nodes"] for q_, b_, g_ in zip(pair_q, pair_b, pair_g)], bool)


def run_chain(ch, top, arms):
    """frozen SAFE (RT.build_cache + KB.contexts + KB.f6_select) and the section-16 patch with the balanced beam on the
    block ranking `top` (None = the served BASE ranking).  arms = (P50 arm, SAFE arm, PRIMARY arm)."""
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
    EXPO[ch] = {"P50_nodes_mean": round(float(np.mean([sizes[list(p50[qi])].sum() for qi in rowsA])), 1),
                "SAFE_nodes_mean (= the PRIMARY served nodes, matched)": round(float(np.mean([sizes[list(safe[qi])].sum() for qi in rowsA])), 1),
                "patch_novel_nodes_mean": round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1),
                "patch_queries_truncated": int(sum(patch[qi]["truncated"] for qi in rowsA))}
    if ch == "QMAX":
        QREF.update(p50=p50, safe=safe, kept={qi: patch[qi]["kept"] for qi in rowsA}, nodes={qi: patch[qi]["nodes"] for qi in rowsA})
    if not DRY:
        a50, asafe, aprim = arms
        FLAGS[a50], FLAGS[asafe], FLAGS[aprim] = flags_from_sets(p50), flags_from_sets(safe), flags_from_patch(patch)
        if ch != "QMAX":
            o50 = np.array([len(p50[qi] & QREF["p50"][qi]) for qi in rowsA])
            osf = np.array([len(safe[qi] & QREF["safe"][qi]) for qi in rowsA])
            okp = np.array([len(patch[qi]["kept"] & QREF["kept"][qi]) for qi in rowsA])
            same = int(sum(1 for qi in rowsA if patch[qi]["kept"] == QREF["kept"][qi] and patch[qi]["nodes"] == QREF["nodes"][qi]))
            OVL[ch] = {"F0": {"recall_blocks@50 (mean |P50 & P50_QMAX_F0| / 50)": round(float(o50.mean()) / P_MAIN, 4), "rows_identical_P50": int((o50 == P_MAIN).sum())},
                       "SAFE": {"mean |SAFE & QMAX_SAFE| / 50": round(float(osf.mean()) / P_MAIN, 4), "rows_identical_SAFE": int((osf == P_MAIN).sum())},
                       "PRIMARY": {"kept_blocks_overlap_mean (of 49)": round(float(okp.mean()), 2), "rows_identical_served_set": same}}
    CHAIN_SECS[ch] = round(time.time() - t, 1)
    G.log("chain %-10s SAFE + patch in %5.1fs (RSS %.0f MB)" % (ch, CHAIN_SECS[ch], _rss_mb()))


t_ = time.time()
top_q = G.F0([Cd, Cs, Cmax], npart)[:, :TOPP]
run_chain("QMAX", top_q, (ARM_Q50, ARM_QSAFE, ARM_QPRIM))
run_chain("BASE", None, ("L1", "SAFE", ARM_BAL))
SC, PS = {}, {}
MQ = mask_top(Cmax)
OVL_ALONE = {}
if not DRY:
    FLAGS["QMAX_ALONE"] = MQ[pair_q, pair_b]
    FLAGS["DENSE_VOTE_ALONE"] = mask_top(Cd)[pair_q, pair_b]
    FLAGS["SPLADE_VOTE_ALONE"] = mask_top(Cs)[pair_q, pair_b]
    SC["QMAX"] = qb_ref[pair_q, pair_b]
    PS["QMAX"] = G.positions(Cmax)[pair_q, pair_b]
    for nm_, rk_ in (("DENSE_VOTE", Cd), ("SPLADE_VOTE", Cs), ("NODE_VOTE_L1 (F0 of Cd, Cs)", base_full)):
        OVL_ALONE[nm_] = round(float((mask_top(rk_) & MQ)[rowsA].sum(axis=1).mean()) / P_MAIN, 4)
EXPO_ALONE = {"QMAX": round(float(np.mean([sizes[Cmax[qi, :P_MAIN]].sum() for qi in rowsA])), 1)}
for v in VARIANTS:
    S_ = QB.pop(v)
    Cv = np.argsort(-S_, axis=1, kind="stable").astype(np.int64)
    EXPO_ALONE[v] = round(float(np.mean([sizes[Cv[qi, :P_MAIN]].sum() for qi in rowsA])), 1)
    if not DRY:
        Mv = mask_top(Cv)
        FLAGS[v + "_ALONE"] = Mv[pair_q, pair_b]
        OVL_ALONE[v] = round(float((Mv & MQ)[rowsA].sum(axis=1).mean()) / P_MAIN, 4)
        SC[v] = S_[pair_q, pair_b]
        PS[v] = G.positions(Cv)[pair_q, pair_b]
        del Mv
    top_v = G.F0([Cd, Cs, Cv], npart)[:, :TOPP]
    del S_, Cv
    run_chain(v, top_v, (v + "_F0", v + "_SAFE", v + "_PRIMARY"))
TIMES["chains"] = round(time.time() - t_, 1)

if DRY:
    G.log("DRY RUN (%.0fs, RSS %.0f MB): pins, F0(Cd, Cs) == served base_rank, stage 2 == the accepted streamed record, all-nodes subset "
          "== stage 2 bitwise, FPS / RAND nested and <= QMAX (== on |B| <= m), P50 == first 50 of every ranking, |SAFE| = 50, challengers "
          "independent of the ranking, patch exposure == SAFE exposure: all asserted on %d chains; no coverage number, overlap or diagnostic "
          "of any representative arm computed; no record.  times %s" % (time.time() - T_ALL, _rss_mb(), 2 + len(VARIANTS), json.dumps(TIMES)))
    sys.exit(0)

# ---------------------------------------------------------------- coverage of every arm on split A
ARMS = (["DENSE_VOTE_ALONE", "SPLADE_VOTE_ALONE", "QMAX_ALONE", "L1", "SAFE", ARM_BAL, ARM_Q50, ARM_QSAFE, ARM_QPRIM]
        + [v + "_" + L for v in VARIANTS for L in LEVELS])
IND, ANYV = {}, {}
for a in ARMS:
    f = FLAGS[a]
    IND[a] = np.array([pair_ptr[j + 1] > pair_ptr[j] and bool(f[pair_ptr[j]:pair_ptr[j + 1]].all()) for j in range(nA)], bool)
    ANYV[a] = np.array([bool(f[pair_ptr[j]:pair_ptr[j + 1]].any()) for j in range(nA)], bool)
hopsA = np.asarray(hops)[rowsA]
HOPS = sorted(set(int(x) for x in hopsA if x >= 0))


def mc(ref, arm, s=None):
    s = np.ones(nA, bool) if s is None else s
    g_, l_, p_ = G.X.mcnemar(IND[ref][s], IND[arm][s])
    return {"gained": g_, "lost": l_, "p": p_}


def mcl(ref, arm, s=None):
    o = mc(ref, arm, s)
    o["label"] = label(o)
    return o


def arm_out(a, refs):
    o = {"ALL": round(float(IND[a].mean()), 4), "ANY": round(float(ANYV[a].mean()), 4), "per_hop": {}}
    for h in HOPS:
        s = hopsA == h
        e = {"n": int(s.sum()), "ALL": round(float(IND[a][s].mean()), 4)}
        for r in refs:
            e["vs_" + r] = mc(r, a, s)
        o["per_hop"]["hop%d" % h] = e
    for r in refs:
        o["ALL_vs_" + r] = mcl(r, a)
    return o


res_arms = {}
for a in ("DENSE_VOTE_ALONE", "SPLADE_VOTE_ALONE", "L1", "SAFE", ARM_BAL):
    res_arms[a] = arm_out(a, [])
res_arms["QMAX_ALONE"] = arm_out("QMAX_ALONE", ["L1", "DENSE_VOTE_ALONE"])
res_arms[ARM_Q50] = arm_out(ARM_Q50, ["L1"])
res_arms[ARM_QSAFE] = arm_out(ARM_QSAFE, ["SAFE"])
res_arms[ARM_QPRIM] = arm_out(ARM_QPRIM, ["SAFE", "L1", ARM_BAL])
for v in VARIANTS:
    for L in LEVELS:
        refs = [REF[L]] + ([REF0[L]] if L in REF0 else ["L1", "DENSE_VOTE_ALONE"])
        res_arms[v + "_" + L] = arm_out(v + "_" + L, refs)

# ---------------------------------------------------------------- the reference chains reproduce the accepted record (asserted) and the replay record
recA = rec["arms"]
for a in ("L1", "SAFE", ARM_BAL, ARM_Q50, ARM_QSAFE, ARM_QPRIM):
    o, r = res_arms[a], recA[a]
    assert o["ALL"] == r["ALL"] and o["ANY"] == r["ANY"], (a, o["ALL"], r["ALL"])
    for h, e in r["per_hop"].items():
        assert o["per_hop"][h]["ALL"] == e["ALL"], (a, h)
for key, ref in (("ALL_vs_SAFE", "SAFE"), ("ALL_vs_L1", "L1"), ("ALL_vs_BALANCED_H2_PATCH1", ARM_BAL), ("ALL_vs_QMAX_F0", ARM_Q50), ("ALL_vs_QMAX_SAFE", ARM_QSAFE)):
    if key in recA[ARM_QPRIM]:
        o = mc(ref, ARM_QPRIM)
        assert (o["gained"], o["lost"]) == (recA[ARM_QPRIM][key]["gained"], recA[ARM_QPRIM][key]["lost"]), key
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert (IND["L1"] == np.asarray(recS["_ind_BASE"], bool)[rowsA]).all(), "BASE differs from the frozen replay record"
assert (IND["SAFE"] == np.asarray(recS["_ind_SAFE"], bool)[rowsA]).all(), "canonical SAFE not reproduced"
REPRODUCED = ("L1 / SAFE / BALANCED_H2_PATCH1 / QMAX_F0 / QMAX_SAFE / QMAX_BALANCED_H2_PATCH1 == transfer_repro_%s.json (ALL, ANY, per hop, "
              "PRIMARY's paired comparisons); L1 and SAFE == %s (split A)" % (name, SAFE_RECORD[name]))
G.log("reference reproduced: %s" % REPRODUCED)

# ---------------------------------------------------------------- the pre-registered per-level outcome
TOL = max(2, int(round(TOL_FRAC * nA)))


def preserves(o):
    net = o["lost"] - o["gained"]
    return bool(net <= TOL and not (o["lost"] > o["gained"] and o["p"] < ALPHA_LOSS))


levels = {}
for L in LEVELS:
    ref = REF[L]
    lv = {}
    for v in VARIANTS:
        a = v + "_" + L
        o = mc(ref, a)
        e = {"ALL": res_arms[a]["ALL"], "ANY": res_arms[a]["ANY"],
             "vs_QMAX": {**o, "net_loss": o["lost"] - o["gained"], "label": label(o)}, "PRESERVES": preserves(o),
             "vectors_scored_per_query": VECTORS[v], "dot_product_reduction_vs_QMAX": round(N / float(VECTORS[v]), 2)}
        if L in REF0:
            r0 = REF0[L]
            dq = int(IND[ref].sum()) - int(IND[r0].sum())            # covered queries QMAX adds at this level
            e["vs_no_stage2 (%s)" % r0] = mcl(r0, a)
            e["share_of_QMAX_gain_retained ((covered - no_stage2) / (QMAX covered - no_stage2); null if |QMAX - no_stage2| < %d queries)" % TOL] = (
                round((int(IND[a].sum()) - int(IND[r0].sum())) / float(dq), 3) if abs(dq) >= TOL else None)
            e["overlap_with_QMAX_chain"] = OVL[v][L]
            e["exposure"] = EXPO[v]
        else:
            e["recall_blocks@50 (mean |P50_alone & P50_QMAX_alone| / 50)"] = OVL_ALONE[v]
            e["exposure"] = {"P50_nodes_mean": EXPO_ALONE[v], "QMAX_ALONE_P50_nodes_mean": EXPO_ALONE["QMAX"]}
        lv[v] = e
    fps_ok = [k for k in LADDER if lv["FPS_%d" % k]["PRESERVES"]]
    rnd_ok = [k for k in LADDER if lv["RAND_%d" % k]["PRESERVES"]]
    levels[L] = {"reference": ref, "ALL_reference": res_arms[ref]["ALL"], "tolerance_net_loss": TOL,
                 "no_stage2_reference": REF0.get(L), "ALL_no_stage2_reference": res_arms[REF0[L]]["ALL"] if L in REF0 else None,
                 "variants": lv,
                 "FPS_m_preserving": fps_ok, "smallest_preserving_FPS_m": (min(fps_ok) if fps_ok else None),
                 "FPS_preserving_from_smallest_m_upward": (bool(fps_ok) and fps_ok == [k for k in LADDER if k >= min(fps_ok)]),
                 "RAND_m_preserving": rnd_ok, "smallest_preserving_RAND_m": (min(rnd_ok) if rnd_ok else None),
                 "CENT_MEAN_PRESERVES": lv["CENT_MEAN"]["PRESERVES"], "CENT_NORM_PRESERVES": lv["CENT_NORM"]["PRESERVES"]}
    G.log("level %-7s ref %s %.4f | %s | FPS preserving %s, RAND preserving %s" % (
        L, ref, res_arms[ref]["ALL"], " ".join("%s %.4f(%+d)" % (v, lv[v]["ALL"], -lv[v]["vs_QMAX"]["net_loss"]) for v in VARIANTS), fps_ok, rnd_ok))
ALONE_REFS = {a: res_arms[a]["ALL"] for a in ("DENSE_VOTE_ALONE", "SPLADE_VOTE_ALONE", "L1", "QMAX_ALONE")}
ALONE_REFS["recall_blocks@50 vs QMAX_ALONE"] = {k_: OVL_ALONE[k_] for k_ in ("DENSE_VOTE", "SPLADE_VOTE", "NODE_VOTE_L1 (F0 of Cd, Cs)")}

# ---------------------------------------------------------------- the outlier diagnostic (descriptive): were the dropped gold nodes peripheral in their block?
t_ = time.time()
Qn = D.Q
g_rank = np.zeros(len(pair_g), np.int64)                     # rank of the gold node among its block's nodes by q . e (1 = the block's max node); fp32, descriptive
for i in range(len(pair_g)):
    nodes = nodes_of(int(pair_b[i]))
    s_ = np.asarray(D._E16[nodes], np.float32) @ Qn[int(pair_q[i])]
    pos_ = int(np.searchsorted(nodes, pair_g[i]))
    assert nodes[pos_] == pair_g[i]
    g_rank[i] = int((s_ > s_[pos_]).sum()) + 1
pct_p, cos_p = pct_c[pair_g].astype(np.float64), cos_c[pair_g].astype(np.float64)


def pop_stats(idx, v, L):
    if len(idx) == 0:
        return {"n": 0}
    o = {"n": int(len(idx)),
         "centroid_distance_percentile_median (0 = most central node of its block, 1 = most peripheral)": round(float(np.median(pct_p[idx])), 3),
         "centroid_distance_percentile_mean": round(float(pct_p[idx].mean()), 3),
         "share_in_peripheral_quartile (percentile >= 0.75)": round(float((pct_p[idx] >= 0.75).mean()), 3),
         "cos_to_normalized_block_centroid_median": round(float(np.median(cos_p[idx])), 4),
         "share_that_is_the_block_max_node_for_its_query": round(float((g_rank[idx] == 1).mean()), 3),
         "rank_in_block_by_q.e_median": float(np.median(g_rank[idx]))}
    if v in NODE_VARIANTS:
        gap = SC["QMAX"][idx].astype(np.float64) - SC[v][idx].astype(np.float64)
        o["share_that_is_a_representative"] = round(float(members[v][pair_g[idx]].mean()), 3)
        o["share_block_score_equal_to_QMAX"] = round(float((gap == 0).mean()), 3)
        o["block_score_gap_QMAX_minus_variant_median"] = round(float(np.median(gap)), 4)
    if L == "ALONE" and v is not None:
        o["gold_block_position_median (variant / QMAX)"] = [float(np.median(PS[v][idx])), float(np.median(PS["QMAX"][idx]))]
    return o


diag = {"all_gold_nodes_split_A": pop_stats(np.arange(len(pair_g)), None, None), "per_level": {}}
try:
    from scipy.stats import mannwhitneyu
except Exception:                                            # descriptive only
    mannwhitneyu = None
for L in LEVELS:
    ref = REF[L]
    dl = {}
    for v in VARIANTS:
        a = v + "_" + L
        fr, fv = FLAGS[ref], FLAGS[a]
        lostq = IND[ref] & ~IND[a]
        dropped = np.nonzero(lostq[pair_j] & fr & ~fv)[0]
        retained = np.nonzero(fr & fv)[0]
        e = {"lost_queries (QMAX covers, variant not)": int(lostq.sum()), "gained_queries (variant covers, QMAX not)": int((IND[a] & ~IND[ref]).sum()),
             "dropped_gold_nodes (served by QMAX, not by the variant, in lost queries)": pop_stats(dropped, v, L),
             "retained_gold_nodes (served by both, all queries)": pop_stats(retained, v, L)}
        if mannwhitneyu is not None and len(dropped) >= 5 and len(retained) >= 5:
            e["mann_whitney_p (centroid-distance percentile, dropped vs retained; two-sided, descriptive)"] = float(mannwhitneyu(pct_p[dropped], pct_p[retained], alternative="two-sided").pvalue)
        dl[v] = e
    diag["per_level"][L] = dl
TIMES["outlier_diagnostic"] = round(time.time() - t_, 1)

# ---------------------------------------------------------------- partition cohesion (query-free): actual blocks vs a size-matched random partition
t_ = time.time()
rng2 = np.random.default_rng(COH_SEED)
permN = rng2.permutation(N)
R_rnd = np.zeros(npart)
rad_rnd = np.zeros((npart, len(LADDER)))
qe_rnd = np.zeros((npart, len(LADDER)))
for bi in range(npart):
    R, s, rad, qe = represent(np.sort(permN[ptr[bi]:ptr[bi + 1]]), False)
    R_rnd[bi], rad_rnd[bi], qe_rnd[bi] = R, rad, qe


def coh_summary(R, rad, qe):
    w = sizes / float(sizes.sum())
    multi = sizes >= 2
    pair_cos = np.where(multi, (sizes * R ** 2 - 1.0) / np.maximum(sizes - 1, 1), np.nan)
    r1 = rad[:, 0]
    ok = r1 > 0
    return {"resultant_length_size_weighted_mean (1 = identical vectors)": round(float((R * w).sum()), 4),
            "resultant_length_quartiles (blocks with >= 2 nodes)": [round(float(x), 4) for x in np.percentile(R[multi], [25, 50, 75])],
            "mean_pairwise_cosine_inside_a_block_size_weighted": round(float(np.nansum(pair_cos * w) / w[multi].sum()), 4),
            "fps_covering_radius_size_weighted_mean_by_m (Euclidean, stored vectors)": {str(k): round(float((rad[:, i] * w).sum()), 4) for i, k in enumerate(LADDER)},
            "fps_mean_distance_to_nearest_representative_by_m": {str(k): round(float((qe[:, i] * w).sum()), 4) for i, k in enumerate(LADDER)},
            "covering_radius_ratio_r_m_over_r_1_mean": {str(k): round(float(np.mean(rad[ok, i] / r1[ok])), 4) for i, k in enumerate(LADDER)}}


cohesion = {"global_resultant_length (all nodes; the anisotropy floor)": round(float(np.sqrt(S_sum @ S_sum)) / N, 4),
            "actual_partition": coh_summary(R_act, rad_act, qe_act),
            "size_matched_random_partition (seed %d)" % COH_SEED: coh_summary(R_rnd, rad_rnd, qe_rnd)}
TIMES["cohesion"] = round(time.time() - t_, 1)
G.log("cohesion: %s" % json.dumps(cohesion))

# ---------------------------------------------------------------- record
res = {"cache": name, "dataset": G.X.DS_OF[name], "partition": G.PARTITION_OF.get(name), "n_split_A": nA, "n_cache_rows": nq, "N": int(N), "npart": int(npart),
       "block_size_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
       "mode": "PREREGISTERED_PROTOTYPE_LADDER_SPLIT_A", "status": "DEV_A_EFFICIENCY_CALIBRATION (no selection: DEV_A is exhausted for selection)",
       "request_sha256": REQUEST_SHA256,
       "preregistration": {"path": os.path.relpath(PRE, G.X.REPO).replace("\\", "/"), "sha256": pre_sha},
       "cache_file": {"path": os.path.relpath(C.path, G.X.REPO).replace("\\", "/"), "sha256": sha_file(C.path), "row_query_ids_sha256": meta.get("row_query_ids_sha256")},
       "code": {"path": os.path.relpath(os.path.abspath(__file__), G.X.REPO).replace("\\", "/"), "sha256": sha_file(os.path.abspath(__file__))},
       "pinned": PINNED, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "platform": {"python": platform.python_version(), "numpy": np.__version__},
       "host_at_start": HOST0,
       "reference_reproduced": REPRODUCED,
       "stage2_reference": {"source": "_l1c_transfer_blockmax.blockmax_scores_streamed (accepted, transfer_repro_%s.json)" % name,
                            "sha256_scores": rec["stage2"]["streamed"]["sha256_scores"], "sha256_ranking": rec["stage2"]["streamed"]["sha256_ranking"]},
       "representatives": {"ladder": list(LADDER), "rand_seed": RAND_SEED, "vectors_scored_per_query": VECTORS,
                           "dot_product_reduction_vs_QMAX": {v: round(N / float(VECTORS[v]), 2) for v in VARIANTS},
                           "identities": IDENT},
       "alone_references (IVF-like endpoint comparators)": ALONE_REFS,
       "levels": levels, "arms": res_arms, "outlier_diagnostic": diag, "cohesion": cohesion,
       "seconds_by_stage": TIMES, "chain_seconds": CHAIN_SECS}
res["seconds"] = round(time.time() - T_ALL, 1)
res["process_rss_mb_at_end"] = round(_rss_mb(), 1)
G.S.wj(fp_out, res)
G.log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], os.path.relpath(fp_out, G.X.REPO), sha_file(fp_out)[:12]))
