"""STEP 2 -- FULL STRUCTURAL PATH PROVENANCE CACHE (KB multi-hop / set-aware phase).

Re-runs the FROZEN directional expansion (identical decisions, MAX_HOPS=3, BEAM=64) but records
the provenance chain of every returned structural node:

    seed_row -> a -> b -> node          (ordered NODES)
    P_seed   -> P_a -> P_b -> P_node    (ordered canonical PARTITIONS, derived at analysis time)

Stored aligned to the existing cache's s_node[qi, j] order, so every downstream script can join
on (qi, j) with zero ambiguity.

Parity gate: the recomputed `added` list must equal the cached s_node row EXACTLY -- that is the
frozen expansion reproduced decision-for-decision.  The cached s_hop is NOT the path length: the
frozen mechanism records the first hop at which a node was ever SCORED, while a node can be scored
at hop h, lose the beam, and only enter scope later.  p_len below is the true scope-entry depth
(== the length of the recorded provenance chain); the disagreement rate is reported, not asserted.

NEW_ENCODER_PASSES = 0 (reads the existing gte_qwen node/query embeddings).
Single owner: a .lock file per corpus; an existing output is never rewritten (resume-safe).

  python scratchpad/_l1kb_paths.py <dataset>|ALL
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
from _l1ps_cache import F32View

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
OUTD = f"{ROOT}/KB_MULTI_HOP_SET_ROUTER/paths"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
SMAX = 256
BEAM_MAIN = 64
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def expand_dir_prov(seed_rows, r_q, adjp, adji, deg, Xn, M, budget=TA.MAX_EDGES_SCORED):
    """Byte-for-byte the frozen TA.expand_dir, plus a parent pointer for every scoped node.
    The parent recorded is the frontier node that ATTAINED the node's best directional score
    (ties broken by smallest parent row id), so provenance is deterministic."""
    scope = set(int(x) for x in seed_rows); frontier = [int(x) for x in seed_rows]
    order = []; vmeta = {}; escored = 0
    par = {}
    rq = r_q.astype(np.float32)
    for hop in range(TA.MAX_HOPS):
        Vs = []; Ss = []; Ps = []
        for e in frontier:
            if deg[e] > TA.DEG_CAP:
                continue
            nb = adji[adjp[e]:adjp[e + 1]]
            if len(nb) == 0:
                continue
            escored += len(nb)
            nb = nb[deg[nb] <= TA.DEG_CAP]
            if len(nb) == 0:
                continue
            delta = Xn[nb] - Xn[e]; nd = np.linalg.norm(delta, axis=1)
            ok = nd > 1e-9
            if not ok.any():
                continue
            Vs.append(nb[ok]); Ss.append((delta[ok] / nd[ok, None]) @ rq)
            Ps.append(np.full(int(ok.sum()), e, np.int64))
        if not Vs:
            break
        V = np.concatenate(Vs); S = np.concatenate(Ss).astype(np.float64)
        PA = np.concatenate(Ps)
        uv, inv = np.unique(V, return_inverse=True)
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), -1e30); np.maximum.at(best_s, inv, S)
        # deterministic argmax parent per unique node
        idx = np.lexsort((PA, -S, inv))
        first = np.ones(len(idx), bool)
        if len(idx) > 1:
            first[1:] = inv[idx[1:]] != inv[idx[:-1]]
        bestpar = np.zeros(len(uv), np.int64)
        bestpar[inv[idx[first]]] = PA[idx[first]]
        for k in range(len(uv)):
            v = int(uv[k]); m = vmeta.get(v)
            if m is None:
                vmeta[v] = [hop + 1, float(best_s[k]), int(cnt[k])]
            else:
                m[2] += int(cnt[k])
                if best_s[k] > m[1]:
                    m[1] = float(best_s[k])
        new_mask = np.fromiter((int(v) not in scope for v in uv), bool, len(uv))
        if not new_mask.any():
            break
        cv = uv[new_mask]; cs = best_s[new_mask]; cp = bestpar[new_mask]
        keep = np.argsort(-cs, kind="stable")[:M]
        for i2 in keep:
            v = int(cv[i2])
            if v not in scope:
                scope.add(v); order.append((float(cs[i2]), v)); par[v] = int(cp[i2])
        frontier = [int(cv[i]) for i in keep]
        if len(scope) >= TA.M_MAX * 4 or escored >= budget:
            break
    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:TA.M_MAX]
    return added, vmeta, escored, par


def build(DS):
    out = f"{OUTD}/path_{DS}.npz"
    if os.path.exists(out):
        log(f"{DS}: path cache already complete -- nothing to do")
        return
    lock = out + ".lock"
    if os.path.exists(lock):
        log(f"{DS}: LOCK PRESENT -- another owner is building, refusing to duplicate")
        return
    os.makedirs(OUTD, exist_ok=True)
    open(lock, "w").write(f"{os.getpid()} {time.time()}")
    try:
        _build(DS, out)
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def _build(DS, out):
    log(f"=== PATH PROVENANCE {DS} ===")
    z = np.load(f"{ROOT}/runs/cache_{DS}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    rows = z["rows"]; seeds = z["seeds"]; s_node = z["s_node"]; s_hop = z["s_hop"]
    nq = meta["n_dev_queries"]
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    ndocs = len(hard)
    BIG = bool(meta["fp16_path"])
    log(f"docs={ndocs} nq={nq} fp16={BIG} predicted_peak~{meta['predicted_peak_gb']}GB")

    EMB = f"data/ukb_storage/{DS}/gte_qwen/"
    nodes = np.load(EMB + "nodes.npy", mmap_mode="r")
    qall = np.load(EMB + "queries_all.npy", mmap_mode="r")
    if BIG:
        X16 = np.empty((ndocs, nodes.shape[1]), np.float16)
        for a in range(0, ndocs, 50000):
            b = min(a + 50000, ndocs)
            blk = np.array(nodes[a:b], np.float32)
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
            X16[a:b] = blk.astype(np.float16); del blk
        Xn = F32View(X16)
    else:
        Xn = np.array(nodes, np.float32)
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    Qm = np.stack([np.asarray(qall[int(i)], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)

    p_root = np.full((nq, SMAX), -1, np.int32)   # originating retrieval seed (node row)
    p_a = np.full((nq, SMAX), -1, np.int32)      # 1st intermediate  (seed -> a)
    p_b = np.full((nq, SMAX), -1, np.int32)      # 2nd intermediate  (a -> b)
    p_seedslot = np.full((nq, SMAX), -1, np.int8)  # which of the SEED_K seeds the chain starts at
    p_len = np.zeros((nq, SMAX), np.int8)        # chain length == hop
    t = time.time(); bad_node = bad_hop = 0
    for qi in range(nq):
        sd = [int(x) for x in seeds[qi] if int(x) >= 0]
        r_q = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        added, vm, e, par = expand_dir_prov(sd, r_q, adjp, adji, deg, Xn, BEAM_MAIN)
        sslot = {v: k for k, v in enumerate(sd)}
        k = min(len(added), SMAX)
        cached = s_node[qi]
        for jj in range(k):
            v = int(added[jj])
            if int(cached[jj]) != v:
                bad_node += 1
            chain = []
            cur = v
            while cur in par and len(chain) < TA.MAX_HOPS + 1:
                cur = par[cur]; chain.append(cur)
            chain.reverse()          # [seed, a, b] ancestors, seed first
            if not chain:
                continue
            p_root[qi, jj] = chain[0]
            p_seedslot[qi, jj] = sslot.get(chain[0], -1)
            if len(chain) > 1:
                p_a[qi, jj] = chain[1]
            if len(chain) > 2:
                p_b[qi, jj] = chain[2]
            p_len[qi, jj] = len(chain)
            if len(chain) != int(s_hop[qi, jj]):
                bad_hop += 1
        if (qi + 1) % 500 == 0:
            log(f"   {qi+1}/{nq}  node_mismatch={bad_node} hop_mismatch={bad_hop}")
    dt = time.time() - t
    avail = int((s_node >= 0).sum())
    par_status = "EXACT" if bad_node == 0 else "MISMATCH"
    log(f"{DS}: FROZEN_EXPANSION_PARITY = {par_status} (node_mismatch={bad_node} of {avail}); "
        f"chain_depth_deeper_than_first_scored_hop={bad_hop} {dt:.1f}s")
    assert par_status == "EXACT", f"{DS}: provenance run diverged from the frozen cache"
    pmeta = {"dataset": DS, "n_dev_queries": nq, "SMAX": SMAX, "BEAM": BEAM_MAIN,
             "MAX_HOPS": TA.MAX_HOPS, "FROZEN_EXPANSION_PARITY": par_status,
             "n_nodes_scope_depth_gt_first_scored_hop": int(bad_hop),
             "n_struct_nodes_with_provenance": int((p_len > 0).sum()),
             "runtime_sec": round(dt, 1), "NEW_ENCODER_PASSES": 0}
    np.savez_compressed(out, p_root=p_root, p_a=p_a, p_b=p_b, p_seedslot=p_seedslot,
                        p_len=p_len, meta_json=json.dumps(pmeta))
    log(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    mp = f"{ROOT}/KB_MULTI_HOP_SET_ROUTER/paths_manifest.json"
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[DS] = pmeta
    json.dump(man, open(mp, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    for d in (DSETS if a == "ALL" else [a]):
        build(d)
    log("ALL PATH CACHES DONE")
