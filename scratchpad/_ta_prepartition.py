"""G2 Track-A CORRECTED — PRE-PARTITION structural evidence (parameter-free).

Canonical objective (user architecture lock):
    FULL CORPUS -> Dense/SPLADE -> parameter-free structural continuation -> augmented NODE EVIDENCE
    -> C-partition voting -> UPDATED partition ranking -> top P=50 -> hard union -> L2.

Three configurations, all sharing ONE deterministic compute contract and ZERO learned parameters:

  BASE  dense[:K] -> PR_d ;  splade[:K] -> PR_s ;  RRF(PR_d, PR_s) -> top-P
        (bit-exact replica of scratchpad/l1_eval_phase1.py evaluate_dataset, topo=C, router=dense+splade)

  A1    RRF BEFORE STRUCTURE
        node-level RRF(dense[:K], splade[:K]) -> fused node evidence -> seeds = fused[:SEED_K]
        -> parameter-free directional expansion -> structural node evidence (ordered by s_dir)
        -> PR_g ;  RRF(PR_d, PR_s, PR_g) -> top-P
        Structural nodes vote; they never occupy a candidate slot. A query with no structural
        evidence contributes no third channel (per-query mask) so M=0 collapses to BASE exactly.

  A2    STRUCTURE BEFORE RRF
        dense[:K]  -> seeds -> expansion -> augmented dense node list  -> PR_d'
        splade[:K] -> seeds -> expansion -> augmented splade node list -> PR_s'
        RRF(PR_d', PR_s') -> top-P
        Structural nodes are appended after the retrieval nodes in their own channel, so they
        receive strictly lower vote weight 1/(K0+K+j). M=0 -> lists unchanged -> BASE exactly.

Every constant below is an EXISTING frozen constant (canonical router or frozen Track-A compute
contract). No new free continuous parameter is introduced and nothing is tuned against labels.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

# ---- FROZEN CONSTANTS (canonical L1 router) ----
K0 = 60          # RRF constant, l1_eval_phase1.K0
K_LOCK = 100     # router K, L1_LOCKED_MANIFEST
P_MAIN = 50      # top-P partitions, L1_LOCKED_MANIFEST
# ---- FROZEN CONSTANTS (Track-A compute contract, _b12_build.py) ----
SEED_K = 5
M_MAX = 256
MAX_HOPS = 3
DEG_CAP = 300
MAX_EDGES_SCORED = 400000
TOP200 = 200

CACHE = "scratchpad/_ta"


# ================================================================= topology + graph (one pass, cached)
def load_topology(ds, log=print):
    """doc-row space, hard partition array, mem_idx (own + 1-hop neighbours' partitions) and the
    undirected adjacency -- ALL from master_nodes_{ds}.json + variant_C/partition_map.json.

    NOTE (deliberate deviation from _b12_build.build_adj): the adjacency is master_nodes.neighbors
    for EVERY dataset. _b12_build branched to kb.txt for metaqa; the corrected mainline forbids
    dataset-specific compute branches, and neighbors is exactly the graph the canonical mem_idx is
    already built from, so structural evidence and partition membership share one object."""
    os.makedirs(CACHE, exist_ok=True)
    cp = f"{CACHE}/{ds}_topo.npz"
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        log(f"[topo] cache hit {cp}")
        return (z["hard"], (z["mem_ptr"], z["mem_idx"]), int(z["npart"]),
                (z["adj_ptr"], z["adj_idx"]), z["deg"], json.loads(str(z["id2row_json"])))
    from src.pipeline.standardizer import load_nodes
    pm = json.load(open(f"scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json"))
    mpath = f"data/processed/master_nodes_{ds}.json"
    if not os.path.exists(mpath):
        mpath = "data/processed/master_nodes.json"
    all_nodes = load_nodes(mpath)
    srcs = set(n.metadata.get("source", "") for n in all_nodes)
    if len(srcs) > 1:
        all_nodes = [n for n in all_nodes if n.metadata.get("source") == ds]
    docs = [n for n in all_nodes if n.metadata.get("type") != "question"]
    n = len(docs)
    assert len(pm) == n, f"partition_map {len(pm)} vs docs {n}"
    id2row = {nd.node_id: i for i, nd in enumerate(docs)}
    hard = np.array([int(pm.get(docs[i].node_id, -1)) for i in range(n)], np.int64)
    npart = int(hard.max()) + 1
    # adjacency (undirected, deduped)
    adj_sets = [set() for _ in range(n)]
    for i, nd in enumerate(docs):
        for x in nd.neighbors:
            j = id2row.get(x)
            if j is not None and j != i:
                adj_sets[i].add(j); adj_sets[j].add(i)
    deg = np.array([len(s) for s in adj_sets], np.int32)
    adj_ptr = np.zeros(n + 1, np.int64); adj_ptr[1:] = np.cumsum(deg)
    adj_idx = np.empty(int(adj_ptr[-1]), np.int32)
    for i, s in enumerate(adj_sets):
        if s:
            adj_idx[adj_ptr[i]:adj_ptr[i + 1]] = np.fromiter(sorted(s), np.int32, len(s))
    # mem_idx = own partition + the partitions of this node's OUTGOING neighbours.
    # canonical load_partition_topology / _onehop_membership iterate nd.neighbors WITHOUT symmetrising,
    # so mem_idx must be built from the DIRECTED edge list; the undirected adj above is for structural
    # EXPANSION only (what the frozen Track-A contract _b12_build.build_adj uses). Verified bit-identical
    # to the reference by _ta_diff.py (0/19029 rows differ on SQuAD). HONEST NOTE: this was changed while
    # chasing the SQuAD external-parity mismatch and it did NOT move it -- directed vs symmetrised makes
    # no difference to the selection on any dev corpus. It is kept because it is the faithful replica.
    mem_lists = []
    for i, nd in enumerate(docs):
        s = {int(hard[i])}
        for x in nd.neighbors:
            jj = id2row.get(x)
            if jj is not None:
                s.add(int(hard[jj]))
        mem_lists.append(sorted(s))
    mlen = np.array([len(x) for x in mem_lists], np.int64)
    mem_ptr = np.zeros(n + 1, np.int64); mem_ptr[1:] = np.cumsum(mlen)
    mem_idx = np.concatenate([np.array(x, np.int32) for x in mem_lists])
    np.savez(cp, hard=hard, mem_ptr=mem_ptr, mem_idx=mem_idx, npart=npart,
             adj_ptr=adj_ptr, adj_idx=adj_idx, deg=deg, id2row_json=json.dumps(id2row))
    log(f"[topo] {ds} docs={n} npart={npart} edges_undirected={int(adj_ptr[-1])//2} -> cached")
    return hard, (mem_ptr, mem_idx), npart, (adj_ptr, adj_idx), deg, id2row


# ================================================================= canonical partition voting
def partition_ranking(node_lists, mem, npart):
    """EXACT replica of l1_eval_phase1.evaluate_dataset.partition_ranking, generalised to
    variable-length per-query node lists. Vote weight for the node at position r is 1/(K0+r);
    an out-of-range node still consumes its rank slot (canonical behaviour)."""
    mem_ptr, mem_flat = mem
    nq = len(node_lists); nnodes = len(mem_ptr) - 1
    S = np.zeros((nq, npart), np.float32); M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(node_lists[qi]):
            nd = int(nd)
            if nd < 0 or nd >= nnodes:
                continue
            # w stays a float64 Python scalar because the reference accumulates float32_array += float64
            # (promote, add in float64, cast back on store); casting w to float32 first would change the
            # rounding. HONEST NOTE: this was also changed while chasing the SQuAD external-parity
            # mismatch and it too was a no-op for it -- kept only for faithfulness to the reference.
            w = 1.0 / (K0 + r)
            ps = mem_flat[mem_ptr[nd]:mem_ptr[nd + 1]]
            S[qi, ps] += w
            np.maximum.at(M[qi], ps, w)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32); rows = np.arange(nq)[:, None]
        rank[rows, order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int32)


def rrf_partitions(rankings, npart, masks=None):
    """Canonical partition-level RRF. rankings[0] supplies the deterministic tie-break order, which
    at n=2 reproduces l1_eval_phase1's dense-insertion-order tie-break bit-for-bit.
    masks[c][q]=0 suppresses channel c for query q (no evidence -> no vote)."""
    nq = rankings[0].shape[0]
    rows = np.arange(nq)[:, None]; cols = np.arange(npart)[None, :]
    fv = np.zeros((nq, npart), np.float64)
    for c, rk in enumerate(rankings):
        pos = np.empty((nq, npart), np.int32); pos[rows, rk] = cols
        contrib = 1.0 / (K0 + pos)
        if masks is not None and masks[c] is not None:
            contrib = contrib * masks[c][:, None]
        fv += contrib
    base = rankings[0]
    fv_in = np.take_along_axis(fv, base, axis=1)
    idx2 = np.argsort(-fv_in, axis=1, kind="stable")
    return np.take_along_axis(base, idx2, axis=1).astype(np.int32)


# ================================================================= parameter-free geometry (frozen)
def residual(qvec, seed_rows, Xn):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not len(seed_rows):
        return q
    E = Xn[seed_rows]
    U, s, _ = np.linalg.svd(E.T, full_matrices=False); U = U[:, s > 1e-6]
    if U.shape[1] == 0:
        return q
    r = q - U @ (U.T @ q); nn = np.linalg.norm(r)
    return r / nn if nn > 1e-6 else q


def expand_dir(seed_rows, r_q, adjp, adji, deg, Xn, M, budget=MAX_EDGES_SCORED):
    """Frozen Track-A directional beam (_b12_build.expand_dir, CSR adjacency).
    Returns (added rows in s_dir order, vmeta, edges_scored)."""
    scope = set(int(x) for x in seed_rows); frontier = [int(x) for x in seed_rows]
    order = []; vmeta = {}; escored = 0
    rq = r_q.astype(np.float32)
    for hop in range(MAX_HOPS):
        Vs = []; Ss = []
        for e in frontier:
            if deg[e] > DEG_CAP:
                continue
            nb = adji[adjp[e]:adjp[e + 1]]
            if len(nb) == 0:
                continue
            escored += len(nb)
            nb = nb[deg[nb] <= DEG_CAP]
            if len(nb) == 0:
                continue
            delta = Xn[nb] - Xn[e]; nd = np.linalg.norm(delta, axis=1)
            ok = nd > 1e-9
            if not ok.any():
                continue
            Vs.append(nb[ok]); Ss.append((delta[ok] / nd[ok, None]) @ rq)
        if not Vs:
            break
        V = np.concatenate(Vs); S = np.concatenate(Ss).astype(np.float64)
        uv, inv = np.unique(V, return_inverse=True)
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), -1e30); np.maximum.at(best_s, inv, S)
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
        cv = uv[new_mask]; cs = best_s[new_mask]
        keep = np.argsort(-cs, kind="stable")[:M]
        for idx in keep:
            v = int(cv[idx])
            if v not in scope:
                scope.add(v); order.append((float(cs[idx]), v))
        frontier = [int(cv[i]) for i in keep]
        if len(scope) >= M_MAX * 4 or escored >= budget:
            break
    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:M_MAX]
    return added, vmeta, escored


def node_rrf(dense_k, splade_k):
    """Parameter-free node-level RRF over the two retrieval channels (A1 seed evidence)."""
    dr = {int(r): k for k, r in enumerate(dense_k)}
    sr = {int(r): k for k, r in enumerate(splade_k)}
    uni = list(dict.fromkeys([int(x) for x in dense_k] + [int(x) for x in splade_k]))
    sc = np.array([1.0 / (K0 + dr.get(c, TOP200)) + 1.0 / (K0 + sr.get(c, TOP200)) for c in uni])
    return [uni[k] for k in np.argsort(-sc, kind="stable")]
