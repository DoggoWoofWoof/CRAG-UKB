"""Query-level offset ladder on DEV_A (metaqa): predicted answer location from static embedding offsets, then
direct node retrieval around it, block votes through BASE's static table, RRF with BASE.  No traversal.

  O0  BASE
  O1  raw offset            Delta = q - c,  c = mean of the frozen SEED_K=5 RRF seeds
  O2  structural offset     Delta = G (q - c),  G = sum d d^T / sum |d|^2 over undirected STRUCT edges
      O2s  same direction, raw magnitude (G shrinks: trace 1, top eigenvalue 0.09)
  O3  typed offset          Delta_r = mean_{(u,r,v)} (e_v - e_u) by relation ID (labels never read);
                            r* = argmax_r sum_i w_i cos(q - e_{s_i}, Delta_r)  (multi-anchor consensus)
  O4  oracle relation       r* = the answer relation of the query (qtype annotation; DIAGNOSTIC ceiling only)
  O5  oracle point          e_hat = mean gold embedding (DIAGNOSTIC ceiling only)
Each predicted point e_hat_i = norm(e_{s_i} + Delta) retrieves its 100 nearest nodes; node score = max_i cosine."""
import json
import os
import sys

import numpy as np

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name)
C = D.C
E, Q = D.E, D.Q
N, dim = E.shape
npart, nq = D.npart, D.nq
res = {"cache": name, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
seeds = C.seeds.astype(np.int64)                        # (nq, 5) frozen RRF seeds
sw = 1.0 / (S.K0 + np.arange(seeds.shape[1]))           # anchor weights from the frozen seed ranks (RRF-like)
KN = S.K_LOCK
S.log("%s DEV_A n=%d BASE %.4f" % (name, int(D.A.sum()), res["BASE_A"]["ALL"]))

# ---------------------------------------------------------------- static offsets
s_, d_, r_, _ = D.struct_edges()
nrel = int(r_.max()) + 1
Delta = np.zeros((nrel, dim), np.float32)
for rel in range(nrel):
    m = r_ == rel
    if m.any():
        Delta[rel] = (E[d_[m]] - E[s_[m]]).mean(axis=0)
Nn, STRUCT, KNN, NERX = D.cd.keysets()
u = (STRUCT // np.int64(N)).astype(np.int64)
v = (STRUCT % np.int64(N)).astype(np.int64)
G = np.zeros((dim, dim), np.float64)
den = 0.0
for a in range(0, len(u), 20000):
    dd = (E[v[a:a + 20000]] - E[u[a:a + 20000]]).astype(np.float64)
    G += dd.T @ dd
    den += float((dd * dd).sum())
G /= den
G = G.astype(np.float32)

rd0 = S.canon_channel_rank([D.d_ids[i, :KN] for i in range(nq)], D.mem, npart)
rs0 = S.canon_channel_rank([D.s_ids[i, :KN] for i in range(nq)], D.mem, npart)
r0 = S.rrf_ranks([rd0, rs0])


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    return allv


def retrieve(points):
    """points (nq, m, dim) unit -> (ids (nq, KN), scores) nearest nodes by max_i cosine."""
    ids = np.zeros((nq, KN), np.int64)
    sc = np.zeros((nq, KN), np.float32)
    m = points.shape[1]
    for a in range(0, nq, 200):
        b = min(nq, a + 200)
        P = points[a:b].reshape(-1, dim)
        sims = (P @ E.T).reshape(b - a, m, N).max(axis=1)
        top = np.argpartition(-sims, KN, axis=1)[:, :KN]
        o = np.take_along_axis(sims, top, axis=1)
        srt = np.argsort(-o, axis=1)
        ids[a:b] = np.take_along_axis(top, srt, axis=1)
        sc[a:b] = np.take_along_axis(o, srt, axis=1)
    return ids, sc


def block_channel(ids):
    return S.canon_channel_rank([ids[i] for i in range(nq)], D.mem, npart)


def points_from_delta(Dl):
    """Dl (nq, dim) or (nq, 5, dim) -> unit points e_hat_i = norm(e_{s_i} + Delta_i)."""
    P = E[seeds] + (Dl[:, None, :] if Dl.ndim == 2 else Dl)
    return P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-9)


def arm(tag, points):
    ids, sc = retrieve(points)
    rk = block_channel(ids)
    run("%s (direct channel alone)" % tag, rk)
    run("%s + BASE (RRF 3)" % tag, S.rrf_ranks([rd0, rs0, rk]))
    return ids


run("O0 BASE", r0)
# seeds alone (no offset): nearest nodes of the seeds = a re-retrieval; isolates the offset's contribution
arm("O0' seeds only (Delta = 0)", points_from_delta(np.zeros((nq, dim), np.float32)))
c = E[seeds].mean(axis=1)
rho = Q - c
arm("O1 raw offset Delta = q - mean(seeds)", points_from_delta(rho))
Gr = rho @ G.T
arm("O2 structural offset Delta = G(q - c)", points_from_delta(Gr))
Gs = Gr * (np.linalg.norm(rho, axis=1, keepdims=True) / (np.linalg.norm(Gr, axis=1, keepdims=True) + 1e-9))
arm("O2s structural direction, raw magnitude", points_from_delta(Gs))
# O3 typed consensus: S(r) = sum_i w_i cos(q - e_{s_i}, Delta_r)
Dn = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
rho_i = Q[:, None, :] - E[seeds]                                        # (nq, 5, dim)
rho_i = rho_i / (np.linalg.norm(rho_i, axis=2, keepdims=True) + 1e-9)
Sr = np.einsum("qsd,rd->qsr", rho_i, Dn)                                # (nq, 5, nrel)
Scons = (Sr * sw[None, :, None]).sum(axis=1)                            # (nq, nrel)
rstar = Scons.argmax(axis=1)
arm("O3 typed consensus offset (argmax relation)", points_from_delta(Delta[rstar]))
# per-anchor relation choice (no consensus)
r_i = Sr.argmax(axis=2)                                                 # (nq, 5)
arm("O3' typed per-anchor offset", points_from_delta(Delta[r_i]))
# reverse offsets too (answer may be the source of the relation): choose over +/-Delta_r
Dboth = np.concatenate([Delta, -Delta], axis=0)
Dbn = Dboth / (np.linalg.norm(Dboth, axis=1, keepdims=True) + 1e-9)
Sr2 = np.einsum("qsd,rd->qsr", rho_i, Dbn)
rstar2 = (Sr2 * sw[None, :, None]).sum(axis=1).argmax(axis=1)
arm("O3b typed consensus over +/-Delta_r", points_from_delta(Dboth[rstar2]))

# ---- diagnostics with annotations (never used by the method)
qt = {}
with open(os.path.join(S.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        qt[o["query_id"]] = o["qtype"]
vocab = D.relation_vocab()
REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre", "year": "release_year",
       "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating", "tag": "has_tags"}
rid = {lab: i for i, lab in enumerate(vocab)}
first_rel = np.full(nq, -1, np.int64)
last_rel = np.full(nq, -1, np.int64)
for i in range(nq):
    parts = [REL[p] for p in qt[C.qids[i]].split("_to_") if p in REL]
    if parts:
        first_rel[i] = rid[parts[0]]
        last_rel[i] = rid[parts[-1]]
A = D.A
acc_first = float((rstar == first_rel)[A & (first_rel >= 0)].mean())
acc_last = float((rstar == last_rel)[A & (last_rel >= 0)].mean())
S.log("O3 consensus relation == first relation %.3f, == answer relation %.3f (DEV_A; diagnostic)" % (acc_first, acc_last))
res["diag"] = {"consensus_relation_acc_first": acc_first, "consensus_relation_acc_last": acc_last}
# O4 oracle relation (first hop from the topic): ceiling of the typed offset with a perfect relation reader
Dl = np.where(first_rel[:, None] >= 0, Delta[np.maximum(first_rel, 0)], 0.0).astype(np.float32)
arm("O4 ORACLE first-relation offset (diagnostic)", points_from_delta(Dl))
# O5 oracle point
gc = np.zeros((nq, 1, dim), np.float32)
for i in range(nq):
    gc[i, 0] = E[C.gold_nodes[i]].mean(axis=0)
gc /= np.linalg.norm(gc, axis=2, keepdims=True) + 1e-9
arm("O5 ORACLE answer-centroid point (diagnostic)", gc)
S.wj(os.path.join(S.OUT, "offsets_q_A_%s.json" % name), res)
