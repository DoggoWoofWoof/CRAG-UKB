"""Bounded Step-9 L2 signal profiler (C/P50 corpus).

Profiles DENSE / SPLADE / OFFSET / MIXTURE / GRAPH on a bounded sample of TRAIN queries
for the pilot datasets (2wiki_clean, musique_clean), on the *actual* frozen C/P50 corpus
(data/l2_corpus/<ds>/train). Plus a light, encode-free structural feasibility+coverage
probe for RELATION / PATH.

NO training. NO re-encode. NO full-corpus recompute. NO hotpot/squad. NO full metaqa.
All scores derive from cached artifacts:
  - nodes.npy / queries_all.npy   (dense embeddings, proven doc-order)
  - dense_top200_all / splade_top200_all
  - splade_doc_embs.pkl (doc CSR)  x  splade_q_shards/*.npz (query CSR, all-order)
  - cached OffsetHead / MixtureHead checkpoints (compat probe only, NOT canonical L2 model)

Writes: results/L2/L2_SIGNAL_PROFILE.json  (+ .md written by companion step).
"""
import os, sys, json, time, glob, pickle, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp

sys.path.insert(0, os.path.abspath("."))

DATASETS = ["2wiki_clean", "musique_clean"]
SPLIT = "train"
SAMPLE = 300          # dense/splade/offset/mixture sample
GRAPH_SAMPLE = 120    # graph + rel/path structural probe sample (heavier)
SEED = 1234
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UKB = "data/ukb_storage"
CORP = "data/l2_corpus"
OUT = "results/L2/L2_SIGNAL_PROFILE.json"

rng = np.random.default_rng(SEED)


def log(*a):
    print(*a, flush=True)


def load_corpus(ds, split):
    d = f"{CORP}/{ds}/{split}"
    off = np.load(f"{d}/query_offsets.npy")
    cand = np.load(f"{d}/cand_ids.npy", mmap_mode="r")
    lab = np.load(f"{d}/labels.npy", mmap_mode="r")
    dsc = np.load(f"{d}/dense_score.npy", mmap_mode="r")
    drk = np.load(f"{d}/dense_rank.npy", mmap_mode="r")
    srk = np.load(f"{d}/splade_rank.npy", mmap_mode="r")
    meta = json.load(open(f"{d}/query_meta.json"))
    return dict(off=off, cand=cand, lab=lab, dsc=dsc, drk=drk, srk=srk, meta=meta)


def gold_rank_stats(order_scores_desc_idx, cand_labels):
    """Given candidate scores, return the best (lowest) rank of any gold among the scope."""
    order = np.argsort(-order_scores_desc_idx, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    gold_pos = np.where(cand_labels == 1)[0]
    if len(gold_pos) == 0:
        return None, None
    gr = ranks[gold_pos]
    return int(gr.min()), sorted(int(x) for x in gr)


def load_head(kind):
    """kind in {'offset','mixture'}. Pick the most-recent cached checkpoint whose
    state_dict shapes match the architecture. Returns (module, path) or (None,None)."""
    import torch.nn as nn
    files = sorted(glob.glob(f"{UKB}/_head_cache/head_*.pt"), key=os.path.getmtime, reverse=True)

    class OffsetHead(nn.Module):
        def __init__(self, d=1536):
            super().__init__(); self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))
        def forward(self, qn, seed):
            return F.normalize(seed + self.net(qn), dim=-1)

    class MixtureHead(nn.Module):
        def __init__(self, d=1536, K=8):
            super().__init__(); self.K, self.d = K, d
            self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, K * d))
        def forward(self, qn, seed):
            off = self.net(qn).view(-1, self.K, self.d)
            return F.normalize(seed.unsqueeze(1) + off, dim=-1)

    want_last = 1536 if kind == "offset" else 8 * 1536
    for f in files:
        try:
            sd = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            continue
        # find the net.2.weight shape
        keys = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
        last = None
        for k, shp in keys.items():
            if k.endswith("net.2.weight") or k.endswith(".2.weight"):
                last = shp
        if last is None:
            continue
        if last[0] == want_last and last[1] == 512:
            mod = OffsetHead() if kind == "offset" else MixtureHead()
            # strip possible prefix
            clean = {}
            for k, v in sd.items():
                kk = k
                for pre in ("net.",):
                    pass
                clean[k.split("net.", 1)[-1] if "net." in k else k] = v
            # rebuild with matching keys: our module expects 'net.0.weight' etc.
            target = mod.state_dict()
            newsd = {}
            for tk in target:
                # find a source key ending with tk
                match = [k for k in sd if k.endswith(tk)]
                if match:
                    newsd[tk] = sd[match[0]]
            if set(newsd) == set(target):
                mod.load_state_dict(newsd)
                mod.eval().to(DEV)
                return mod, f
    return None, None


def profile_dataset(ds):
    log(f"\n================ {ds} ================")
    base = f"{UKB}/{ds}/gte_qwen"
    res = {"dataset": ds, "split": SPLIT}

    C = load_corpus(ds, SPLIT)
    Nq = len(C["meta"])
    log(f"  corpus queries={Nq}")

    # sample deterministically
    idx_all = np.arange(Nq)
    samp = rng.choice(idx_all, size=min(SAMPLE, Nq), replace=False)
    samp.sort()

    # dense embeddings
    docemb = np.load(f"{base}/nodes.npy").astype("float32")
    N = docemb.shape[0]
    Dn = F.normalize(torch.from_numpy(docemb).to(DEV), dim=1)
    qall = np.load(f"{base}/queries_all.npy").astype("float32")
    Qn = F.normalize(torch.from_numpy(qall).to(DEV), dim=1)
    dense_top200 = np.load(f"{base}/dense_top200_all.npy")
    splade_top200 = np.load(f"{base}/splade_top200_all.npy")
    log(f"  N_docs={N} qall={qall.shape} dense_top200={dense_top200.shape}")

    # splade doc matrix + concatenated query shards (all-order)
    from src.core.splade_scorer import SpladeScorer
    scr = SpladeScorer(ds); scr._ensure_matrix()
    doc_mat = scr._matrix.tocsr()
    shards = sorted(glob.glob(f"{base}/splade_q_shards/q_shard_*.npz"))
    Q = sp.vstack([sp.load_npz(s) for s in shards]).tocsr()
    log(f"  splade doc_mat={doc_mat.shape} nnz={doc_mat.nnz}  Q(all-order)={Q.shape} nnz={Q.nnz}")

    def scope(qi):
        s, e = C["off"][qi], C["off"][qi + 1]
        return np.asarray(C["cand"][s:e]), np.asarray(C["lab"][s:e]), np.asarray(C["dsc"][s:e]), \
               np.asarray(C["drk"][s:e]), np.asarray(C["srk"][s:e]), s, e

    # ---------- SPLADE parity gate (proves q-vector + doc-row alignment jointly) ----------
    par_ok = 0; par_tot = 0
    for qi in samp[:60]:
        r = C["meta"][qi]["row_all"]
        q = Q[r]
        full = doc_mat.dot(q.T)
        full = np.asarray(full.todense()).ravel() if sp.issparse(full) else np.asarray(full).ravel()
        am = int(np.argmax(full))
        par_tot += 1
        if am == int(splade_top200[r, 0]):
            par_ok += 1
    parity = par_ok / max(1, par_tot)
    res["SPLADE_PARITY_argmax_vs_top200"] = {"rate": parity, "checked": par_tot}
    log(f"  [SPLADE parity] global-argmax==splade_top200[:,0]  rate={parity:.4f} on {par_tot} queries")

    # ---------- per-signal profiling ----------
    prof = {}

    # DENSE (from corpus dense_score, canonical order == dense desc)
    t0 = time.time(); dense_gr = []; n_pos = 0; n_cov_q = 0
    for qi in samp:
        cand, lab, dsc, drk, srk, s, e = scope(qi)
        best, _ = gold_rank_stats(dsc.astype("float32"), lab)
        if best is not None:
            dense_gr.append(best); n_pos += 1
    dt = time.time() - t0
    tot_cand = int(sum(C["off"][qi + 1] - C["off"][qi] for qi in samp))
    prof["DENSE"] = {
        "impl": "cached nodes.npy·queries_all.npy cosine (corpus dense_score)", "device": "GPU-precomputed",
        "output_dim": 1, "coverage": "100% of scope (every candidate has a real score)",
        "gold_best_rank_median": float(np.median(dense_gr)) if dense_gr else None,
        "gold_best_rank_p90": float(np.percentile(dense_gr, 90)) if dense_gr else None,
        "queries_with_gold": n_pos, "sec_per_1k_cand": 1000 * dt / max(1, tot_cand),
        "bytes_per_pair": 2, "cacheable": True, "batchable": True,
    }
    log(f"  [DENSE] gold best-rank median={np.median(dense_gr) if dense_gr else None}  q_with_gold={n_pos}")

    # SPLADE exact over scope
    t0 = time.time(); spl_gr = []; nonzero_frac = []; new_cov = []; n_pos = 0
    prev_missing = []  # frac candidates that had splade_rank==-1 (no global-top200 info) now given a real score
    scand = 0
    for qi in samp:
        cand, lab, dsc, drk, srk, s, e = scope(qi)
        r = C["meta"][qi]["row_all"]
        sub = doc_mat[cand]                       # (scope, vocab) CSR
        sc = sub.dot(Q[r].T)
        sc = np.asarray(sc.todense()).ravel() if sp.issparse(sc) else np.asarray(sc).ravel()
        scand += len(cand)
        nonzero_frac.append(float((sc > 0).mean()))
        prev_missing.append(float((srk < 0).mean()))
        best, _ = gold_rank_stats(sc.astype("float32"), lab)
        if best is not None:
            spl_gr.append(best); n_pos += 1
    dt = time.time() - t0
    prof["SPLADE"] = {
        "impl": "exact sparse dot: splade_doc_embs.pkl[cand] · splade_q_shards[row_all]  (NO re-encode)",
        "device": "CPU scipy sparse", "output_dim": 1,
        "coverage": "100% of scope — every P50 candidate gets an EXACT score",
        "prev_corpus_had_only_rank": "splade_rank was top-200 membership; mean frac of scope with rank==-1 (now scored) = %.3f" % float(np.mean(prev_missing)),
        "mean_nonzero_score_frac": float(np.mean(nonzero_frac)),
        "gold_best_rank_median": float(np.median(spl_gr)) if spl_gr else None,
        "gold_best_rank_p90": float(np.percentile(spl_gr, 90)) if spl_gr else None,
        "queries_with_gold": n_pos, "sec_per_1k_cand": 1000 * dt / max(1, scand),
        "bytes_per_pair": 2, "cacheable": True, "batchable": True,
        "reusable_vectors_exist": True, "reencode_required": False,
    }
    log(f"  [SPLADE] exact scope score: gold best-rank median={np.median(spl_gr) if spl_gr else None} "
        f"nonzero={np.mean(nonzero_frac):.3f} was-missing={np.mean(prev_missing):.3f} sec/1k={1000*dt/max(1,scand):.4f}")

    # OFFSET / MIXTURE
    for kind in ("offset", "mixture"):
        mod, path = load_head(kind)
        if mod is None:
            prof[kind.upper()] = {"impl": "cached head", "status": "NO_MATCHING_CHECKPOINT",
                                  "compatible": True, "note": "mechanism is pool-agnostic (pred·docemb); no checkpoint matched for empirical probe"}
            log(f"  [{kind.upper()}] no matching cached checkpoint (mechanism still compatible)")
            continue
        t0 = time.time(); gr = []; n_pos = 0; scand = 0
        for qi in samp:
            cand, lab, dsc, drk, srk, s, e = scope(qi)
            r = C["meta"][qi]["row_all"]
            seed_idx = int(dense_top200[r, 0])
            qn = Qn[r:r + 1]; seed = Dn[seed_idx:seed_idx + 1]
            with torch.no_grad():
                pred = mod(qn, seed)                      # offset:(1,d)  mixture:(1,K,d)
                cd = Dn[torch.from_numpy(cand).long().to(DEV)]   # (scope,d)
                if pred.dim() == 2:
                    sc = (cd @ pred[0])                   # (scope,)
                else:
                    sc = (cd @ pred[0].T).max(dim=1).values  # (scope,)
            sc = sc.cpu().numpy()
            scand += len(cand)
            best, _ = gold_rank_stats(sc.astype("float32"), lab)
            if best is not None:
                gr.append(best); n_pos += 1
        dt = time.time() - t0
        prof[kind.upper()] = {
            "impl": f"cached {kind} head forward -> pred vector(s); score = pred·docemb[cand]",
            "checkpoint": os.path.basename(path), "checkpoint_note": "AVAILABLE cache — compat/runtime probe, NOT the canonical L2 checkpoint",
            "device": "GPU", "output_dim": (1 if kind == "offset" else 8),
            "coverage": "100% of scope (dense-family dot; works on any candidate set)",
            "gold_best_rank_median": float(np.median(gr)) if gr else None,
            "gold_best_rank_p90": float(np.percentile(gr, 90)) if gr else None,
            "queries_with_gold": n_pos, "sec_per_1k_cand": 1000 * dt / max(1, scand),
            "bytes_per_pair": 2, "cacheable": True, "batchable": True, "compatible": True,
        }
        log(f"  [{kind.upper()}] ckpt={os.path.basename(path)} gold best-rank median={np.median(gr) if gr else None} "
            f"sec/1k={1000*dt/max(1,scand):.4f}")

    res["signals"] = prof

    # ---------- GRAPH (cheap) + RELATION/PATH structural coverage probe ----------
    graph_res = profile_graph(ds, C, samp[:GRAPH_SAMPLE], base)
    res["GRAPH"] = graph_res

    # free GPU
    del Dn, Qn
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return res


def profile_graph(ds, C, samp, base):
    """Cheap graph features from C edge families + encode-free RELATION/PATH structural
    coverage (upper-bound proxy: fraction of P50 candidates that carry a title-mention (A)
    or NER edge to another in-scope candidate — i.e., could receive a connecting-sentence
    relation/path score)."""
    sys.path.insert(0, os.path.abspath("scratchpad"))
    from ac_graph_diagnostic import load_graph_families
    t0 = time.time()
    fams, N = load_graph_families(ds)
    STRUCT, NER, Cg = fams["STRUCT"], fams["NER"], fams["C"]
    A = fams["A"]
    load_dt = time.time() - t0
    log(f"  [GRAPH] families loaded in {load_dt:.1f}s  STRUCT_nnz={STRUCT.nnz} NER_nnz={NER.nnz} C_nnz={Cg.nnz}")

    def scope(qi):
        s, e = C["off"][qi], C["off"][qi + 1]
        return np.asarray(C["cand"][s:e]), np.asarray(C["lab"][s:e])

    deg_struct_glob = np.asarray(STRUCT.sum(1)).ravel()
    deg_ner_glob = np.asarray(NER.sum(1)).ravel()
    deg_C_glob = np.asarray(Cg.sum(1)).ravel()

    rel_cov = []; path_cov = []; ncomp_frac = []; ingraph_frac = []
    t0 = time.time()
    for qi in samp:
        cand, lab = scope(qi)
        cmask = np.zeros(N, bool); cmask[cand] = True
        # within-scope A-degree (relation upper-bound: candidate has >=1 title-mention edge in scope)
        subA = A[cand][:, cand]
        a_deg = np.asarray(subA.sum(1)).ravel()
        rel_cov.append(float((a_deg > 0).mean()))
        # path upper-bound: reachable in exactly 2 hops via A within scope but not 1-hop
        subA2 = (subA @ subA)
        two = np.asarray(subA2.sum(1)).ravel()
        path_only = ((two > 0) & (a_deg == 0))
        path_cov.append(float(path_only.mean()))
        # any-graph membership (C): candidate has >=1 C edge in scope
        subC = Cg[cand][:, cand]
        c_deg = np.asarray(subC.sum(1)).ravel()
        ingraph_frac.append(float((c_deg > 0).mean()))
        # induced components on C within scope
        from scipy.sparse.csgraph import connected_components
        ncomp, lbl = connected_components(subC, directed=False)
        _, cnt = np.unique(lbl, return_counts=True)
        ncomp_frac.append(cnt.max() / len(cand))
    dt = time.time() - t0
    tot = int(sum(C["off"][qi + 1] - C["off"][qi] for qi in samp))
    out = {
        "impl": "C edge families (STRUCT/NER/C) reconstructed; per-scope degree, 2-hop, components",
        "device": "CPU scipy sparse", "families_load_sec": load_dt,
        "cheap_features": ["global_degree(struct/ner/C)", "in_scope_degree", "2hop_reach", "induced_components", "seed_geodesic(available)"],
        "sec_per_1k_cand": 1000 * dt / max(1, tot), "bytes_per_pair": 2, "cacheable": True, "compatible": True,
        "mean_global_deg_struct": float(deg_struct_glob.mean()), "mean_global_deg_ner": float(deg_ner_glob.mean()),
        "mean_global_deg_C": float(deg_C_glob.mean()),
        # encode-free RELATION/PATH structural coverage (upper bound on scorable candidates)
        "RELATION_structural_coverage_meanfrac": float(np.mean(rel_cov)),
        "PATH_structural_coverage_meanfrac": float(np.mean(path_cov)),
        "C_in_scope_membership_meanfrac": float(np.mean(ingraph_frac)),
        "C_largest_component_meanfrac": float(np.mean(ncomp_frac)),
        "coverage_note": "RELATION/PATH coverage = UPPER BOUND (structural A-edge/2-hop within scope, no topic-detection, no encode). "
                         "Actual relation/path signal is nonzero only for candidates whose title-entity is 1-hop/2-hop from a DETECTED query topic AND whose connecting sentence is text-encoded (gte-Qwen). Encode pass = the expensive deferred build.",
    }
    log(f"  [GRAPH] rel_cov(ub)={np.mean(rel_cov):.3f} path_cov(ub)={np.mean(path_cov):.3f} "
        f"C_membership={np.mean(ingraph_frac):.3f} largest_comp={np.mean(ncomp_frac):.3f} sec/1k={1000*dt/max(1,tot):.4f}")
    return out


def main():
    all_res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "sample_dense_splade_offset_mixture": SAMPLE,
               "sample_graph_relpath": GRAPH_SAMPLE, "split": SPLIT, "device": str(DEV), "datasets": {}}
    for ds in DATASETS:
        all_res["datasets"][ds] = profile_dataset(ds)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(all_res, open(OUT, "w"), indent=2)
        log(f"  wrote {OUT}")
    log("DONE_PROFILE")


if __name__ == "__main__":
    main()
