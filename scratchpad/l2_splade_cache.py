"""Step-1 of the E0-E3 phase: materialize EXACT candidate-level SPLADE for the C/P50 pilot corpus.

For every (dataset, split) in the pilot, compute per-candidate (aligned to the corpus CSR cand_ids order):
  - splade_scope_score.npy  (float16)  = exact sparse dot  doc_mat[cand] . Q_all[row_all]^T   over the P50 scope
  - splade_scope_rank.npy   (int16)    = within-scope rank by that score (0 = best)

We KEEP the existing splade_rank.npy (global top-200 membership rank) untouched — it means something different
(SPLADE's global retrieval rank). Provenance recorded in _splade_scope_manifest.json.

No re-encode. No full-corpus SPLADE retrieval. Reuses cached splade_doc_embs.pkl (order PASS) x splade_q_shards
(all-order). Method + alignment proven in results/L2/L2_SIGNAL_PROFILE.md (parity bit-exact).
"""
import os, sys, json, glob, time
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.abspath("."))

DATASETS = ["2wiki_clean", "musique_clean"]
SPLITS = ["train", "val", "test"]
CORP = "data/l2_corpus"
UKB = "data/ukb_storage"


def log(*a): print(*a, flush=True)


def main():
    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "doc_mat[cand] . Q_all[row_all]^T over P50 scope", "datasets": {}}
    for ds in DATASETS:
        base = f"{UKB}/{ds}/gte_qwen"
        from src.core.splade_scorer import SpladeScorer
        scr = SpladeScorer(ds); scr._ensure_matrix()
        doc_mat = scr._matrix.tocsr()
        shards = sorted(glob.glob(f"{base}/splade_q_shards/q_shard_*.npz"))
        Q = sp.vstack([sp.load_npz(s) for s in shards]).tocsr()
        splade_top200 = np.load(f"{base}/splade_top200_all.npy")
        log(f"[{ds}] doc_mat={doc_mat.shape} Q(all-order)={Q.shape}")
        manifest["datasets"][ds] = {}
        for split in SPLITS:
            d = f"{CORP}/{ds}/{split}"
            if not os.path.exists(f"{d}/query_offsets.npy"):
                continue
            off = np.load(f"{d}/query_offsets.npy")
            cand = np.load(f"{d}/cand_ids.npy")
            meta = json.load(open(f"{d}/query_meta.json"))
            Nq = len(meta); Npair = len(cand)
            score = np.zeros(Npair, dtype=np.float16)
            rank = np.full(Npair, -1, dtype=np.int16)
            par_ok = par_tot = 0
            t0 = time.time()
            for qi in range(Nq):
                s, e = int(off[qi]), int(off[qi + 1])
                c = cand[s:e]
                r = meta[qi]["row_all"]
                q = Q[r]
                sc = doc_mat[c].dot(q.T)
                sc = np.asarray(sc.todense()).ravel() if sp.issparse(sc) else np.asarray(sc).ravel()
                score[s:e] = sc.astype(np.float16)
                # within-scope rank (0 = best); stable
                order = np.argsort(-sc, kind="stable")
                rk = np.empty(len(order), dtype=np.int64); rk[order] = np.arange(len(order))
                rank[s:e] = rk.astype(np.int16)
                # parity spot-check on first 40 queries: global argmax == splade_top200[:,0]
                if qi < 40:
                    full = doc_mat.dot(q.T)
                    full = np.asarray(full.todense()).ravel() if sp.issparse(full) else np.asarray(full).ravel()
                    par_tot += 1
                    if int(np.argmax(full)) == int(splade_top200[r, 0]):
                        par_ok += 1
            np.save(f"{d}/splade_scope_score.npy", score)
            np.save(f"{d}/splade_scope_rank.npy", rank)
            dt = time.time() - t0
            # how many scope candidates had NO global-top200 info (splade_rank==-1) but now have a real score
            gtr = np.load(f"{d}/splade_rank.npy")
            was_missing = float((gtr < 0).mean())
            nonzero = float((score > 0).mean())
            manifest["datasets"][ds][split] = {
                "Nq": Nq, "Npair": Npair, "parity_argmax_rate": par_ok / max(1, par_tot), "parity_checked": par_tot,
                "frac_scope_missing_global_top200_now_scored": was_missing, "frac_nonzero_score": nonzero,
                "seconds": dt, "files": ["splade_scope_score.npy(f16)", "splade_scope_rank.npy(int16)"],
                "kept_untouched": "splade_rank.npy (== splade_global_top200_rank)",
            }
            log(f"  [{ds}/{split}] scored {Npair} pairs parity={par_ok}/{par_tot} was_missing={was_missing:.3f} nonzero={nonzero:.3f} {dt:.1f}s")
    os.makedirs("results/L2", exist_ok=True)
    json.dump(manifest, open("results/L2/_splade_scope_manifest.json", "w"), indent=2)
    log("WROTE results/L2/_splade_scope_manifest.json")
    log("DONE_SPLADE_CACHE")


if __name__ == "__main__":
    main()
