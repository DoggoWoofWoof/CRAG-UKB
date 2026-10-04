"""Canonical C/P50 L2 corpus builder (STEP 1-5,8).
Interface = MASTER_TOPOLOGY C, C partitions, Dense+SPLADE partition-RRF K0=60, K=100, P=50.
Three separate objects:
  (1) FULL SCOPE  : every query's complete ~5k P50 candidate universe (CSR arrays). NEVER truncated.
  (2) TRAIN SUBSET: all in-scope positives + tagged sampled negatives (separate index arrays).
  (3) EXPENSIVE   : relation/path/graph features — NOT built here (Step 9 profiling first).
Reuses cached embeddings/top200/partition maps. NO Qwen/SPLADE re-encode. Routing bit-identical to
scratchpad/ac_scope_analysis.py (which is bit-identical to the L1 evaluator).
Compact dtypes, memmap .npy, CSR layout (query_offsets / cand_ids / labels / aligned feature arrays).
"""
import os, sys, json, time, argparse, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from l1_eval_phase1 import load_partition_topology
from ac_scope_analysis import _fusion_ranking

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False

K, P, K0 = 100, 50, 60
NEG_BUDGET = 128          # per-query negatives in the TRAIN subset (all positives always kept)
SEED = 1234

def _splits_for(dataset, j):
    si = j["split_indices"]
    if dataset == "webqsp":
        return {"all": si.get("all", list(range(len(j["ids"]))))}
    out = {}
    for s in ("train", "val", "test"):
        if si.get(s):
            out[s] = si[s]
    return out

def _norm_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True); n[n == 0] = 1.0
    return A / n

def _dense_scores_for_scope(qvecs, X, cand_lists):
    """dense cosine score for each query's scope candidates, from CACHED embeddings (no re-encode)."""
    out = []
    if _HAS_TORCH:
        Xt = torch.tensor(X, device="cuda")
        for qi, cand in enumerate(cand_lists):
            q = torch.tensor(qvecs[qi], device="cuda")
            s = (Xt[torch.tensor(cand, device="cuda")] @ q).float().cpu().numpy()
            out.append(s.astype(np.float16))
        del Xt; torch.cuda.empty_cache()
    else:
        for qi, cand in enumerate(cand_lists):
            out.append((X[cand] @ qvecs[qi]).astype(np.float16))
    return out

def build_split(dataset, split, qrows, j, hard, memC, npC, docs_C, X, dense_all, splade_all,
                id2idx, outdir):
    golds = j["golds"]; hops = j.get("hops", [None] * len(j["ids"]))
    qvecs = _norm_rows(np.load(f"data/ukb_storage/{dataset}/gte_qwen/queries_all.npy")[qrows].astype(np.float32))
    dense_t = dense_all[qrows]; splade_t = splade_all[qrows]
    rankC = _fusion_ranking(dense_t, splade_t, memC, npC, K)     # partition ranking, K=100

    nq = len(qrows)
    cand_lists, part_rank_lists = [], []
    q_meta = []
    for qi in range(nq):
        sel = rankC[qi][:P]                                       # top-50 partitions (fusion order)
        cand = np.concatenate([docs_C[p] for p in sel]) if len(sel) else np.array([], np.int64)
        # part_rank per candidate = fusion rank (0..49) of its hard partition
        rank_of_part = {int(p): r for r, p in enumerate(sel)}
        prank = np.array([rank_of_part[int(hard[c])] for c in cand], np.int16)
        cand_lists.append(cand.astype(np.int64)); part_rank_lists.append(prank)

    dscore_lists = _dense_scores_for_scope(qvecs, X, cand_lists)

    # global dense/splade ranks (top200 membership) -> map doc->rank
    q_off = [0]; cand_ids = []; labels = []; part_id = []; part_rank = []
    dense_score = []; dense_rank = []; splade_rank = []
    for qi in range(nq):
        gi = qrows[qi]
        cand = cand_lists[qi]
        # canonical order = dense score DESC (reproducible primary retrievability within scope)
        order = np.argsort(-dscore_lists[qi].astype(np.float32), kind="stable")
        cand = cand[order]; prank = part_rank_lists[qi][order]; dsc = dscore_lists[qi][order]
        gold_doc = set(id2idx[g] for g in golds[gi] if g in id2idx)
        lab = np.array([1 if int(c) in gold_doc else 0 for c in cand], np.uint8)
        dmap = {int(d): r for r, d in enumerate(dense_t[qi])}     # global dense rank (0..199) else -1
        smap = {int(d): r for r, d in enumerate(splade_t[qi])}
        drank = np.array([dmap.get(int(c), -1) for c in cand], np.int16)
        srank = np.array([smap.get(int(c), -1) for c in cand], np.int16)

        n_exp = len(gold_doc)                                    # golds mapped to docs (in-corpus)
        n_in = int(lab.sum())
        n_exp_raw = len(golds[gi])                               # raw expected (pre doc-map)
        if n_in == 0:
            status = "L1_ANY_FAIL"
        elif n_in < n_exp:
            status = "L1_PARTIAL"
        else:
            status = "L1_ALL_SUCCESS"
        q_meta.append({"query_id": j["ids"][gi], "dataset": dataset, "split": split,
                       "row_all": int(gi), "N_SCOPE": int(len(cand)),
                       "N_GOLD_EXPECTED_RAW": n_exp_raw, "N_GOLD_EXPECTED_INCORP": n_exp,
                       "N_GOLD_IN_SCOPE": n_in,
                       "ANY_GOLD_PRESENT": bool(n_in > 0), "ALL_GOLD_PRESENT": bool(n_in == n_exp and n_exp > 0),
                       "hop": hops[gi], "L1_STATUS": status})
        cand_ids.append(cand.astype(np.int32)); labels.append(lab); part_id.append(hard[cand].astype(np.int16))
        part_rank.append(prank); dense_score.append(dsc); dense_rank.append(drank); splade_rank.append(srank)
        q_off.append(q_off[-1] + len(cand))

    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/query_offsets.npy", np.array(q_off, np.int64))
    np.save(f"{outdir}/cand_ids.npy", np.concatenate(cand_ids))
    np.save(f"{outdir}/labels.npy", np.concatenate(labels))
    np.save(f"{outdir}/part_id.npy", np.concatenate(part_id))
    np.save(f"{outdir}/part_rank.npy", np.concatenate(part_rank))
    np.save(f"{outdir}/dense_score.npy", np.concatenate(dense_score))
    np.save(f"{outdir}/dense_rank.npy", np.concatenate(dense_rank))
    np.save(f"{outdir}/splade_rank.npy", np.concatenate(splade_rank))
    json.dump(q_meta, open(f"{outdir}/query_meta.json", "w"))

    # ---- (2) TRAIN SUBSET sampler: all positives + tagged negatives (deterministic) ----
    sampler = build_train_subset(q_off, cand_ids, labels, dense_rank, splade_rank, part_rank, split, outdir)

    npairs = q_off[-1]
    scopes = np.array([m["N_SCOPE"] for m in q_meta])
    from collections import Counter
    stt = Counter(m["L1_STATUS"] for m in q_meta)
    summ = {"dataset": dataset, "split": split, "N_queries": nq, "N_pairs": int(npairs),
            "mean_scope": float(scopes.mean()), "median_scope": float(np.median(scopes)),
            "p95_scope": float(np.percentile(scopes, 95)),
            "pct_ANY_GOLD_PRESENT": round(100 * np.mean([m["ANY_GOLD_PRESENT"] for m in q_meta]), 2),
            "pct_ALL_GOLD_PRESENT": round(100 * np.mean([m["ALL_GOLD_PRESENT"] for m in q_meta]), 2),
            "n_zero_ingold": int(stt.get("L1_ANY_FAIL", 0)),
            "L1_status": dict(stt), "N_positives": int(np.concatenate(labels).sum()),
            "train_subset": sampler,
            "bytes_on_disk": sum(os.path.getsize(f"{outdir}/{f}") for f in os.listdir(outdir))}
    json.dump(summ, open(f"{outdir}/_summary.json", "w"), indent=2)
    return summ

def build_train_subset(q_off, cand_ids_list, labels_list, dense_rank_list, splade_rank_list,
                       part_rank_list, split, outdir):
    """All positives + <=NEG_BUDGET tagged negatives/query, sampled deterministically from the FULL scope.
    Buckets (only those whose info exists now): DENSE_HARD, SPLADE_HARD, DENSE_SPLADE_DISAGREEMENT,
    HIGH_PARTITION_VOTE, SAME_PARTITION, RANDOM. relation/path/kNN confusers deferred (features absent)."""
    rng = np.random.RandomState(SEED)
    BUCKETS = ["DENSE_HARD", "SPLADE_HARD", "DENSE_SPLADE_DISAGREEMENT", "HIGH_PARTITION_VOTE", "SAME_PARTITION", "RANDOM"]
    sub_qoff = [0]; sub_local = []; sub_src = []
    bucket_counts = {b: 0 for b in BUCKETS}
    n_pos_total = 0
    for qi in range(len(q_off) - 1):
        lab = labels_list[qi]; drank = dense_rank_list[qi]; srank = splade_rank_list[qi]; prank = part_rank_list[qi]
        n = len(lab)
        pos = np.where(lab == 1)[0]; neg = np.where(lab == 0)[0]
        n_pos_total += len(pos)
        chosen = {}                                              # local_idx -> source tag (first wins)
        def add(idxs, tag, k):
            idxs = [int(i) for i in idxs if int(i) not in chosen and lab[int(i)] == 0]
            for i in idxs[:k]:
                chosen[i] = tag; bucket_counts[tag] += 1
        per = max(1, NEG_BUDGET // len(BUCKETS))
        # DENSE_HARD: best dense rank among negs (candidate order is already dense-desc -> lowest local idx)
        add(list(neg[np.argsort(np.where(drank[neg] < 0, 9999, drank[neg]))]), "DENSE_HARD", per)
        # SPLADE_HARD: best splade rank among negs
        add(list(neg[np.argsort(np.where(srank[neg] < 0, 9999, srank[neg]))]), "SPLADE_HARD", per)
        # DENSE_SPLADE_DISAGREEMENT: |drank - srank| large (both present)
        both = neg[(drank[neg] >= 0) & (srank[neg] >= 0)]
        if len(both):
            add(list(both[np.argsort(-np.abs(drank[both].astype(int) - srank[both].astype(int)))]), "DENSE_SPLADE_DISAGREEMENT", per)
        # HIGH_PARTITION_VOTE: candidates in the top selected partitions (lowest part_rank)
        add(list(neg[np.argsort(prank[neg])]), "HIGH_PARTITION_VOTE", per)
        # SAME_PARTITION: negatives sharing a positive's partition
        if len(pos):
            pos_parts = set(int(prank[p]) for p in pos)
            sp = [int(i) for i in neg if int(prank[i]) in pos_parts]
            rng.shuffle(sp); add(sp, "SAME_PARTITION", per)
        # RANDOM: fill remainder
        rem = NEG_BUDGET - len(chosen)
        if rem > 0:
            pool = [int(i) for i in neg if int(i) not in chosen]; rng.shuffle(pool); add(pool, "RANDOM", rem)
        # assemble: all positives + chosen negatives
        loc = list(pos) + list(chosen.keys())
        src = ["POSITIVE"] * len(pos) + [chosen[i] for i in chosen.keys()]
        sub_local.extend(loc); sub_src.extend(src); sub_qoff.append(len(sub_local))
    np.save(f"{outdir}/train_sub_offsets.npy", np.array(sub_qoff, np.int64))
    np.save(f"{outdir}/train_sub_local.npy", np.array(sub_local, np.int32))     # local idx within each query's scope
    json.dump(sub_src, open(f"{outdir}/train_sub_source.json", "w"))
    return {"n_pairs": len(sub_local), "n_positives": int(n_pos_total), "neg_budget": NEG_BUDGET,
            "bucket_counts": bucket_counts}

def run(datasets, splits, out_root):
    report = {}
    for dataset in datasets:
        j = json.load(open(f"data/ukb_storage/{dataset}/gte_qwen/query_ids_all.json"))
        hard, memC, npC, doc_nodes, _ = load_partition_topology(dataset, "C")
        id2idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
        docs_C = [np.where(hard == p)[0] for p in range(npC)]
        X = _norm_rows(np.load(f"data/ukb_storage/{dataset}/gte_qwen/nodes.npy").astype(np.float32))
        dense_all = np.load(f"data/ukb_storage/{dataset}/gte_qwen/dense_top200_all.npy")
        splade_all = np.load(f"data/ukb_storage/{dataset}/gte_qwen/splade_top200_all.npy")
        # ---- ALIGNMENT GATE: cached query emb argmax == dense_top200[:,0] on a sample ----
        qall = _norm_rows(np.load(f"data/ukb_storage/{dataset}/gte_qwen/queries_all.npy").astype(np.float32))
        samp = np.random.RandomState(0).choice(len(qall), size=min(200, len(qall)), replace=False)
        top1 = (qall[samp] @ X.T).argmax(1)
        agree = float(np.mean(top1 == dense_all[samp, 0]))
        assert agree >= 0.98, f"[{dataset}] ALIGNMENT_GATE FAIL: qemb-argmax vs dense_top200[:,0] agree={agree:.3f}"
        print(f"[{dataset}] ALIGNMENT_GATE PASS (dense-argmax parity {agree:.3f}) npart_C={npC} docs={len(hard)}", flush=True)
        avail = _splits_for(dataset, j)
        for split in splits:
            if split not in avail:
                continue
            t0 = time.time()
            outdir = os.path.join(out_root, dataset, split)
            s = build_split(dataset, split, np.array(avail[split]), j, hard, memC, npC, docs_C, X,
                            dense_all, splade_all, id2idx, outdir)
            s["build_sec"] = round(time.time() - t0, 1)
            report[f"{dataset}/{split}"] = s
            print(f"  [{dataset}/{split}] nq={s['N_queries']} pairs={s['N_pairs']} scope~{s['mean_scope']:.0f} "
                  f"ANY {s['pct_ANY_GOLD_PRESENT']}% ALL {s['pct_ALL_GOLD_PRESENT']}% zero-gold {s['n_zero_ingold']} "
                  f"disk {s['bytes_on_disk']/1e6:.1f}MB {s['build_sec']}s", flush=True)
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-root", default="data/l2_corpus")
    ap.add_argument("--report", default="results/L2/_l2_corpus_pilot_report.json")
    a = ap.parse_args()
    rep = run(a.datasets, a.splits, a.out_root)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=2)
    print(f"WROTE {a.report}", flush=True)
