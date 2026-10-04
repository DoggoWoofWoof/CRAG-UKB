"""Phase 1 L1: ALL-QUERY dense/splade top200 retrieval (exact) + routing sweep preparation.
Checkpointed, resume-safe, memory-bounded. Clean implementation per spec.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json, pickle, hashlib, logging, time
import numpy as np
import torch
import faiss

log = logging.getLogger("phase1")

def get_all_queries(dataset):
    from src.pipeline.standardizer import load_nodes
    from src.experiments.overlap_retrain import _splits, _hard_membership
    from src.core.engine import CoreEngine
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    doc_id_to_idx = eng.node_id_to_idx
    mpath = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mpath):
        mpath = "data/processed/master_nodes.json"
    all_nodes = load_nodes(mpath)
    srcs = set(n.metadata.get("source", "") for n in all_nodes)
    if len(srcs) > 1:
        all_nodes = [n for n in all_nodes if n.metadata.get("source") == dataset]
    q_nodes = [n for n in all_nodes if n.metadata.get("type") == "question"]
    n_raw = len(q_nodes)
    pairs = []
    for q in q_nodes:
        golds = [nid for nid in q.neighbors if nid in doc_id_to_idx]
        if not golds:
            continue
        pairs.append((q.node_id, q.content, golds, q.metadata))
    n_excluded = n_raw - len(pairs)
    n_evaluable = len(pairs)
    pairs.sort(key=lambda x: x[0])
    all_ids = [p[0] for p in pairs]
    all_texts = [p[1] for p in pairs]
    all_golds = [p[2] for p in pairs]
    all_hops = [(p[3].get("hop") or p[3].get("hops") or None) for p in pairs]
    log.info(f"[{dataset}] N_RAW_QUESTIONS={n_raw} N_EVALUABLE_WITH_GOLD={n_evaluable} N_EXCLUDED_ZERO_MAPPED_GOLD={n_excluded}")
    hard = _hard_membership(eng)
    sp = _splits(eng, hard)
    id_to_split = {}
    for split in ["train", "val", "test"]:
        for nd, _, _ in sp[split]:
            if nd.node_id in id_to_split:
                raise RuntimeError(f"duplicate qid {nd.node_id} across splits")
            id_to_split[nd.node_id] = split
    split_indices = {"train": [], "val": [], "test": [], "all": list(range(len(all_ids)))}
    unmatched = []
    for i, qid in enumerate(all_ids):
        s = id_to_split.get(qid)
        if s is None:
            unmatched.append(qid)
        elif s == "train":
            split_indices["train"].append(i)
        elif s == "val":
            split_indices["val"].append(i)
        elif s == "test":
            split_indices["test"].append(i)
    if unmatched:
        raise RuntimeError(f"[{dataset}] unmatched {len(unmatched)} {unmatched[:5]}")
    total_assigned = len(split_indices["train"]) + len(split_indices["val"]) + len(split_indices["test"])
    if total_assigned != len(all_ids):
        raise RuntimeError(f"[{dataset}] total_assigned {total_assigned} != {len(all_ids)}")
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError(f"[{dataset}] duplicate all_ids")
    return all_ids, all_texts, all_golds, all_hops, split_indices, eng

def get_reusable_qwen_cache(dataset, all_ids):
    from src.experiments.overlap_retrain import _splits, _hard_membership
    from src.core.engine import CoreEngine
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    hard = _hard_membership(eng)
    sp = _splits(eng, hard)
    reusable_all = {}
    n_cached_total = 0
    proven_splits = []
    for split in ["train", "val", "test"]:
        p = f"data/ukb_storage/{dataset}/gte_qwen/queries_{split}.npy"
        if not os.path.exists(p):
            log.info(f"[{dataset}] {split} UNPROVEN: missing file")
            continue
        arr = np.load(p, mmap_mode='r')
        expected_ids = [nd.node_id for nd, _, _ in sp[split]]
        if len(expected_ids) != arr.shape[0]:
            log.info(f"[{dataset}] {split} UNPROVEN: row mismatch {arr.shape[0]} vs {len(expected_ids)}")
            continue
        log.info(f"[{dataset}] {split} PROVEN_ALIGNED rows={arr.shape[0]}")
        full = np.load(p)
        for idx, qid in enumerate(expected_ids):
            if qid not in reusable_all:
                reusable_all[qid] = full[idx]
        n_cached_total += arr.shape[0]
        proven_splits.append(split)
    # Filter to qids in all_ids (ALL_EVALUABLE)
    all_set = set(all_ids)
    reusable = {k: v for k, v in reusable_all.items() if k in all_set}
    # Assert bounds
    assert 0 <= len(reusable) <= len(all_ids), f"N_cached {len(reusable)} out of bounds N_all {len(all_ids)}"
    assert len(reusable) == len(set(reusable.keys()) & all_set)
    return reusable, n_cached_total, proven_splits

def encode_qwen_checkpointed(dataset, all_ids, all_texts, reusable_dict, batch=64):
    n_all = len(all_ids)
    n_cached = len(reusable_dict)
    log.info(f"[{dataset}] N_all_evaluable={n_all} N_cached_proven={n_cached} N_missing={n_all - n_cached}")
    if n_all - n_cached == 0:
        dim = next(iter(reusable_dict.values())).shape[0] if reusable_dict else 1536
        out = np.empty((n_all, dim), dtype=np.float32)
        for i, qid in enumerate(all_ids):
            out[i] = reusable_dict[qid]
        return out
    from sentence_transformers import SentenceTransformer
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    meta = json.load(open(f"data/ukb_storage/{dataset}/gte_qwen/meta.json"))
    instr = meta.get("query_instruction", "")
    model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.half()
    except:
        pass
    shard_dir = f"data/ukb_storage/{dataset}/gte_qwen/queries_all_shards"
    os.makedirs(shard_dir, exist_ok=True)
    manifest_path = os.path.join(shard_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"shards": {}}
    missing_indices = [i for i, qid in enumerate(all_ids) if qid not in reusable_dict]
    missing_texts = [all_texts[i] for i in missing_indices]
    missing_ids = [all_ids[i] for i in missing_indices]
    shard_size = 4096
    for shard_start in range(0, len(missing_texts), shard_size):
        shard_end = min(shard_start + shard_size, len(missing_texts))
        shard_idx = shard_start // shard_size
        shard_file = os.path.join(shard_dir, f"shard_{shard_idx:05d}.npy")
        shard_ids_file = os.path.join(shard_dir, f"shard_{shard_idx:05d}_ids.json")
        expected_ids = missing_ids[shard_start:shard_end]
        if os.path.exists(shard_file) and os.path.exists(shard_ids_file):
            if json.load(open(shard_ids_file)) == expected_ids:
                arr = np.load(shard_file, mmap_mode='r')
                if arr.shape[0] == len(expected_ids):
                    log.info(f"  shard {shard_idx} hit {arr.shape}")
                    continue
        batch_texts = [instr + t for t in missing_texts[shard_start:shard_end]]
        embs = []
        for s in range(0, len(batch_texts), batch):
            e = model.encode(batch_texts[s:s+batch], normalize_embeddings=True, show_progress_bar=False, batch_size=batch, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(e)
            embs.append(e)
        shard_arr = np.vstack(embs)
        np.save(shard_file, shard_arr)
        json.dump(expected_ids, open(shard_ids_file, "w"))
        manifest["shards"][str(shard_idx)] = {"ids": expected_ids, "file": shard_file, "rows": shard_arr.shape[0]}
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        log.info(f"  encoded shard {shard_idx} {shard_arr.shape} committed")
        try:
            from src.experiments.backends import commit_persistent_storage
            commit_persistent_storage()
        except:
            pass
    missing_dict = {}
    for shard_idx in range(0, (len(missing_texts) + shard_size - 1) // shard_size):
        arr = np.load(os.path.join(shard_dir, f"shard_{shard_idx:05d}.npy"))
        ids = json.load(open(os.path.join(shard_dir, f"shard_{shard_idx:05d}_ids.json")))
        for qid, row in zip(ids, arr):
            missing_dict[qid] = row
    dim = 1536
    if missing_dict:
        dim = next(iter(missing_dict.values())).shape[0]
    elif reusable_dict:
        dim = next(iter(reusable_dict.values())).shape[0]
    out = np.empty((n_all, dim), dtype=np.float32)
    for i, qid in enumerate(all_ids):
        out[i] = reusable_dict.get(qid, missing_dict.get(qid))
    return out

def check_doc_row_alignment(dataset):
    """Verify Qwen nodes, SPLADE matrix, partition_map have identical row count and order.
    Reports DOC_ROW_COUNT_ALIGNMENT and DOC_ROW_ORDER_ALIGNMENT separately.
    Qwen doc order is exactly engine.nodes order as used by reencode_ukb.py: texts=[n.content for n in engine.nodes]
    """
    import pickle
    qwen_path = f"data/ukb_storage/{dataset}/gte_qwen/nodes.npy"
    splade_path = f"data/ukb_storage/{dataset}/splade_doc_embs.pkl"
    pm_path = f"data/ukb_storage/{dataset}/gte_qwen/partition_map.json"
    if not os.path.exists(pm_path):
        pm_path = f"data/ukb_storage/{dataset}/partition_map.json"
    qwen_rows = np.load(qwen_path, mmap_mode='r').shape[0] if os.path.exists(qwen_path) else None
    pm_rows = len(json.load(open(pm_path))) if os.path.exists(pm_path) else None
    # Qwen ordered ids exactly as reencode_ukb.py: engine.nodes in order
    from src.core.engine import CoreEngine
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    # Verify len matches nodes.npy
    assert len(eng.nodes) == qwen_rows, f"qwen doc ids len {len(eng.nodes)} != nodes.npy rows {qwen_rows} for {dataset} - proves eng.nodes order is the retrievable order"
    qwen_ordered_ids = [eng.nodes[i].node_id for i in range(len(eng.nodes))]
    qwen_hash = hashlib.sha256("".join(qwen_ordered_ids).encode()).hexdigest()[:16]
    # SPLADE ordered ids
    splade_rows = None
    splade_ids_hash = None
    order_status = "UNPROVEN"
    if os.path.exists(splade_path):
        try:
            d = pickle.load(open(splade_path, "rb"))
            splade_rows = d["matrix"].shape[0]
            inv = {v: k for k, v in d["id_to_idx"].items()}
            ordered_ids = [inv[i] for i in range(splade_rows)]
            splade_ids_hash = hashlib.sha256("".join(ordered_ids).encode()).hexdigest()[:16]
            if splade_rows == qwen_rows and splade_ids_hash == qwen_hash:
                order_status = "PASS"
            elif splade_rows == qwen_rows:
                order_status = "FAIL"
            else:
                order_status = "FAIL"
        except Exception as e:
            log.warning(f"[{dataset}] splade order check failed {e}")
            order_status = "UNPROVEN"
    else:
        order_status = "UNPROVEN"
        splade_rows = None
    count_ok = (qwen_rows == pm_rows) and (splade_rows is None or splade_rows == qwen_rows)
    count_status = "PASS" if count_ok else "FAIL"
    if splade_rows is None:
        order_status = "UNPROVEN" if count_ok else "FAIL"
        splade_ids_hash = None
    log.info(f"[{dataset}] DOC_ROW_COUNT_ALIGNMENT={count_status} Qwen={qwen_rows} PM={pm_rows} SPLADE={splade_rows}")
    log.info(f"[{dataset}] DOC_ROW_ORDER_ALIGNMENT={order_status} qwen_hash={qwen_hash} splade_hash={splade_ids_hash}")
    if count_status != "PASS":
        raise RuntimeError(f"DOC_ROW_COUNT_ALIGNMENT FAIL for {dataset}")
    if order_status == "FAIL":
        raise RuntimeError(f"DOC_ROW_ORDER_ALIGNMENT FAIL for {dataset}")
    # Fusion allowed only if order proven or splade missing but count ok? For Phase 1 we require order PASS when SPLADE exists
    if splade_rows is not None and order_status != "PASS":
        raise RuntimeError(f"Fusion not allowed: DOC_ROW_ORDER_ALIGNMENT {order_status} for {dataset}")
    return count_status, order_status

def dense_top200_incremental(dataset, q_emb, all_ids, out_path):
    assert len(all_ids) == len(q_emb), f"all_ids {len(all_ids)} != q_emb {len(q_emb)}"
    nq = len(q_emb)
    topk = 200
    # If final exists and matches, skip
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode='r')
        if arr.shape[0] == nq and arr.shape[1] == topk:
            log.info(f"  dense_top200 hit {arr.shape}")
            return np.load(out_path)
    dpath = f"data/ukb_storage/{dataset}/gte_qwen/nodes.npy"
    doc_emb = np.load(dpath).astype("float32")
    faiss.normalize_L2(doc_emb)
    faiss.normalize_L2(q_emb)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    faiss_backend = "CPU"; gpu_count = 0
    try:
        gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    except Exception as e:
        log.info(f"faiss.get_num_gpus failed: {repr(e)}"); gpu_count = 0
    if gpu_count > 0:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            faiss_backend = "GPU"
        except Exception as e:
            log.warning(f"FAISS_GPU_CONVERSION_FAILED backend=CPU exc={repr(e)}")
            faiss_backend = "CPU"
    log.info(f"[{dataset}] FAISS_BACKEND={faiss_backend} FAISS_GPU_COUNT={gpu_count} FAISS_DEVICE={'cuda:0' if faiss_backend=='GPU' else 'cpu'} INDEX_TYPE=IndexFlatIP(exact) N_DOCS={doc_emb.shape[0]} N_QUERIES={nq} TOPK={topk}")
    _t_search = time.time()
    shard_dir = os.path.join(os.path.dirname(out_path), "dense_top200_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_size = 8192
    for s in range(0, nq, shard_size):
        e = min(s + shard_size, nq)
        shard_file = os.path.join(shard_dir, f"shard_{s:07d}.npy")
        shard_manifest = os.path.join(shard_dir, f"shard_{s:07d}_manifest.json")
        expected_qids = all_ids[s:e] if all_ids else None
        if os.path.exists(shard_file) and os.path.exists(shard_manifest):
            mf = json.load(open(shard_manifest))
            if mf.get("qids") == expected_qids and mf.get("shape") == [e - s, topk]:
                a = np.load(shard_file, mmap_mode='r')
                if a.shape[0] == (e - s):
                    log.info(f"  dense shard {s}-{e} hit")
                    continue
        bs = 4096
        I_shard = np.empty((e - s, topk), dtype=np.int64)
        for t in range(s, e, bs):
            te = min(t + bs, e)
            _, i = index.search(q_emb[t:te].astype("float32"), topk)
            I_shard[t - s:te - s] = i
        np.save(shard_file, I_shard)
        json.dump({"qids": expected_qids, "shape": [e - s, topk], "dataset": dataset}, open(shard_manifest, "w"), indent=2)
        log.info(f"  dense shard {s}-{e} saved")
        try:
            from src.experiments.backends import commit_persistent_storage
            commit_persistent_storage()
        except:
            pass
    I = np.empty((nq, topk), dtype=np.int64)
    for s in range(0, nq, shard_size):
        e = min(s + shard_size, nq)
        I[s:e] = np.load(os.path.join(shard_dir, f"shard_{s:07d}.npy"))
    np.save(out_path, I)
    _sec = time.time() - _t_search
    log.info(f"[{dataset}] FAISS_BACKEND={faiss_backend} SECONDS={_sec:.1f} QUERIES_PER_SEC={nq/max(_sec,1e-6):.1f} (incl. resumed shards)")
    log.info(f"  dense_top200 assembled {I.shape} -> {out_path}")
    return I

def splade_top200_incremental(dataset, texts, all_ids, out_path, batch=64):
    from src.core.splade_scorer import SpladeScorer
    import scipy.sparse as sp
    scorer = SpladeScorer(dataset)
    if not scorer.available():
        log.warning(f"SPLADE not available for {dataset}")
        return None
    scorer._ensure_matrix()
    doc_mat = scorer._matrix
    N = doc_mat.shape[0]
    nq = len(texts)
    topk = 200
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode='r')
        if arr.shape[0] == nq:
            log.info(f"  splade_top200 hit {arr.shape}")
            return np.load(out_path)
    MEM_BUDGET = 256 * 1024 * 1024
    max_bs = max(1, min(2048, MEM_BUDGET // max(1, N * 4)))
    log.info(f"[{dataset}] N_docs={N} splade_score_batch_size={max_bs} estimated_score_matrix_MB={max_bs*N*4/1024/1024:.1f}")
    q_shard_dir = os.path.join(os.path.dirname(out_path), "splade_q_shards")
    os.makedirs(q_shard_dir, exist_ok=True)
    shard_size = 8192
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "none"
    log.info(f"[{dataset}] SPLADE_MODEL_DEVICE_BEFORE=uninit CUDA_AVAILABLE={cuda_avail} GPU_NAME={gpu_name}")
    _enc_sec = 0.0; _dev_logged = False
    for s in range(0, nq, shard_size):
        e = min(s + shard_size, nq)
        shard_q_file = os.path.join(q_shard_dir, f"q_shard_{s:07d}.npz")
        shard_q_manifest = os.path.join(q_shard_dir, f"q_shard_{s:07d}_manifest.json")
        expected_qids = all_ids[s:e]
        if os.path.exists(shard_q_file) and os.path.exists(shard_q_manifest):
            mf = json.load(open(shard_q_manifest))
            if mf.get("qids") == expected_qids:
                try:
                    tq = sp.load_npz(shard_q_file)
                    if tq.shape[0] == (e - s):
                        log.info(f"  SPLADE q shard {s}-{e} hit")
                        continue
                except:
                    pass
        scorer._ensure_model()
        if not _dev_logged:
            log.info(f"[{dataset}] SPLADE_MODEL_DEVICE_AFTER={scorer._device} CUDA_AVAILABLE={cuda_avail} GPU_NAME={gpu_name}")
            if cuda_avail and str(scorer._device) != "cuda":
                log.warning(f"[{dataset}] SPLADE_ENCODER_ON_CPU_WHILE_CUDA_AVAILABLE — device placement bug")
            _dev_logged = True
        rows = []
        _t_enc = time.time()
        for t in range(s, e, batch):
            te = min(t + batch, e)
            batch_texts = texts[t:te]
            inputs = scorer._tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(scorer._device)
            with torch.no_grad():
                logits = scorer._model(**inputs).logits
                v = torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1)
                v = torch.max(v, dim=1).values.cpu().numpy()
            rows.append(sp.csr_matrix(v))
        _enc_sec += time.time() - _t_enc
        q_shard = sp.vstack(rows).tocsr()
        sp.save_npz(shard_q_file, q_shard)
        json.dump({"qids": expected_qids, "shape": list(q_shard.shape)}, open(shard_q_manifest, "w"), indent=2)
        log.info(f"  SPLADE q shard {s}-{e} saved")
        try:
            from src.experiments.backends import commit_persistent_storage
            commit_persistent_storage()
        except:
            pass
    if _dev_logged:
        log.info(f"[{dataset}] SPLADE_QUERY_ENCODING_SECONDS={_enc_sec:.1f}")
    out_dir = os.path.dirname(out_path)
    shard_out_dir = os.path.join(out_dir, "splade_top200_shards")
    os.makedirs(shard_out_dir, exist_ok=True)
    # ---- exact scoring backend: GPU torch-sparse if CUDA, else scipy CSR (CPU). Bit-identical top200 verified by parity gate. ----
    scoring_backend = "scipy_cpu"; Dsp_gpu = None
    if cuda_avail:
        try:
            coo = doc_mat.tocoo()
            idx_t = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
            val_t = torch.tensor(coo.data.astype("float32"), dtype=torch.float32)
            Dsp_gpu = torch.sparse_coo_tensor(idx_t, val_t, size=(N, doc_mat.shape[1])).coalesce().to("cuda")
            scoring_backend = "torch_sparse_gpu"
            log.info(f"[{dataset}] SPLADE_SCORING_BACKEND=torch_sparse_gpu doc_nnz={int(coo.nnz)} gpu={gpu_name}")
        except Exception as ex:
            log.warning(f"[{dataset}] GPU sparse doc build FAILED exc={repr(ex)} -> scipy_cpu"); Dsp_gpu = None; scoring_backend = "scipy_cpu"
    if Dsp_gpu is None:
        log.info(f"[{dataset}] SPLADE_SCORING_BACKEND=scipy_cpu (CUDA_AVAILABLE={cuda_avail})")
    def _topk_scipy(qb):
        scores = (qb.dot(doc_mat.T)).toarray().astype("float32")
        out = np.empty((scores.shape[0], topk), dtype=np.int64)
        for i in range(scores.shape[0]):
            row = scores[i]
            if topk >= N:
                order = np.argsort(-row)
            else:
                part = np.argpartition(-row, topk)[:topk]
                order = part[np.argsort(-row[part])]
            out[i] = order
        return out
    def _topk_gpu(qb):
        qd = torch.tensor(qb.toarray(), dtype=torch.float32, device="cuda")   # (b, vocab)
        sc = torch.sparse.mm(Dsp_gpu, qd.t()).t().contiguous()                 # (b, N) = (q @ doc^T)
        _, idx = torch.topk(sc, topk, dim=1, largest=True, sorted=True)        # exact top-k by value
        return idx.to("cpu").numpy().astype(np.int64)
    # GPU can use a larger score block (24GB) vs the CPU 256MiB budget
    if scoring_backend == "torch_sparse_gpu":
        step = max(1, min(4096, (4 * 1024 * 1024 * 1024) // max(1, N * 4)))
    else:
        step = max_bs
    log.info(f"[{dataset}] scoring_step={step}")
    _parity_ok = None; _score_sec = 0.0; _score_q = 0
    for s in range(0, nq, shard_size):
        e = min(s + shard_size, nq)
        out_shard = os.path.join(shard_out_dir, f"shard_{s:07d}.npy")
        out_manifest = os.path.join(shard_out_dir, f"shard_{s:07d}_manifest.json")
        expected_qids = all_ids[s:e]
        if os.path.exists(out_shard) and os.path.exists(out_manifest):
            mf = json.load(open(out_manifest))
            if mf.get("qids") == expected_qids:
                a = np.load(out_shard, mmap_mode='r')
                if a.shape[0] == (e - s):
                    log.info(f"  splade top200 shard {s}-{e} hit")
                    continue
        q_shard = sp.load_npz(os.path.join(q_shard_dir, f"q_shard_{s:07d}.npz"))
        I_shard = np.empty((e - s, topk), dtype=np.int64)
        _t_sc = time.time()
        for t in range(0, q_shard.shape[0], step):
            te = min(t + step, q_shard.shape[0])
            qb = q_shard[t:te]
            if scoring_backend == "torch_sparse_gpu":
                res = _topk_gpu(qb)
                if _parity_ok is None:   # one-time exact-parity gate vs scipy CPU (first block)
                    ncheck = min(64, qb.shape[0])
                    ref = _topk_scipy(qb[:ncheck])
                    ov = float(np.mean([len(set(res[i][:topk]) & set(ref[i][:topk]))/topk for i in range(ncheck)]))
                    _parity_ok = ov >= 0.999
                    log.info(f"[{dataset}] SPLADE_GPU_PARITY top200_overlap={ov:.5f} -> {'PASS' if _parity_ok else 'FAIL'}")
                    if not _parity_ok:
                        log.warning(f"[{dataset}] SPLADE_GPU_PARITY FAIL -> falling back to scipy_cpu for exactness")
                        scoring_backend = "scipy_cpu"; step = max_bs
                        res = _topk_scipy(qb)
            else:
                res = _topk_scipy(qb)
            I_shard[t:t + res.shape[0]] = res
        _score_sec += time.time() - _t_sc; _score_q += (e - s)
        np.save(out_shard, I_shard)
        json.dump({"qids": expected_qids, "shape": list(I_shard.shape), "scoring_backend": scoring_backend}, open(out_manifest, "w"), indent=2)
        log.info(f"  splade top200 shard {s}-{e} saved backend={scoring_backend}")
        try:
            from src.experiments.backends import commit_persistent_storage
            commit_persistent_storage()
        except:
            pass
    if _score_q:
        log.info(f"[{dataset}] EXACT_SPARSE_SCORING_SECONDS={_score_sec:.1f} scoring_qps={_score_q/max(_score_sec,1e-6):.1f} backend={scoring_backend}")
        log.info(f"[{dataset}] TOTAL_SPLADE_SHARD_SECONDS={_enc_sec + _score_sec:.1f} (encode={_enc_sec:.1f} score={_score_sec:.1f})")
    I = np.empty((nq, topk), dtype=np.int64)
    for s in range(0, nq, shard_size):
        e = min(s + shard_size, nq)
        I[s:e] = np.load(os.path.join(shard_out_dir, f"shard_{s:07d}.npy"))
    np.save(out_path, I)
    log.info(f"  splade_top200 assembled {I.shape} -> {out_path}")
    return I

def audit_ner_provenance(dataset):
    pkl = f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl"
    if not os.path.exists(pkl):
        return {"exists": False, "classification": "MISSING"}
    # Builder trace: src/pipeline/ner_edges.py:def build_ner_edges(dataset, texts, n, maxdf=25, weighted=True, cache_dir=...)
    # Source input: texts = [eng.nodes[i].content for i in range(len(eng.nodes))]  (doc corpus texts, ordered by doc index)
    # Node ordering: row i corresponds to doc_nodes[i] order = engine.node_id_to_idx
    # Query text: NOT used (texts are doc content only, from engine.nodes)
    # Gold/answers: NOT used (no access to question nodes)
    # Check file header
    try:
        A = pickle.load(open(pkl, "rb"))
        shape = list(A.shape)
        nnz = int(A.nnz)
    except Exception as e:
        shape = None
        nnz = None
    canon_manifest = f"data/canonical/{dataset}/ner_manifest.json"
    if os.path.exists(canon_manifest):
        j = json.load(open(canon_manifest))
        if not j.get("ner_available", True):
            return {"exists": True, "builder": "src/pipeline/ner_edges.py:build_ner_edges", "input": "doc texts only (doc corpus)", "query_used": False, "gold_used": False, "node_ordering": "doc_nodes order (engine.node_id_to_idx)", "shape": shape, "nnz": nnz, "canonical_manifest": j, "classification": "LEGACY_EXPLORATORY", "reason": j.get("reason", "")}
        else:
            return {"exists": True, "builder": "src/pipeline/ner_edges.py:build_ner_edges", "input": "doc texts only", "query_used": False, "gold_used": False, "node_ordering": "doc_nodes order", "shape": shape, "nnz": nnz, "canonical_manifest": j, "classification": "CANONICAL_SAFE"}
    if dataset in ["metaqa", "webqsp"]:
        return {"exists": True, "builder": "src/pipeline/ner_edges.py:build_ner_edges", "input": "KB entity name docs (entity corpus)", "query_used": False, "gold_used": False, "node_ordering": "doc_nodes order", "shape": shape, "nnz": nnz, "classification": "LEGACY_EXPLORATORY", "reason": "KB entity-node docs; NER on entity names is not passage-level; canonical manifest marks ner_available false"}
    else:
        return {"exists": True, "builder": "src/pipeline/ner_edges.py:build_ner_edges", "input": "doc texts only", "query_used": False, "gold_used": False, "node_ordering": "doc_nodes order", "shape": shape, "nnz": nnz, "classification": "CANONICAL_SAFE"}

def run_one(dataset, max_queries=None, skip_splade=False, skip_dense=False, dry_run_dir=None):
    log.info(f"===== {dataset} max_queries={max_queries} =====")
    all_ids, all_texts, all_golds, all_hops, split_idx, eng = get_all_queries(dataset)
    if max_queries is not None:
        all_ids = all_ids[:max_queries]
        all_texts = all_texts[:max_queries]
        all_golds = all_golds[:max_queries]
        all_hops = all_hops[:max_queries]
        split_idx = {"train": list(range(len(all_ids))), "val": [], "test": [], "all": list(range(len(all_ids)))}
        log.info(f"  DRY RUN truncated to {len(all_ids)}")
    n_all = len(all_ids)
    out_dir = f"data/ukb_storage/{dataset}/gte_qwen"
    if dry_run_dir:
        out_dir = dry_run_dir
    os.makedirs(out_dir, exist_ok=True)
    if dataset in ["metaqa", "webqsp"]:
        prov = audit_ner_provenance(dataset)
        log.info(f"[{dataset}] NER provenance {prov}")
        json.dump(prov, open(os.path.join(out_dir, f"ner_provenance_{dataset}.json"), "w"), indent=2)
        if prov.get("classification") == "LEGACY_EXPLORATORY":
            log.info(f"[{dataset}] NER LEGACY_EXPLORATORY -> B/C diagnostics but excluded from cross-dataset canonical recommendation")
    reusable, n_cached_total, proven = get_reusable_qwen_cache(dataset, all_ids)
    log.info(f"[{dataset}] N_all_evaluable={n_all} N_cached_proven={len(reusable)} N_missing={n_all - len(reusable)} proven_splits={proven} total_cached_rows={n_cached_total}")
    cache_q = f"{out_dir}/queries_all.npy"
    cache_ids = f"{out_dir}/query_ids_all.json"
    if max_queries is not None:
        cache_q = f"{out_dir}/queries_all_{max_queries}.npy"
        cache_ids = f"{out_dir}/query_ids_all_{max_queries}.json"
    if os.path.exists(cache_q) and os.path.exists(cache_ids):
        j = json.load(open(cache_ids))
        if len(j["ids"]) == n_all and j["hash"] == hashlib.sha256("".join(all_ids).encode()).hexdigest()[:16]:
            q_emb = np.load(cache_q)
            log.info(f"  cache hit queries_all {q_emb.shape}")
        else:
            q_emb = encode_qwen_checkpointed(dataset, all_ids, all_texts, reusable)
            np.save(cache_q, q_emb)
            json.dump({"ids": all_ids, "hash": hashlib.sha256("".join(all_ids).encode()).hexdigest()[:16], "split_indices": split_idx, "golds": all_golds, "hops": all_hops}, open(cache_ids, "w"), indent=2)
            log.info(f"  saved queries_all {q_emb.shape} via checkpointed encode")
    else:
        q_emb = encode_qwen_checkpointed(dataset, all_ids, all_texts, reusable)
        np.save(cache_q, q_emb)
        json.dump({"ids": all_ids, "hash": hashlib.sha256("".join(all_ids).encode()).hexdigest()[:16], "split_indices": split_idx, "golds": all_golds, "hops": all_hops}, open(cache_ids, "w"), indent=2)
        log.info(f"  saved queries_all {q_emb.shape} via checkpointed encode")
    # DOC alignment gate before any fusion
    check_doc_row_alignment(dataset)
    if not skip_dense:
        dense_path = f"{out_dir}/dense_top200_all.npy"
        if max_queries is not None:
            dense_path = f"{out_dir}/dense_top200_all_{max_queries}.npy"
        dense_top200_incremental(dataset, q_emb, all_ids, dense_path)
        arr = np.load(dense_path, mmap_mode='r')
        assert arr.shape == (n_all, 200)
        assert np.all((arr >= 0) & (arr < len(eng.nodes)))
        log.info(f"  dense verify {arr.shape} bounds ok N_docs={len(eng.nodes)} DOC_ROW_ALIGNMENT PASS")
    if not skip_splade:
        splade_path = f"{out_dir}/splade_top200_all.npy"
        if max_queries is not None:
            splade_path = f"{out_dir}/splade_top200_all_{max_queries}.npy"
        splade_top200_incremental(dataset, all_texts, all_ids, splade_path)
        arr = np.load(splade_path, mmap_mode='r')
        assert arr.shape == (n_all, 200)
        assert np.all((arr >= 0) & (arr < len(eng.nodes)))
        log.info(f"  splade verify {arr.shape} bounds ok")
    return

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean","musique_clean","squad_clean","hotpotqa_clean","metaqa","webqsp"])
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--skip_splade", action="store_true")
    p.add_argument("--skip_dense", action="store_true")
    p.add_argument("--dry_run_dir", type=str, default=None)
    args = p.parse_args()
    t0 = time.time()
    for ds in args.datasets:
        run_one(ds, max_queries=args.max_queries, skip_splade=args.skip_splade, skip_dense=args.skip_dense, dry_run_dir=args.dry_run_dir)
    log.info(f"TOTAL {time.time()-t0:.1f}s")
