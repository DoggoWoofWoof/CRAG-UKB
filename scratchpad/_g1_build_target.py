"""G1 target-substrate builder (frozen dataset-agnostic method; runs on Modal GPU).

Assembles the SAME C11a 5-expert interface for a TARGET dataset that data/l2_corpus/<dev> already holds,
using ONLY frozen, dataset-agnostic methods (no target learning, no interface change):

  Stage A  base C/P50 scope   : build_l2_corpus.build_split  -> query_offsets/cand_ids/labels/part_*/
                                 dense_score/dense_rank/splade_rank + train subset + query_meta
  Stage B  SPLADE scope score : exact per-candidate SPLADE dot over the P50 scope, queries encoded
                                 on-the-fly with naver/splade-cocondenser (no pre-built shards needed);
                                 parity spot-check argmax == splade_top200[:,0]
  Stage C  offset + mixture   : UNIVERSAL source heads (results/L2/_heads/universal_{offset,mixture}_src_gteqwen.pt)
                                 applied WITHOUT refit  -> offset_score.npy / mixture_score.npy (f16)
  Stage D  masked relation    : title-mention topology mask (CPU, from master_nodes) + fresh gte-Qwen2
                                 connecting-sentence encode  -> relation_mask.npy / relation_qwen_score.npy
  Stage E  expert_meta.npz    : per-query behavioral metadata (gold_count, L1_status, dense/splade
                                 disagreement, relation coverage)

Every stage is idempotent (skips when its final output exists) so the job is fully resumable.
TRAIN is a seeded subset (G1_TRAIN_CAP, default 25000) comparable to the dev training budget; VAL is full.
TEST is never built. Controlled by env: G1_DS (default squad_clean), G1_SPLITS (default "train,val"),
G1_TRAIN_CAP (default 25000).
"""
import os, sys, json, time, pickle, hashlib
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scratchpad"))
import numpy as np

DS = os.environ.get("G1_DS", "squad_clean")
SPLITS = os.environ.get("G1_SPLITS", "train,val").split(",")
TRAIN_CAP = int(os.environ.get("G1_TRAIN_CAP", "25000"))
CORP = "data/l2_corpus"
UKB = "data/ukb_storage"
HEADS = "results/L2/_heads"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:.0f}s][{DS}]", *a, flush=True)


# =====================================================================  Stage A: base C/P50 scope
def stage_a():
    import build_l2_corpus as B
    from l1_eval_phase1 import load_partition_topology
    def _norm(A):
        n = np.linalg.norm(A, axis=1, keepdims=True); n[n == 0] = 1.0; return A / n
    j = json.load(open(f"{UKB}/{DS}/gte_qwen/query_ids_all.json"))
    hard, memC, npC, doc_nodes, _ = load_partition_topology(DS, "C")
    id2idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
    docs_C = [np.where(hard == p)[0] for p in range(npC)]
    X = _norm(np.load(f"{UKB}/{DS}/gte_qwen/nodes.npy").astype(np.float32))
    dense_all = np.load(f"{UKB}/{DS}/gte_qwen/dense_top200_all.npy")
    splade_all = np.load(f"{UKB}/{DS}/gte_qwen/splade_top200_all.npy")
    # alignment gate — TIE-AWARE. Checks nodes.npy (X) and the precomputed dense_top200
    # ranking are the same embedding space/order. A raw argmax==precomputed check
    # spuriously fails on KB corpora with duplicate entity embeddings (webqsp: many
    # exact-cosine ties, verified score-gap <=1.2e-7), so a disagreement whose local
    # top-1 score TIES the precomputed top-1 score is not a misalignment — the same
    # tie-invariance used for the W0 dense/kNN parity gates.
    qall = _norm(np.load(f"{UKB}/{DS}/gte_qwen/queries_all.npy").astype(np.float32))
    samp = np.random.RandomState(0).choice(len(qall), size=min(500, len(qall)), replace=False)
    S = qall[samp] @ X.T
    top1 = S.argmax(1); pre = dense_all[samp, 0]
    raw_agree = float(np.mean(top1 == pre))
    dis = np.where(top1 != pre)[0]
    ar = np.arange(len(samp))
    real_mis = int(np.sum(np.abs(S[ar, top1][dis] - S[ar, pre][dis]) > 1e-4)) if len(dis) else 0
    agree = 1.0 - real_mis / len(samp)
    assert agree >= 0.98, f"ALIGNMENT_GATE FAIL tie_aware={agree:.3f} raw={raw_agree:.3f} real_mis={real_mis}"
    log(f"Stage A: ALIGNMENT_GATE PASS tie_aware={agree:.3f} raw={raw_agree:.3f} real_mis={real_mis} npart_C={npC} docs={len(hard)}")
    si = j["split_indices"]
    out = {}
    for split in SPLITS:
        d = f"{CORP}/{DS}/{split}"
        if os.path.exists(f"{d}/query_offsets.npy") and os.path.exists(f"{d}/train_sub_local.npy" if split == "train" else f"{d}/query_offsets.npy"):
            log(f"Stage A: {split} exists, skip"); continue
        rows = np.array(si[split], dtype=np.int64)
        if split == "train" and TRAIN_CAP and len(rows) > TRAIN_CAP:
            rows = np.sort(np.random.RandomState(1234).choice(rows, size=TRAIN_CAP, replace=False))
            log(f"Stage A: {split} capped {len(si[split])} -> {len(rows)} (seed 1234)")
        t = time.time()
        s = B.build_split(DS, split, rows, j, hard, memC, npC, docs_C, X, dense_all, splade_all, id2idx, d)
        out[split] = {"nq": s["N_queries"], "pairs": s["N_pairs"], "mean_scope": round(s["mean_scope"], 1),
                      "ANY": s["pct_ANY_GOLD_PRESENT"], "ALL": s["pct_ALL_GOLD_PRESENT"], "sec": round(time.time() - t, 1)}
        log(f"Stage A: {split} nq={s['N_queries']} pairs={s['N_pairs']} scope~{s['mean_scope']:.0f} "
            f"ANY {s['pct_ANY_GOLD_PRESENT']}% ALL {s['pct_ALL_GOLD_PRESENT']}% {out[split]['sec']}s")
    return out


# =====================================================================  Stage B: SPLADE scope score
def _splade_encode_batch(scorer, texts, bs=256):
    import torch
    scorer._ensure_model()
    tok, mdl, dev = scorer._tokenizer, scorer._model, scorer._device
    outs = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        with torch.no_grad():
            logits = mdl(**inp).logits
            rl = torch.log(1 + torch.relu(logits))
            m = inp["attention_mask"].unsqueeze(-1)
            q = torch.max(rl * m, dim=1).values
        outs.append(q.float().cpu().numpy())
    return np.concatenate(outs, 0)


def stage_b():
    from src.core.splade_scorer import SpladeScorer
    import l2_relation as R
    scr = SpladeScorer(DS); scr._ensure_matrix(); doc_mat = scr._matrix.tocsr()
    # --- align SPLADE-pkl doc rows to the corpus id2idx / nodes.npy doc order ---
    # cand_ids and splade_top200 are in the corpus id2idx space (enumerate(doc_nodes)); the SPLADE
    # pkl carries its OWN id_to_idx. For integer-suffixed KBs (webqsp) the two orders coincide, so
    # this remap is the identity; for entity-name KBs (metaqa: metaqa_ent_$ ...) they differ and the
    # unremapped doc_mat makes both the scoring (doc_mat[c]) and the parity check wrong (par=0.000).
    from l1_eval_phase1 import load_partition_topology
    _, _, _, _doc_nodes, _ = load_partition_topology(DS, "C")
    _sp_id2idx = {nid: row for row, nid in scr._idx_to_id.items()}
    _miss = [n.node_id for n in _doc_nodes if n.node_id not in _sp_id2idx]
    assert not _miss, f"SPLADE pkl missing {len(_miss)} corpus doc ids (e.g. {_miss[:3]})"
    _remap = np.fromiter((_sp_id2idx[n.node_id] for n in _doc_nodes), dtype=np.int64, count=len(_doc_nodes))
    assert len(_remap) == doc_mat.shape[0], f"SPLADE remap {len(_remap)} != doc_mat rows {doc_mat.shape[0]}"
    doc_mat = doc_mat[_remap]        # row r now == corpus doc row r (id2idx / nodes.npy space)
    qtxt = R.qtext_all(DS)
    splade_top200 = np.load(f"{UKB}/{DS}/gte_qwen/splade_top200_all.npy")
    log(f"Stage B: doc_mat={doc_mat.shape} nqtxt={len(qtxt)}")
    out = {}
    for split in SPLITS:
        d = f"{CORP}/{DS}/{split}"
        if os.path.exists(f"{d}/splade_scope_score.npy"):
            log(f"Stage B: {split} exists, skip"); continue
        off = np.load(f"{d}/query_offsets.npy"); cand = np.load(f"{d}/cand_ids.npy")
        meta = json.load(open(f"{d}/query_meta.json")); nq = len(meta)
        rows = [meta[qi]["row_all"] for qi in range(nq)]
        score = np.zeros(len(cand), np.float16); rank = np.full(len(cand), -1, np.int16)
        par_ok = par_tot = 0; t = time.time(); BS = 256
        for b in range(0, nq, BS):
            qb = _splade_encode_batch(scr, [qtxt[rows[qi]] for qi in range(b, min(b + BS, nq))], bs=BS)
            for k, qi in enumerate(range(b, min(b + BS, nq))):
                s, e = int(off[qi]), int(off[qi + 1]); c = cand[s:e]; q = qb[k]
                sc = np.asarray(doc_mat[c].dot(q)).ravel()
                score[s:e] = sc.astype(np.float16)
                order = np.argsort(-sc, kind="stable"); rk = np.empty(len(order), np.int64); rk[order] = np.arange(len(order))
                rank[s:e] = rk.astype(np.int16)
                if qi < 40:
                    full = np.asarray(doc_mat.dot(q)).ravel(); par_tot += 1
                    par_ok += int(int(np.argmax(full)) == int(splade_top200[rows[qi], 0]))
            if b % (BS * 20) == 0:
                log(f"Stage B: {split} {min(b + BS, nq)}/{nq} ({time.time()-t:.0f}s)")
        np.save(f"{d}/splade_scope_score.npy", score); np.save(f"{d}/splade_scope_rank.npy", rank)
        par = par_ok / max(1, par_tot)
        assert par >= 0.90, f"SPLADE_PARITY FAIL {split} {par:.3f} (row-order misalignment)"
        out[split] = {"pairs": int(len(cand)), "parity": round(par, 3), "sec": round(time.time() - t, 1)}
        log(f"Stage B: {split} scored {len(cand)} parity={par_ok}/{par_tot} {out[split]['sec']}s")
    return out


# =====================================================================  Stage C: universal offset + mixture
def stage_c():
    import torch
    import l2_lib as L
    from l2_lib import OffsetHead, MixtureHead
    Dn, Qn, dtop = L.load_embeddings(DS)
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"{HEADS}/universal_offset_src_gteqwen.pt", map_location="cpu")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"{HEADS}/universal_mixture_src_gteqwen.pt", map_location="cpu")); mh.eval()
    out = {}
    for split in SPLITS:
        d = f"{CORP}/{DS}/{split}"
        if os.path.exists(f"{d}/offset_score.npy") and os.path.exists(f"{d}/mixture_score.npy"):
            log(f"Stage C: {split} exists, skip"); continue
        S = L.load_split(DS, split); t = time.time()
        oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn).astype("float16")
        mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn).astype("float16")
        np.save(f"{d}/offset_score.npy", oh_sc); np.save(f"{d}/mixture_score.npy", mh_sc)
        out[split] = {"pairs": int(len(oh_sc)), "sec": round(time.time() - t, 1)}
        log(f"Stage C: {split} offset+mixture {len(oh_sc)} pairs {out[split]['sec']}s")
    return out


# =====================================================================  Stage D: masked relation (fresh Qwen)
def _encode_qwen(sentences, model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"):
    import torch
    from sentence_transformers import SentenceTransformer
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    m = SentenceTransformer(model_name, trust_remote_code=True); m.max_seq_length = 256; m = m.half()
    emb = m.encode(sentences, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")


def stage_d():
    import l2_lib as L
    from l2_relation_qwen import build_sparse_edges
    emb_path = f"scratchpad/_g1_qwen_sent_emb_{DS}.npy"
    sents_pkl = f"scratchpad/_g1_qwen_sentences_{DS}.pkl"
    # need masks for every split?
    need = [sp for sp in SPLITS if not os.path.exists(f"{CORP}/{DS}/{sp}/relation_qwen_score.npy")]
    if not need:
        log("Stage D: all relation arrays exist, skip"); return {"skipped": True}
    # D1 topology masks (CPU, resumable-to-completion)
    for sp in SPLITS:
        cache = f"scratchpad/_reledges_sparse_{DS}_{sp}.pkl"
        if os.path.exists(cache + ".done"):
            log(f"Stage D1: {sp} edges done, skip"); continue
        t = time.time()
        while True:
            r = build_sparse_edges(DS, sp, budget_s=100000, verbose=True)
            if r is not None:
                break
        log(f"Stage D1: {sp} edges built ({time.time()-t:.0f}s)")
    # D2 collect global unique sentences
    if not os.path.exists(sents_pkl):
        global_t2g = {}; sentences = []; per_split = {}
        for sp in SPLITS:
            st = pickle.load(open(f"scratchpad/_reledges_sparse_{DS}_{sp}.pkl", "rb"))
            l2g = []
            for txt in st["sent_texts"]:
                g = global_t2g.get(txt)
                if g is None:
                    g = global_t2g[txt] = len(sentences); sentences.append(txt)
                l2g.append(g)
            per_split[f"{DS}/{sp}"] = {"local2global": l2g}
        pickle.dump({"sentences": sentences, "per_split": per_split,
                     "sha1": [hashlib.sha1(s.encode()).hexdigest() for s in sentences]},
                    open(sents_pkl, "wb"), protocol=4)
        log(f"Stage D2: {len(sentences)} unique connecting sentences")
    g = pickle.load(open(sents_pkl, "rb"))
    # D3 encode with gte-Qwen2 (fresh, canonical doc semantics)
    if not os.path.exists(emb_path):
        t = time.time()
        emb = _encode_qwen(g["sentences"]) if g["sentences"] else np.zeros((0, 1536), np.float32)
        np.save(emb_path, emb)
        log(f"Stage D3: encoded {emb.shape} sentences ({time.time()-t:.0f}s)")
    Sent = np.load(emb_path)
    # D4 materialize per split
    Q = np.load(f"{UKB}/{DS}/gte_qwen/queries_all.npy").astype("float32")
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    out = {}
    for sp in SPLITS:
        d = f"{CORP}/{DS}/{sp}"
        if os.path.exists(f"{d}/relation_qwen_score.npy"):
            log(f"Stage D4: {sp} exists, skip"); continue
        st = pickle.load(open(f"scratchpad/_reledges_sparse_{DS}_{sp}.pkl", "rb"))
        l2g = g["per_split"][f"{DS}/{sp}"]["local2global"]
        S = L.load_split(DS, sp); off = S["off"]; meta = S["meta"]; npairs = len(S["cand"])
        mask = np.zeros(npairs, bool); score = np.zeros(npairs, np.float32)
        for qi, lst in st["per_q"].items():
            base = off[qi]; qv = Q[meta[qi]["row_all"]]
            for (jj, sid) in lst:
                gsid = l2g[sid]; p = base + jj; mask[p] = True; score[p] = float(Sent[gsid] @ qv)
        np.save(f"{d}/relation_mask.npy", mask); np.save(f"{d}/relation_qwen_score.npy", score)
        out[sp] = {"eligible_pairs": int(mask.sum()), "eligible_frac": round(float(mask.mean()), 5)}
        log(f"Stage D4: {sp} eligible_pairs={int(mask.sum())} ({100*mask.mean():.3f}% of scope)")
    return out


# =====================================================================  Stage E: expert_meta.npz
_L1CODE = {"L1_ANY_FAIL": 0, "L1_PARTIAL": 1, "L1_ALL_SUCCESS": 2}
def stage_e():
    import l2_lib as L
    out = {}
    for split in SPLITS:
        d = f"{CORP}/{DS}/{split}"
        if os.path.exists(f"{d}/expert_meta.npz"):
            log(f"Stage E: {split} exists, skip"); continue
        S = L.load_split(DS, split); off = S["off"]; labels = S["labels"]; meta = S["meta"]; nq = len(off) - 1
        mask = np.load(f"{d}/relation_mask.npy")
        row_all = np.array([meta[qi]["row_all"] for qi in range(nq)], np.int64)
        gold_count = np.array([int((labels[off[qi]:off[qi + 1]] == 1).sum()) for qi in range(nq)], np.int64)
        l1 = np.array([_L1CODE.get(meta[qi]["L1_STATUS"], -1) for qi in range(nq)], np.int8)
        dense = S["dense_score"]; splade = S["splade_scope_score"]
        dsg = np.zeros(nq, np.float32); rel_any = np.zeros(nq, bool); rel_goldp = np.zeros(nq, bool); rel_nc = np.zeros(nq, np.int64)
        for qi in range(nq):
            s, e = off[qi], off[qi + 1]; dd = dense[s:e]; sp_ = splade[s:e]; k = min(20, e - s)
            td = set(np.argpartition(-dd, k - 1)[:k].tolist()); ts = set(np.argpartition(-sp_, k - 1)[:k].tolist())
            u = len(td | ts); dsg[qi] = 1.0 - (len(td & ts) / u if u else 0.0)
            m = mask[s:e]; rel_any[qi] = m.any(); rel_nc[qi] = int(m.sum()); rel_goldp[qi] = bool((m & (labels[s:e] == 1)).any())
        np.savez(f"{d}/expert_meta.npz", row_all=row_all, gold_count=gold_count, L1_status=l1,
                 dense_splade_disagreement=dsg, relation_any_signal_present=rel_any,
                 relation_gold_present=rel_goldp, relation_num_candidates=rel_nc)
        out[split] = {"nq": int(nq), "rel_any_q": int(rel_any.sum()), "rel_gold_q": int(rel_goldp.sum())}
        log(f"Stage E: {split} meta nq={nq} rel_any_q={int(rel_any.sum())} rel_gold_q={int(rel_goldp.sum())}")
    return out


# stage registry (name -> fn); consumed by main() and the Modal wrapper (which commits the volume between stages)
STAGES = [("A_scope", stage_a), ("B_splade", stage_b), ("C_heads", stage_c), ("D_relation", stage_d), ("E_meta", stage_e)]


def finalize(man):
    files = ["query_offsets.npy", "cand_ids.npy", "labels.npy", "dense_score.npy", "splade_scope_score.npy",
             "offset_score.npy", "mixture_score.npy", "relation_qwen_score.npy", "relation_mask.npy", "expert_meta.npz"]
    man["integrity"] = {}
    for split in SPLITS:
        d = f"{CORP}/{DS}/{split}"
        present = {f: os.path.exists(f"{d}/{f}") for f in files}
        man["integrity"][split] = {"all_present": all(present.values()), "missing": [f for f, v in present.items() if not v]}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(man, open(f"results/GENERALIZATION/_g1_build_{DS}.json", "w"), indent=1)
    log("BUILD_MANIFEST -> results/GENERALIZATION/_g1_build_" + DS + ".json")
    log("ALL_STAGES_DONE integrity=" + json.dumps(man["integrity"]))
    return man


def main():
    log(f"=== G1 substrate build DS={DS} splits={SPLITS} train_cap={TRAIN_CAP} ===")
    man = {"dataset": DS, "splits": SPLITS, "train_cap": TRAIN_CAP}
    for name, fn in STAGES:
        man[name] = fn()
    finalize(man)


if __name__ == "__main__":
    main()
