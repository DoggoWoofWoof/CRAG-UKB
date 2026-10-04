"""
A/B/C scoped-L2 vs D_L2 for 2wiki_clean
Variants:
 A = structural + semantic-kNN (canonical gte_qwen partitions)
 B = structural + NER (scratchpad/ablation variant_B)
 C = structural + NER + semantic-kNN (variant_C)
 D = full/global (scope_topk=0) reference 71.5

For each topology x P in [20,50,100]:
  n_queries, scope oracle, mean/median/p95 N_scope,
  L2 R@2/5/20/50 hit@, routing loss, ranking loss, success,
  plus top500 persist

Headroom: scope oracle, scoped L2, best single expert oracle, cooperative if cheap
Locks: L1_vote_K=100, L2_output_K=500, K=8, heads 06a9/3240
Reuse: topology once, L1 vote once, query features once
"""
import os, json, pickle, logging, pathlib, sys, hashlib, time
import numpy as np, torch, faiss
from collections import Counter

# Ensure src on path
sys.path.insert(0, os.getcwd())
from src.experiments.l1_universal_head import _load as load_head_data
from src.experiments.l1_universal_head import _train_universal
from src.experiments.l2_seed import _topP, _scoped_order, _recall, _pool_stats, MAXK, KS, _splade_scoped_order
from src.experiments.l1_rerank100 import _feats, _rr
from src.core.engine import CoreEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_abc_2wiki")

DATASET = "2wiki_clean"
SUBDIR = "gte_qwen"
LIMIT = 8000
TR_CAP = 3000
TE_CAP = 0  # full validation uncapped like D ; for quick test use 100
EPOCHS = 15
K = 8
HEADS = {
    "hard": "data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt",
    "mix": "data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt",
}
VARIANTS = {
    "A": "structural+semantic-kNN (canonical)",
    "B": "structural+NER",
    "C": "structural+NER+semantic-kNN",
}
P_VALUES = [20,50,100]
L1_VOTE_K = 100  # for partition voting (MAXK in faiss search)
L2_OUTPUT_K = 500
OUTDIR = pathlib.Path("results/L2/abc_2wiki")
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_variant_partitions(dataset, variant):
    """Return partition_map dict, npart"""
    if variant == "A":
        engine = CoreEngine(source=dataset, index_subdir=SUBDIR)
        pm = engine.partition_map
        npart = max(pm.values())+1 if pm else 0
        # Also need mem_idx and hard? Engine loads those via _load; but partition_map from engine is canonical
        return pm, npart, engine
    elif variant in ("B","C"):
        p = f"scratchpad/ablation/{dataset}/variant_{variant}/partition_map.json"
        if not os.path.exists(p):
            log.error(f"Missing {p} for {variant}")
            raise FileNotFoundError(f"[artifact/partition] frozen but absent {p} -> FAIL")
        pm = json.load(open(p))
        pm = {k:int(v) for k,v in pm.items()}
        npart = max(pm.values())+1
        engine = CoreEngine(source=dataset, index_subdir=SUBDIR)
        # override
        engine.partition_map = pm
        return pm, npart, engine
    else:
        raise ValueError(variant)

def preflight():
    log.info("=== PREFLIGHT ===")
    checks = []
    # heads
    for kind, path in HEADS.items():
        exists = os.path.exists(path)
        log.info(f"HEAD {kind} {path} exists={exists} size={os.path.getsize(path) if exists else -1}")
        if not exists:
            return False
    # dataset artifacts
    for variant in ["A","B","C"]:
        try:
            pm,npart,eng = load_variant_partitions(DATASET, variant)
            log.info(f"variant {variant} npart={npart} pm_len={len(pm)} ok")
            # graph
            graph_path = f"data/ukb_storage/{DATASET}/gte_qwen/graph.pt" if variant=="A" else f"scratchpad/ablation/{DATASET}/variant_{variant}/graph.pt"
            log.info(f"  graph {graph_path} exists={os.path.exists(graph_path)} size={os.path.getsize(graph_path) if os.path.exists(graph_path) else -1}")
            # ner
            ner_path = f"data/ukb_storage/{DATASET}/ner_edges_w_df25.pkl"
            log.info(f"  NER {ner_path} exists={os.path.exists(ner_path)}")
            # dense
            nodes = f"data/ukb_storage/{DATASET}/gte_qwen/nodes.npy"
            log.info(f"  dense nodes {nodes} exists={os.path.exists(nodes)}")
            qtest = f"data/ukb_storage/{DATASET}/gte_qwen/queries_test.npy"
            log.info(f"  dense qtest {qtest} exists={os.path.exists(qtest)}")
            splade = f"data/ukb_storage/{DATASET}/splade_doc_embs.pkl"
            log.info(f"  splade {splade} exists={os.path.exists(splade)}")
            part = f"data/ukb_storage/{DATASET}/gte_qwen/partition_map.json" if variant=="A" else f"scratchpad/ablation/{DATASET}/variant_{variant}/partition_map.json"
            log.info(f"  partition {part} exists={os.path.exists(part)}")
            centroids = f"data/ukb_storage/{DATASET}/gte_qwen/centroids.index" if variant=="A" else f"scratchpad/ablation/{DATASET}/variant_{variant}/centroids.index"
            log.info(f"  centroids {centroids} exists={os.path.exists(centroids)}")
        except Exception as e:
            log.error(f"preflight variant {variant} failed {e}")
            return False
    # output paths unique
    for variant in VARIANTS:
        for p in P_VALUES:
            out = OUTDIR / f"rankings_{DATASET}_{variant}_P{p}_gte_qwen.npz"
            log.info(f"  output {out} exists={out.exists()} will_overwrite={out.exists()} (unique check: ok if not exists or overwrite is intentional for fresh run)")
    log.info("PREFLIGHT PASS")
    return True

def compute_metrics(dataset, variant, p_value, topP, hard, gte, orders, npart):
    """Compute required metrics for one P"""
    # hard: doc->partition
    # gte: list of gold doc idx lists per query
    # orders: dict of expert->order (nq,500)
    # topP: list of sets
    harr = np.asarray(hard)
    # scope stats
    psize = np.bincount(harr, minlength=npart)
    sizes = np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else len(harr) for tp in topP], dtype=np.int64)
    mean_scope = float(sizes.mean()) if len(sizes) else 0
    median_scope = float(np.median(sizes)) if len(sizes) else 0
    p95_scope = float(np.percentile(sizes,95)) if len(sizes) else 0
    # scope oracle (pool ceiling)
    # fraction of golds whose partition in topP
    # per query: if any gold partitioned in scope? Actually gold doc's partition in topP
    # compute scope_recall as mean over queries of (fraction of golds reachable)
    ceils = []
    routing_loss_count = 0
    for qi, gg in enumerate(gte):
        if not gg:
            continue
        reachable = sum(1 for g in gg if hard[g] in topP[qi]) / len(gg)
        ceils.append(reachable)
        if reachable < 1.0:
            # if not all gold reachable, count as routing loss if any gold outside?
            # routing loss = gold outside scope (any gold not in scope)
            # We'll compute per query: if no gold in scope -> routing loss
            # For now, count queries where 0 gold reachable?
            pass
    scope_oracle = round(100*float(np.mean(ceils)) if ceils else 0, 2)
    # For routing loss vs ranking loss, need D_L2 order
    # Assume orders contains "D_L2" as minrank or bestof
    # We'll compute per query:
    #  routing loss: gold outside scope (not in topP)
    #  ranking loss: gold in scope but outside retrieved topK (500)
    #  success: gold in final topK
    nq = len(gte)
    # Use D_L2 order if present else dense
    d_order = orders.get("D_L2") if "D_L2" in orders else orders.get("dense+rel_hard+mlpT") if "dense+rel_hard+mlpT" in orders else list(orders.values())[0]
    # Compute hit/recall via _recall
    rec = _recall(d_order, gte, ks=KS)
    # routing vs ranking breakdown
    routing_loss = 0
    ranking_loss = 0
    success = 0
    for qi, gg in enumerate(gte):
        if not gg:
            continue
        gs = set(gg)
        # scope check
        in_scope = any(hard[g] in topP[qi] for g in gs)
        if not in_scope:
            routing_loss += 1
            continue
        # in scope, check if in top500
        row = set(int(x) for x in d_order[qi] if x>=0)
        if gs & row:
            success += 1
        else:
            ranking_loss += 1
    total_gold_queries = sum(1 for g in gte if g)
    routing_loss_pct = round(100*routing_loss/max(1,total_gold_queries),2)
    ranking_loss_pct = round(100*ranking_loss/max(1,total_gold_queries),2)
    success_pct = round(100*success/max(1,total_gold_queries),2)
    return {
        "n_queries": nq,
        "scope_oracle": scope_oracle,
        "mean_N_scope": round(mean_scope,1),
        "median_N_scope": round(median_scope,1),
        "p95_N_scope": round(p95_scope,1),
        "routing_loss_pct": routing_loss_pct,
        "ranking_loss_pct": ranking_loss_pct,
        "success_pct": success_pct,
        "R@": rec,
        "sizes": sizes.tolist()[:5], # sample
    }

def run_one_variant(variant, te_cap=0, device=None):
    log.info(f"=== VARIANT {variant} {VARIANTS[variant]} te_cap={te_cap} ===")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device {device}")
    # Load partition map
    pm, npart, engine = load_variant_partitions(DATASET, variant)
    log.info(f"loaded pm npart={npart} len={len(pm)}")
    # Load L2 data via _load (same as D)
    from src.experiments.l1_universal_head import _load as _load_head
    data = _load_head(DATASET, SUBDIR, LIMIT, TR_CAP, 1)  # te_cap 1 for _load internal? But we need test split full
    # _load returns dict with keys: X, hard, mem_idx, npart, train, val, test etc.
    # Actually _load returns data[d] structure: need to inspect
    # Let's use the same _load signature as l2_seed: _load(d, subdir, limit, tr_cap, te_cap)
    # But _load from l1_universal_head expects different args? Check.
    # Instead use l2_seed's _load via l1_universal_head._load
    # For now, reload via l2_seed's internal: we already loaded via _load_head? Let's directly call _load_head with te_cap
    # Simpler: use l2_seed's _load via import
    from src.experiments.l1_universal_head import DATASETS
    # Use _load from l1_universal_head with te_cap param? Let's inspect its signature
    import inspect
    sig = inspect.signature(_load_head)
    log.info(f"_load signature {sig}")
    # try calling with te_cap
    try:
        # _load(dataset, subdir, limit, tr_cap, te_cap) ?
        ddata = _load_head(DATASET, SUBDIR, LIMIT, TR_CAP, te_cap if te_cap!=0 else 2000)
    except Exception as e:
        log.error(f"_load failed {e}")
        # fallback to l2_seed's _load which is same?
        from src.experiments.l2_seed import _load as _load_l2
        ddata = _load_l2(DATASET, SUBDIR, LIMIT, TR_CAP, te_cap)
    # ddata structure: need to extract test
    # Check keys
    log.info(f"ddata keys {list(ddata.keys()) if isinstance(ddata, dict) else type(ddata)}")
    if isinstance(ddata, dict) and "test" in ddata:
        qte, ste, gte = ddata["test"]
        X = ddata["X"]
        hard = ddata["hard"]
        mem_idx = ddata["mem_idx"]
        npart_data = ddata["npart"]
        # npart from data may differ from pm npart; use pm npart
        npart = npart
        texts = ddata.get("test_texts", None)
        bm25 = ddata.get("bm25", None)
        splade = ddata.get("splade", None)
    else:
        # if _load returns per_ds style, adapt
        log.error(f"unexpected ddata {ddata}")
        return None
    log.info(f"X {X.shape} qte {qte.shape} npart {npart} hard {len(hard)} gte {len(gte)} sample gold {gte[0] if gte else []}")
    # Override hard/mem_idx with variant pm? The hard array is doc->partition from original load; we need to replace with variant pm mapping
    # hard array order corresponds to X order (doc order). pm maps doc_id -> pid. Need to map via node ids
    # The engine and _load should have same doc ordering: sorted node_id
    # Let's get doc ids order from engine or ddata
    # ddata may have "node_ids" ?
    if "node_ids" in ddata:
        node_ids = ddata["node_ids"]
    else:
        # try to get from engine via data files
        # For 2wiki_clean, doc ids are like 2wiki_clean_doc_0 etc sorted
        # pm keys are those ids, so order is sorted keys
        node_ids = sorted(pm.keys())
        # But X order may be sorted node_ids as well
        # Assume X order matches sorted pm keys
        pass
    # Build hard_variant array in X order
    # Need to map X index -> pid via pm
    # X index i corresponds to node_ids[i]
    if "node_ids" in ddata:
        hard_variant = np.array([pm[nid] for nid in node_ids], dtype=np.int64)
    else:
        # assume pm order is sorted and X order is same sorted
        # So we can just create array in sorted pm key order
        sorted_ids = sorted(pm.keys())
        hard_variant = np.array([pm[nid] for nid in sorted_ids], dtype=np.int64)
        # Check length matches X
        if len(hard_variant) != X.shape[0]:
            log.warning(f"hard_variant len {len(hard_variant)} vs X {X.shape[0]} mismatch, using original hard")
            hard_variant = np.array(hard)
        else:
            log.info(f"built hard_variant len {len(hard_variant)} matches X")
    hard = hard_variant
    # mem_idx for _feats: list per doc of partition ids (hard membership single)
    # Original mem_idx from _load is list length N, each entry is list of partitions for that doc
    # For our variants, use hard membership: [[pid] for pid in hard_variant]
    log.info(f"mem_idx type {type(mem_idx)} sample {mem_idx[:2] if isinstance(mem_idx, list) else list(mem_idx.items())[:2] if isinstance(mem_idx, dict) else mem_idx}")
    # Build variant mem_idx from hard (hard = variant hard)
    mem_idx_variant = [[int(pid)] for pid in hard]
    log.info(f"mem_idx_variant sample {mem_idx_variant[:2]} len {len(mem_idx_variant)} npart {npart}")
    # Now load heads
    import torch
    # Determine dim
    dim = X.shape[1]
    # Load hard head
    sd_hard = torch.load(HEADS["hard"], map_location="cpu")
    # Infer head class
    from src.experiments.query_relation import OffsetHead
    from src.experiments.l1_ablate import MixtureHead
    hard_head = OffsetHead(dim)
    hard_head.load_state_dict(sd_hard)
    hard_head.eval()
    sd_mix = torch.load(HEADS["mix"], map_location="cpu")
    mix_head = MixtureHead(dim, K=8)
    mix_head.load_state_dict(sd_mix)
    mix_head.eval()
    log.info(f"heads loaded hard {HEADS['hard']} mix {HEADS['mix']}")
    # Prepare faiss index for dense voting
    X_t = torch.tensor(X, device=device) if device.type=="cuda" else None
    # For voting, need dense_order via faiss search
    # Use faiss IndexFlatIP on X
    # Normalize?
    # Check if X is already normalized (gte_qwen is normalized)
    # Build faiss index
    idx = faiss.IndexFlatIP(dim)
    # Normalize for IP? Assume already normalized via encode
    # But to be safe, normalize?
    # The _load X is already L2 normalized?
    # We'll use as is
    idx.add(X)
    # Prepare query embeddings
    # qte is already normalized?
    log.info(f"faiss search for L1 voting top {MAXK}")
    _, dense_order = idx.search(qte, MAXK)  # (nq,100)
    # Compute L1 vote ordering once per topology
    # Use _topP logic but we need S,M via _feats
    # _feats expects dense_order, mem_idx, npart, topn=200
    # Let's compute votes once
    S, M = _feats(dense_order, mem_idx_variant, npart, topn=200)
    votes = _rr(S) + _rr(M)
    # votes shape (nq, npart)
    # partition ranking per query
    part_ranking = np.argsort(-votes, axis=1)  # (nq, npart) sorted pid
    log.info(f"part_ranking {part_ranking.shape} votes {votes.shape}")
    # Compute query features once
    with torch.no_grad():
        qt = torch.tensor(qte, device=device)
        ste_t = torch.tensor(ste, device=device) if len(ste)>0 else None
        # Need sv = X[ste]? For offset head, need seed docs embeddings
        # ste is list of seed doc indices per query? Actually ste is seed doc indices?
        # In _load, train/test are (q, seed_doc_idx, gold_list) - check
        # qte, ste, gte where ste is seed doc idx array (nq,) or (nq, K?) Let's inspect
        log.info(f"ste {ste[:3] if len(ste)>0 else []} type {type(ste)} shape {np.array(ste).shape if hasattr(ste,'__len__') else 'unknown'}")
        # For offset head: heads["hard"](qt, sv) where sv = X[ste]
        if ste is not None and len(ste)>0:
            # ste may be array of doc indices
            sv = torch.tensor(X[ste], device=device) if isinstance(ste, np.ndarray) else torch.tensor(X[np.array(ste)], device=device)
        else:
            sv = qt  # fallback
        pos_dense = qt
        pos_hard = hard_head(qt, sv) if 'hard_head' in locals() else qt
        pos_mix = mix_head(qt, sv) if 'mix_head' in locals() else qt
        # For mix, pos shape (nq, K, dim)
        log.info(f"pos_dense {pos_dense.shape} pos_hard {pos_hard.shape} pos_mix {pos_mix.shape if hasattr(pos_mix,'shape') else type(pos_mix)}")
    # Now for each P compute scope and scoped L2
    results = {}
    for p in P_VALUES:
        log.info(f"--- P={p} ---")
        topP = [set(int(pid) for pid in part_ranking[qi, :p]) for qi in range(len(qte))]
        # pool stats
        harr = np.asarray(hard)
        psize = np.bincount(harr, minlength=npart)
        sizes = np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else len(harr) for tp in topP], dtype=np.int64)
        mean_pool = float(sizes.mean())
        median_pool = float(np.median(sizes))
        p95_pool = float(np.percentile(sizes,95))
        corpus_N = len(harr)
        log.info(f"pool mean {mean_pool:.1f} median {median_pool:.1f} p95 {p95_pool:.1f} N {corpus_N} reduction {corpus_N/max(1,mean_pool):.2f}")
        # scope oracle
        ceils = []
        for qi, gg in enumerate(gte):
            if not gg: continue
            reachable = sum(1 for g in gg if hard[g] in topP[qi]) / len(gg)
            ceils.append(reachable)
        scope_oracle = round(100*float(np.mean(ceils)) if ceils else 0,2)
        log.info(f"scope oracle {scope_oracle}")
        # Scoped orders
        # Use _scoped_order for dense/hard/mix
        # Need X_t and hard_t
        hard_t = torch.tensor(hard, device=device)
        # For dense/hard: pos is (nq, dim) ; for mix: (nq,K,dim)
        # We'll compute orders for each expert
        # For CPU device, _scoped_order still works but we have X_t on cpu? In l2_seed, X_t is on device
        # If device is cpu, X_t is cpu, still works
        # Prepare X_t for scoring
        if device.type == "cpu":
            X_t_cpu = torch.tensor(X, device=device)
        else:
            X_t_cpu = X_t  # already on cuda
        # Compute scoped orders
        # dense
        # Convert pos to cpu numpy then to torch for _scoped_order expects pos as torch tensor? In l2_seed, pos is torch tensor on device, then _scoped_order does pos.cpu() ?
        # Actually _scoped_order expects pos as tensor (nq, dim) or (nq,K,dim) and X_t as tensor on device
        # It loops over pos batches and does einsum
        # So we can call similarly
        # For dense/hard
        # pos_dense is (nq, dim) already on device
        # We'll need to move pos to cpu for _scoped_order? No, it expects pos as cpu then moves to device inside
        # Let's just call _scoped_order with pos_dense.cpu() etc. Need to check impl: it does pos[s:s+bs].to(device) inside, so pos should be cpu tensor
        # Our pos_dense is already on device, so we can pass pos_dense.cpu()
        orders = {}
        # dense
        orders_dense, scores_dense = _scoped_order(pos_dense.cpu(), X_t_cpu, hard_t, topP, False, device, k=L2_OUTPUT_K)
        orders["dense"] = orders_dense
        # hard
        orders_hard, _ = _scoped_order(pos_hard.cpu(), X_t_cpu, hard_t, topP, False, device, k=L2_OUTPUT_K)
        orders["rel_hard"] = orders_hard
        # mix (mlpT)
        # For mix, pos_mix is (nq,K,dim), need _scoped_order with is_mix True? Actually _scoped_order handles is_mix via einsum
        # But for simplicity use _scoped_order with is_mix=True for pos_mix
        # However _scoped_order for mix does max over K? It does einsum then max(1). That's correct for single order.
        # For per-direction orders, we might not need
        # Let's compute mix as is_mix True
        orders_mix, _ = _scoped_order(pos_mix.cpu(), X_t_cpu, hard_t, topP, True, device, k=L2_OUTPUT_K)
        orders["mlpT"] = orders_mix
        # splade
        if splade is not None and texts is not None:
            try:
                splade_orders = _splade_scoped_order(splade, texts, hard, topP, dataset=DATASET, k=L2_OUTPUT_K)
                orders["splade"] = splade_orders
            except Exception as e:
                log.warning(f"splade failed {e}")
        # D_L2 as bestof dense+rel_hard+mlpT? In D, D_L2 is minrank of those?
        # In l2_seed, 3way is bestof dense,rel_hard,mlpT
        # We'll define D_L2 as that
        if "dense" in orders and "rel_hard" in orders and "mlpT" in orders:
            from src.experiments.l2_seed import _bestof
            three = _bestof([orders["dense"], orders["rel_hard"], orders["mlpT"]])
            # three is ragged? _bestof returns np array (nq, m) where m is max len
            # Need to ensure shape is (nq, 500) with -1 padding
            # Pad to 500
            if three.shape[1] < L2_OUTPUT_K:
                pad = np.full((three.shape[0], L2_OUTPUT_K - three.shape[1]), -1, dtype=np.int64)
                three = np.concatenate([three, pad], axis=1)
            orders["D_L2"] = three
        # Also compute best single expert oracle among available
        # For headroom, compute R@ for each single expert within scope
        single_best = 0
        best_expert = None
        for m in ["dense","rel_hard","mlpT","splade"]:
            if m in orders:
                r = _recall(orders[m], gte, ks=[5])
                # r is dict {5:..., "hit5":...}
                if r[5] > single_best:
                    single_best = r[5]
                    best_expert = m
        log.info(f"best single expert {best_expert} {single_best}")
        # Compute metrics for D_L2 if available
        d_order = orders.get("D_L2", orders["dense"])
        rec = _recall(d_order, gte, ks=KS)
        # routing vs ranking
        routing_loss = 0
        ranking_loss = 0
        success = 0
        total = 0
        for qi, gg in enumerate(gte):
            if not gg: continue
            total += 1
            gs = set(gg)
            in_scope = any(hard[g] in topP[qi] for g in gs)
            if not in_scope:
                routing_loss += 1
                continue
            row = set(int(x) for x in d_order[qi] if x>=0)
            if gs & row:
                success += 1
            else:
                ranking_loss += 1
        routing_loss_pct = round(100*routing_loss/max(1,total),2)
        ranking_loss_pct = round(100*ranking_loss/max(1,total),2)
        success_pct = round(100*success/max(1,total),2)
        log.info(f"P{p} scope_oracle {scope_oracle} D_L2 R@5 {rec[5]} routing {routing_loss_pct} ranking {ranking_loss_pct} success {success_pct} mean_scope {mean_pool:.1f}")
        # Persist rankings
        out_npz = OUTDIR / f"rankings_{DATASET}_{variant}_P{p}_gte_qwen.npz"
        # Use schema 1.0: query_ids, gold_ids, dense_order, rel_hard_order, mlpT_order, splade_order, D_L2_order
        # Need query_ids: from ddata, maybe have query ids? Use indices
        # For now, use np.arange for query_ids
        query_ids = np.arange(len(qte), dtype=np.int32)
        # gold_ids: need to store as object? Use padded?
        # Simplify: save gold_lists as array of python objects? npz can't store variable length easily
        # Instead save gold_ids as list of lists via pickle? We'll save as object array
        gold_ids = np.array(gte, dtype=object)
        np.savez_compressed(str(out_npz),
            query_ids=query_ids,
            gold_ids=gold_ids,
            dense_order=orders.get("dense", np.zeros((len(qte), L2_OUTPUT_K), dtype=np.int64)),
            rel_hard_order=orders.get("rel_hard", np.zeros((len(qte), L2_OUTPUT_K), dtype=np.int64)),
            mlpT_order=orders.get("mlpT", np.zeros((len(qte), L2_OUTPUT_K), dtype=np.int64)),
            splade_order=orders.get("splade", np.zeros((len(qte), L2_OUTPUT_K), dtype=np.int64)) if "splade" in orders else np.zeros((len(qte), L2_OUTPUT_K), dtype=np.int64),
            D_L2_order=d_order,
            topP=np.array([list(tp) for tp in topP], dtype=object),
            hard=hard,
            npart=np.array([npart]),
            P=np.array([p]),
        )
        log.info(f"saved {out_npz}")
        # Store results
        results[p] = {
            "variant": variant,
            "P": p,
            "n_queries": len(qte),
            "scope_oracle": scope_oracle,
            "mean_N_scope": round(mean_pool,1),
            "median_N_scope": round(median_pool,1),
            "p95_N_scope": round(p95_pool,1),
            "corpus_N": corpus_N,
            "pool_reduction": round(corpus_N/max(1,mean_pool),2),
            "R@": rec,
            "routing_loss_pct": routing_loss_pct,
            "ranking_loss_pct": ranking_loss_pct,
            "success_pct": success_pct,
            "best_single_expert": best_expert,
            "best_single_R@5": single_best,
        }
        log.info(json.dumps(results[p], indent=2))
    # Save json summary
    json.dump(results, open(OUTDIR / f"summary_{DATASET}_{variant}_gte_qwen.json","w"), indent=2)
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--te-cap", type=int, default=0, help="0 = full validation uncapped, else cap for quick test")
    p.add_argument("--quick", action="store_true", help="quick test with 200 queries")
    args = p.parse_args()
    te_cap = args.te_cap
    if args.quick:
        te_cap = 200
    if not preflight():
        log.error("PREFLIGHT FAIL")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device {device}")
    all_results = {}
    for variant in ["A","B","C"]:
        res = run_one_variant(variant, te_cap=te_cap, device=device)
        all_results[variant] = res
        # Early failure check after first variant
        if variant == "A":
            # Check if scope oracle low
            for p in P_VALUES:
                sc = res[p]["scope_oracle"]
                log.info(f"EARLY CHECK A P{p} scope_oracle {sc} vs threshold 50")
                if sc < 30:
                    log.warning(f"A P{p} scope oracle low {sc} -> routing failure")
    # Overall summary table
    print("\n=== FINAL TABLE topology | P | mean scope | scope oracle R@5 | L2 R@5 | routing loss | ranking loss | R@2/5/20/50 ===")
    for variant in ["A","B","C"]:
        for p in P_VALUES:
            r = all_results[variant][p]
            print(f"{variant} | P{p:3d} | {r['mean_N_scope']:7.1f} | {r['scope_oracle']:6.2f} | {r['R@'][5]:6.2f} | {r['routing_loss_pct']:6.2f} | {r['ranking_loss_pct']:6.2f} | {r['R@'][2]:5.2f}/{r['R@'][5]:5.2f}/{r['R@'][20]:5.2f}/{r['R@'][50]:5.2f}")
    # Report ETA etc.
    log.info("DONE")
