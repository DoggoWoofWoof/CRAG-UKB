"""
L1 Routing-Graph Ablation — controlled variants A/B/C/D on small/medium corpora.
- A: structural + dense kNN (historical G_routing, exact k=3 + isolates, pymetis, centroid degree-weighted)
- B: structural + NER (no kNN) — corpus-global NER artifact per universe, N/A for KB
- C: structural + NER + dense kNN
- D: direct global dense/SPLADE control at matched candidate budget

Fixed: document universe, query split, partition algorithm, target=100, METIS, encoders,
voting (dense/SPLADE count + RRF), top-partition budget, L2 candidate budget.
Only routing-graph topology changes. Writes to scratchpad/ablation/ without overwriting canonical.

Thorough metrics: partition counts, membership sizes, centroid, dense/SPLADE/RRF voting,
Partition Recall@1/3/5/10/20, Candidate Recall@top-1/3/5/10/20, mean/p95 docs to L2,
edge-cut, latency, RAM, storage, routing ceiling (gold_reachable).
For D: global Recall@K at matched candidate counts.
"""
import os, json, pickle, time, random, logging, csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import faiss
import torch
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

# project imports
from src.pipeline.standardizer import load_nodes
from src.core.splade_scorer import SpladeScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation")

SPLIT_SEED=42
TRAIN_RATIO=0.70
VAL_RATIO=0.20
TEST_RATIO=0.10
TARGET_PER_PARTITION=100
VOTE_K=100
K_VALUES=[1,3,5,10,20]
COVERAGE_K=[1,3,5,10,20,50,100,200]
CENTROID_K_VALUES=K_VALUES  # for partition recall
CANDIDATE_K_PARTITIONS=K_VALUES  # top partitions for candidate pool
RRF_K=60
TOP_PARTITION_BUDGETS=K_VALUES  # same
L2_CANDIDATE_BUDGET=100  # not swept, just reference

DATASETS_PRIORITY=["squad_clean","2wiki_clean","musique_clean"]
DATASETS_SECONDARY=["metaqa","webqsp"]  # KB, B/C marked N/A per spec
DATASETS_ALL=DATASETS_PRIORITY+DATASETS_SECONDARY

# Map canonical NER vs ukb_storage NER
def ner_available(dataset: str) -> bool:
    return os.path.exists(f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl")

def load_embeddings_faiss(dataset: str) -> tuple[np.ndarray, faiss.Index]:
    idx = faiss.read_index(f"data/ukb_storage/{dataset}/nodes.index")
    ntotal = idx.ntotal
    dim = idx.d
    # reconstruct_n is faster
    try:
        embs = idx.reconstruct_n(0, ntotal).astype(np.float32)
    except Exception:
        # fallback loop
        embs = np.stack([idx.reconstruct(i) for i in range(ntotal)]).astype(np.float32)
    # ensure normalized (already)
    faiss.normalize_L2(embs)
    return embs, idx

def load_master(dataset: str):
    path = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(path):
        # fallback canonical?
        path = f"data/processed/master_nodes.json"
    nodes = load_nodes(path)
    # filter to source
    # master_nodes_{ds} already filtered, but double-check
    # For squad_clean etc, source == dataset string
    # If generic, filter
    if dataset not in path:
        # already per-source
        pass
    else:
        # if file is per-source, all nodes have source=dataset, but we keep as is
        pass
    # Ensure we only keep nodes with source == dataset or if file is per-source keep all
    # For squad_clean file, source is squad_clean, so keep all
    # For generic, filter
    # We'll just keep all docs+Qs from file; if file contains multiple sources, filter
    # Detect if mixing: check unique sources
    uniq_sources = set(n.metadata.get("source","") for n in nodes)
    if len(uniq_sources)>1:
        nodes = [n for n in nodes if n.metadata.get("source")==dataset]
    doc_nodes = [n for n in nodes if n.metadata.get("type")!="question"]
    q_nodes = [n for n in nodes if n.metadata.get("type")=="question"]
    # Build doc_id->idx
    doc_id_to_idx = {n.node_id:i for i,n in enumerate(doc_nodes)}
    return nodes, doc_nodes, q_nodes, doc_id_to_idx

def get_split_queries(q_nodes, partition_map, doc_id_to_idx, dataset):
    # same as benchmark_partition_selection._get_split_queries but using partition_map gold partitions
    # q_nodes: list of StandardNode questions
    # partition_map: dict node_id->pid (for gold reachable check, but for split we need gt_pids if available)
    # For ablation, we need gold doc ids, not just partitions
    # We'll return list of (q_node, gt_doc_ids, gt_pids) where gt_doc_ids are neighbors that are doc ids
    all_pairs = []
    for q in q_nodes:
        gt_doc_ids = [nid for nid in q.neighbors if nid in doc_id_to_idx]
        if not gt_doc_ids:
            continue
        gt_pids = list(set(partition_map.get(nid) for nid in gt_doc_ids if partition_map.get(nid) is not None))
        # keep also even if pid missing? but then unreachable
        all_pairs.append((q.node_id, q, gt_doc_ids, gt_pids))
    if not all_pairs:
        return {"train":[],"val":[],"test":[]}
    all_pairs.sort(key=lambda x: x[0])
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_pairs)
    n=len(all_pairs)
    train_end=int(n*TRAIN_RATIO)
    val_end=train_end+int(n*VAL_RATIO)
    def to_q(pairs):
        return [(q, docs, pids) for _,q,docs,pids in pairs]
    return {"train":to_q(all_pairs[:train_end]), "val":to_q(all_pairs[train_end:val_end]), "test":to_q(all_pairs[val_end:])}

def build_structural_adj(doc_nodes, doc_id_to_idx):
    # adjacency as set per node
    N=len(doc_nodes)
    adj=[set() for _ in range(N)]
    for i,node in enumerate(doc_nodes):
        for nid in node.neighbors:
            j=doc_id_to_idx.get(nid)
            if j is not None and j!=i:
                adj[i].add(j)
                adj[j].add(i)  # undirected mirror already via loop but ensure symmetry
    return adj

def load_ner_edges(dataset, doc_id_to_idx):
    pkl=f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl"
    if not os.path.exists(pkl):
        return None, 0
    A=pickle.load(open(pkl,"rb"))
    # A is csr (N,N) with 1/df weights, symmetric. Row order matches doc_nodes order used at build time.
    # Verify shape matches
    N=len(doc_id_to_idx)
    if A.shape[0]!=N:
        log.warning(f"NER shape mismatch {dataset}: {A.shape} vs N={N}")
        # attempt to remap via ids? but fallback to ignoring
        return None,0
    # extract edges
    # convert to adjacency sets + edge list
    # Use coo
    coo=A.tocoo()
    # keep only upper triangular to deduplicate? but we need undirected adjacency
    adj=[set() for _ in range(N)]
    # For metrics, count undirected edges (i<j) where weight>0
    edge_set=set()
    for r,c in zip(coo.row, coo.col):
        if r==c: continue
        if r<c:
            edge_set.add((r,c))
        adj[r].add(int(c))
    return adj, len(edge_set)

def build_knn_edges(embeddings: np.ndarray, structural_adj: List[set], ner_adj: Optional[List[set]], include_knn: bool) -> Tuple[List[set], int, int]:
    """Build universal k=3 exact + isolates k=3. Returns adjacency, universal_count, isolate_count."""
    N, dim = embeddings.shape
    # start with base adjacency: structural + ner if provided
    base_adj=[set(s) for s in structural_adj]
    if ner_adj is not None:
        for i in range(N):
            base_adj[i].update(ner_adj[i])
    # if not include_knn, return base only (no isolate step)
    if not include_knn:
        # count edges undirected
        edges=sum(len(s) for s in base_adj)//2
        return base_adj, 0, 0

    # Universal semantic k=3 exact via IndexFlatIP
    vectors = embeddings.astype(np.float32)
    faiss.normalize_L2(vectors)
    knn_index = faiss.IndexFlatIP(dim)
    knn_index.add(vectors)
    search_index=knn_index
    try:
        if hasattr(faiss,"get_num_gpus") and faiss.get_num_gpus()>0:
            import faiss as _f
            search_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(),0,knn_index)
            log.info(f"  kNN universal: exact IndexFlatIP on GPU for {N} nodes")
    except Exception as e:
        log.warning(f"  kNN GPU unavailable {e}, CPU brute force")

    distances, indices = search_index.search(vectors, 4)  # self +3
    universal_added=0
    # Track synthetic edges
    # Add to base_adj undirected
    for i in range(N):
        for local_idx in indices[i][1:4]:
            if local_idx==-1: continue
            j=int(local_idx)
            if i!=j and j not in base_adj[i]:
                base_adj[i].add(j)
                base_adj[j].add(i)
                universal_added+=1
    # find isolates after universal+structural+ner
    G_tmp = nx.Graph()
    G_tmp.add_nodes_from(range(N))
    for i in range(N):
        for j in base_adj[i]:
            if i<j:
                G_tmp.add_edge(i,j)
    isolates=list(nx.isolates(G_tmp))
    isolate_added=0
    if isolates:
        # IVFFlat for isolates
        nlist=int(np.sqrt(N))
        # Need to train on full vectors
        quantizer=faiss.IndexFlatIP(dim)
        knn_ivf=faiss.IndexIVFFlat(quantizer, dim, nlist)
        knn_ivf.train(vectors)
        knn_ivf.add(vectors)
        knn_ivf.nprobe=max(1, nlist//10)
        # batch isolates as in original
        batch_size=20000
        for b in range(0,len(isolates),batch_size):
            batch_isol=isolates[b:b+batch_size]
            iso_embs=embeddings[batch_isol].astype(np.float32)
            faiss.normalize_L2(iso_embs)
            d2, idx2 = knn_ivf.search(iso_embs, 4)
            for idx_in_batch, orig_i in enumerate(batch_isol):
                for j in idx2[idx_in_batch]:
                    if j==-1 or j==orig_i: continue
                    j=int(j)
                    if j not in base_adj[orig_i]:
                        base_adj[orig_i].add(j)
                        base_adj[j].add(orig_i)
                        isolate_added+=1
        log.info(f"  kNN isolates: {len(isolates)} isolates, added {isolate_added} edges")
    else:
        log.info(f"  kNN: no isolates after universal (N={N})")
    return base_adj, universal_added, isolate_added

def build_graph_and_partitions(dataset: str, variant: str, embeddings: np.ndarray, doc_nodes, doc_id_to_idx, target=100, out_dir=None):
    """
    variant: A (struct+knn), B (struct+ner), C (struct+ner+knn)
    Returns: G_nx, parts, n_cuts, partition_map, centroids, stats
    """
    N=len(doc_nodes)
    structural_adj=build_structural_adj(doc_nodes, doc_id_to_idx)
    # NER
    ner_adj, ner_edges = (None,0)
    if variant in ("B","C"):
        if not ner_available(dataset):
            return None  # N/A
        ner_adj, ner_edges = load_ner_edges(dataset, doc_id_to_idx)
        if ner_adj is None:
            return None
    else:
        ner_adj=None

    include_knn = variant in ("A","C")
    # Build adjacency with flags
    if variant=="A":
        base_adj, univ, iso = build_knn_edges(embeddings, structural_adj, None, include_knn=True)
    elif variant=="B":
        base_adj, univ, iso = build_knn_edges(embeddings, structural_adj, ner_adj, include_knn=False)
        univ=iso=0
    elif variant=="C":
        base_adj, univ, iso = build_knn_edges(embeddings, structural_adj, ner_adj, include_knn=True)
    else:
        raise ValueError(variant)

    # Build NetworkX for METIS
    G_nx=nx.Graph()
    G_nx.add_nodes_from(range(N))
    total_undirected=0
    for i in range(N):
        for j in base_adj[i]:
            if i<j:
                G_nx.add_edge(i,j)
                total_undirected+=1

    # METIS partition
    n_parts=max(1, N//target)
    # adjacency list for pymetis
    adjacency_list=[list(base_adj[i]) for i in range(N)]
    n_cuts=None
    try:
        import pymetis
        n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
        parts=list(membership)
        log.info(f"  METIS: {n_parts} parts, n_cuts={n_cuts}, variant={variant}")
    except Exception as e:
        log.warning(f"  METIS failed {e}, fallback naive")
        n_cuts=0
        parts=[i//target for i in range(N)]
        # compute cut approx by counting cross edges
        # fallback: count edges crossing partitions
        for i in range(N):
            pi=parts[i]
            for j in base_adj[i]:
                if j>i and parts[j]!=pi:
                    n_cuts+=1
        # also need edge count
    # Build partition_map node_id->pid
    partition_map={doc_nodes[i].node_id: int(parts[i]) for i in range(N)}
    # Partition stats
    counts=[parts.count(p) for p in range(n_parts)]
    # degree-weighted centroids
    # weight = len(neighbors)+1 where neighbors is original structural? But spec says degree-weighted centroid as historical (len(neighbors)+1) where neighbors includes all edges? Original indexers uses len(nodes[i].neighbors)+1 where neighbors after synthetic insertion (i.e., current graph degree). We'll use current base_adj degree.
    partition_data=defaultdict(list)
    # But we need embeddings and weights
    # weights based on base_adj degree +1
    for i,pid in enumerate(parts):
        w=float(len(base_adj[i]))+1.0
        partition_data[pid].append((embeddings[i], w))
    centroids=[]
    pids_sorted=sorted(partition_data.keys())
    for pid in pids_sorted:
        vecs=[x[0] for x in partition_data[pid]]
        ws=[x[1] for x in partition_data[pid]]
        centroids.append(np.average(vecs, axis=0, weights=ws))
    centroids=np.stack(centroids).astype(np.float32) if centroids else np.zeros((0,embeddings.shape[1]),dtype=np.float32)
    faiss.normalize_L2(centroids)
    # Save if out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Save graph.pt (torch)
        data_torch=None
        try:
            from torch_geometric.data import Data
            edge_src=[]; edge_dst=[]
            for i in range(N):
                for j in base_adj[i]:
                    edge_src.append(i); edge_dst.append(j)
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long)
            data_torch=Data(edge_index=edge_index, num_nodes=N)
            torch.save(data_torch, os.path.join(out_dir,"graph.pt"))
        except Exception as e:
            log.warning(f"Could not save graph.pt {e}")
            # fallback: save edgelist json
            pass
        # Save partition_map.json
        json.dump(partition_map, open(os.path.join(out_dir,"partition_map.json"),"w"), indent=2)
        # Save centroids.index + pids
        if len(pids_sorted)>0:
            dim=centroids.shape[1]
            idx=faiss.IndexFlatIP(dim)
            idx.add(centroids)
            faiss.write_index(idx, os.path.join(out_dir,"centroids.index"))
            json.dump(pids_sorted, open(os.path.join(out_dir,"centroid_pids.json"),"w"))
        # Save stats json
        stats={
            "dataset":dataset,
            "variant":variant,
            "n_nodes":N,
            "n_edges_undirected": total_undirected,
            "n_edges_directed": total_undirected*2,
            "universal_knn_added": univ,
            "isolate_knn_added": iso,
            "ner_edges": ner_edges if variant in ("B","C") else 0,
            "structural_edges": sum(len(s) for s in structural_adj)//2,
            "n_parts": n_parts,
            "n_cuts": n_cuts,
            "edge_cut_ratio": n_cuts/(total_undirected*2) if total_undirected else 0,
            "partition_counts": counts,
            "min_size": min(counts) if counts else 0,
            "max_size": max(counts) if counts else 0,
            "mean_size": float(np.mean(counts)) if counts else 0,
            "median_size": float(np.median(counts)) if counts else 0,
            "std_size": float(np.std(counts)) if counts else 0,
            "centroid_dim": int(centroids.shape[1]) if centroids.size else 0,
        }
        json.dump(stats, open(os.path.join(out_dir,"stats.json"),"w"), indent=2)
        # approximate storage
        # Also write README variant
    # compute storage sizes if out_dir
    return G_nx, parts, n_cuts, partition_map, centroids, pids_sorted, base_adj, total_undirected, structural_adj, ner_adj

def compute_partition_metrics_for_queries(queries: List[Tuple[Any, List[str], List[int]]],
                                         partition_map: Dict[str,int],
                                         doc_id_to_idx: Dict[str,int],
                                         partition_to_nodes: Dict[int, List[int]],
                                         embeddings: np.ndarray,
                                         doc_nodes: List[Any],
                                         centroids: np.ndarray,
                                         centroid_pids: List[int],
                                         dense_index: faiss.Index,
                                         splade_scorer: Optional[SpladeScorer],
                                         variant_stats: Dict[str,Any],
                                         dataset: str,
                                         topKs=K_VALUES):
    """Returns dict with partition recall, candidate recall, docs to L2 etc for each voting method."""
    # Precompute dense query embeddings for all queries (MiniLM)
    # Use DenseEncoder would be similar but we can directly use faiss index for docs and SentenceTransformer for queries?
    # Instead use DenseEncoder from src.core.encoders but that loads same model multi-qa-MiniLM-L6
    from src.core.encoders import DenseEncoder
    encoder=DenseEncoder()
    # Build structures for voting
    # For each query, we need:
    # - centroid ranking
    # - dense_vote ranking (top VOTE_K docs -> count partitions)
    # - splade_vote ranking
    # - RRF fusion
    # We'll also need global dense ranking for D later, but here compute voting
    # To avoid re-encoding per query inside loop, batch encode
    q_texts=[q.content for q,_,_ in queries]
    # Batch dense encode
    log.info(f"  Encoding {len(q_texts)} queries (dense MiniLM) for {dataset}")
    q_dense_embs=encoder.encode(q_texts)  # already normalized
    faiss.normalize_L2(q_dense_embs)
    # For SPLADE, batch encode queries if scorer available
    q_splade_embs=None
    if splade_scorer and splade_scorer.available():
        log.info(f"  Encoding {len(q_texts)} queries (SPLADE) for {dataset}")
        # Batch encode via scorer's tokenizer/model; we'll do batched via private method
        # Use scorer._ensure_model and manual batch
        splade_scorer._ensure_model()
        # use globally imported torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tok, mdl = splade_scorer._tokenizer, splade_scorer._model
        device = splade_scorer._device
        q_splade_embs=[]
        batch=32
        for i in range(0,len(q_texts),batch):
            batch_texts=q_texts[i:i+batch]
            inputs=tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits=mdl(**inputs).logits
                relu_log=torch.log1p(torch.relu(logits))
                mask=inputs["attention_mask"].unsqueeze(-1)
                qvecs=torch.max(relu_log*mask, dim=1).values.cpu().numpy()
                q_splade_embs.append(qvecs)
        q_splade_embs=np.vstack(q_splade_embs)
        # doc matrix
        splade_scorer._ensure_matrix()
        doc_matrix=splade_scorer._matrix  # CSR (N, vocab)
        # For fast per-query top docs, we can compute dot via matrix.dot(qvec) per query but we will loop
    else:
        doc_matrix=None

    # Build centroid index for search (faiss)
    centroid_index=None
    if centroids.size and len(centroid_pids)>0:
        centroid_index=faiss.IndexFlatIP(centroids.shape[1])
        centroid_index.add(centroids.astype(np.float32))

    # For dense global index, we already have dense_index (FAISS nodes.index)
    # For candidate pool, need mapping from pid -> list of node idxs
    # Already have partition_to_nodes

    # Metrics accumulators per method
    methods=["centroid","dense_vote","splade_vote","rrf"]
    # Also we will compute per K for each method: partition_recall, gt_recall, full_coverage, mrr, first_hit, weakest, latency
    # For candidate: pool recall
    results={m: {f"partition_recall@{k}":[] for k in K_VALUES} for m in methods}
    # extend to hold other metrics
    for m in methods:
        for k in K_VALUES:
            results[m][f"partition_gt_recall@{k}"]=[]
            results[m][f"partition_full_coverage@{k}"]=[]
            results[m][f"candidate_recall@{k}"]=[]
            results[m][f"candidate_gt_recall@{k}"]=[]
            results[m][f"candidate_full_coverage@{k}"]=[]
            results[m][f"docs_to_l2@{k}"]=[]
        results[m]["latency_ms"]=[]
        results[m]["mrr"]=[]
        results[m]["avg_first_hit"]=[]
        results[m]["avg_weakest"]=[]

    # D global metrics will be separate
    # For latency measurement per query
    # Iterate queries
    n_parts=len(centroid_pids) if centroid_pids else max(partition_map.values())+1 if partition_map else 0
    # Precompute id->idx for splade? Need mapping from doc_id to row
    splade_id_to_row=None
    if splade_scorer and splade_scorer.available():
        # data contains id_to_idx mapping
        with open(splade_scorer.cache_path,"rb") as f:
            data=pickle.load(f)
            # data["id_to_idx"] is node_id->row
            splade_id_to_row=data["id_to_idx"]
            # also need idx_to_id for doc_matrix rows? Already have.
    # For each query
    for qi, (qnode, gt_doc_ids, gt_pids) in enumerate(tqdm(queries, desc=f"Eval {dataset}")):
        gt_set=set(gt_pids)
        gt_doc_set=set(gt_doc_ids)
        num_gt=len(gt_set)
        # need per method ranking
        # centroid ranking
        rankings={}
        latencies={}
        # centroid
        t0=time.time()
        if centroid_index is not None:
            qv=q_dense_embs[qi:qi+1].astype(np.float32)
            dists, idxs = centroid_index.search(qv, n_parts)  # full ranking
            # idxs are indices into centroids/pids_sorted
            cent_ranked=[int(centroid_pids[idx]) for idx in idxs[0] if 0 <= idx < len(centroid_pids)]
            # If n_parts ranking not full due to smaller index, pad with remaining pids
            if len(cent_ranked)<n_parts:
                remaining=[p for p in centroid_pids if p not in cent_ranked]
                cent_ranked.extend(remaining)
        else:
            cent_ranked=[]
        t1=time.time()
        rankings["centroid"]=cent_ranked
        latencies["centroid"]=(t1-t0)*1000

        # dense vote
        t0=time.time()
        # dense search top VOTE_K docs
        qv_dense=q_dense_embs[qi:qi+1].astype(np.float32)
        _, dense_idxs = dense_index.search(qv_dense, VOTE_K)
        vote_counts={}
        for didx in dense_idxs[0]:
            if 0 <= didx < len(doc_nodes):
                nid=doc_nodes[int(didx)].node_id
                pid=partition_map.get(nid)
                if pid is not None:
                    vote_counts[int(pid)]=vote_counts.get(int(pid),0)+1
        # sort by count descending, then pid
        sorted_pids=sorted(vote_counts.keys(), key=lambda p: vote_counts[p], reverse=True)
        # Need full ranking: voted partitions first, then remaining partitions by centroid? But spec says vote ranking only includes voted partitions, then remaining unranked considered miss.
        # For metric with miss_sentinel = n_parts+1, we need full ranking. We'll create full ranking: voted sorted, then remaining pids sorted by id
        remaining=[p for p in range(n_parts) if p not in vote_counts]
        dense_ranked=sorted_pids+remaining
        t1=time.time()
        rankings["dense_vote"]=dense_ranked
        latencies["dense_vote"]=(t1-t0)*1000

        # splade vote
        if splade_scorer and splade_scorer.available():
            t0=time.time()
            # For query, compute scores = doc_matrix.dot(qvec)
            qvec=q_splade_embs[qi]  # vocab dim ~30522
            scores=doc_matrix.dot(qvec)  # (N,)
            # top VOTE_K doc ids
            if VOTE_K >= scores.shape[0]:
                order=np.argsort(-scores)
            else:
                top=np.argpartition(-scores, VOTE_K)[:VOTE_K]
                order=top[np.argsort(-scores[top])]
            vote_counts_s={}
            for idx in order:
                nid=doc_nodes[int(idx)].node_id  # assuming row order matches doc_nodes order; need verify id_to_idx alignment
                # Actually doc_matrix row order is by id_to_idx mapping, which should correspond to doc_nodes order if id_to_idx built from doc_nodes order
                # But to be safe, map via stored id_to_idx reverse: row->node_id via idx_to_id
                # However we used doc_nodes index order == row order only if splade cache built on same ordering (via ukb_storage build). It should be.
                # Use direct mapping via doc_nodes[idx] if idx < len(doc_nodes)
                pid=partition_map.get(nid)
                if pid is not None:
                    vote_counts_s[int(pid)]=vote_counts_s.get(int(pid),0)+1
            sorted_pids_s=sorted(vote_counts_s.keys(), key=lambda p: vote_counts_s[p], reverse=True)
            remaining_s=[p for p in range(n_parts) if p not in vote_counts_s]
            splade_ranked=sorted_pids_s+remaining_s
            t1=time.time()
            rankings["splade_vote"]=splade_ranked
            latencies["splade_vote"]=(t1-t0)*1000
        else:
            rankings["splade_vote"]=[]
            latencies["splade_vote"]=0

        # RRF fusion of dense + splade (if splade available, else same as dense)
        t0=time.time()
        if rankings["splade_vote"] and rankings["dense_vote"]:
            rrf={}
            for rank_list in [rankings["dense_vote"], rankings["splade_vote"]]:
                for r, pid in enumerate(rank_list):
                    rrf[pid]=rrf.get(pid,0)+1.0/(RRF_K + r +1)
            rrf_ranked=sorted(rrf.keys(), key=lambda p: rrf[p], reverse=True)
            # pad remaining if any missing (should be all)
            if len(rrf_ranked)<n_parts:
                remaining_rrf=[p for p in range(n_parts) if p not in rrf]
                rrf_ranked.extend(remaining_rrf)
        else:
            rrf_ranked=rankings["dense_vote"]
        t1=time.time()
        # if RRF already computed, adjust latency to include fusion only
        # We'll set latency as sum of dense+splade + fusion (approx)
        if latencies.get("dense_vote") and latencies.get("splade_vote"):
            lat_rrf = latencies["dense_vote"]+latencies["splade_vote"]+ (t1-t0)*1000
        else:
            lat_rrf = (t1-t0)*1000 + latencies.get("dense_vote",0)
        rankings["rrf"]=rrf_ranked
        latencies["rrf"]=lat_rrf

        # Now compute metrics per method per K
        for method in methods:
            ranked=rankings.get(method, [])
            if not ranked:
                # method unavailable (e.g., splade)
                for k in K_VALUES:
                    results[method][f"partition_recall@{k}"].append(0.0)
                    results[method][f"partition_gt_recall@{k}"].append(0.0)
                    results[method][f"partition_full_coverage@{k}"].append(0.0)
                    results[method][f"candidate_recall@{k}"].append(0.0)
                    results[method][f"candidate_gt_recall@{k}"].append(0.0)
                    results[method][f"candidate_full_coverage@{k}"].append(0.0)
                    results[method][f"docs_to_l2@{k}"].append(0.0)
                results[method]["latency_ms"].append(latencies.get(method,0))
                results[method]["mrr"].append(0.0)
                results[method]["avg_first_hit"].append(float(n_parts+1))
                results[method]["avg_weakest"].append(float(n_parts+1))
                continue
            # partition metrics
            # compute mrr, first_hit, weakest
            # mrr: first hit in full ranking
            mrr=0.0
            first_hit=n_parts+1
            for i, pid in enumerate(ranked):
                if pid in gt_set:
                    mrr=1.0/(i+1)
                    first_hit=i+1
                    break
            # weakest: max rank among gt_pids
            rank_of={}
            for i,pid in enumerate(ranked):
                if pid in gt_set and pid not in rank_of:
                    rank_of[pid]=i+1
            if gt_set and all(pid in rank_of for pid in gt_set):
                weakest=max(rank_of[pid] for pid in gt_set)
            else:
                weakest=float(n_parts+1)
            results[method]["mrr"].append(mrr)
            results[method]["avg_first_hit"].append(float(first_hit))
            results[method]["avg_weakest"].append(float(weakest))
            results[method]["latency_ms"].append(latencies[method])
            for k in K_VALUES:
                topk=set(ranked[:k])
                hits=len(gt_set & topk)
                recall=1.0 if hits>0 else 0.0
                gt_recall=hits/num_gt if num_gt else 0.0
                full_cov=1.0 if gt_set and gt_set.issubset(topk) else 0.0
                results[method][f"partition_recall@{k}"].append(recall)
                results[method][f"partition_gt_recall@{k}"].append(gt_recall)
                results[method][f"partition_full_coverage@{k}"].append(full_cov)
                # candidate pool: union of docs in topk partitions
                pool=set()
                for pid in ranked[:k]:
                    pool.update(partition_to_nodes.get(pid, []))
                # pool indices to doc ids
                pool_doc_ids=set(doc_nodes[idx].node_id for idx in pool)
                cand_hits=len(gt_doc_set & pool_doc_ids)
                cand_recall=1.0 if cand_hits>0 else 0.0
                cand_gt_recall=cand_hits/len(gt_doc_set) if gt_doc_set else 0.0
                cand_full=1.0 if gt_doc_set and gt_doc_set.issubset(pool_doc_ids) else 0.0
                docs_to_l2=len(pool)
                results[method][f"candidate_recall@{k}"].append(cand_recall)
                results[method][f"candidate_gt_recall@{k}"].append(cand_gt_recall)
                results[method][f"candidate_full_coverage@{k}"].append(cand_full)
                results[method][f"docs_to_l2@{k}"].append(float(docs_to_l2))

    # aggregate
    agg={}
    for method in methods:
        agg[method]={}
        for key, vals in results[method].items():
            if not vals:
                continue
            if "latency" in key or "first_hit" in key or "weakest" in key:
                agg[method][f"avg_{key}"]=float(np.mean(vals))
                agg[method][f"median_{key}"]=float(np.median(vals))
                agg[method][f"p95_{key}"]=float(np.percentile(vals,95))
                agg[method][f"p99_{key}"]=float(np.percentile(vals,99))
            elif "docs_to_l2" in key:
                agg[method][f"mean_{key}"]=float(np.mean(vals))
                agg[method][f"median_{key}"]=float(np.median(vals))
                agg[method][f"p95_{key}"]=float(np.percentile(vals,95))
            else:
                # recall etc as percentage
                agg[method][key]=float(np.mean(vals)*100)
                agg[method][key+"_raw"]=float(np.mean(vals))
    return agg, results

def compute_global_recall(queries, doc_nodes, doc_id_to_idx, dense_index, splade_scorer, q_dense_embs, q_splade_embs):
    """Compute global dense/splade Recall@K at standard Ks and at matched candidate counts."""
    # For each query, compute global ranking via dense FAISS and via SPLADE dot
    # Evaluate at Ks = 1,3,5,10,20,50,100,200 and also at pool-size matched Ks (mean docs_to_l2 from earlier)
    Ks_std=[1,3,5,10,20,50,100,200]
    # Also compute max 500?
    dense_recall={f"recall@{k}":[] for k in Ks_std}
    splade_recall={f"recall@{k}":[] for k in Ks_std} if splade_scorer and splade_scorer.available() else {}
    dense_gt_recall={f"gt_recall@{k}":[] for k in Ks_std}
    splade_gt_recall={f"gt_recall@{k}":[] for k in Ks_std} if splade_recall else {}
    N=len(doc_nodes)
    # doc_matrix for splade if available
    doc_matrix=None
    if splade_scorer and splade_scorer.available():
        splade_scorer._ensure_matrix()
        doc_matrix=splade_scorer._matrix
    for qi, (qnode, gt_doc_ids, _) in enumerate(tqdm(queries, desc="Global recall")):
        gt_set=set(gt_doc_ids)
        num_gt=len(gt_set)
        # dense global
        qv=q_dense_embs[qi:qi+1].astype(np.float32)
        _, idxs = dense_index.search(qv, max(Ks_std))
        ranked_dense=[doc_nodes[int(i)].node_id for i in idxs[0] if 0 <= i < N]
        for k in Ks_std:
            topk=set(ranked_dense[:k])
            hits=len(gt_set & topk)
            dense_recall[f"recall@{k}"].append(1.0 if hits>0 else 0.0)
            dense_gt_recall[f"gt_recall@{k}"].append(hits/num_gt if num_gt else 0.0)
        # splade global
        if doc_matrix is not None:
            qvec=q_splade_embs[qi]
            scores=doc_matrix.dot(qvec)
            if max(Ks_std) >= N:
                order=np.argsort(-scores)
            else:
                top=np.argpartition(-scores, max(Ks_std))[:max(Ks_std)]
                order=top[np.argsort(-scores[top])]
            ranked_splade=[doc_nodes[int(i)].node_id for i in order]
            for k in Ks_std:
                topk=set(ranked_splade[:k])
                hits=len(gt_set & topk)
                splade_recall[f"recall@{k}"].append(1.0 if hits>0 else 0.0)
                splade_gt_recall[f"gt_recall@{k}"].append(hits/num_gt if num_gt else 0.0)
    # aggregate
    agg_dense={k: float(np.mean(v)*100) for k,v in dense_recall.items()}
    agg_dense.update({k+"_gt": float(np.mean(v)*100) for k,v in dense_gt_recall.items()})
    agg_splade={k: float(np.mean(v)*100) for k,v in splade_recall.items()} if splade_recall else {}
    if splade_recall:
        agg_splade.update({k+"_gt": float(np.mean(v)*100) for k,v in splade_gt_recall.items()})
    return {"dense":agg_dense, "splade":agg_splade}

def main():
    import argparse
    parser=argparse.ArgumentParser(description="L1 routing-graph ablation")
    parser.add_argument("--datasets", nargs="+", default=DATASETS_PRIORITY, help="datasets to run")
    parser.add_argument("--variants", nargs="+", default=["A","B","C"], choices=["A","B","C"])
    parser.add_argument("--target", type=int, default=TARGET_PER_PARTITION)
    parser.add_argument("--limit", type=int, default=None, help="max queries per split (for speed)")
    parser.add_argument("--out", default="results/l1_routing_ablation")
    args=parser.parse_args()

    os.makedirs("scratchpad/ablation", exist_ok=True)
    # Root out
    base_out=args.out
    os.makedirs(base_out, exist_ok=True)

    all_results={}
    for dataset in args.datasets:
        log.info(f"===== Dataset {dataset} =====")
        # Load master and embeddings
        try:
            nodes, doc_nodes, q_nodes, doc_id_to_idx = load_master(dataset)
        except Exception as e:
            log.error(f"Failed load_master {dataset}: {e}")
            continue
        N=len(doc_nodes)
        log.info(f"  doc_nodes {N}, q_nodes {len(q_nodes)}")
        # Load embeddings
        try:
            embeddings, dense_index = load_embeddings_faiss(dataset)
        except Exception as e:
            log.error(f"Failed load embeddings {dataset}: {e}")
            continue
        # Partition map for split: need one to determine gold reachable? Use existing A partition_map for split determination
        # But for consistent split across variants, we use deterministic shuffle on query ids regardless of partition_map
        # So we can generate split without partition_map, just using doc_id_to_idx
        # For gold_reachable we need to know doc existence
        # Create dummy partition_map for split: use existing A if available, else empty
        dummy_pm={}
        pm_path=f"data/ukb_storage/{dataset}/partition_map.json"
        if os.path.exists(pm_path):
            dummy_pm=json.load(open(pm_path))
        splits=get_split_queries(q_nodes, dummy_pm, doc_id_to_idx, dataset)
        # Optionally limit
        if args.limit:
            for k in splits:
                splits[k]=splits[k][:args.limit]
        test_queries=splits["test"]
        val_queries=splits["val"]
        log.info(f"  splits: train {len(splits['train'])} val {len(val_queries)} test {len(test_queries)}")
        # Compute gold_reachable (doc existence) for test
        reachable_test=sum(1 for _,docs,_ in test_queries if all(d in doc_id_to_idx for d in docs))/len(test_queries) if test_queries else 0
        reachable_all=sum(1 for q in q_nodes if all(nid in doc_id_to_idx for nid in q.neighbors if q.neighbors))/len(q_nodes) if q_nodes else 0
        # Actually gold_reachable = fraction where all gold docs are in doc_id_to_idx (should be 100% for clean)
        dataset_results={"dataset":dataset, "n_docs":N, "n_queries_total":len(q_nodes), "test_queries":len(test_queries), "gold_reachable_test":reachable_test}
        # Load splade scorer
        splade_scorer=SpladeScorer(dataset)
        # For each variant
        for variant in args.variants:
            # KB handling: B/C N/A for metaqa/webqsp per spec
            if dataset in ["metaqa","webqsp"] and variant in ["B","C"]:
                log.info(f"  Variant {variant} N/A for KB dataset {dataset}, marking unavailable")
                dataset_results[variant]={"available":False, "reason":"KB dataset, NER not applicable per spec"}
                continue
            if variant in ["B","C"] and not ner_available(dataset):
                log.info(f"  Variant {variant} N/A for {dataset}, no NER artifact")
                dataset_results[variant]={"available":False, "reason":"no NER artifact for this universe"}
                continue
            log.info(f"  --- Variant {variant} ---")
            out_dir=f"scratchpad/ablation/{dataset}/variant_{variant}"
            stats_path=os.path.join(out_dir,"stats.json")
            # If variant A, use canonical existing
            if variant=="A":
                # Load existing graph/partition stats
                # Use existing out_dir canonical? We'll compute stats from existing
                pm_path=f"data/ukb_storage/{dataset}/partition_map.json"
                graph_path=f"data/ukb_storage/{dataset}/graph.pt"
                cent_path=f"data/ukb_storage/{dataset}/centroids.index"
                cent_pids_path=f"data/ukb_storage/{dataset}/centroid_pids.json"
                if not os.path.exists(pm_path):
                    log.error(f"  Missing partition_map for A {dataset}")
                    continue
                partition_map=json.load(open(pm_path))
                # Compute N, parts, counts
                parts=[partition_map[doc_nodes[i].node_id] for i in range(N) if doc_nodes[i].node_id in partition_map]
                # Need n_parts from max
                n_parts=max(partition_map.values())+1
                counts=[list(partition_map.values()).count(p) for p in range(n_parts)]
                # centroids
                try:
                    cent_idx=faiss.read_index(cent_path) if os.path.exists(cent_path) else None
                    cent_pids=json.load(open(cent_pids_path)) if os.path.exists(cent_pids_path) else list(range(n_parts))
                    if cent_idx is not None:
                        centroids=cent_idx.reconstruct_n(0, cent_idx.ntotal)
                    else:
                        centroids=np.zeros((n_parts, embeddings.shape[1]),dtype=np.float32)
                except Exception as e:
                    log.warning(f"  centroid load failed {e}")
                    centroids=np.zeros((n_parts, embeddings.shape[1]),dtype=np.float32)
                    cent_pids=list(range(n_parts))
                    cent_idx=None
                # Graph stats
                try:
                    g=torch.load(graph_path, map_location="cpu", weights_only=False)
                    edge_undirected=g.edge_index.shape[1]//2 if hasattr(g,"edge_index") else 0
                except Exception as e:
                    log.warning(f"  graph load failed {e}")
                    edge_undirected=0
                # n_cuts: need to compute from pymetis? We have previous stats? Use stored? We can estimate via edge cut ratio from G_nx? For A we could rebuild to get n_cuts, but we have no stored n_cuts. We'll approximate 0 or compute quickly by loading graph and partition_map
                # For now, compute n_cuts by counting cross-partition edges via adjacency
                # Build adjacency quickly from graph.pt if available
                n_cuts=0
                # quick: if we have G_nx not loaded, we can compute from edge_index
                try:
                    if 'g' in locals() and hasattr(g,"edge_index"):
                        edge_index=g.edge_index.numpy() if isinstance(g.edge_index, torch.Tensor) else np.array(g.edge_index)
                        # edge_index shape 2 x 2E
                        for k in range(edge_index.shape[1]//2):
                            i=int(edge_index[0,2*k]); j=int(edge_index[1,2*k])  # undirected mirror, take one direction
                            # need to map i->pid via doc_nodes[i].node_id
                            # i is doc idx
                            if i < N and j < N:
                                pi=partition_map.get(doc_nodes[i].node_id, -1)
                                pj=partition_map.get(doc_nodes[j].node_id, -1)
                                if pi!=pj:
                                    n_cuts+=1
                except Exception as e:
                    n_cuts=0
                # Build partition_to_nodes
                partition_to_nodes=defaultdict(list)
                for idx, node in enumerate(doc_nodes):
                    pid=partition_map.get(node.node_id)
                    if pid is not None:
                        partition_to_nodes[int(pid)].append(idx)
                # For metrics, we need G_nx? Not needed for eval
                G_nx=None
                base_adj=None
                # Store stats
                # Compute storage sizes
                storage_graph=os.path.getsize(graph_path) if os.path.exists(graph_path) else 0
                storage_cent=os.path.getsize(cent_path) if os.path.exists(cent_path) else 0
                storage_pm=os.path.getsize(pm_path) if os.path.exists(pm_path) else 0
                variant_stats={
                    "variant":"A",
                    "available":True,
                    "n_nodes":N,
                    "n_edges_undirected":edge_undirected,
                    "n_parts":n_parts,
                    "n_cuts":n_cuts,
                    "edge_cut_ratio": n_cuts/(edge_undirected*2) if edge_undirected else 0,
                    "partition_counts":counts,
                    "min_size":min(counts) if counts else 0,
                    "max_size":max(counts) if counts else 0,
                    "mean_size":float(np.mean(counts)) if counts else 0,
                    "median_size":float(np.median(counts)) if counts else 0,
                    "std_size":float(np.std(counts)) if counts else 0,
                    "storage_bytes":{"graph":storage_graph,"centroids":storage_cent,"partition_map":storage_pm,"total":storage_graph+storage_cent+storage_pm},
                    "ram_estimate_mb": (embeddings.nbytes + edge_undirected*2*8 + n_parts*embeddings.shape[1]*4)/1e6,
                }
                dataset_results["A_stat"]=variant_stats
                # Evaluate
                # Need centroids array and pids for search
                if centroids is None or len(centroids)==0:
                    centroids=np.zeros((n_parts, embeddings.shape[1]),dtype=np.float32)
                    cent_pids=list(range(n_parts))
                agg, _ = compute_partition_metrics_for_queries(test_queries, partition_map, doc_id_to_idx, partition_to_nodes, embeddings, doc_nodes, centroids, cent_pids, dense_index, splade_scorer, variant_stats, dataset)
                dataset_results["A"]=agg
                dataset_results["A_stat"].update({"eval":agg})
            else:
                # Build B/C
                # Check if already built
                if os.path.exists(stats_path):
                    try:
                        variant_stats=json.load(open(stats_path))
                        # Load partition_map etc for eval
                        pm=json.load(open(os.path.join(out_dir,"partition_map.json")))
                        # centroids
                        cent_pids=json.load(open(os.path.join(out_dir,"centroid_pids.json")))
                        cent_idx=faiss.read_index(os.path.join(out_dir,"centroids.index"))
                        centroids=cent_idx.reconstruct_n(0, cent_idx.ntotal)
                        partition_to_nodes=defaultdict(list)
                        for idx, node in enumerate(doc_nodes):
                            pid=pm.get(node.node_id)
                            if pid is not None:
                                partition_to_nodes[int(pid)].append(idx)
                        # Deduplicate stats vs agg
                        log.info(f"  Reusing built {variant} from {out_dir}")
                    except Exception as e:
                        log.warning(f"  Failed reuse {e}, rebuilding")
                        variant_stats=None
                        pm=None
                    if 'agg' not in locals() or variant_stats is None:
                        pass
                    else:
                        # Need to recompute agg? maybe cached results json
                        results_path=os.path.join(out_dir,"eval.json")
                        if os.path.exists(results_path):
                            agg=json.load(open(results_path))
                            dataset_results[variant]=agg
                            dataset_results[f"{variant}_stat"]=variant_stats
                            continue
                # Build fresh
                G_nx, parts, n_cuts, partition_map, centroids, centroid_pids, base_adj, total_undirected, structural_adj, ner_adj = build_graph_and_partitions(dataset, variant, embeddings, doc_nodes, doc_id_to_idx, target=args.target, out_dir=out_dir)
                if G_nx is None:
                    dataset_results[variant]={"available":False}
                    continue
                # Build partition_to_nodes
                partition_to_nodes=defaultdict(list)
                for idx, node in enumerate(doc_nodes):
                    pid=partition_map.get(node.node_id)
                    if pid is not None:
                        partition_to_nodes[int(pid)].append(idx)
                # storage
                storage_graph=os.path.getsize(os.path.join(out_dir,"graph.pt")) if os.path.exists(os.path.join(out_dir,"graph.pt")) else 0
                storage_cent=os.path.getsize(os.path.join(out_dir,"centroids.index")) if os.path.exists(os.path.join(out_dir,"centroids.index")) else 0
                storage_pm=os.path.getsize(os.path.join(out_dir,"partition_map.json"))
                variant_stats={
                    "variant":variant,
                    "available":True,
                    "n_nodes":N,
                    "n_edges_undirected":total_undirected,
                    "n_parts":len(set(parts)),
                    "n_cuts":int(n_cuts) if n_cuts is not None else 0,
                    "edge_cut_ratio": float(n_cuts/(total_undirected*2)) if total_undirected else 0,
                    "partition_counts":[parts.count(p) for p in range(max(parts)+1)],
                    "min_size":min([parts.count(p) for p in range(max(parts)+1)]) if parts else 0,
                    "max_size":max([parts.count(p) for p in range(max(parts)+1)]) if parts else 0,
                    "mean_size":float(np.mean([parts.count(p) for p in range(max(parts)+1)])) if parts else 0,
                    "median_size":float(np.median([parts.count(p) for p in range(max(parts)+1)])) if parts else 0,
                    "std_size":float(np.std([parts.count(p) for p in range(max(parts)+1)])) if parts else 0,
                    "storage_bytes":{"graph":storage_graph,"centroids":storage_cent,"partition_map":storage_pm,"total":storage_graph+storage_cent+storage_pm},
                    "ram_estimate_mb": (embeddings.nbytes + total_undirected*2*8 + len(set(parts))*embeddings.shape[1]*4)/1e6,
                }
                json.dump(variant_stats, open(stats_path,"w"), indent=2)
                # Evaluate
                agg, _ = compute_partition_metrics_for_queries(test_queries, partition_map, doc_id_to_idx, partition_to_nodes, embeddings, doc_nodes, centroids, centroid_pids, dense_index, splade_scorer, variant_stats, dataset)
                json.dump(agg, open(os.path.join(out_dir,"eval.json"),"w"), indent=2)
                dataset_results[variant]=agg
                dataset_results[f"{variant}_stat"]=variant_stats
        # D global control
        log.info(f"  === D global control {dataset} ===")
        # For D we need q_dense_embs and q_splade_embs; compute via same encoder? Reuse logic from evaluation but we can call compute_global_recall
        # Need to get embeddings for test queries again (we already have inside compute but we need outside)
        from src.core.encoders import DenseEncoder
        encoder=DenseEncoder()
        q_texts=[q.content for q,_,_ in test_queries]
        q_dense_embs=encoder.encode(q_texts)
        faiss.normalize_L2(q_dense_embs)
        q_splade_embs=None
        if splade_scorer.available():
            splade_scorer._ensure_model()
            tok, mdl = splade_scorer._tokenizer, splade_scorer._model
            device=splade_scorer._device
            q_splade_embs=[]
            batch=32
            for i in range(0,len(q_texts),batch):
                batch_texts=q_texts[i:i+batch]
                inputs=tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    logits=mdl(**inputs).logits
                    relu_log=torch.log1p(torch.relu(logits))
                    mask=inputs["attention_mask"].unsqueeze(-1)
                    qvecs=torch.max(relu_log*mask, dim=1).values.cpu().numpy()
                    q_splade_embs.append(qvecs)
            q_splade_embs=np.vstack(q_splade_embs)
        global_agg=compute_global_recall(test_queries, doc_nodes, doc_id_to_idx, dense_index, splade_scorer, q_dense_embs, q_splade_embs)
        # Also matched candidate budget: for each variant's mean docs_to_l2 at K, compute global recall at that K
        # We'll compute for each variant and each K's mean docs_to_l2, the global recall at that pool size
        # For simplicity, we add to global_agg the mapping
        matched={}
        for variant in ["A","B","C"]:
            if variant not in dataset_results or not isinstance(dataset_results[variant], dict) or "dense_vote" not in dataset_results[variant]:
                continue
            for k in K_VALUES:
                key=f"mean_docs_to_l2@{k}"
                # get from dense_vote aggregation
                # agg stores mean_docs_to_l2@k inside dense_vote? Actually per method we stored mean_docs_to_l2@k
                # Access dataset_results[variant]["dense_vote"][key] etc but our agg is nested: dataset_results[variant][method][mean_...]
                # Let's extract
                try:
                    mean_docs=dataset_results[variant]["dense_vote"][f"mean_docs_to_l2@{k}"]
                    # Compute global dense recall at K = round(mean_docs)
                    # Need to compute global recall at that K specifically (not just std Ks)
                    # We can compute directly: for each query, check if gold in top mean_docs global
                    # Simplify: approximate using interpolation of std Ks? For now compute exact per query mean? Better compute exact per query pool size? But we have mean, we can compute recall at that integer K
                    k_global=int(round(mean_docs))
                    # Compute recall at that K via full ranking already? We have dense ranking up to 200, but mean_docs may be >200 (e.g., 500). Need to compute for that k individually.
                    # We'll recompute global recall at that k
                    hits=[]
                    for qi,(qnode, gt_docs, _) in enumerate(test_queries):
                        qv=q_dense_embs[qi:qi+1].astype(np.float32)
                        _, idxs = dense_index.search(qv, k_global)
                        ranked=[doc_nodes[int(i)].node_id for i in idxs[0] if 0 <= i < len(doc_nodes)]
                        hits.append(1.0 if set(gt_docs) & set(ranked[:k_global]) else 0.0)
                    matched[f"{variant}_dense_global_recall_at_match_{k}parts_mean{int(mean_docs)}"]=float(np.mean(hits)*100)
                except Exception as e:
                    pass
        global_agg["matched"]=matched
        dataset_results["D"]=global_agg
        # Save dataset result
        out_path=os.path.join(base_out, f"{dataset}.json")
        json.dump(dataset_results, open(out_path,"w"), indent=2)
        # Also save csv per method
        all_results[dataset]=dataset_results
        # Update combined
        json.dump(all_results, open(os.path.join(base_out, "combined.json"),"w"), indent=2)
        log.info(f"  Saved {out_path}")

    # Summary markdown
    summary_path=os.path.join(base_out, "SUMMARY.md")
    with open(summary_path,"w", encoding="utf-8") as f:
        f.write("# L1 Routing-Graph Ablation Summary\n\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, target={args.target}, vote_k={VOTE_K}\n\n")
        for ds, res in all_results.items():
            f.write(f"## {ds}\n\n")
            f.write(f"- n_docs {res.get('n_docs')} test {res.get('test_queries')} gold_reachable {res.get('gold_reachable_test',0):.3f}\n")
            for var in ["A","B","C"]:
                stat=res.get(f"{var}_stat") or res.get(var)
                if isinstance(stat, dict) and stat.get("available",True)==False:
                    f.write(f"- **{var}**: N/A ({stat.get('reason')})\n")
                    continue
                if var in res and isinstance(res[var], dict):
                    # find dense_vote recall@5 etc
                    dv=res[var].get("dense_vote",{})
                    f.write(f"- **{var}** edges {stat.get('n_edges_undirected')} parts {stat.get('n_parts')} cuts {stat.get('n_cuts')} "
                            f"storage {stat.get('storage_bytes',{}).get('total',0)//1024}KB "
                            f"dense_vote R@5 {dv.get('partition_recall@5',0):.1f} cand R@5 {dv.get('candidate_recall@5',0):.1f} "
                            f"docs_to_L2@5 mean {dv.get('mean_docs_to_l2@5',0):.1f}\n")
            if "D" in res:
                d=res["D"]["dense"]
                f.write(f"- **D global dense** R@5 {d.get('recall@5',0):.1f} R@20 {d.get('recall@20',0):.1f} R@100 {d.get('recall@100',0):.1f}\n")
            f.write("\n")
    log.info(f"Summary written to {summary_path}")
    # Also produce machine-readable CSV
    csv_path=os.path.join(base_out, "table.csv")
    # Flatten per dataset/variant/method/K
    rows=[]
    for ds, res in all_results.items():
        for var in ["A","B","C"]:
            if var not in res or not isinstance(res[var], dict) or "dense_vote" not in res[var]:
                continue
            stat=res.get(f"{var}_stat",{})
            for method in ["centroid","dense_vote","splade_vote","rrf"]:
                if method not in res[var]: continue
                m=res[var][method]
                row={"dataset":ds,"variant":var,"method":method,
                     "n_parts":stat.get("n_parts"),"n_edges":stat.get("n_edges_undirected"),"n_cuts":stat.get("n_cuts"),
                     "storage_total_kb":stat.get("storage_bytes",{}).get("total",0)//1024,
                     "ram_mb":stat.get("ram_estimate_mb")}
                for k in K_VALUES:
                    row[f"part_R@{k}"]=m.get(f"partition_recall@{k}",0)
                    row[f"cand_R@{k}"]=m.get(f"candidate_recall@{k}",0)
                    row[f"docs_to_L2@{k}"]=m.get(f"mean_docs_to_l2@{k}",0)
                row["mrr"]=m.get("mrr",0)
                row["lat_p95"]=m.get("p95_latency_ms",0)
                rows.append(row)
    if rows:
        import csv
        with open(csv_path,"w", newline="", encoding="utf-8") as cf:
            writer=csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"CSV written to {csv_path}")

if __name__=="__main__":
    main()
