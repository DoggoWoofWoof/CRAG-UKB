"""Lane B: L1 routing evaluation for TEST/EVAL populations.
108 cells: K 25/50/100/200 x Router Dense/SPLADE/Dense+SPLADE x Topology A/B/C x P 20/50/100
Uses exact top200 caches and partition maps from ablation_qwen.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json, pickle, hashlib, logging
import numpy as np
from collections import defaultdict

log = logging.getLogger("l1eval")

K_VALUES = [25,50,100,200]
P_VALUES = [20,50,100]
ROUTERS = ["dense","splade","dense+splade"]
TOPOS = ["A","B","C"]
K0=60; TOPN_K = {25:25,50:50,100:100,200:200}

def load_partition_topology(dataset, topo):
    if topo=="A":
        pm_path=f"data/ukb_storage/{dataset}/gte_qwen/partition_map.json"
    else:
        pm_path=f"scratchpad/ablation_qwen/{dataset}/variant_{topo}/partition_map.json"
    if not os.path.exists(pm_path):
        raise FileNotFoundError(f"Canonical Qwen B/C missing: {pm_path} for {dataset} {topo} — no MiniLM fallback allowed")
    pm=json.load(open(pm_path))
    # Build hard array aligned to doc order: need doc_nodes order
    from src.pipeline.standardizer import load_nodes
    mpath=f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mpath):
        mpath="data/processed/master_nodes.json"
    all_nodes=load_nodes(mpath)
    srcs=set(n.metadata.get("source","") for n in all_nodes)
    if len(srcs)>1:
        all_nodes=[n for n in all_nodes if n.metadata.get("source")==dataset]
    doc_nodes=[n for n in all_nodes if n.metadata.get("type")!="question"]
    # Ensure order matches engine.nodes order? Use doc_nodes order as is (load_nodes preserves file order)
    # For gte_qwen, engine.nodes order is same as load_nodes filtered order, so we can use this
    N=len(doc_nodes)
    # Verify pm size
    assert len(pm)==N, f"pm {len(pm)} vs N {N} for {dataset} {topo} {pm_path}"
    # Build hard array
    hard=np.array([int(pm.get(doc_nodes[i].node_id, -1)) for i in range(N)], dtype=np.int64)
    npart=int(max(hard))+1
    # Build mem_idx: own + 1-hop neighbors' partitions
    doc_id_to_idx={n.node_id:i for i,n in enumerate(doc_nodes)}
    mem_idx=[]
    for i, nd in enumerate(doc_nodes):
        s=set([int(hard[i])])
        for nb in nd.neighbors:
            j=doc_id_to_idx.get(nb)
            if j is not None:
                s.add(int(hard[j]))
        mem_idx.append(sorted(s))
    # Partition size stats
    from collections import Counter
    cnt=Counter(hard.tolist())
    sizes=list(cnt.values())
    stats={"npart":npart, "mean":float(np.mean(sizes)), "median":float(np.median(sizes)), "p95":float(np.percentile(sizes,95)), "min":min(sizes), "max":max(sizes)}
    return hard, mem_idx, npart, doc_nodes, stats

def rrf_fuse(rank_lists, k0=60, topp=None):
    score=defaultdict(float)
    for lst in rank_lists:
        for r,pid in enumerate(lst):
            score[int(pid)]+=1.0/(k0 + r)
    out=sorted(score.keys(), key=lambda p: -score[p])
    return out[:topp] if topp else out

def check_canonical_parity():
    """Compare new evaluator vs existing canonical _feats/_rr/membership for 2wiki_clean TEST 32."""
    import numpy as np
    from src.experiments.l1_rerank100 import _feats as canon_feats, _rr as canon_rr
    from src.experiments.overlap_retrain import _onehop_membership
    from src.core.engine import CoreEngine
    from src.pipeline.standardizer import load_nodes
    dataset="2wiki_clean"
    # Load canonical engine for A and C
    for topo, K in [("A",25),("A",100),("C",100)]:
        # Load partition topology via new method
        hard_new, mem_idx_new, npart_new, _, _ = load_partition_topology(dataset, topo)
        # Load canonical engine and mem
        # For canonical, need to construct engine with appropriate partition_map
        # For A, use gte_qwen engine; for C, need ablation_qwen engine but we can mimic via loading pm and building mem via _onehop_membership on a mock engine
        # Instead directly compute canonical mem via _onehop_membership on an engine that has the same partition_map
        # Create a mock engine object with nodes and partition_map
        from src.pipeline.standardizer import load_nodes
        mpath=f"data/processed/master_nodes_{dataset}.json"
        all_nodes=load_nodes(mpath)
        doc_nodes=[n for n in all_nodes if n.metadata.get("type")!="question"]
        # Build mock engine
        class MockEng:
            def __init__(self, doc_nodes, pm):
                self.nodes=doc_nodes
                self.partition_map=pm
        pm_path = f"data/ukb_storage/{dataset}/gte_qwen/partition_map.json" if topo=="A" else f"scratchpad/ablation_qwen/{dataset}/variant_{topo}/partition_map.json"
        pm=json.load(open(pm_path))
        mock=MockEng(doc_nodes, pm)
        mem_canon_dict=_onehop_membership(mock)
        # Convert to mem_idx list aligned to doc order
        mem_idx_canon=[sorted(mem_canon_dict.get(doc_nodes[i].node_id, {int(hard_new[i])})) for i in range(len(doc_nodes))]
        # Compare mem_idx
        match_mem = sum(1 for a,b in zip(mem_idx_new, mem_idx_canon) if a==b)
        log.info(f"[PARITY {dataset}/{topo}] mem_idx match {match_mem}/{len(mem_idx_new)}")
        # Now test _feats/_rr vs new evaluator's partition_ranking for first 32 test queries
        # Load test dense order
        qids_path=f"data/ukb_storage/{dataset}/gte_qwen/query_ids_all.json"
        j=json.load(open(qids_path))
        test_idx=j["split_indices"]["test"][:32]
        dense_all=np.load(f"data/ukb_storage/{dataset}/gte_qwen/dense_top200_all.npy")
        order_K=np.array([dense_all[i,:K] for i in test_idx])
        # Canon feats
        S_c, M_c = canon_feats(order_K, mem_idx_canon, npart_new, topn=K, k0=60)
        votes_c = canon_rr(S_c) + canon_rr(M_c)
        ranking_c = np.argsort(-votes_c, axis=1)
        # New evaluator feats (replicate)
        S_n=np.zeros((len(order_K), npart_new), dtype=np.float32)
        M_n=np.zeros((len(order_K), npart_new), dtype=np.float32)
        for qi in range(len(order_K)):
            for r, nd in enumerate(order_K[qi][:K]):
                w=1.0/(60+r)
                if nd<0 or nd>=len(mem_idx_new):
                    continue
                for p in mem_idx_new[int(nd)]:
                    S_n[qi,p]+=w
                    if w> M_n[qi,p]:
                        M_n[qi,p]=w
        def rr_new(score):
            order=np.argsort(-score, axis=1)
            rank=np.empty((len(order_K), npart_new), dtype=np.int32)
            rows=np.arange(len(order_K))[:,None]
            rank[rows, order]=np.arange(npart_new)[None,:]
            return 1.0/(60+rank)
        votes_n=rr_new(S_n)+rr_new(M_n)
        ranking_n=np.argsort(-votes_n, axis=1)
        # Compare top100
        top20_match=np.mean([len(set(ranking_c[qi,:20]) & set(ranking_n[qi,:20]))/20 for qi in range(len(order_K))])
        top50_match=np.mean([len(set(ranking_c[qi,:50]) & set(ranking_n[qi,:50]))/50 for qi in range(len(order_K))])
        top100_match=np.mean([len(set(ranking_c[qi,:100]) & set(ranking_n[qi,:100]))/100 for qi in range(len(order_K))])
        log.info(f"[PARITY {dataset}/{topo} K={K}] TOP20 {top20_match:.4f} TOP50 {top50_match:.4f} TOP100 {top100_match:.4f} CANONICAL_ROUTING_PARITY={'PASS' if top20_match>0.99 and top50_match>0.99 else 'FAIL'}")
        if top20_match<0.99 or top50_match<0.99:
            return False
    log.info("CANONICAL_ROUTING_PARITY=PASS TOP20>0.99 TOP50>0.99")
    return True

def evaluate_dataset(dataset, max_k=200):
    from src.core.engine import CoreEngine
    from src.experiments.overlap_retrain import _splits, _hard_membership
    # Load test ids/golds
    # Use queries_all.json split_indices to get test rows
    qids_path=f"data/ukb_storage/{dataset}/gte_qwen/query_ids_all.json"
    if not os.path.exists(qids_path):
        # fallback to dry32? but for full we need all
        qids_path=f"data/ukb_storage/{dataset}/gte_qwen/queries_all.json"  # alternative?
        pass
    # Actually queries_all.json is at data/ukb_storage/{ds}/gte_qwen/query_ids_all.json per phase1
    # Let's find
    candidates=[f"data/ukb_storage/{dataset}/gte_qwen/query_ids_all.json", f"data/ukb_storage/{dataset}/gte_qwen/queries_all.json"]
    # Use the one that exists
    for cand in candidates:
        if os.path.exists(cand):
            qids_path=cand
            break
    j=json.load(open(qids_path))
    all_ids=j["ids"]
    split_idx=j["split_indices"]
    golds=j["golds"]  # list of lists of doc ids
    hops=j.get("hops", [None]*len(all_ids))
    # Evaluation population selector per spec
    if dataset == "webqsp":
        test_idx = split_idx["all"]
        eval_protocol = "ALL_EVALUABLE_CURRENT_SUBSTRATE"
    elif dataset in ["2wiki_clean","musique_clean","squad_clean","hotpotqa_clean"]:
        test_idx = split_idx["test"]
        eval_protocol = "DERIVED_TEST"
    elif dataset == "metaqa":
        test_idx = split_idx["test"]
        eval_protocol = "NATIVE_TEST"
    else:
        test_idx = split_idx["test"] if split_idx["test"] else split_idx["all"]
        eval_protocol = "DERIVED_TEST"
    n_test=len(test_idx)
    log.info(f"[{dataset}] EVAL_PROTOCOL={eval_protocol} N_EVAL={n_test} all={len(all_ids)}")
    # Map gold doc ids to doc indices for this dataset (need doc_id_to_idx)
    from src.pipeline.standardizer import load_nodes
    mpath=f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mpath):
        mpath="data/processed/master_nodes.json"
    all_nodes=load_nodes(mpath)
    srcs=set(n.metadata.get("source","") for n in all_nodes)
    if len(srcs)>1:
        all_nodes=[n for n in all_nodes if n.metadata.get("source")==dataset]
    doc_nodes=[n for n in all_nodes if n.metadata.get("type")!="question"]
    doc_id_to_idx={n.node_id:i for i,n in enumerate(doc_nodes)}
    # Build test gold indices
    test_golds=[]
    test_hops=[]
    for idx in test_idx:
        gids=golds[idx]
        # Convert to doc idx
        g_idx=[doc_id_to_idx[g] for g in gids if g in doc_id_to_idx]
        test_golds.append(g_idx)
        test_hops.append(hops[idx])
    # Load dense/splade top200 for all, then slice test
    dense_all_path=f"data/ukb_storage/{dataset}/gte_qwen/dense_top200_all.npy"
    splade_all_path=f"data/ukb_storage/{dataset}/gte_qwen/splade_top200_all.npy"
    # For 2wiki etc, these exist; for hotpot, dense exists, splade may be partial
    dense_all=None
    if os.path.exists(dense_all_path):
        arr=np.load(dense_all_path, mmap_mode='r')
        if len(arr)==len(all_ids):
            dense_all=np.array([arr[i] for i in test_idx])
        else:
            # Try to slice via dense shards? For now handle
            dense_all=np.load(dense_all_path)[test_idx]
    splade_all=None
    if os.path.exists(splade_all_path):
        arr=np.load(splade_all_path, mmap_mode='r')
        if len(arr)==len(all_ids):
            splade_all=np.array([arr[i] for i in test_idx])
    log.info(f"[{dataset}] dense_test {dense_all.shape if dense_all is not None else None} splade_test {splade_all.shape if splade_all is not None else None}")
    # For each topology, compute partition rankings per K
    results=[]
    # Preload topologies
    topos_data={}
    for topo in TOPOS:
        try:
            hard, mem_idx, npart, doc_nodes_topo, stats = load_partition_topology(dataset, topo)
            topos_data[topo]=(hard, mem_idx, npart, stats)
            log.info(f"[{dataset}/{topo}] npart={npart} mean={stats['mean']:.1f}")
        except Exception as e:
            log.warning(f"[{dataset}/{topo}] missing {e}")
            topos_data[topo]=None
    # For each config, compute coverage
    # Need also hard for N_scope: use hard array
    for K in K_VALUES:
        # Prepare dense order_K and splade order_K for this K
        # order_K = top200[:, :K]
        dense_K = dense_all[:,:K] if dense_all is not None else None
        splade_K = splade_all[:,:K] if splade_all is not None else None
        for topo in TOPOS:
            if topos_data[topo] is None:
                continue
            hard, mem_idx, npart, stats = topos_data[topo]
            # For each router, compute partition ranking
            # Dense router
            routers_to_eval=[]
            if dense_K is not None:
                routers_to_eval.append(("dense", dense_K))
            if splade_K is not None:
                routers_to_eval.append(("splade", splade_K))
            # For dense+splade, need both, will compute fused
            # First compute dense and splade rankings individually
            dense_ranking=None
            splade_ranking=None
            # Helper to compute partition ranking from order_K
            def partition_ranking(order_K):
                nq=len(order_K)
                # S,M via _feats with topn=K
                S=np.zeros((nq, npart), dtype=np.float32)
                M=np.zeros((nq, npart), dtype=np.float32)
                for qi in range(nq):
                    for r, nd in enumerate(order_K[qi][:K]):
                        w=1.0/(K0 + r)
                        if nd<0 or nd>=len(mem_idx):
                            continue
                        for p in mem_idx[int(nd)]:
                            S[qi,p]+=w
                            if w> M[qi,p]:
                                M[qi,p]=w
                # rr
                def rr(score):
                    order=np.argsort(-score, axis=1)
                    rank=np.empty((nq,npart), dtype=np.int32)
                    rows=np.arange(nq)[:,None]
                    rank[rows, order]=np.arange(npart)[None,:]
                    return 1.0/(K0+rank)
                votes=rr(S)+rr(M)
                # ranking per query (compact numpy int32 array, not python list-of-lists; argsort on
                # float64 votes is identical to the original implementation)
                return np.argsort(-votes, axis=1).astype(np.int32)
            if dense_K is not None:
                dense_ranking=partition_ranking(dense_K)
            if splade_K is not None:
                splade_ranking=partition_ranking(splade_K)
            # Now evaluate each router
            eval_routers=[]
            if dense_ranking is not None:
                eval_routers.append(("dense", dense_ranking))
            if splade_ranking is not None:
                eval_routers.append(("splade", splade_ranking))
            if dense_ranking is not None and splade_ranking is not None:
                # Exact RRF fusion, vectorized: score(p) = 1/(K0+pos_dense(p)) + 1/(K0+pos_splade(p)),
                # where pos_*(p) is the rank of partition p in that router's ranking (== enumerate index
                # in the Python rrf_fuse over full permutations). Bit-identical to rrf_fuse (validated on 2wiki).
                _nq = dense_ranking.shape[0]; _rows = np.arange(_nq)[:, None]; _cols = np.arange(npart)[None, :]
                pos_d = np.empty((_nq, npart), dtype=np.int32); pos_d[_rows, dense_ranking] = _cols
                pos_s = np.empty((_nq, npart), dtype=np.int32); pos_s[_rows, splade_ranking] = _cols
                fused_votes = 1.0/(K0 + pos_d) + 1.0/(K0 + pos_s)   # float64, matches rrf_fuse score exactly
                # Exact tie-break parity with Python rrf_fuse: ties broken by dense insertion order
                # (== dense-position ascending == dense_ranking order). Stable sort of fused desc over
                # partitions already in dense_ranking order reproduces sorted(keys, key=-score) exactly.
                _fv_in_dorder = np.take_along_axis(fused_votes, dense_ranking, axis=1)
                _idx2 = np.argsort(-_fv_in_dorder, axis=1, kind="stable")
                fused = np.take_along_axis(dense_ranking, _idx2, axis=1).astype(np.int32)
                eval_routers.append(("dense+splade", fused))
            for router, ranking in eval_routers:
                for P in P_VALUES:
                    # Compute N_scope and coverage per query
                    # ranking is list of partition ids sorted
                    # For each query, selected partitions = ranking[qi][:P]
                    # N_scope = sum of partition sizes for those partitions
                    # Need partition sizes: from hard
                    part_sizes=np.bincount(hard, minlength=npart)  # exact partition sizes (== Counter(hard))
                    # Also need gold partition ids for diagnostic
                    # Compute per query metrics
                    nq=len(ranking)
                    any_cov=0
                    all_cov=0
                    gold_frac_sum=0
                    full_evidence_cov=0  # for evidence datasets
                    scope_sizes=[]
                    hop_any=defaultdict(int); hop_all=defaultdict(int); hop_tot=defaultdict(int)  # per-hop coverage (metaqa)
                    for qi in range(nq):
                        sel=set(int(x) for x in ranking[qi][:P])
                        # N_scope = docs whose hard partition is in the selected top-P partitions
                        sz=int(part_sizes[list(sel)].sum()) if sel else 0
                        scope_sizes.append(sz)
                        gold_idx=test_golds[qi]
                        if not gold_idx:
                            continue
                        # gold doc count
                        n_gold=len(gold_idx)
                        # Deduce if gold retained: check if doc's hard partition in sel OR via onehop? Actually scope is defined as union of docs whose hard partition in sel
                        # So we need to know which docs are in scope: hard in sel
                        # For coverage we check if gold doc's hard partition in sel OR if any of its mem partitions in sel? Wait full scope includes onehop? No, scope is docs whose hard partition in sel, not mem. But ANY_GOLD_COV should be gold doc in scope (hard in sel) OR via onehop? Let's use hard only for scope definition per spec: N_scope = docs whose hard partition in sel. Gold retained if gold hard in sel.
                        # However earlier we used mem for voting but scope is hard union.
                        # For diagnostic, we also compute partition-level recall: gold partition = hard[gold]
                        # For simplicity, retained if hard in sel
                        retained=[1 for g in gold_idx if int(hard[g]) in sel]
                        cnt_ret=sum(retained)
                        gold_frac=cnt_ret/n_gold if n_gold else 0
                        gold_frac_sum+=gold_frac
                        is_any=cnt_ret>0; is_all=cnt_ret==n_gold
                        if is_any:
                            any_cov+=1
                        if is_all:
                            all_cov+=1
                        h=test_hops[qi]
                        if h is not None:
                            hk=str(h); hop_tot[hk]+=1
                            if is_any: hop_any[hk]+=1
                            if is_all: hop_all[hk]+=1
                    nq_eff=nq
                    any_cov_pct=100*any_cov/nq_eff if nq_eff else 0
                    all_cov_pct=100*all_cov/nq_eff if nq_eff else 0
                    gold_frac_pct=100*gold_frac_sum/nq_eff if nq_eff else 0
                    mean_scope=float(np.mean(scope_sizes)) if scope_sizes else 0
                    median_scope=float(np.median(scope_sizes)) if scope_sizes else 0
                    p95_scope=float(np.percentile(scope_sizes,95)) if scope_sizes else 0
                    scope_frac=mean_scope/len(hard) if len(hard) else 0
                    reduction=len(hard)/max(1,mean_scope)
                    # For evidence datasets, full_evidence_cov == all_cov (since golds are evidence)
                    # For answer datasets, all_cov is diagnostic
                    results.append({
                        "dataset":dataset, "K":K, "router":router, "topology":topo, "P":P,
                        "n_test":nq,
                        "ANY_GOLD_COV":round(any_cov_pct,2),
                        "ALL_GOLD_COV":round(all_cov_pct,2),
                        "GOLD_FRAC":round(gold_frac_pct,2),
                        "mean_scope":round(mean_scope,1),
                        "median_scope":round(median_scope,1),
                        "p95_scope":round(p95_scope,1),
                        "scope_frac":round(scope_frac,4),
                        "reduction":round(reduction,2)
                    })
                    if hop_tot:
                        hk_sorted=sorted(hop_tot)
                        results[-1]["ANY_by_hop"]={h:round(100*hop_any[h]/hop_tot[h],2) for h in hk_sorted}
                        results[-1]["ALL_by_hop"]={h:round(100*hop_all[h]/hop_tot[h],2) for h in hk_sorted}
                        results[-1]["N_by_hop"]={h:hop_tot[h] for h in hk_sorted}
    return results

if __name__=="__main__":
    import argparse, time, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p=argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean"])
    p.add_argument("--out", default="results/L1/phase1_test.json")
    args=p.parse_args()
    all_res=[]
    for ds in args.datasets:
        res=evaluate_dataset(ds)
        all_res.extend(res)
        # Save per dataset
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(all_res, open(args.out,"w"), indent=2)
        log.info(f"[{ds}] {len(res)} configs done -> {args.out}")
    # Also save CSV
    import csv
    if all_res:
        _fields=[]
        for r in all_res:
            for k in r:
                if k not in _fields: _fields.append(k)
        with open(args.out.replace(".json",".csv"),"w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_res)
