import modal, os, sys, json, time, pickle, hashlib, numpy as np, torch, faiss

MODAL_VOLUME="crag-data-volume"
REMOTE_ROOT="/root/CRAG"

app = modal.App("crag-abc-full")
volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)

def _modal_base_image(modal):
    return (
        modal.Image.micromamba(python_version="3.11")
        .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
        .apt_install("git", "build-essential", "ninja-build")
        .pip_install("torch==2.2.1", "numpy<2.0")
        .pip_install("torch-geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18", find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
        .pip_install("networkx==3.2.1", "rank_bm25", "spacy", "pyyaml", "pandas", "tqdm", "scipy", "sentence-transformers<3.0", "transformers==4.44.2")
        .pip_install("colbert-ai>=0.2.19", extra_options="--no-deps")
        .pip_install("ragatouille==0.0.9", "langchain<0.2")
        .run_commands("pip uninstall -y faiss-cpu faiss-gpu")
        .pip_install("faiss-gpu-cu12==1.8.0.1")
        .micromamba_install("pymetis=2022.1", "pytorch-cuda=12.1", "cuda-nvcc", "cuda-cudart-dev", channels=["conda-forge", "pytorch", "nvidia"])
        .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
        .run_commands("python -m spacy download en_core_web_sm")
    )

def _modal_image(modal):
    return (
        _modal_base_image(modal)
        .pip_install("xgboost==2.1.1")
        .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs")
        .add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py")
        .add_local_dir("scratchpad", remote_path=f"{REMOTE_ROOT}/scratchpad")
    )

image = _modal_image(modal)

@app.function(image=image, gpu="A10G", volumes={f"{REMOTE_ROOT}/storage": volume}, timeout=86400)
def run_full():
    import os, sys
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.append(REMOTE_ROOT)
    # symlink storage
    import shutil
    for sd in ["data","checkpoints","results"]:
        remote_sd=os.path.join(REMOTE_ROOT,"storage",sd)
        local_sd=os.path.join(REMOTE_ROOT,sd)
        os.makedirs(remote_sd, exist_ok=True)
        if os.path.islink(local_sd):
            continue
        if os.path.exists(local_sd):
            for item in os.listdir(local_sd):
                s,d=os.path.join(local_sd,item), os.path.join(remote_sd,item)
                if not os.path.exists(d):
                    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s,d)
            shutil.rmtree(local_sd) if os.path.isdir(local_sd) else os.remove(local_sd)
        os.symlink(remote_sd, local_sd)
    # now actual logic
    import json, pickle, time, hashlib, numpy as np, torch, faiss, random
    from pathlib import Path
    from collections import Counter
    from src.core.engine import CoreEngine
    from src.experiments.l1_universal_head import _load
    from src.experiments.l1_rerank100 import _feats, _rr
    from src.experiments.l2_seed import _scoped_order, _splade_scoped_order, _recall, MAXK, KS
    from src.experiments.e2e_pipeline import _merge_minrank
    from src.experiments.query_relation import OffsetHead
    from src.experiments.l1_ablate import MixtureHead
    from src.experiments.overlap_retrain import _onehop_membership

    DATASET="2wiki_clean"
    SUBDIR="gte_qwen"
    L1_VOTE_K=100
    P_VALUES=[20,50,100]
    OUTDIR=Path("results/L2/abc_2wiki_full")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} cuda {torch.cuda.is_available()}")
    if device.type=="cuda":
        print(f"GPU {torch.cuda.get_device_name(0)} mem {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # Phase 0 single load
    t0=time.time()
    print("=== PHASE 0 SINGLE LOAD ===")
    data=_load(DATASET, SUBDIR, 8000, 3000, 0)  # 0 uncapped => 1500
    X=data["X"]; n=data["n"]; npart=data["npart"]; hard=data["hard"]; mem_idx=data["mem_idx"]; id2idx=data["id2idx"]
    qte, ste, gte = data["test"]
    texts=data["test_texts"]
    splade=data["splade"]
    # query_ids real
    from src.experiments.overlap_retrain import _splits, _hard_membership
    eng_tmp=CoreEngine(source=DATASET, index_subdir=SUBDIR)
    sp=_splits(eng_tmp, _hard_membership(eng_tmp))
    # data test is sp["test"][:cap] where cap 0 => 1500
    cap=len(sp["test"])
    test_nodes=sp["test"][:cap]
    query_ids=[nd.node_id for nd,_,_ in test_nodes]
    qhash=hashlib.sha256("".join(query_ids).encode()).hexdigest()[:16]
    print(f"query_ids {len(query_ids)} hash {qhash} X {X.shape} qte {qte.shape} npart {npart}")
    assert len(query_ids)==1500, f"expected 1500 got {len(query_ids)}"
    assert len(qte)==1500
    # heads
    print("loading heads")
    sd_hard=torch.load("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt", map_location=device)
    hard_head=OffsetHead(1536).to(device); hard_head.load_state_dict(sd_hard, strict=True); hard_head.eval()
    sd_mix=torch.load("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt", map_location=device)
    mix_head=MixtureHead(1536,K=8).to(device); mix_head.load_state_dict(sd_mix, strict=True); mix_head.eval()
    # pos
    X_t=torch.tensor(X, device=device)
    hard_t=torch.tensor(hard, device=device)
    with torch.no_grad():
        qt=torch.tensor(qte, device=device)
        sv=X_t[torch.tensor(ste, device=device)]
        pos_dense=qt
        pos_hard=hard_head(qt,sv)
        pos_mix=mix_head(qt,sv)
    print(f"pos {pos_dense.shape} {pos_hard.shape} {pos_mix.shape}")
    # SPLADE
    splade_q=None
    if splade:
        from src.experiments.l2_seed import _splade_query_vecs
        splade_q=_splade_query_vecs(splade[0], texts, dataset=DATASET)
        print(f"splade_q {splade_q.shape} doc matrix {splade[1].shape}")
    # FAISS index
    faiss_idx=faiss.IndexFlatIP(1536)
    faiss_idx.add(X)
    t1=time.time()
    print(f"PHASE0 done {t1-t0:.1f}s DATASET_LOADS=1")

    # Phase1 global validation
    print("=== PHASE1 GLOBAL D VALIDATION ===")
    # Load canonical D
    canon_path="results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json"
    canon=json.load(open(canon_path))
    # 2wiki_clean is key? Check
    print(f"canon keys {list(canon.keys())[:5]}")
    # Find 2wiki entry
    # The file has per dataset entries: expect "2wiki_clean"
    canon_entry=canon.get("2wiki_clean") or canon.get("2wiki") or {}
    # It has L2_minrank etc
    # Extract expected R@
    # Try to find L2_minrank
    exp=None
    if "L2_minrank" in canon_entry:
        exp=canon_entry["L2_minrank"]
    elif "2wiki_clean" in canon:
        exp=canon["2wiki_clean"].get("L2_minrank",{})
    print(f"canon 2wiki L2_minrank {exp}")
    # If not found, try top level
    if not exp:
        # Search
        for k,v in canon.items():
            if isinstance(v, dict) and "L2_minrank" in v:
                print(f"found {k} {v['L2_minrank']}")
                if k=="2wiki_clean":
                    exp=v["L2_minrank"]
    # Also check alternative file e2e_full6...
    if not exp or 5 not in exp:
        try:
            alt=json.load(open("results/L2/e2e_full6_universal_gte_qwen.json"))
            print(f"alt keys {list(alt.keys())[:3]}")
            if "2wiki_clean" in alt:
                exp=alt["2wiki_clean"].get("L2_minrank", exp)
                print(f"alt exp {exp}")
        except: pass
    # Fallback to known 71.5
    if not exp:
        exp={2:55,5:71.5,20:78,50:82}
        print(f"using fallback exp {exp}")

    # Compute global D_L2
    nq=len(qte)
    topP_global=[set()]*nq
    sigs_global=[]
    # dense
    od,_=_scoped_order(pos_dense.cpu(), X_t, hard_t, topP_global, False, device, k=500)
    sigs_global.append(od)
    od,_=_scoped_order(pos_hard.cpu(), X_t, hard_t, topP_global, False, device, k=500)
    sigs_global.append(od)
    od,_=_scoped_order(pos_mix.cpu(), X_t, hard_t, topP_global, True, device, k=500)
    sigs_global.append(od)
    if splade:
        od=_splade_scoped_order(splade, texts, hard, topP_global, dataset=DATASET, k=500)
        sigs_global.append(od)
    l2_global=[_merge_minrank([s[qi] for s in sigs_global]) for qi in range(nq)]
    l2_arr=np.full((nq,500), -1, dtype=np.int64)
    for qi,lst in enumerate(l2_global):
        arr=np.array(lst[:500], dtype=np.int64)
        l2_arr[qi,:len(arr)]=arr
    rec_global=_recall(l2_arr, gte)
    print(f"GLOBAL reproduced R@ {rec_global} vs canon {exp}")
    # Compare
    ok=True
    for k in [2,5,20,50]:
        exp_val=exp.get(k) or exp.get(str(k)) or 0
        rec_val=rec_global.get(k,0)
        diff=abs(float(exp_val)-float(rec_val))
        print(f"  K{k} exp {exp_val} rec {rec_val} diff {diff:.2f} {'PASS' if diff<=0.1 else 'FAIL'}")
        if diff>0.1:
            ok=False
    if not ok:
        print("GLOBAL_SCOPE_D_EQUIVALENCE FAIL - aborting")
        # Still persist? but abort
        # Write failure file
        Path("results/L2/abc_2wiki_full/GLOBAL_FAIL.json").write_text(json.dumps({"exp":exp,"rec":rec_global}, indent=2))
        volume.commit()
        return {"status":"GLOBAL_FAIL","exp":exp,"rec":rec_global}
    print("GLOBAL_SCOPE_D_EQUIVALENCE PASS")
    # Persist global
    Path("results/L2/abc_2wiki_full").mkdir(parents=True, exist_ok=True)

    # Phase2 routing once
    print("=== PHASE2 ROUTING ONCE ===")
    _, dense_order100 = faiss_idx.search(qte, L1_VOTE_K)
    # Helper for mem_idx per variant
    def build_mem_idx_variant(partition_map, hard_arr, doc_id_to_idx, variant):
        # Use onehop structural+NER for B/C, structural only for A
        # For A, use canonical mem_idx (already onehop structural)
        # For B/C, build structural+NER
        # To avoid recomputing structural, reuse mem_idx for A, for B/C build via ner
        if variant=="A":
            return mem_idx  # canonical onehop structural
        else:
            # Build structural+NER onehop
            # Need structural adj from engine nodes
            # Use eng_tmp nodes
            N=len(hard_arr)
            # Build structural adj
            structural_adj=[set() for _ in range(N)]
            for i, node in enumerate(eng_tmp.nodes):
                for nid in node.neighbors:
                    j=doc_id_to_idx.get(nid)
                    if j is not None:
                        structural_adj[i].add(j)
            # NER
            import pickle
            ner_adj=pickle.load(open(f"data/ukb_storage/{DATASET}/ner_edges_w_df25.pkl","rb"))
            coo=ner_adj.tocoo()
            ner_sets=[set() for _ in range(N)]
            for r,c in zip(coo.row, coo.col):
                if r!=c:
                    ner_sets[r].add(int(c))
            combined=[structural_adj[i].union(ner_sets[i]) for i in range(N)]
            mem=[]
            for i in range(N):
                s={int(hard_arr[i])}
                for nb in combined[i]:
                    s.add(int(hard_arr[nb]))
                mem.append(sorted(s))
            return mem

    # Build hard and mem per variant
    variants={}
    for variant in ["A","B","C"]:
        if variant=="A":
            pm_path="data/ukb_storage/2wiki_clean/gte_qwen/partition_map.json"
        elif variant=="B":
            pm_path="scratchpad/ablation/2wiki_clean/variant_B/partition_map.json"
        else:
            pm_path="scratchpad/ablation/2wiki_clean/variant_C/partition_map.json"
        pm=json.load(open(pm_path))
        pm={k:int(v) for k,v in pm.items()}
        N=len(id2idx) if 'id2idx' in locals() else len(hard)
        # Build hard_variant via id2idx
        # Use data id2idx
        id2idx_local=data["id2idx"]
        hard_variant=np.full(N, -1, dtype=np.int64)
        for nid,pid in pm.items():
            if nid in id2idx_local:
                hard_variant[id2idx_local[nid]]=pid
        assert np.all(hard_variant>=0)
        mem_variant=build_mem_idx_variant(pm, hard_variant, id2idx_local, variant)
        npart_v=max(hard_variant)+1
        variants[variant]={"hard":hard_variant, "mem":mem_variant, "npart":npart_v, "pm":pm}
        print(f"variant {variant} npart {npart_v} mem mean {sum(len(x) for x in mem_variant)/len(mem_variant):.2f}")

    # Compute partition voting once per variant
    part_rankings={}
    for variant in ["A","B","C"]:
        mem=variants[variant]["mem"]
        npart=variants[variant]["npart"]
        S,M=_feats(dense_order100, mem, npart, topn=200)
        votes=_rr(S)+_rr(M)
        part_rank=np.argsort(-votes, axis=1)
        part_rankings[variant]=part_rank
        print(f"routing {variant} part_rank {part_rank.shape}")
    print("PARTITION_ROUTING_PASSES=3 FAISS_DENSE_SEARCHES=1")

    # Phase3 scoped L2
    print("=== PHASE3 SCOPED L2 ===")
    # For each variant, for each P, score
    # Use full score reuse: score P100 pool fully (not truncated) then filter
    # Implement per variant per expert: score P100 pool fully, then derive
    all_results={}
    for variant in ["A","B","C"]:
        hard_v=variants[variant]["hard"]
        hard_t_v=torch.tensor(hard_v, device=device)
        mem_v=variants[variant]["mem"]
        npart_v=variants[variant]["npart"]
        part_rank=part_rankings[variant]
        # For this variant, precompute topP sets
        topPs={}
        for P in P_VALUES:
            topPs[P]=[set(int(pid) for pid in part_rank[qi,:P]) for qi in range(nq)]
        # For each expert, compute P100 full ordering once, then derive
        # We'll do per expert per query batch
        # Helper to compute full ordering for P100 pool
        def full_order_for_expert(name, pos):
            is_mix=(name=="mlpT")
            # For each query, pool = docs where hard in topP100
            full_orders=[]
            harr=np.array(hard_v)
            for qi in range(nq):
                pool=np.where(np.isin(harr, list(topPs[100][qi])))[0]
                # Score
                if is_mix:
                    p = pos[qi:qi+1].to(device)
                    sim=torch.einsum("bkd,nd->bkn", p, X_t).max(1).values[0]
                    sim_pool=sim[torch.tensor(pool, device=device)]
                    order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                    order=pool[order_idx]
                    full_orders.append(order)
                elif name=="splade":
                    # splade handled separately
                    pass
                else:
                    p = pos[qi].to(device)
                    sim=p @ X_t.T
                    sim_pool=sim[torch.tensor(pool, device=device)]
                    order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                    order=pool[order_idx]
                    full_orders.append(order)
            return full_orders

        # For dense, rel_hard, mlpT
        expert_orders_full={}
        for name, pos in [("dense",pos_dense),("rel_hard",pos_hard),("mlpT",pos_mix)]:
            is_mix=(name=="mlpT")
            full=[]
            harr=np.array(hard_v)
            for qi in range(nq):
                pool=np.where(np.isin(harr, list(topPs[100][qi])))[0]
                if is_mix:
                    p = pos[qi:qi+1].to(device)
                    sim=torch.einsum("bkd,nd->bkn", p, X_t).max(1).values[0]
                    sim_pool=sim[torch.tensor(pool, device=device)]
                    order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                    order=pool[order_idx]
                    full.append(order)
                else:
                    p = pos[qi].to(device)
                    sim=p @ X_t.T
                    sim_pool=sim[torch.tensor(pool, device=device)]
                    order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                    order=pool[order_idx]
                    full.append(order)
            expert_orders_full[name]=full
            print(f"{variant} {name} full P100 computed {len(full)} queries mean pool {np.mean([len(x) for x in full]):.1f}")

        # Splade full
        if splade:
            doc_matrix=splade[1]  # CSR (N, vocab)
            # splade_q is CSR (1500, vocab)
            full_splade=[]
            harr=np.array(hard_v)
            for qi in range(nq):
                pool=np.where(np.isin(harr, list(topPs[100][qi])))[0]
                qvec=splade_q[qi]  # CSR row (1, vocab)
                # Use same logic as l2_seed: doc_matrix.dot(qvec.T) -> (N,1)
                scores=np.asarray(doc_matrix.dot(qvec.T).todense()).ravel()
                pool_scores=scores[pool]
                order_idx=np.argsort(-pool_scores)
                order=pool[order_idx]
                full_splade.append(order)
            expert_orders_full["splade"]=full_splade
            print(f"{variant} splade full P100 computed")

        # Now for each P, derive top500 by filtering full_orders
        variant_results={}
        for P in P_VALUES:
            topP=topPs[P]
            # For each expert, derive top500 for this P by filtering full P100 order
            derived_orders={}
            for name in ["dense","rel_hard","mlpT","splade"]:
                if name not in expert_orders_full:
                    continue
                full=expert_orders_full[name]
                derived=[]
                harr=np.array(hard_v)
                for qi in range(nq):
                    # Filter full order (which is sorted P100) to keep docs where hard in topP[qi]
                    keep=[int(doc) for doc in full[qi] if harr[int(doc)] in topP[qi]]
                    keep=keep[:500]+[-1]*(500-len(keep))
                    derived.append(keep)
                derived=np.array(derived, dtype=np.int64)
                derived_orders[name]=derived
            # D_L2 via merge
            sigs=[derived_orders[k] for k in ["dense","rel_hard","mlpT"] if k in derived_orders]
            if "splade" in derived_orders:
                sigs.append(derived_orders["splade"])
            l2_orders=[_merge_minrank([s[qi] for s in sigs]) for qi in range(nq)]
            l2_arr=np.full((nq,500), -1, dtype=np.int64)
            for qi,lst in enumerate(l2_orders):
                arr=np.array(lst[:500], dtype=np.int64)
                l2_arr[qi,:len(arr)]=arr
            derived_orders["D_L2"]=l2_arr
            # Metrics
            # N_scope
            harr=np.array(hard_v)
            psize=np.bincount(harr, minlength=variants[variant]["npart"])
            sizes=np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else len(harr) for tp in topP], dtype=np.int64)
            mean_scope=float(sizes.mean()); median_scope=float(np.median(sizes)); p95_scope=float(np.percentile(sizes,95)); reduction=len(harr)/max(1,mean_scope)
            # scope oracles
            hit_oracle=[]; gold_recall=[]
            for qi, gg in enumerate(gte):
                if not gg: continue
                in_scope=[1 if hard_v[g] in topP[qi] else 0 for g in gg]
                hit_oracle.append(1.0 if any(in_scope) else 0.0)
                gold_recall.append(sum(in_scope)/len(gg))
            scope_hit=round(100*float(np.mean(hit_oracle)) if hit_oracle else 0,2)
            scope_gold=round(100*float(np.mean(gold_recall)) if gold_recall else 0,2)
            # Expert R@
            expert_rec={}
            for k in ["dense","rel_hard","mlpT","splade"]:
                if k in derived_orders:
                    expert_rec[k]=_recall(derived_orders[k], gte)
            rec_D=_recall(l2_arr, gte)
            # routing/ranking per K
            routing={}; ranking={}; success={}
            for kk in [2,5,20,50]:
                rout=rank=succ=0
                total=0
                for qi, gg in enumerate(gte):
                    if not gg: continue
                    total+=1
                    gs=set(gg)
                    in_scope=any(hard_v[g] in topP[qi] for g in gs)
                    if not in_scope:
                        rout+=1
                        continue
                    row=set(int(x) for x in l2_arr[qi][:kk] if x>=0)
                    if gs & row:
                        succ+=1
                    else:
                        rank+=1
                routing[kk]=round(100*rout/max(1,total),2)
                ranking[kk]=round(100*rank/max(1,total),2)
                success[kk]=round(100*succ/max(1,total),2)
            # best single
            best_k=None; best_val=-1
            for k,v in expert_rec.items():
                if v[5]>best_val:
                    best_val=v[5]; best_k=k
            # coop
            coop={}
            for kk in [2,5,20,50]:
                hits=0; total=0
                for qi, gg in enumerate(gte):
                    if not gg: continue
                    total+=1
                    gs=set(gg)
                    hit_any=any(gs & set(int(x) for x in derived_orders[exp][qi][:kk] if x>=0) for exp in expert_rec.keys())
                    if hit_any:
                        hits+=1
                coop[kk]=round(100*hits/max(1,total),2)
            print(f"{variant} P{P} mean {mean_scope:.1f} hit {scope_hit} gold {scope_gold} D_R5 {rec_D[5]} rout {routing[5]} rank {ranking[5]}")
            # Persist
            out_npz=Path(f"results/L2/abc_2wiki_full/rankings_{DATASET}_{variant}_P{P}_gte_qwen.npz")
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_npz),
                schema_version=np.array("1.0"),
                dataset=np.array(DATASET),
                query_ids=np.array(query_ids, dtype=object),
                gold_ids=np.array([[int(x) for x in g] for g in gte], dtype=object),
                dense_order=derived_orders["dense"].astype(np.int32),
                rel_hard_order=derived_orders["rel_hard"].astype(np.int32),
                mlpT_order=derived_orders["mlpT"].astype(np.int32),
                splade_order=derived_orders.get("splade", np.full((nq,500), -1, dtype=np.int32)).astype(np.int32),
                D_L2_order=l2_arr.astype(np.int32),
                partition_ranking=part_rank.astype(np.int32),
                topP=np.array([np.array(list(tp), dtype=np.int32) for tp in topP], dtype=object),
                N_scope=sizes.astype(np.int32),
                hard_variant=hard_v.astype(np.int32),
            )
            variant_results[P]={
                "mean_scope":round(mean_scope,1),"median_scope":round(median_scope,1),"p95_scope":round(p95_scope,1),"reduction":round(reduction,2),
                "scope_hit":scope_hit,"scope_gold":scope_gold,
                "rec_D":rec_D,"expert_rec":expert_rec,"best_single":best_k,"best_R5":best_val,"coop":coop,
                "routing":routing,"ranking":ranking,"success":success, "nq":nq
            }
        all_results[variant]=variant_results
        # Persist per variant summary
        Path(f"results/L2/abc_2wiki_full/summary_{variant}.json").write_text(json.dumps(variant_results, indent=2))

    # Final tables
    print("=== FINAL TABLE ===")
    for variant in ["A","B","C"]:
        for P in P_VALUES:
            r=all_results[variant][P]
            print(f"{variant} P{P} mean {r['mean_scope']} hit {r['scope_hit']} gold {r['scope_gold']} R5 {r['rec_D'][5]} rout {r['routing'][5]}")

    # Compute best
    best=None; best_val=-1
    for variant in ["A","B","C"]:
        for P in P_VALUES:
            v=all_results[variant][P]["rec_D"][5]
            if v>best_val:
                best_val=v; best=(variant,P)
    print(f"BEST {best} {best_val}")

    # Save overall
    Path("results/L2/abc_2wiki_full/overall.json").write_text(json.dumps(all_results, indent=2))
    volume.commit()
    return all_results

@app.local_entrypoint()
def main():
    run_full.remote()

if __name__=="__main__":
    main()
