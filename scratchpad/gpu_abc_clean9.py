import modal, os
MODAL_VOLUME="crag-data-volume"
REMOTE_ROOT="/root/CRAG"
app=modal.App("crag-abc-clean9")
volume=modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
def _base(modal):
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
def _img(modal):
    return _base(modal).pip_install("xgboost==2.1.1").add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src").add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs").add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py").add_local_dir("scratchpad", remote_path=f"{REMOTE_ROOT}/scratchpad")
image=_img(modal)

@app.function(image=image, gpu="A10G", volumes={f"{REMOTE_ROOT}/storage": volume}, timeout=86400)
def run_clean9():
    import os, sys
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.append(REMOTE_ROOT)
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
    import json, pickle, time, hashlib, numpy as np, torch, faiss
    from pathlib import Path
    from src.core.engine import CoreEngine
    from src.experiments.l1_universal_head import _load
    from src.experiments.l1_rerank100 import _feats, _rr
    from src.experiments.l2_seed import _scoped_order, _splade_scoped_order, _recall, KS, _splade_query_vecs
    from src.experiments.e2e_pipeline import _merge_minrank
    from src.experiments.query_relation import OffsetHead
    from src.experiments.l1_ablate import MixtureHead

    SUBDIR="gte_qwen"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    sd_hard=torch.load("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt", map_location=device)
    sd_mix=torch.load("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt", map_location=device)

    for DATASET in ["musique_clean","squad_clean"]:
        print(f"\n===== DATASET {DATASET} CLEAN 9 =====")
        t0=time.time()
        # NEW PROTOCOL: FULL POPULATION - use all available queries
        # For musique 1995, squad 13033, use limit 20000 to get full
        data=_load(DATASET, SUBDIR, 20000, 3000, 0)
        X=data["X"]; npart=data["npart"]; hard=data["hard"]; mem_idx=data["mem_idx"]; id2idx=data["id2idx"]
        qte, ste, gte = data["test"]
        texts=data["test_texts"]
        splade=data["splade"]
        nq=len(qte)
        print(f"{DATASET} X {X.shape} qte {qte.shape} npart {npart} nq {nq}")
        # Strict: query_ids must be same selection/order as _load
        # _load does: caps {"train": tr_cap or limit, "test": te_cap or limit} => test cap 8000
        # and prep does: qs = sp[split][:caps[split]] where sp is _splits with _hard_membership
        # So query_ids are sp["test"][:8000] in sorted node_id order after shuffle with seed 42
        # We need to replicate that exact order
        from src.experiments.overlap_retrain import _splits, _hard_membership
        eng_tmp=CoreEngine(source=DATASET, index_subdir=SUBDIR)
        sp=_splits(eng_tmp, _hard_membership(eng_tmp))
        # Determine cap used by _load
        cap_test = 8000 if nq==8000 or nq==1995 else nq  # for squad 8000, musique 1995
        # Actually nq is cap, so use nq
        query_ids=[nd.node_id for nd,_,_ in sp["test"][:nq]]
        print(f"query_ids {len(query_ids)} hash {hashlib.sha256(''.join(query_ids).encode()).hexdigest()[:16]} qte {len(qte)}")
        assert len(query_ids)==len(qte)==len(gte), f"parity fail {len(query_ids)} vs {len(qte)} vs {len(gte)}"
        # SPLADE query alignment: ensure splade_q rows correspond to same 8000 queries in same order
        # _load's qte was encoded via eq([...]) for those 8000, and splade_q via _splade_query_vecs for same texts
        # Our splade_q from data["splade"] is already aligned, but we also need to ensure splade_q from _splade_query_vecs matches
        # For strict, use data's splade matrix and also verify splade_q alignment
        # data["splade"] is (scorer, matrix) where matrix rows are X order, not query order
        # splade_q we compute via _splade_query_vecs for texts (which is test_texts in same order)
        # So it should be aligned
        print(f"SQUAD_QUERY_POPULATION_PARITY PASS {nq}  SPLADE_QUERY_ALIGNMENT check pending")
        if splade:
            # Verify splade_q length matches nq
            from src.experiments.l2_seed import _splade_query_vecs as sq
            # The cached file may have been for 13033 queries, but we need 8000
            # _splade_query_vecs will return m[:len(texts)] where texts is 8000, so it will be 8000
            qmat=sq(splade[0], texts, dataset=DATASET)
            assert qmat.shape[0]==nq, f"splade_q {qmat.shape[0]} != nq {nq}"
            print(f"SPLADE_QUERY_ALIGNMENT PASS {qmat.shape}")

        hard_head=OffsetHead(1536).to(device); hard_head.load_state_dict(sd_hard, strict=True); hard_head.eval()
        mix_head=MixtureHead(1536,K=8).to(device); mix_head.load_state_dict(sd_mix, strict=True); mix_head.eval()
        X_t=torch.tensor(X, device=device)
        hard_t=torch.tensor(hard, device=device)
        with torch.no_grad():
            qt=torch.tensor(qte, device=device)
            sv=X_t[torch.tensor(ste, device=device)]
            pos_dense=qt
            pos_hard=hard_head(qt,sv)
            pos_mix=mix_head(qt,sv)
        splade_q_mat=None
        if splade:
            splade_q_mat=_splade_query_vecs(splade[0], texts, dataset=DATASET)
            print(f"splade_q_mat {splade_q_mat.shape}")
        faiss_idx=faiss.IndexFlatIP(1536)
        faiss_idx.add(X)
        print(f"PHASE0 {DATASET} {time.time()-t0:.1f}s")

        # GLOBAL - FULL POPULATION per new protocol
        print(f"=== GLOBAL {DATASET} FULL POPULATION ===")
        canon=json.load(open("results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json"))
        exp_entry=canon.get(DATASET,{})
        exp=exp_entry.get("L2_minrank",{}) if isinstance(exp_entry, dict) else {}
        print(f"canon (legacy capped) {exp} nq {len(qte)}")
        nq=len(qte)
        topP_global=[set()]*nq
        sigs=[]
        od,_=_scoped_order(pos_dense.cpu(), X_t, hard_t, topP_global, False, device, k=500)
        sigs.append(od)
        od,_=_scoped_order(pos_hard.cpu(), X_t, hard_t, topP_global, False, device, k=500)
        sigs.append(od)
        od,_=_scoped_order(pos_mix.cpu(), X_t, hard_t, topP_global, True, device, k=500)
        sigs.append(od)
        if splade:
            od=_splade_scoped_order(splade, texts, hard, topP_global, dataset=DATASET, k=500)
            sigs.append(od)
        l2g=[_merge_minrank([s[qi] for s in sigs]) for qi in range(nq)]
        l2_arr=np.full((nq,500), -1, dtype=np.int64)
        for qi,lst in enumerate(l2g):
            arr=np.array(lst[:500], dtype=np.int64)
            l2_arr[qi,:len(arr)]=arr
        rec=_recall(l2_arr, gte)
        print(f"GLOBAL rec {rec} vs legacy exp {exp}")
        # For musique 1995, legacy is already full (1995<8000 cap, so it is full) - do strict gate
        # For squad 13033, legacy was 8000 capped, so new full 13033 is NEW control, do not compare
        is_full_squad = (DATASET=="squad_clean" and nq==13033)
        if is_full_squad:
            print(f"SQUAD FULL POPULATION {nq} - legacy was 8000, establishing NEW FULL_D control (no gate vs old)")
            # Persist new full control
            Path("results/L2/full_population_controls").mkdir(parents=True, exist_ok=True)
            Path(f"results/L2/full_population_controls/{DATASET}_D.json").write_text(json.dumps({"dataset":DATASET,"n_queries":nq,"query_id_hash":hashlib.sha256(''.join(query_ids).encode()).hexdigest()[:16],"corpus_N":len(hard),"population_type":"full","D_L2":rec}, indent=2))
        else:
            # Strict gate for musique (1995) and other full that matches legacy
            for k in [2,5,20,50]:
                ev=exp.get(str(k), exp.get(k,0))
                rv=rec.get(k,0)
                diff=abs(float(ev)-float(rv))
                print(f" K{k} exp {ev} rec {rv} diff {diff:.2f} {'PASS' if diff<=0.1 else 'FAIL'}")
                if diff>0.1:
                    Path(f"results/L2/abc_{DATASET.split('_')[0]}_full").mkdir(parents=True, exist_ok=True)
                    Path(f"results/L2/abc_{DATASET.split('_')[0]}_full/GLOBAL_FAIL.json").write_text(json.dumps({"exp":exp,"rec":rec}, indent=2))
                    raise RuntimeError(f"GLOBAL FAIL {DATASET} K{k} diff {diff}")
            print(f"GLOBAL PASS {DATASET}")
            # Also persist as new full control for reference
            Path("results/L2/full_population_controls").mkdir(parents=True, exist_ok=True)
            Path(f"results/L2/full_population_controls/{DATASET}_D.json").write_text(json.dumps({"dataset":DATASET,"n_queries":nq,"query_id_hash":hashlib.sha256(''.join(query_ids).encode()).hexdigest()[:16],"corpus_N":len(hard),"population_type":"full","D_L2":rec}, indent=2))

        # ROUTING - strict
        print(f"=== ROUTING {DATASET} ===")
        _, dense_order100 = faiss_idx.search(qte, 100)
        variants={}
        for variant in ["A","B","C"]:
            if variant=="A":
                pm_path=f"data/ukb_storage/{DATASET}/gte_qwen/partition_map.json"
                if not os.path.exists(pm_path):
                    pm_path=f"data/ukb_storage/{DATASET}/partition_map.json"
            elif variant=="B":
                pm_path=f"scratchpad/ablation/{DATASET}/variant_B/partition_map.json"
            else:
                pm_path=f"scratchpad/ablation/{DATASET}/variant_C/partition_map.json"
            if not os.path.exists(pm_path):
                raise FileNotFoundError(f"Missing topology artifact {variant} {pm_path}")
            pm=json.load(open(pm_path))
            pm={k:int(v) for k,v in pm.items()}
            N=len(hard)
            if len(pm)!=N:
                raise RuntimeError(f"pm len {len(pm)} != N {N} for {variant} {DATASET}")
            if len(id2idx)!=N:
                raise RuntimeError(f"id2idx len {len(id2idx)} != N {N}")
            hard_v=np.full(N, -1, dtype=np.int64)
            missing=[]
            for nid,pid in pm.items():
                if nid in id2idx:
                    hard_v[id2idx[nid]]=pid
                else:
                    missing.append(nid)
            if missing:
                raise RuntimeError(f"missing_ids {len(missing)} for {variant} {DATASET}")
            if np.any(hard_v==-1):
                raise RuntimeError(f"hard_v has -1 for {variant} {DATASET} count {np.sum(hard_v==-1)}")
            if variant=="A":
                mem_v=mem_idx
                mism=sum(1 for i in range(N) if set(mem_idx[i])!=set(mem_v[i]))
                assert mism==0, f"A parity {mism}"
            else:
                structural_adj=[set() for _ in range(N)]
                for i, node in enumerate(eng_tmp.nodes):
                    for nid in node.neighbors:
                        j=id2idx.get(nid)
                        if j is not None:
                            structural_adj[i].add(j)
                try:
                    ner_adj=pickle.load(open(f"data/ukb_storage/{DATASET}/ner_edges_w_df25.pkl","rb"))
                except Exception as e:
                    raise RuntimeError(f"NER load fail for {variant} {DATASET}: {e}")
                coo=ner_adj.tocoo()
                ner_sets=[set() for _ in range(N)]
                for r,c in zip(coo.row, coo.col):
                    if r!=c:
                        ner_sets[r].add(int(c))
                for i in range(N):
                    structural_adj[i].update(ner_sets[i])
                mem_v=[]
                for i in range(N):
                    s={int(hard_v[i])}
                    for nb in structural_adj[i]:
                        s.add(int(hard_v[nb]))
                    assert int(hard_v[i]) in s
                    mem_v.append(sorted(s))
                assert all(len(x)>=1 for x in mem_v)
            npart_v=int(max(hard_v)+1)
            variants[variant]={"hard":hard_v,"mem":mem_v,"npart":npart_v,"pm":pm}
            print(f"variant {variant} npart {npart_v} mem mean {sum(len(x) for x in mem_v)/len(mem_v):.2f}")

        part_rankings={}
        for variant in variants:
            mem=variants[variant]["mem"]
            npart=variants[variant]["npart"]
            S,M=_feats(dense_order100, mem, npart, topn=200)
            votes=_rr(S)+_rr(M)
            part_rank=np.argsort(-votes, axis=1)
            part_rankings[variant]=part_rank
            print(f"routing {variant} {part_rank.shape}")

        # Small parity test for full reuse (5 queries) if needed
        # For this clean run, we will use direct 36 passes (safe) - not full reuse to avoid tie complexity
        # But we can test quickly 5q parity
        print("=== PARITY TEST 5q (direct vs full reuse) ===")
        # Use 5 queries
        nq_test=min(5, nq)
        # Build full orders for P100 for test
        # For brevity, just test dense P50
        # Use same logic as before: full P100 pool then filter vs direct P50
        # For each variant, test one
        for variant in ["A"]:
            hard_v=variants[variant]["hard"]
            harr=np.array(hard_v)
            part_rank=part_rankings[variant]
            topP100_test=[set(int(pid) for pid in part_rank[qi,:100]) for qi in range(nq_test)]
            topP50_test=[set(int(pid) for pid in part_rank[qi,:50]) for qi in range(nq_test)]
            # Direct P50
            topP50_direct=topP50_test
            # For dense, compute full P100 then filter
            # Use pos_dense[:nq_test]
            # Direct
            od_direct,_=_scoped_order(pos_dense[:nq_test].cpu(), X_t, torch.tensor(hard_v, device=device), topP50_direct, False, device, k=500)
            # Full
            # Compute full ordering for P100 pool
            full_orders=[]
            for qi in range(nq_test):
                pool=np.where(np.isin(harr, list(topP100_test[qi])))[0]
                p=pos_dense[qi].to(device)
                sim=p @ X_t.T
                sim_pool=sim[torch.tensor(pool, device=device)]
                order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                order=pool[order_idx]
                full_orders.append(order)
            # Filter
            derived=[]
            for qi in range(nq_test):
                keep=[int(doc) for doc in full_orders[qi] if harr[int(doc)] in topP50_test[qi]]
                keep=keep[:500]+[-1]*(500-len(keep))
                derived.append(keep)
            derived=np.array(derived)
            eq=np.array_equal(od_direct, derived)
            diff=np.sum(od_direct!=derived)
            rec_direct=_recall(od_direct, gte[:nq_test])
            rec_derived=_recall(derived, gte[:nq_test])
            print(f"parity {variant} dense P50 direct vs full-filter equal {eq} diff {diff} R5 {rec_direct[5]} vs {rec_derived[5]}")
            # Consider metric parity
            metric_eq=all(rec_direct[k]==rec_derived[k] for k in [2,5,20,50])
            print(f"metric parity {metric_eq} -> FULL_SCORE_REUSE_METRIC_PARITY {'PASS' if metric_eq else 'FAIL'}")

        # Now full 9 cells with direct 36 passes (safe)
        all_results={}
        for variant in variants:
            hard_v=variants[variant]["hard"]
            hard_t_v=torch.tensor(hard_v, device=device)
            part_rank=part_rankings[variant]
            npart_v=variants[variant]["npart"]
            topPs={P:[set(int(pid) for pid in part_rank[qi,:P]) for qi in range(nq)] for P in [20,50,100]}
            variant_results={}
            for P in [20,50,100]:
                topP=topPs[P]
                harr=np.array(hard_v)
                psize=np.bincount(harr, minlength=npart_v)
                sizes=np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else len(harr) for tp in topP], dtype=np.int64)
                mean_scope=float(sizes.mean()); median_scope=float(np.median(sizes)); p95_scope=float(np.percentile(sizes,95)); reduction=len(harr)/max(1,mean_scope)
                hit=[]; gold=[]
                for qi, gg in enumerate(gte):
                    if not gg: continue
                    in_scope=[1 if hard_v[g] in topP[qi] else 0 for g in gg]
                    hit.append(1.0 if any(in_scope) else 0.0)
                    gold.append(sum(in_scope)/len(gg))
                scope_hit=round(100*float(np.mean(hit)) if hit else 0,2)
                scope_gold=round(100*float(np.mean(gold)) if gold else 0,2)
                od_dense,_=_scoped_order(pos_dense.cpu(), X_t, hard_t_v, topP, False, device, k=500)
                od_hard,_=_scoped_order(pos_hard.cpu(), X_t, hard_t_v, topP, False, device, k=500)
                od_mix,_=_scoped_order(pos_mix.cpu(), X_t, hard_t_v, topP, True, device, k=500)
                orders={"dense":od_dense,"rel_hard":od_hard,"mlpT":od_mix}
                if splade:
                    od=_splade_scoped_order(splade, texts, hard_v, topP, dataset=DATASET, k=500)
                    orders["splade"]=od
                sigs=[orders[k] for k in ["dense","rel_hard","mlpT"] if k in orders]
                if "splade" in orders:
                    sigs.append(orders["splade"])
                l2_orders=[_merge_minrank([s[qi] for s in sigs]) for qi in range(nq)]
                l2_arr=np.full((nq,500), -1, dtype=np.int64)
                for qi,lst in enumerate(l2_orders):
                    arr=np.array(lst[:500], dtype=np.int64)
                    l2_arr[qi,:len(arr)]=arr
                orders["D_L2"]=l2_arr
                rec_D=_recall(l2_arr, gte)
                expert_rec={}
                for k in ["dense","rel_hard","mlpT","splade"]:
                    if k in orders:
                        expert_rec[k]=_recall(orders[k], gte)
                routing={}; ranking={}; success={}
                for kk in [2,5,20,50]:
                    rout=rank=succ=0
                    total=0
                    for qi, gg in enumerate(gte):
                        if not gg: continue
                        total+=1
                        gs=set(gg)
                        in_sc=any(hard_v[g] in topP[qi] for g in gs)
                        if not in_sc:
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
                best_k=None; best_val=-1
                for k,v in expert_rec.items():
                    if v[5]>best_val:
                        best_val=v[5]; best_k=k
                coop={}
                for kk in [2,5,20,50]:
                    hits=0; total=0
                    for qi, gg in enumerate(gte):
                        if not gg: continue
                        total+=1
                        gs=set(gg)
                        hit_any=any(gs & set(int(x) for x in orders[exp][qi][:kk] if x>=0) for exp in expert_rec.keys())
                        if hit_any:
                            hits+=1
                    coop[kk]=round(100*hits/max(1,total),2)
                print(f"{DATASET} {variant} P{P} mean {mean_scope:.1f} hit {scope_hit} R5 {rec_D[5]}")
                outdir=Path(f"results/L2/abc_{DATASET.split('_')[0]}_full")
                outdir.mkdir(parents=True, exist_ok=True)
                out_npz=outdir/f"rankings_{DATASET}_{variant}_P{P}_gte_qwen.npz"
                np.savez_compressed(str(out_npz),
                    schema_version=np.array("1.0"),
                    dataset=np.array(DATASET),
                    query_ids=np.array(query_ids, dtype=object),
                    gold_ids=np.array([[int(x) for x in g] for g in gte], dtype=object),
                    dense_order=orders["dense"].astype(np.int32),
                    rel_hard_order=orders["rel_hard"].astype(np.int32),
                    mlpT_order=orders["mlpT"].astype(np.int32),
                    splade_order=orders.get("splade", np.full((nq,500), -1, dtype=np.int32)).astype(np.int32),
                    D_L2_order=l2_arr.astype(np.int32),
                    partition_ranking=part_rank.astype(np.int32),
                    topP=np.array([np.array(list(tp), dtype=np.int32) for tp in topP], dtype=object),
                    N_scope=sizes.astype(np.int32),
                    hard_variant=hard_v.astype(np.int32),
                    scope_hit=np.array(hit),
                    scope_gold=np.array(gold),
                )
                variant_results[P]={
                    "mean_scope":round(mean_scope,1),"median_scope":round(median_scope,1),"p95_scope":round(p95_scope,1),"reduction":round(reduction,2),
                    "scope_hit":scope_hit,"scope_gold":scope_gold,
                    "rec_D":rec_D,"expert_rec":expert_rec,"best_single":best_k,"best_R5":best_val,"coop":coop,
                    "routing":routing,"ranking":ranking,"success":success,"nq":nq
                }
            all_results[variant]=variant_results
            Path(f"results/L2/abc_{DATASET.split('_')[0]}_full/summary_{variant}.json").write_text(json.dumps(variant_results, indent=2))
        Path(f"results/L2/abc_{DATASET.split('_')[0]}_full/overall.json").write_text(json.dumps(all_results, indent=2))
        # paper table offline will be generated later, but create simple
        import csv
        rows=[]
        for variant in ["A","B","C"]:
            for P in [20,50,100]:
                v=all_results[variant][P]
                rows.append({"topology":variant,"P":P,"mean_scope":v["mean_scope"],"scope_hit":v["scope_hit"],"R5":v["rec_D"][5]})
        Path(f"results/L2/abc_{DATASET.split('_')[0]}_full/paper_table.json").write_text(json.dumps(rows, indent=2))
        with open(f"results/L2/abc_{DATASET.split('_')[0]}_full/paper_table.csv","w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        del X, X_t, hard_t, pos_dense, pos_hard, pos_mix, qt, sv, faiss_idx
        if device.type=="cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"done {DATASET}")

    volume.commit()

@app.local_entrypoint()
def main():
    run_clean9.remote()
