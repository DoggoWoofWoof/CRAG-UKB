import modal, os

MODAL_VOLUME="crag-data-volume"
REMOTE_ROOT="/root/CRAG"
app = modal.App("crag-abc-ms")
volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)

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
def run_ms():
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
    from src.experiments.l2_seed import _scoped_order, _splade_scoped_order, _recall, KS
    from src.experiments.e2e_pipeline import _merge_minrank
    from src.experiments.query_relation import OffsetHead
    from src.experiments.l1_ablate import MixtureHead
    from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership

    DATASETS=["musique_clean","squad_clean"]
    SUBDIR="gte_qwen"
    P_ALL=[20,50,100]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    # Load heads once
    sd_hard=torch.load("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt", map_location=device)
    sd_mix=torch.load("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt", map_location=device)
    # Will instantiate per dataset dim (same 1536)
    overall_all={}
    for DATASET in DATASETS:
        print(f"\n===== DATASET {DATASET} =====")
        t0=time.time()
        data=_load(DATASET, SUBDIR, 8000, 3000, 0)
        X=data["X"]; npart=data["npart"]; hard=data["hard"]; mem_idx=data["mem_idx"]; id2idx=data["id2idx"]
        qte, ste, gte = data["test"]
        texts=data["test_texts"]
        splade=data["splade"]
        nq=len(qte)
        print(f"{DATASET} X {X.shape} qte {qte.shape} npart {npart} nq {nq} mem mean {sum(len(x) for x in mem_idx)/len(mem_idx):.2f}")
        # query_ids
        eng_tmp=CoreEngine(source=DATASET, index_subdir=SUBDIR)
        sp=_splits(eng_tmp, _hard_membership(eng_tmp))
        cap=len(sp["test"])
        query_ids=[nd.node_id for nd,_,_ in sp["test"][:cap]]
        qhash=hashlib.sha256("".join(query_ids).encode()).hexdigest()[:16]
        print(f"query_ids {len(query_ids)} hash {qhash}")
        # heads
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
        # SPLADE
        splade_q=None
        if splade:
            from src.experiments.l2_seed import _splade_query_vecs
            splade_q=_splade_query_vecs(splade[0], texts, dataset=DATASET)
            print(f"splade_q {splade_q.shape if hasattr(splade_q,'shape') else type(splade_q)}")
        faiss_idx=faiss.IndexFlatIP(1536)
        faiss_idx.add(X)
        t1=time.time()
        print(f"PHASE0 {DATASET} {t1-t0:.1f}s")

        # GLOBAL validation
        print(f"=== GLOBAL {DATASET} ===")
        canon=json.load(open("results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json"))
        # Find dataset entry; keys are like musique_clean etc
        exp_entry=canon.get(DATASET, {})
        # It may be under different name? For musique, check
        if not exp_entry:
            # Search case-insensitive
            for k,v in canon.items():
                if DATASET.split("_")[0] in k:
                    exp_entry=v
                    break
        exp=exp_entry.get("L2_minrank") if isinstance(exp_entry, dict) else {}
        print(f"canon {DATASET} {exp}")
        if not exp:
            exp={2:50,5:70,20:80,50:85}
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
        print(f"GLOBAL rec {rec} vs exp {exp}")
        ok=True
        for k in [2,5,20,50]:
            ev=exp.get(str(k)) if isinstance(exp, dict) else exp.get(k,0)
            if isinstance(ev, dict):
                ev=0
            # exp may have string keys
            try:
                ev=float(exp.get(str(k), exp.get(k,0)))
            except: ev=0
            rv=rec.get(k,0)
            diff=abs(ev-rv)
            print(f" K{k} exp {ev} rec {rv} diff {diff:.2f} {'PASS' if diff<=0.1 else 'FAIL'}")
            if diff>0.1:
                ok=False
        if not ok:
            print(f"GLOBAL FAIL {DATASET} abort")
            Path(f"results/L2/abc_{DATASET.split('_')[0]}/GLOBAL_FAIL.json").write_text(json.dumps({"exp":exp,"rec":rec}, indent=2))
            volume.commit()
            continue
        print(f"GLOBAL PASS {DATASET}")

        # Routing
        print(f"=== ROUTING {DATASET} ===")
        _, dense_order100 = faiss_idx.search(qte, 100)
        # Build hard/mem per variant
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
                print(f"variant {variant} missing {pm_path} skip")
                continue
            pm=json.load(open(pm_path))
            pm={k:int(v) for k,v in pm.items()}
            N=len(hard)
            hard_v=np.full(N, -1, dtype=np.int64)
            for nid,pid in pm.items():
                if nid in id2idx:
                    hard_v[id2idx[nid]]=pid
            # If any -1, fallback to original hard
            if np.any(hard_v==-1):
                print(f"variant {variant} hard missing {np.sum(hard_v==-1)} fallback")
                hard_v=np.where(hard_v==-1, hard, hard_v)
            # mem
            if variant=="A":
                mem_v=mem_idx
            else:
                # structural+NER onehop
                # Build structural adj
                structural_adj=[set() for _ in range(N)]
                for i, node in enumerate(eng_tmp.nodes):
                    for nid in node.neighbors:
                        j=id2idx.get(nid)
                        if j is not None:
                            structural_adj[i].add(j)
                # NER
                try:
                    ner_adj=pickle.load(open(f"data/ukb_storage/{DATASET}/ner_edges_w_df25.pkl","rb"))
                    coo=ner_adj.tocoo()
                    ner_sets=[set() for _ in range(N)]
                    for r,c in zip(coo.row, coo.col):
                        if r!=c:
                            ner_sets[r].add(int(c))
                    for i in range(N):
                        structural_adj[i].update(ner_sets[i])
                except Exception as e:
                    print(f"ner load fail {e}")
                mem_v=[]
                for i in range(N):
                    s={int(hard_v[i])}
                    for nb in structural_adj[i]:
                        s.add(int(hard_v[nb]))
                    mem_v.append(sorted(s))
            npart_v=max(hard_v)+1
            variants[variant]={"hard":hard_v,"mem":mem_v,"npart":npart_v,"pm":pm}
            print(f"variant {variant} npart {npart_v} mem mean {sum(len(x) for x in mem_v)/len(mem_v):.2f}")

        # Part routing
        part_rankings={}
        for variant in variants:
            mem=variants[variant]["mem"]
            npart=variants[variant]["npart"]
            S,M=_feats(dense_order100, mem, npart, topn=200)
            votes=_rr(S)+_rr(M)
            part_rank=np.argsort(-votes, axis=1)
            part_rankings[variant]=part_rank
            print(f"routing {variant} {part_rank.shape}")

        # Scoped L2 adaptive: A P20/50/100, B/C P50 only, expand if needed
        # Determine which Ps to run
        plan={}
        for variant in variants:
            if variant=="A":
                plan[variant]=[20,50,100]
            else:
                plan[variant]=[50]
        all_results={}
        for variant in variants:
            hard_v=variants[variant]["hard"]
            hard_t_v=torch.tensor(hard_v, device=device)
            mem_v=variants[variant]["mem"]
            npart_v=variants[variant]["npart"]
            part_rank=part_rankings[variant]
            # Precompute topP sets for all P
            topPs={}
            for P in [20,50,100]:
                topPs[P]=[set(int(pid) for pid in part_rank[qi,:P]) for qi in range(nq)]
            # For each P in plan, score
            variant_results={}
            for P in plan[variant]:
                topP=topPs[P]
                # N_scope
                harr=np.array(hard_v)
                psize=np.bincount(harr, minlength=npart_v)
                sizes=np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else len(harr) for tp in topP], dtype=np.int64)
                mean_scope=float(sizes.mean()); median_scope=float(np.median(sizes)); p95_scope=float(np.percentile(sizes,95)); reduction=len(harr)/max(1,mean_scope)
                # scope oracles
                hit=[]; gold=[]
                for qi, gg in enumerate(gte):
                    if not gg: continue
                    in_scope=[1 if hard_v[g] in topP[qi] else 0 for g in gg]
                    hit.append(1.0 if any(in_scope) else 0.0)
                    gold.append(sum(in_scope)/len(gg))
                scope_hit=round(100*float(np.mean(hit)) if hit else 0,2)
                scope_gold=round(100*float(np.mean(gold)) if gold else 0,2)
                # Scoped orders
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
                # expert rec
                expert_rec={}
                for k in ["dense","rel_hard","mlpT","splade"]:
                    if k in orders:
                        expert_rec[k]=_recall(orders[k], gte)
                # routing/ranking
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
                        hit_any=any(gs & set(int(x) for x in orders[exp][qi][:kk] if x>=0) for exp in expert_rec.keys())
                        if hit_any:
                            hits+=1
                    coop[kk]=round(100*hits/max(1,total),2)
                print(f"{DATASET} {variant} P{P} mean {mean_scope:.1f} hit {scope_hit} R5 {rec_D[5]} rout {routing[5]}")
                # Persist
                outdir=Path(f"results/L2/abc_{DATASET.split('_')[0]}")
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
                )
                variant_results[P]={
                    "mean_scope":round(mean_scope,1),"median_scope":round(median_scope,1),"p95_scope":round(p95_scope,1),"reduction":round(reduction,2),
                    "scope_hit":scope_hit,"scope_gold":scope_gold,
                    "rec_D":rec_D,"expert_rec":expert_rec,"best_single":best_k,"best_R5":best_val,"coop":coop,
                    "routing":routing,"ranking":ranking,"success":success,"nq":nq
                }
            all_results[variant]=variant_results
            # Save per variant summary
            Path(f"results/L2/abc_{DATASET.split('_')[0]}/summary_{variant}.json").write_text(json.dumps(variant_results, indent=2))
        # Overall per dataset
        Path(f"results/L2/abc_{DATASET.split('_')[0]}/overall.json").write_text(json.dumps(all_results, indent=2))
        overall_all[DATASET]=all_results
        # Release
        del X, X_t, hard_t, pos_dense, pos_hard, pos_mix, qt, sv
        if device.type=="cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"done {DATASET}")

    # Save combined
    Path("results/L2/abc_musique_squad_combined.json").write_text(json.dumps(overall_all, indent=2))
    volume.commit()
    return overall_all

@app.local_entrypoint()
def main():
    run_ms.remote()
