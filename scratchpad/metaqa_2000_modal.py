import modal
app = modal.App("crag-metaqa2000")
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = modal.Image.micromamba(python_version="3.11").env({"CONDA_OVERRIDE_CUDA": "12.1"}).apt_install("git").pip_install("torch==2.2.1","numpy<2.0","faiss-gpu-cu12==1.8.0.1","scipy","sentence-transformers<3.0","transformers==4.44.2","networkx==3.2.1","spacy","pyyaml","pandas","tqdm").run_commands("python -m spacy download en_core_web_sm").add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")

@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": vol}, timeout=3600)
def run():
    import os, sys
    os.chdir("/root/CRAG")
    sys.path.insert(0, "/root/CRAG")
    # Ensure data symlink via storage volume
    import pathlib
    for sd in ["data","results"]:
        storage_sd=f"/root/CRAG/storage/{sd}"
        local_sd=f"/root/CRAG/{sd}"
        if os.path.exists(storage_sd) and not os.path.islink(local_sd):
            if os.path.exists(local_sd):
                import shutil
                if os.path.isdir(local_sd):
                    import shutil as sh
                    for item in os.listdir(local_sd):
                        s=os.path.join(local_sd,item); d=os.path.join(storage_sd,item)
                        if not os.path.exists(d):
                            (sh.copytree if os.path.isdir(s) else sh.copy2)(s,d)
                    sh.rmtree(local_sd)
                else:
                    os.remove(local_sd)
            os.symlink(storage_sd, local_sd)
    import json, torch, faiss, numpy as np
    from src.core.engine import CoreEngine
    from src.experiments.overlap_retrain import _splits, _hard_membership
    from src.experiments.l1_universal_head import _load
    from src.experiments.l2_seed import _recall, MAXK
    # Load historical 2000 IDs
    hist=json.load(open("data/canonical/metaqa/splits/historical_2000_ids.json"))
    hist_ids=hist['ids']
    print(f"hist hash {hist['hash']} n {len(hist_ids)}")
    # Master file provenance
    eng=CoreEngine(source='metaqa', index_subdir='gte_qwen')
    print(f"master {eng.master_nodes_path}")
    # Load heads directly from exact files
    hard_path="data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt"
    mix_path="data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt"
    # Check existence on volume (via storage symlink)
    import pathlib
    # The volume is mounted at /root/CRAG/storage, but _symlink_storage not called here, so need to handle
    # Instead, check both
    for p in [hard_path, mix_path, f"/root/CRAG/storage/{hard_path}", f"/root/CRAG/storage/{mix_path}"]:
        print(f"check {p} exists {os.path.exists(p)} size {os.path.getsize(p) if os.path.exists(p) else 'NA'}")
    # Try to load via storage path if needed
    def load_head(path):
        for cand in [path, f"/root/CRAG/storage/{path}"]:
            if os.path.exists(cand):
                return torch.load(cand, map_location='cpu')
        raise FileNotFoundError(path)
    # Actually need to use _symlink_storage logic: we are on Modal, data is at /root/CRAG/storage/data...
    # Let's just use storage path
    hard_sd=torch.load(f"/root/CRAG/storage/{hard_path}", map_location='cpu')
    mix_sd=torch.load(f"/root/CRAG/storage/{mix_path}", map_location='cpu')
    print(f"hard keys {list(hard_sd.keys())[:3]} mix keys {list(mix_sd.keys())[:3]}")
    import hashlib
    print(f"hard sha256 {hashlib.sha256(open(f'/root/CRAG/storage/{hard_path}','rb').read()).hexdigest()}")
    print(f"mix sha256 {hashlib.sha256(open(f'/root/CRAG/storage/{mix_path}','rb').read()).hexdigest()}")
    # Now need to evaluate: we will use e2e_pipeline's level2_order but with exact IDs
    # Instead of calling _train_universal, we load heads directly
    from src.experiments.query_relation import OffsetHead
    from src.experiments.l1_ablate import MixtureHead
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Need dim
    # Load metaqa data to get dim
    data=_load('metaqa','gte_qwen',8000,3000,1000000)
    dim=data['X'].shape[1]
    print(f"dim {dim}")
    hard_head=OffsetHead(dim).to(device)
    hard_head.load_state_dict(hard_sd)
    hard_head.eval()
    mix_head=MixtureHead(dim,8).to(device)
    mix_head.load_state_dict(mix_sd)
    mix_head.eval()
    heads={'hard':hard_head,'mix_hard':mix_head}
    # Now evaluate on first 2000
    # Use same data but sliced to 2000
    qte,ste,gte=data['test']
    qte2000=qte[:2000]
    ste2000=ste[:2000]
    gte2000=gte[:2000]
    print(f"qte2000 {qte2000.shape} gte2000 len {len(gte2000)}")
    # Need to do level2_order with scope0
    from src.experiments.e2e_pipeline import level2_order
    # Need per_d
    X=data['X']
    idx=faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    per_d={'Xt': torch.tensor(X, device=device), 'faiss': idx}
    # level2_order expects data_d with test etc, but we have sliced
    data_d={'X':X, 'hard':data['hard'], 'mem_idx':data['mem_idx'], 'npart':data['npart'], 'test':(qte2000,ste2000,gte2000), 'test_texts':data['test_texts'][:2000], 'splade':data.get('splade')}
    sigs, scrs, gte_out = level2_order('metaqa', data_d, per_d, heads, scope_topk=0, device=device)
    print(f"sigs {len(sigs)} {[s.shape for s in sigs]}")
    # Compute recalls
    KS=[2,5,20,50]
    def recall(order, golds):
        out={k:0 for k in KS}
        n=0
        for qi,g in enumerate(golds):
            if not g: continue
            n+=1
            gs=set(g)
            row=order[qi]
            for k in KS:
                found=gs & set(int(x) for x in row[:k] if x>=0)
                out[k]+=len(found)/len(g)
        return {k: round(100*out[k]/max(n,1),2) for k in KS}
    for i,name in enumerate(['dense','rel_hard','mlpT','splade'][:len(sigs)]):
        r=recall(sigs[i], gte2000)
        print(f"{name} {r}")
    # D_L2
    from src.experiments.e2e_pipeline import _merge_minrank, _recall as rec2
    nq=len(gte2000)
    l2_min=[_merge_minrank([sigs[j][qi] for j in range(len(sigs))]) for qi in range(nq)]
    r_l2=rec2(l2_min, gte2000)
    print(f"D_L2 {r_l2}")
    # D_E2E needs L3
    from src.core.engine import CoreEngine as CE
    eng2=CE(source='metaqa', index_subdir='gte_qwen')
    from src.experiments.l1l3_recall import _graph
    n=X.shape[0]
    id2idx=eng2.node_id_to_idx
    _, A_str, _ = _graph(eng2, n, id2idx, sources=("struct",))
    from src.pipeline.ner_edges import build_ner_edges
    A_ner=build_ner_edges('metaqa', [eng2.nodes[i].content for i in range(len(eng2.nodes))], n)
    import scipy.sparse as sp
    def _transition(A):
        A=A.tocsr().astype(np.float32)
        deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1.0
        return sp.diags(1.0/deg) @ A
    P=_transition((A_str+A_ner).tocsr())
    from src.experiments.e2e_pipeline import _ner_compose
    composed=[_ner_compose(l2_min[qi], P, n, n_seed=2) for qi in range(nq)]
    r_e2e=rec2(composed, gte2000)
    print(f"D_E2E {r_e2e}")
    # Save
    out={'hist_hash':hist['hash'],'n':2000,'dense':recall(sigs[0],gte2000),'rel_hard':recall(sigs[1],gte2000) if len(sigs)>1 else {},'mlpT':recall(sigs[2],gte2000) if len(sigs)>2 else {},'splade':recall(sigs[3],gte2000) if len(sigs)>3 else {},'D_L2':r_l2,'D_E2E':r_e2e}
    import json, os
    os.makedirs('results/L2', exist_ok=True)
    json.dump(out, open('results/L2/metaqa_2000_repro.json','w'), indent=2)
    print("saved")

if __name__=="__main__":
    run.remote()
