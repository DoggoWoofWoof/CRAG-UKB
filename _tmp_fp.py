import hashlib, numpy as np, torch
from src.experiments.l1_universal_head import _load

def compute_fp(per_ds, kind, K, epochs):
    dim = per_ds[list(per_ds.keys())[0]]["Xt"].shape[1]
    fp = hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())
    print(f"init {kind}|K{K}|e{epochs}|d{dim} -> {fp.hexdigest()[:16]}")
    for d in sorted(per_ds.keys()):
        print(f"dataset {d}")
        h_before = fp.hexdigest()
        fp.update(d.encode())
        print(f"  after name {fp.hexdigest()[:16]} prev {h_before[:16]}")
        arr1 = np.asarray(per_ds[d]["train"][1])
        print(f"  seeds {arr1.dtype} {arr1.shape} {hashlib.sha256(arr1.tobytes()).hexdigest()[:16]}")
        fp.update(arr1.tobytes())
        print(f"  after seeds {fp.hexdigest()[:16]}")
        flat = np.asarray([g for gl in per_ds[d]["train"][2] for g in gl], dtype=np.int64)
        print(f"  golds {flat.dtype} {len(flat)} {hashlib.sha256(flat.tobytes()).hexdigest()[:16]}")
        fp.update(flat.tobytes())
        print(f"  after golds {fp.hexdigest()[:16]}")
        arrX = np.ascontiguousarray(per_ds[d]["Xt"][:64].detach().cpu().numpy())
        print(f"  Xt {arrX.dtype} {arrX.shape} {hashlib.sha256(arrX.tobytes()).hexdigest()[:16]}")
        fp.update(arrX.tobytes())
        print(f"  after Xt {fp.hexdigest()[:16]}")
    final = fp.hexdigest()
    print(f"final {kind} {final} file head_{final[:16]}.pt")
    return final

head_datasets=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
subdir="gte_qwen"
limit=8000; tr_cap=3000; te_cap=1
epochs=15; K=8
device=torch.device("cpu")
per_ds={}
for d in head_datasets:
    print(f"=== loading {d} ===")
    dd=_load(d, subdir, limit, tr_cap, te_cap)
    import faiss
    idx=faiss.IndexFlatIP(dd["X"].shape[1]); idx.add(dd["X"])
    per_ds[d]={"train": dd["train"], "Xt": torch.tensor(dd["X"], device=device), "index": idx}
    print(f"  X {dd['X'].shape} train {len(dd['train'][0])}")

print("computing hard")
hard_fp=compute_fp(per_ds, "hard", K, epochs)
print("computing mix")
mix_fp=compute_fp(per_ds, "mix_hard", K, epochs)
print(f"CURRENT_HARD_FP={hard_fp[:16]}")
print(f"CURRENT_MIX_FP={mix_fp[:16]}")
print("EXPECTED 06a9fd3a3e39b3d0 32404bf9b65a2d95")
