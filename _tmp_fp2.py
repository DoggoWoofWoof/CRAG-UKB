import hashlib, numpy as np, torch, gc
from src.experiments.l1_universal_head import _load
import faiss

head_datasets=sorted(["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"])
subdir="gte_qwen"
limit=8000; tr_cap=3000; te_cap=1
epochs=15; K=8; dim=1536

for kind in ["hard","mix_hard"]:
    fp = hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())
    print(f"\n=== kind {kind} init {fp.hexdigest()[:16]} ===")
    for d in head_datasets:
        print(f"loading {d}...")
        dd=_load(d, subdir, limit, tr_cap, te_cap)
        # mimic per_ds construction for this single dataset
        # per_ds[d] would be {"train": dd["train"], "Xt": torch.tensor(dd["X"], device=torch.device("cpu")), ...}
        # Need Xt as torch float32
        Xt = torch.tensor(dd["X"], device=torch.device("cpu"))
        train1 = dd["train"][1]  # seeds
        train2 = dd["train"][2]  # golds
        # update fingerprint exactly as production
        h_before = fp.hexdigest()
        fp.update(d.encode())
        print(f"  after name {fp.hexdigest()[:16]} prev {h_before[:16]}")
        arr1 = np.asarray(train1)
        fp.update(arr1.tobytes())
        print(f"  after seeds {fp.hexdigest()[:16]} seeds hash {hashlib.sha256(arr1.tobytes()).hexdigest()[:16]} len {len(arr1)}")
        flat = np.asarray([g for gl in train2 for g in gl], dtype=np.int64)
        fp.update(flat.tobytes())
        print(f"  after golds {fp.hexdigest()[:16]} flat len {len(flat)} hash {hashlib.sha256(flat.tobytes()).hexdigest()[:16]}")
        arrX = np.ascontiguousarray(Xt[:64].detach().cpu().numpy())
        fp.update(arrX.tobytes())
        print(f"  after Xt {fp.hexdigest()[:16]} Xt hash {hashlib.sha256(arrX.tobytes()).hexdigest()[:16]} shape {arrX.shape}")
        # free
        del dd, Xt
        gc.collect()
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    final = fp.hexdigest()
    print(f"FINAL {kind} {final} file head_{final[:16]}.pt")

print("EXPECTED current 06a9fd3a3e39b3d0 and 32404bf9b65a2d95")
