import hashlib, numpy as np, torch, gc
from src.experiments.l1_universal_head import _load

head_datasets_sorted=sorted(["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"])
head_datasets_input=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"] # input order
subdir="gte_qwen"
limit=8000; tr_cap=3000; te_cap=1
epochs=15; K=8; dim=1536

# Store per-dataset bytes
per_dataset_data={}
for d in head_datasets_sorted:
    print(f"loading {d}...")
    dd=_load(d, subdir, limit, tr_cap, te_cap)
    Xt = torch.tensor(dd["X"], device=torch.device("cpu"))
    train1 = dd["train"][1]
    train2 = dd["train"][2]
    # store arrays for later
    arr1 = np.asarray(train1)
    flat = np.asarray([g for gl in train2 for g in gl], dtype=np.int64)
    arrX = np.ascontiguousarray(Xt[:64].detach().cpu().numpy())
    per_dataset_data[d] = {
        "name": d,
        "seeds": arr1,
        "golds_flat": flat,
        "Xt64": arrX,
        "X_shape": dd["X"].shape,
        "train_len": len(train1)
    }
    print(f"  seeds {arr1.dtype} {arr1.shape} hash {hashlib.sha256(arr1.tobytes()).hexdigest()[:16]}")
    print(f"  golds {flat.dtype} {len(flat)} hash {hashlib.sha256(flat.tobytes()).hexdigest()[:16]}")
    print(f"  Xt64 {arrX.dtype} {arrX.shape} hash {hashlib.sha256(arrX.tobytes()).hexdigest()[:16]}")
    del dd, Xt
    gc.collect()

def fp_for(order, kind):
    fp = hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())
    for d in order:
        fp.update(d.encode())
        fp.update(per_dataset_data[d]["seeds"].tobytes())
        fp.update(per_dataset_data[d]["golds_flat"].tobytes())
        fp.update(per_dataset_data[d]["Xt64"].tobytes())
    return fp.hexdigest()

for kind in ["hard","mix_hard"]:
    sorted_fp = fp_for(head_datasets_sorted, kind)
    input_fp = fp_for(head_datasets_input, kind)
    print(f"kind {kind} sorted {sorted_fp} file head_{sorted_fp[:16]}.pt")
    print(f"kind {kind} input_order {input_fp} file head_{input_fp[:16]}.pt")
    print(f"kind {kind} expected current hard 06a9fd3a3e39b3d0 mix 32404bf9b65a2d95")
    # try different K
    for tryK in [8,16]:
        for tryE in [15,20]:
            if tryK==K and tryE==epochs: continue
            fp2 = hashlib.md5(f"{kind}|K{tryK}|e{tryE}|d{dim}".encode())
            for d in head_datasets_sorted:
                fp2.update(d.encode())
                fp2.update(per_dataset_data[d]["seeds"].tobytes())
                fp2.update(per_dataset_data[d]["golds_flat"].tobytes())
                fp2.update(per_dataset_data[d]["Xt64"].tobytes())
            print(f"  try K{tryK} e{tryE} sorted {fp2.hexdigest()[:16]}")

# also try without webqsp (maybe head was trained without webqsp?)
for kind in ["hard"]:
    fp_no_webqsp = hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())
    for d in [x for x in head_datasets_sorted if x!="webqsp"]:
        fp_no_webqsp.update(d.encode())
        fp_no_webqsp.update(per_dataset_data[d]["seeds"].tobytes())
        fp_no_webqsp.update(per_dataset_data[d]["golds_flat"].tobytes())
        fp_no_webqsp.update(per_dataset_data[d]["Xt64"].tobytes())
    print(f"kind {kind} no_webqsp sorted {fp_no_webqsp.hexdigest()[:16]}")

print("done")
