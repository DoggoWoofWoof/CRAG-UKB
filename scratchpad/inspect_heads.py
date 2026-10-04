import modal
app = modal.App("crag-inspect-heads")
vol = modal.Volume.from_name("crag-data-volume")
image = modal.Image.debian_slim().pip_install("torch==2.2.1")

@app.function(image=image, volumes={"/vol": vol})
def inspect():
    import os, hashlib, torch
    for fname in ["head_06a9fd3a3e39b3d0.pt","head_32404bf9b65a2d95.pt"]:
        path=f"/vol/data/ukb_storage/_head_cache/{fname}"
        print(f"FILE {fname}")
        if not os.path.exists(path):
            print("  NOT FOUND")
            continue
        data=open(path,"rb").read()
        print(f"  size {len(data)} sha256 {hashlib.sha256(data).hexdigest()}")
        sd=torch.load(path, map_location="cpu")
        print(f"  keys {list(sd.keys())}")
        total=sum(p.numel() for p in sd.values())
        print(f"  total params {total}")
        for k,v in sd.items():
            print(f"    {k} {tuple(v.shape)}")
