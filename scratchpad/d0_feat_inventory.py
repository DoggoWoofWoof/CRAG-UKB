"""Phase D0 — inventory of canonical CRAG feature dumps (results/L2/relsig_feats_{ds}.pt).
Read-only. Reports split counts, feat_names, pool-size stats, mu/sd. NO mutation, NO training."""
import json, os, glob, torch, numpy as np

OUT = "results/data_audit"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cpu")

def inspect(path):
    d = torch.load(path, map_location=DEV)
    info = {"file": path, "bytes": os.path.getsize(path), "keys": sorted(list(d.keys()))}
    info["feat_names"] = d.get("feat_names")
    mu = d.get("mu"); sd = d.get("sd")
    info["mu_shape"] = list(mu.shape) if torch.is_tensor(mu) else None
    info["mu"] = [round(float(x),4) for x in mu] if torch.is_tensor(mu) else None
    splits = {}
    for k in ("train", "dev", "test"):
        if k not in d:
            continue
        recs = d[k]
        n = len(recs)
        # each record = (x[n,ncol], y[n], ng)  -- pool size n, gold count ng
        pools = []; ngs = []; ncol = None
        for r in recs[:100000]:
            x = r[0]; y = r[1]; ng = r[2]
            pools.append(int(x.shape[0])); ncol = int(x.shape[1])
            ngs.append(int(ng) if not torch.is_tensor(ng) else int(ng))
        pools = np.array(pools); ngs = np.array(ngs)
        splits[k] = {"n_queries": n, "ncol": ncol,
                     "pool_min": int(pools.min()), "pool_max": int(pools.max()),
                     "pool_mean": round(float(pools.mean()),1), "pool_median": int(np.median(pools)),
                     "ng_min": int(ngs.min()), "ng_max": int(ngs.max()), "ng_mean": round(float(ngs.mean()),3),
                     "queries_with_zero_gold": int((ngs==0).sum())}
    info["splits"] = splits
    return info

def main():
    files = sorted(glob.glob("results/L2/relsig_feats_*.pt"))
    out = {}
    for f in files:
        ds = os.path.basename(f)[len("relsig_feats_"):-len(".pt")]
        try:
            out[ds] = inspect(f)
            s = out[ds]["splits"]
            cnt = {k: v["n_queries"] for k,v in s.items()}
            print(f"{ds:22s} feat={out[ds]['feat_names']}  splits={cnt}")
        except Exception as e:
            out[ds] = {"file": f, "error": repr(e)}
            print(f"{ds:22s} ERROR {e!r}")
    json.dump(out, open(f"{OUT}/feat_inventory.json","w"), indent=2)
    print(f"\n-> {OUT}/feat_inventory.json")

if __name__ == "__main__":
    main()
