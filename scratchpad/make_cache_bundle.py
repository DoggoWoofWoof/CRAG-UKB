"""Build the small per-(dataset, model) bundle that the Modal retrieval-cache job consumes.

WHAT GOES REMOTE AND WHAT DOES NOT

  The document stores are already on crag-data-volume -- dense 131/150 shards, splade 131/150
  after this session's upload -- so the 35 GB of dense and 8 GB of splade corpus is NOT
  shipped.  What ships is only what cannot be derived remotely:

    map.npz      u_src / u_row  distinct (store, row) pairs, and pos2u, canonical position ->
                 distinct row.  Duplicate positions are real (the pointer index is
                 many-to-one: 12,581 for hotpotqa, 7,122 for 2wiki) and every one of them must
                 still get its own top-K entry, so scoring is done over POSITIONS via pos2u,
                 never over distinct rows alone.
    store0.json  a FULL sha256 census of every base shard the resolve will read.
    store_N      the flat REV2_PATCH / DENSE_REPAIR arrays -- small, and local-only.
    queries.*    the QUERY matrix, resolved LOCALLY and shipped whole.  It is 323-592 MB dense
                 and 38-49 MB sparse, and resolving it here rather than remotely means the
                 dense query repair patch (4,851 hotpotqa + 1,601 2wiki rows) does not have to
                 be re-plumbed on the other side.

WHY A FULL CENSUS AND NOT A SPOT CHECK

  A remote store has already been wrong in this project in a way a spot check cannot see: 21
  of 281 dense shards were the right SIZE and the wrong BYTES, because `modal volume put`
  skips on name.  The existing kNN bundle censuses 3 shards out of 131.  This censuses every
  shard that will be read, and the Modal side hashes each shard as it streams it during
  resolve -- so the strong check costs no extra remote I/O at all, only this one local pass.

Run:
  PYTHONHASHSEED=0 python scratchpad/make_cache_bundle.py hotpotqa 2wiki --models dense,splade
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

FC = "data/final_canonical"
OUTROOT = "scratchpad/_cache_bundle"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def pointer(ds, model, kind):
    rel = ("pointer_index",) if kind == "docs" else ("queries", "pointer_index")
    z = np.load(os.path.join(FC, ds, *rel, model + ".npz"))
    return z["src"].astype(np.int64), z["row"].astype(np.int64)


def build(ds, model):
    pi = json.load(io.open(os.path.join(FC, "POINTER_INDEX.json"), encoding="utf-8"))
    stores = ((pi["datasets"][ds]) or {})[model]["stores"]
    src, row = pointer(ds, model, "docs")
    n_pos = src.shape[0]

    # distinct (store, row) pairs, and position -> distinct index
    pair = (src << np.int64(40)) | row
    u_pair, pos2u = np.unique(pair, return_inverse=True)
    u_src = (u_pair >> np.int64(40)).astype(np.int16)
    u_row = (u_pair & ((np.int64(1) << np.int64(40)) - 1)).astype(np.int64)
    nu = u_pair.shape[0]
    print("  positions=%d distinct=%d duplicates=%d" % (n_pos, nu, n_pos - nu), flush=True)

    od = os.path.join(OUTROOT, "%s_%s" % (ds, model))
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "map.npz"), u_src=u_src, u_row=u_row,
             pos2u=pos2u.astype(np.int32))

    meta = {"dataset": ds, "model": model, "n_positions": int(n_pos), "n_distinct": int(nu),
            "n_duplicate_positions": int(n_pos - nu), "stores": []}

    base = stores[0]
    ss = int(base["shard_size"])
    need = int(u_row[u_src == 0].max()) // ss
    ext = "npy" if model == "dense" else "npz"
    print("  censusing %d base shards of %s ..." % (need + 1, base["path"]), flush=True)
    t = time.time()
    cens, bytes_ = {}, {}
    for k in range(need + 1):
        p = os.path.join(base["path"], "shard_%05d.%s" % (k, ext))
        if not os.path.isfile(p):
            raise SystemExit("missing local base shard %s" % p)
        cens[str(k)] = sha(p)
        bytes_[str(k)] = os.path.getsize(p)
    print("  census done in %.0fs" % (time.time() - t), flush=True)

    json.dump({"dir": "/root/vol/" + base["path"].replace("\\", "/"),
               "pattern": "shard_%05d." + ext,
               "shard_size": ss, "max_base_shard_needed": need,
               "census_is_complete": True,
               "local_shard_sha256": cens, "local_shard_bytes": bytes_},
              io.open(os.path.join(od, "store0.json"), "w", encoding="utf-8"), indent=1)
    meta["stores"].append({"i": 0, "store": base.get("store"), "path": base["path"],
                           "flat": False, "shard_size": ss,
                           "n_rows_used": int((u_src == 0).sum())})

    for i, st in enumerate(stores):
        if i == 0 or not st.get("flat"):
            continue
        srcp = st["path"]
        dstp = os.path.join(od, "store_%d.%s" % (i, ext))
        with open(srcp, "rb") as a, open(dstp, "wb") as b:
            while True:
                c = a.read(1 << 22)
                if not c:
                    break
                b.write(c)
        meta["stores"].append({"i": i, "store": st.get("store"), "path": srcp, "flat": True,
                               "sha256": sha(dstp), "bytes": os.path.getsize(dstp),
                               "n_rows_used": int((u_src == i).sum())})

    # ---- query matrix, resolved locally and shipped whole
    Q = CanonicalEmbeddings(ds, model, "queries", root=FC)
    nq = len(Q)
    print("  resolving %d queries locally ..." % nq, flush=True)
    if model == "dense":
        qm = np.ascontiguousarray(Q.gather(np.arange(nq)), dtype=np.float16)
        np.save(os.path.join(od, "queries.npy"), qm)
        qmeta = {"n_queries": nq, "dim": int(qm.shape[1]), "dtype": "float16",
                 "file": "queries.npy", "sha256": sha(os.path.join(od, "queries.npy"))}
    else:
        import scipy.sparse as sp
        qm = Q.gather(np.arange(nq)).tocsr().astype(np.float32)
        sp.save_npz(os.path.join(od, "queries.npz"), qm)
        qmeta = {"n_queries": nq, "dim": int(qm.shape[1]), "nnz": int(qm.nnz),
                 "file": "queries.npz", "sha256": sha(os.path.join(od, "queries.npz"))}
    meta["queries"] = qmeta
    meta["map_sha256"] = sha(os.path.join(od, "map.npz"))
    json.dump(meta, io.open(os.path.join(od, "meta.json"), "w", encoding="utf-8"), indent=1)

    tot = sum(os.path.getsize(os.path.join(od, f)) for f in os.listdir(od))
    print("  bundle %s = %.2f GB" % (od, tot / 1e9), flush=True)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--models", default="dense,splade")
    a = ap.parse_args()
    tot = 0
    for ds in a.datasets:
        for model in a.models.split(","):
            print("== %s / %s" % (ds, model), flush=True)
            tot += build(ds, model)
    print("")
    print("total to upload: %.2f GB" % (tot / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
