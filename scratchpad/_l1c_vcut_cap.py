"""STRUCT_VERTEXCUT_V1_TRANSFER step 2 (eleventh ruling, 2026-09-15; PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json): the
section-18 deterministic capacity repair at C applied ONCE to the k* cell of a dataset -- the same call _l1c_vcut_metrics.py makes
after its structural cells, without the structural metrics.  STRUCT edges only; degree-0 nodes are placed later by the transfer
module and nothing is re-repaired afterwards.

    python -u scratchpad/_l1c_vcut_cap.py <ds>   -> parts/<ds>__STRUCT_VCUT_V1_k<k*>__PHG_con__CAP<C>.{npy,json}
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1c_vcut_lib as V  # noqa: E402

X = V.X


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


def main(ds):
    t0 = time.time()
    ksf = os.path.join(V.OUT, "vcut_kstar_%s.json" % ds)
    ks = json.load(open(ksf, encoding="utf-8"))
    k = int(ks["k_star"])
    G = V.Graph(ds)
    C = G.C
    assert ks["dataset"] == ds and ks["C"] == C and ks["N"] == G.N and ks["k_frozen"] == G.k_f
    capf = os.path.join(V.PDIR, "%s__%s_k%d__PHG_con__CAP%d.npy" % (ds, V.TAG, k, C))
    assert not os.path.exists(capf), "CAP file exists -- the repair is applied once"
    z, meta, run = V.load_vcut(ds, k)
    inp = os.path.join(V.PDIR, "%s__%s_k%d__PHG_con.npy" % (ds, V.TAG, k))
    before = V.vcut_size_stats(G, z, k)
    V.log("%s k* %d: before repair %s" % (ds, k, json.dumps(before)))
    zc, rep = V.capacity_repair(G, z, k, C)
    np.save(capf, zc.astype(np.int64))
    after = V.vcut_size_stats(G, zc, k)
    sizes = np.asarray(V.Y_vcut(G, zc, k).sum(axis=0)).ravel()
    after["blocks_gt_1.5C"] = int((sizes > 1.5 * C).sum())
    after["min_block_size"] = int(sizes.min())
    rec = {"RECORD": "STRUCT_VCUT_V1_CAP", "dataset": ds, "k_star": k, "C": C, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "kstar_record": pin(ksf), "input_assignment": pin(inp), "phg_run_status": run["output"]["STATUS"],
           "graph": {"N": G.N, "E": G.E, "k_f": G.k_f, "hubs": int(G.hub.sum()), "degree_0": int((G.deg == 0).sum()), "struct_keys_sha256": G.keys_sha},
           "before": before, "after": after, "capacity_repair": rep, "output_assignment": pin(capf),
           "code": pin(os.path.abspath(__file__)), "lib": pin(os.path.join(HERE, "_l1c_vcut_lib.py")),
           "preregistration": pin(os.path.join(V.OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")), "seconds": round(time.time() - t0, 1)}
    V.X.wj(capf[:-4] + ".json", rec)
    V.log("%s k* %d: after repair %s | repair %s (%.0fs) -> %s" % (ds, k, json.dumps(after), json.dumps(rep), time.time() - t0, os.path.relpath(capf, X.REPO)))


if __name__ == "__main__":
    main(sys.argv[1])
