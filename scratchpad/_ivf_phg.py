"""FBX_SCALE Track C step 2 (addendum 10) -- the FROZEN partitioner on H4_{STRUCT + KNN_IVF<tag>} (WebQSP), at the gate K.

Exactly _pq_phg.py with the keys file of an IVF arm (scratchpad/_ivf_knn.py FULL <nlist> <nprobe> <tag>) in place of the PQ64 keys: STRUCT, the H4_SPLIT_PRESERVE rule (cap = round(N / K)), the driver
binary, parameters, NP = 4, the non-empty repair and the validity gate are scratchpad/_l1h_host.py's, run unchanged.

  python -u scratchpad/_ivf_phg.py CELL <tag> <K> [<K> ...]     # HG then PHG per K; write-once records in results/FREEBASE_SCALE/ivf/parts/
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1h_host as H
import _pq_phg as P

HG = H.HG
OUT = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "ivf", "parts")
DS = P.DS
H.PDIR = OUT
STATE = {"tag": None}


def keys_path(tag):
    return os.path.join(H.REPO, "data", "freebase_scale", "pq", "ivf", tag, "keys_%s.npz" % tag)


def tag_of(K):
    return "H4_%s_k%d" % (STATE["tag"], int(K))


H.tag_of = tag_of


def cmd_hg(Ks):
    tag = STATE["tag"]
    kp = keys_path(tag)
    z, zs = np.load(kp), np.load(P.FROZEN_KEYS)
    N = int(z["N"][0])
    assert N == int(zs["N"][0]) and bool((z["STRUCT"] == zs["STRUCT"]).all()), "STRUCT differs from the frozen family"
    keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
    for K in Ks:
        fp = H.hg_npz(DS, K)
        if os.path.exists(fp[:-4] + ".json"):
            meta = json.load(io.open(fp[:-4] + ".json", encoding="utf-8"))
            assert H.sha_file(fp) == meta["file_sha256"], "%s changed since its record" % fp
            H.log("K %d: H4_%s hypergraph exists (%s)" % (K, tag, meta["file_sha256"][:16]))
            continue
        t = time.time()
        arrays, meta = HG.build_hypergraph(N, keys, K, tag=DS)
        dig = HG.arrays_digest(arrays)
        os.makedirs(OUT, exist_ok=True)
        tmp = fp[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, fp)
        meta.update({"dataset": DS, "file": H.rel(fp), "bytes": os.path.getsize(fp), "file_sha256": H.sha_file(fp), "content_digest": dig,
                     "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1),
                     "inputs": {"keys_npz": H.rel(kp), "keys_npz_sha256": H.sha_file(kp), "STRUCT": int(len(keys["STRUCT"])), "KNN": int(len(keys["KNN"])), "ivf_tag": tag},
                     "scale": {"K": K, "block_size_round_N_over_K": int(round(N / K)),
                               "rule": "the frozen H4_SK rule (H4_SPLIT_PRESERVE, cap = round(N / K)); only the KNN key set differs (IVF search over PQ64 codes)"},
                     "placement": H.placement(), "code": H.code_pins()})
        H.wj(fp[:-4] + ".json", meta)
        H.log("K %d: H4_%s cap %s, %d hyperedges, %d pins, %.1f MB, %.1fs" % (K, tag, meta["cap"], meta["hyperedges"], meta["pins"], meta["bytes"] / 1e6, meta["seconds"]))
        del arrays


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) >= 3 and a[0] == "CELL":
        STATE["tag"] = a[1]
        Ks = [int(x) for x in a[2:]]
        H.log("placement:", json.dumps(H.placement()))
        cmd_hg(Ks)
        H.cmd_phg(DS, Ks)
        H.log("done %.0fs" % (time.time() - H.T0))
    else:
        raise SystemExit(__doc__)
