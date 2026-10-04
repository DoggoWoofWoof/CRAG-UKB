"""FBX_SCALE Track C -- the FROZEN partitioner on H4_{STRUCT + KNN_PQ<m>} (WebQSP), at the gate K.

The KNN family is the only thing that changes: the keys file of a PQ-KNN calibration arm (scratchpad/_pq_knn.py PQ <m>) replaces the frozen KNN keys; STRUCT, the H4_SPLIT_PRESERVE rule (cap =
round(N / K)), the driver binary, parameters, NP = 4, the non-empty repair and the validity gate are scratchpad/_l1h_host.py's, imported and run unchanged (its tag and output-directory
functions are re-pointed at runtime, as _fbx_phg_s.py does).

  python -u scratchpad/_pq_phg.py CONTROL                    # the keys of the frozen family through THIS script's hypergraph path at K = 100: its content digest must equal the frozen H4_SK_k100's
  python -u scratchpad/_pq_phg.py CELL <m> <K> [<K> ...]     # HG then PHG per K on the PQ<m> keys; write-once records in results/FREEBASE_SCALE/pq/parts/
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1h_host as H

HG = H.HG
OUT = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "pq", "parts")
DS = "webqsp"
FROZEN_KEYS = os.path.join(H.REPO, "data", "l1_canonical", DS, "keys.npz")
FROZEN_HG = os.path.join(H.REPO, "results", "L1_HOST", "parts", "%s__H4_SK_k100.json" % DS)
H.PDIR = OUT
STATE = {"m": None, "keys": None}


def keys_path(m):
    return os.path.join(H.REPO, "data", "freebase_scale", "pq", "pq%d" % m, "keys_pq%d.npz" % m)


def tag_of(K):
    return "H4_PQ%d_k%d" % (STATE["m"], int(K))


H.tag_of = tag_of


def cmd_hg(Ks):
    m = STATE["m"]
    kp = keys_path(m)
    z, zs = np.load(kp), np.load(FROZEN_KEYS)
    N = int(z["N"][0])
    assert N == int(zs["N"][0]) and bool((z["STRUCT"] == zs["STRUCT"]).all()), "STRUCT differs from the frozen family"
    keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
    for K in Ks:
        fp = H.hg_npz(DS, K)
        if os.path.exists(fp[:-4] + ".json"):
            meta = json.load(io.open(fp[:-4] + ".json", encoding="utf-8"))
            assert H.sha_file(fp) == meta["file_sha256"], "%s changed since its record" % fp
            H.log("K %d: H4_PQ%d hypergraph exists (%s)" % (K, m, meta["file_sha256"][:16]))
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
                     "inputs": {"keys_npz": H.rel(kp), "keys_npz_sha256": H.sha_file(kp), "STRUCT": int(len(keys["STRUCT"])), "KNN": int(len(keys["KNN"])), "pq_m": m},
                     "scale": {"K": K, "block_size_round_N_over_K": int(round(N / K)),
                               "rule": "the frozen H4_SK rule (H4_SPLIT_PRESERVE, cap = round(N / K)); only the KNN key set differs (compressed-vector KNN)"},
                     "placement": H.placement(), "code": H.code_pins()})
        H.wj(fp[:-4] + ".json", meta)
        H.log("K %d: H4_PQ%d cap %s, %d hyperedges, %d pins, %.1f MB, %.1fs" % (K, m, meta["cap"], meta["hyperedges"], meta["pins"], meta["bytes"] / 1e6, meta["seconds"]))
        del arrays


def cmd_control():
    zs = np.load(FROZEN_KEYS)
    N = int(zs["N"][0])
    arrays, meta = HG.build_hypergraph(N, {"STRUCT": zs["STRUCT"], "KNN": zs["KNN"]}, 100, tag=DS)
    dig = HG.arrays_digest(arrays)
    ref = json.load(io.open(FROZEN_HG, encoding="utf-8"))
    ok = dig == ref["content_digest"]
    H.log("CONTROL: digest %s, frozen %s -> %s (pins %d vs %d)" % (dig[:16], ref["content_digest"][:16], "EQUAL" if ok else "DIFFERENT", meta["pins"], ref["pins"]))
    os.makedirs(OUT, exist_ok=True)
    rec = os.path.join(OUT, "CONTROL__hypergraph_digest.json")
    assert not os.path.exists(rec), "write-once"
    H.wj(rec, {"RECORD": "PQ_CALIB_CONTROL_HYPERGRAPH", "digest": dig, "frozen_digest": ref["content_digest"], "equal": ok, "pins": int(meta["pins"]), "frozen_pins": int(ref["pins"]),
               "code": H.code_pins(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    assert ok, "the keys -> hypergraph path of this script does not reproduce the frozen hypergraph"


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a and a[0] == "CONTROL":
        cmd_control()
    elif len(a) >= 3 and a[0] == "CELL":
        STATE["m"] = int(a[1])
        Ks = [int(x) for x in a[2:]]
        H.log("placement:", json.dumps(H.placement()))
        cmd_hg(Ks)
        H.cmd_phg(DS, Ks)
        H.log("done %.0fs" % (time.time() - H.T0))
    else:
        raise SystemExit(__doc__)
