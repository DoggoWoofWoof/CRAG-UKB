"""FBX_SCALE stage 4A control -- the FROZEN partitioner on graph S alone (WebQSP), at the gate K.

Why: the gate compares graph-S partitions (the deployable arm; Freebase has no KNN family) with the seven PHG maps, which were built on H4_SK (STRUCT u KNN).  No S-only partitioner can
be better than the frozen partitioner on the same S-only hypergraph, so PHG_S is the ceiling of what the S arm can reach and the fair reference for a partitioner comparison on the
same graph semantics.  It is a CONTROL (the frozen algorithm, unchanged), not a candidate: nothing here is tuned and the gate of addendum 1 keeps PHG_SK as its reference.

Everything but the family set is scratchpad/_l1h_host.py's, imported and run unchanged (shard writer, driver binary, parameter list, NP = 4, the non-empty repair, the validity gate);
_l1h_host's tag and output-directory functions are re-pointed at runtime (its file is unchanged).  The hypergraph is src/l1_canonical/hypergraph.build_hypergraph with famset "S":
the frozen H4_SPLIT_PRESERVE rule, cap = round(N / K).

  python -u scratchpad/_fbx_phg_s.py CELL <K> [<K> ...]      # HG then PHG per K; write-once records in results/FREEBASE_SCALE/parts_S/
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1h_host as H

HG = H.HG
OUT = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "parts_S")
DS = "webqsp"
KEYS = os.path.join(H.REPO, "data", "l1_canonical", DS, "keys.npz")
FAMSET = "S"


def tag_of(K):
    return "H4_S_k%d" % int(K)


H.PDIR = OUT
H.tag_of = tag_of


def cmd_hg(Ks):
    z = np.load(KEYS)
    N = int(z["N"][0])
    keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
    for K in Ks:
        fp = H.hg_npz(DS, K)
        if os.path.exists(fp[:-4] + ".json"):
            m = json.load(io.open(fp[:-4] + ".json", encoding="utf-8"))
            assert H.sha_file(fp) == m["file_sha256"], "%s changed since its record" % fp
            H.log("K %d: H4_S hypergraph exists (%s)" % (K, m["file_sha256"][:16]))
            continue
        t = time.time()
        arrays, meta = HG.build_hypergraph(N, keys, K, famset=FAMSET, tag=DS)
        dig = HG.arrays_digest(arrays)
        os.makedirs(OUT, exist_ok=True)
        tmp = fp[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, fp)
        meta.update({"dataset": DS, "file": H.rel(fp), "bytes": os.path.getsize(fp), "file_sha256": H.sha_file(fp), "content_digest": dig,
                     "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1),
                     "inputs": {"keys_npz_sha256": H.sha_file(KEYS), "STRUCT": int(len(keys["STRUCT"]))},
                     "scale": {"K": K, "block_size_round_N_over_K": int(round(N / K)),
                               "rule": "the frozen H4 rule (H4_SPLIT_PRESERVE, cap = round(N / K)) on the STRUCT family only (famset S); nothing else changed"},
                     "placement": H.placement(), "code": H.code_pins()})
        H.wj(fp[:-4] + ".json", meta)
        H.log("K %d: H4_S cap %s, %d hyperedges, %d pins, %.1f MB, %.1fs" % (K, meta["cap"], meta["hyperedges"], meta["pins"], meta["bytes"] / 1e6, meta["seconds"]))
        del arrays


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) < 2 or a[0] != "CELL":
        raise SystemExit(__doc__)
    Ks = [int(x) for x in a[1:]]
    H.log("placement:", json.dumps(H.placement()))
    cmd_hg(Ks)
    H.cmd_phg(DS, Ks)
    H.log("done %.0fs" % (time.time() - H.T0))
