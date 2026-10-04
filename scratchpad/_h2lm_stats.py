"""FBX_SCALE Track B, top-level lever survey (structure only, no partitioner, no gold label, nothing scored):  where do the pins of the H4_SK hypergraph live?

The H4_SPLIT_PRESERVE SK hypergraph has one net per anchor (its closed neighbourhood, deg + 1 pins; nets larger than the cap are split into chunks) with weight
max(1, rint(1000 / deg)) -- a hub's net carries many pins and little weight.  A top-level partition has to read every pin.  This survey measures, per family and in total, the share of
the pins and the share of the weighted KM1 'signal' (sum over nets of weight x (pins - 1)) that belongs to nets of more than c pins, for c in a fixed grid -- i.e. what a net-size cap
on the TOP hypergraph could remove, and what it would cost in signal.  It also counts the vertices left with no net under each cap.  Write-once record:
results/FREEBASE_SCALE/h2l/H2LM_NETSIZE_SURVEY__webqsp__v1.json

  python -u scratchpad/_h2lm_stats.py
"""
import json
import os
import sys
import time

import numpy as np

import _l1h_host as H

REPO = H.REPO
DS = "webqsp"
KEYS = os.path.join(REPO, "data", "l1_canonical", DS, "keys.npz")
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "h2l", "H2LM_NETSIZE_SURVEY__webqsp__v1.json")
CAPS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    t0 = time.time()
    z = np.load(KEYS)
    N = int(z["N"][0])
    deg_f = {}
    for f in ("STRUCT", "KNN"):
        k = z[f]
        a, b = k // N, k % N
        m = a != b
        k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
        d = np.bincount(k // N, minlength=N) + np.bincount(k % N, minlength=N)
        deg_f[f] = d.astype(np.int64)
        H.log("%s: %d pairs, max degree %d" % (f, len(k), int(d.max())))
    rec = {"RECORD": "H2LM_NETSIZE_SURVEY", "dataset": DS, "N": N, "note": "structure only; nets = closed neighbourhoods (deg + 1 pins) before cap-splitting; weight = max(1, rint(1000/deg))",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "families": {}}
    tot = {"pins": 0, "sig": 0.0}
    per = {}
    for f, d in deg_f.items():
        anc = np.flatnonzero(d >= 1)
        sz = d[anc] + 1
        w = np.maximum(1, np.rint(1000.0 / d[anc])).astype(np.float64)
        sig = w * (sz - 1)
        per[f] = (anc, sz, sig)
        tot["pins"] += int(sz.sum())
        tot["sig"] += float(sig.sum())
    for f, (anc, sz, sig) in per.items():
        rows = []
        for c in CAPS:
            big = sz > c
            rows.append({"cap": c, "nets_over_cap": int(big.sum()), "pin_share_over_cap": round(float(sz[big].sum() / sz.sum()), 4), "signal_share_over_cap": round(float(sig[big].sum() / sig.sum()), 4)})
        rec["families"][f] = {"anchors": int(len(anc)), "pins": int(sz.sum()), "max_net": int(sz.max()), "by_cap": rows}
    both = []
    for c in CAPS:
        pins_kept = sum(int(sz[sz <= c].sum()) for (_, sz, _) in per.values())
        sig_kept = sum(float(sg[sz <= c].sum()) for (_, sz, sg) in per.values())
        # vertices that keep at least one net of <= c pins (as anchor or as member): anchor-side only is a lower bound; membership needs the CSR -- computed below from degrees
        both.append({"cap": c, "pins_kept_share": round(pins_kept / tot["pins"], 4), "signal_kept_share": round(sig_kept / tot["sig"], 4), "pin_reduction": round(tot["pins"] / max(pins_kept, 1), 3)})
    rec["total"] = {"pins": tot["pins"], "signal": tot["sig"], "by_cap": both}
    # vertices with no kept net: a vertex is covered if it is the anchor of a kept net or a member of one.  Members: pairs (u, v) with deg(u) + 1 <= c put v in u's net.
    cov_rows = []
    pairs = {}
    for f in ("STRUCT", "KNN"):
        k = z[f]
        a, b = k // N, k % N
        m = a != b
        k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
        pairs[f] = (k // N, k % N)
    for c in CAPS:
        cov = np.zeros(N, bool)
        for f, (u, v) in pairs.items():
            d = deg_f[f]
            ku = d[u] + 1 <= c          # u's net (u anchor) contains v
            kv = d[v] + 1 <= c          # v's net contains u
            cov[u[ku]] = True
            cov[v[ku]] = True
            cov[v[kv]] = True
            cov[u[kv]] = True
        cov_rows.append({"cap": c, "vertices_without_a_kept_net": int((~cov).sum()), "share": round(float((~cov).mean()), 5)})
    rec["coverage"] = cov_rows
    rec["seconds"] = round(time.time() - t0, 1)
    rec["placement"] = H.placement()
    rec["code"] = {"scratchpad/_h2lm_stats.py": H.sha_file(os.path.abspath(__file__))}
    H.wj(OUT, rec)
    H.log("wrote %s" % H.rel(OUT))
    for r in both:
        H.log("cap %5d: pins kept %.4f (x%.2f), signal kept %.4f" % (r["cap"], r["pins_kept_share"], r["pin_reduction"], r["signal_kept_share"]))
    for r in cov_rows:
        H.log("cap %5d: vertices without a kept net %d (%.4f)" % (r["cap"], r["vertices_without_a_kept_net"], r["share"]))


if __name__ == "__main__":
    main()
