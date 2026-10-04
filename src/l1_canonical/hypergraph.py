"""H4_SPLIT_PRESERVE hypergraph over canonical positions -- the frozen rule with its inputs injected.

The numerics are scratchpad/_l1hu_build.py's, imported (ordered / _scatter / resolve_cap / RULES /
FAMSETS / WSCALE), never re-implemented; only the INPUT side differs: the family key sets come
from the caller (CanonicalDataset.keysets over the served graph families) instead of the legacy
`_l1kn` caches, and k follows the frozen rule k = max(1, N // 100) (build_canonical_topo.TARGET)
instead of being read back from a legacy partition map.

    python src/l1_canonical/hypergraph.py parity [ds ...]
        rebuild the legacy <ds>__H4_SPLIT_PRESERVE__SK.npz from scratchpad/_l1kn/keys_<ds>.npz
        and compare every array with the legacy artefact (eptr, eidx, N, k, ew) -- the proof that
        the fork is the same function
    python src/l1_canonical/hypergraph.py build <ds> [ds ...]
        data/l1_canonical/<ds>/hypergraph/H4_SK.npz (+ .json) over canonical positions

Output format is the legacy worker's input format unchanged: eptr int64 [E+1], eidx int32 [pins],
N int64 [1], k int64 [1], ew int32 [E]  (hyperedge weight = max(1, rint(1000 / (|e| - 1)))).
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO, os.path.join(REPO, "scratchpad")):
    if p not in sys.path:
        sys.path.insert(0, p)
import _l1hu_build as HB  # noqa: E402  (frozen rule; imported, not copied)
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

TARGET = 100                              # frozen: build_canonical_topo.TARGET
RULE = "H4_SPLIT_PRESERVE"
FAMSET = "SK"
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def cm(x):
    return "{:,}".format(int(x))


def frozen_k(N):
    return max(1, int(N) // TARGET)


def build_hypergraph(N, keys, k, rule=RULE, famset=FAMSET, log=log, tag=""):
    """_l1hu_build.build with (N, keys_by_family, k) injected.  keys: {"STRUCT": ..., "KNN": ...,
    "NER": ...} sorted unique undirected keys u*N+v.  Returns (arrays, meta)."""
    assert rule in HB.RULES and famset in HB.FAMSETS
    R, fams = HB.RULES[rule], HB.FAMSETS[famset]
    N, k = int(N), int(k)
    tbs = int(round(N / k))
    C = {}
    for f in fams:
        xadj, adj = CanonicalDataset.csr_from_keys(keys[f], N)      # == _l1ep_part._csr_from_keys
        C[f] = (xadj, adj.astype(np.int64), np.diff(xadj).astype(np.int64))
    cap = HB.resolve_cap(rule, np.concatenate([C[f][2][C[f][2] >= 1] + 1 for f in fams]), tbs)

    mark = np.zeros(N, bool)
    parts, szs, fstat = [], [], {}
    for f in fams:
        xadj, adj, deg = C[f]
        anc = np.nonzero(deg >= 1)[0]
        raw = deg[anc] + 1
        pre_n, pre_p = int(len(anc)), int(raw.sum())
        big = raw > cap if cap is not None else np.zeros(len(anc), bool)
        nbig = int(big.sum())

        if R["mode"] == "drop":
            anc, raw, big = anc[~big], raw[~big], big[~big]
            sz = raw.copy()
        elif R["mode"] == "trunc":
            sz = np.minimum(raw, cap) if cap is not None else raw.copy()
        elif R["mode"] == "none":
            sz = raw.copy()
        else:                                             # split: 1 edge per chunk
            per = cap - 1
            nch = np.where(big, (deg[anc] + per - 1) // per, 1)
            sz = np.empty(int(nch.sum()), np.int64)

        if R["mode"] != "split":
            eptr = np.zeros(len(sz) + 1, np.int64)
            eptr[1:] = np.cumsum(sz)
            eidx = np.empty(int(eptr[-1]), np.int64)
            small = ~big
            HB._scatter(eidx, eptr[:-1][small], anc[small], xadj, adj, deg)
            for i in np.nonzero(big)[0]:
                u = int(anc[i])
                eidx[eptr[i]] = u
                eidx[eptr[i] + 1:eptr[i + 1]] = HB.ordered(u, xadj, adj, deg, mark)[:cap - 1]
            kept, dup = int(eptr[-1]), 0
        else:
            eoff = np.zeros(len(nch) + 1, np.int64)
            eoff[1:] = np.cumsum(nch)
            sz[eoff[:-1][~big]] = raw[~big]
            for i in np.nonzero(big)[0]:
                d = int(deg[anc[i]])
                c = np.full(int(nch[i]), per + 1, np.int64)
                r = d % per
                if r:
                    c[-1] = r + 1
                sz[eoff[i]:eoff[i + 1]] = c
            eptr = np.zeros(len(sz) + 1, np.int64)
            eptr[1:] = np.cumsum(sz)
            eidx = np.empty(int(eptr[-1]), np.int64)
            HB._scatter(eidx, eptr[eoff[:-1][~big]], anc[~big], xadj, adj, deg)
            for i in np.nonzero(big)[0]:
                u = int(anc[i])
                om = HB.ordered(u, xadj, adj, deg, mark)
                for j, s in enumerate(range(0, len(om), per)):
                    t = eoff[i] + j
                    eidx[eptr[t]] = u
                    eidx[eptr[t] + 1:eptr[t + 1]] = om[s:s + per]
            kept, dup = int(eptr[-1]), int((nch - 1).sum())

        parts.append(eidx)
        szs.append(sz)
        fstat[f] = {"hyperedges_precap": pre_n, "pins_precap": pre_p,
                    "anchors_over_cap": nbig, "hyperedges_out": int(len(sz)),
                    "pins_written": kept, "anchor_duplication_pins": dup,
                    "pins_discarded": pre_p - (kept - dup),
                    "pin_retention": round((kept - dup) / max(pre_p, 1), 6),
                    "size_max_out": int(sz.max()) if len(sz) else 0}
        log("  %s %s %s: %s anchors (%s over cap) -> %s hyperedges, %s/%s pins (%.4f)"
            % (tag, rule, f, cm(pre_n), cm(nbig), cm(len(sz)), cm(kept - dup), cm(pre_p),
               fstat[f]["pin_retention"]))

    sz = np.concatenate(szs)
    eptr = np.zeros(len(sz) + 1, np.int64)
    eptr[1:] = np.cumsum(sz)
    eidx = np.concatenate(parts)
    assert eidx.min() >= 0 and eidx.max() < N and len(eidx) == eptr[-1]
    w = np.maximum(1, np.rint(HB.WSCALE / (sz - 1)).astype(np.int64))
    arrays = dict(eptr=eptr, eidx=eidx.astype(np.int32), N=np.array([N]), k=np.array([k]))
    if R["w"]:
        arrays["ew"] = w.astype(np.int32)
    pre = sum(v["pins_precap"] for v in fstat.values())
    dup = sum(v["anchor_duplication_pins"] for v in fstat.values())
    meta = {"rule": rule, "famset": famset, "families": list(fams), "N": N, "k": k,
            "k_rule": "max(1, N // %d)" % TARGET, "target_block_size": tbs, "cap": cap, "cap_rule": HB.RULES[rule]["cap"],
            "mode": R["mode"], "weighted": R["w"], "weight_scale": HB.WSCALE,
            "hyperedges": int(len(sz)), "pins": int(len(eidx)),
            "pins_precap_total": pre, "anchor_duplication_pins": dup,
            "pin_retention_total": round((int(len(eidx)) - dup) / max(pre, 1), 6),
            "size_min": int(sz.min()), "size_max": int(sz.max()),
            "size_mean": round(float(sz.mean()), 2),
            "weight_min": int(w.min()), "weight_max": int(w.max()),
            "weight_sum": int(w.sum()), "per_family": fstat}
    return arrays, meta


def arrays_digest(arrays):
    """order-independent digest of the hypergraph content (dtype/shape/bytes of each array)."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(("%s:%s:%s\n" % (key, a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------- parity
LEGACY_KEYS = os.path.join(REPO, "scratchpad", "_l1kn", "keys_%s.npz")
LEGACY_GRAPH = os.path.join(REPO, "scratchpad", "_l1hu", "graphs", "%s__H4_SPLIT_PRESERVE__SK.npz")
LEGACY_DS = ["metaqa", "musique_clean", "squad_clean", "2wiki_clean", "hotpotqa_clean", "webqsp"]


def parity(names):
    """Rebuild each legacy H4_SPLIT_PRESERVE__SK hypergraph from the legacy key sets (as
    _l1hu_build.famkeys read them) with the injected builder and compare with the legacy file."""
    out = {}
    for ds in names:
        kp, gp = LEGACY_KEYS % ds, LEGACY_GRAPH % ds
        if not (os.path.exists(kp) and os.path.exists(gp)):
            out[ds] = {"status": "SKIPPED", "reason": "legacy inputs missing", "keys": kp, "graph": gp}
            log(ds, "skipped (legacy inputs missing)")
            continue
        z = np.load(kp)
        N = int(z["N"][0])
        keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
        g = np.load(gp)
        k = int(g["k"][0])
        assert int(g["N"][0]) == N, (ds, N, g["N"])
        k_rule = frozen_k(N)
        arrays, meta = build_hypergraph(N, keys, k, tag=ds)
        same = {key: bool(np.array_equal(arrays[key], g[key]) and arrays[key].dtype == g[key].dtype) for key in ("eptr", "eidx", "N", "k", "ew")}
        ok = all(same.values()) and set(g.files) == set(arrays)
        out[ds] = {"status": "IDENTICAL" if ok else "DIFFERENT", "arrays_equal": same, "legacy_files": sorted(g.files),
                   "N": N, "k_legacy": k, "k_frozen_rule": k_rule, "k_rule_matches_legacy": k == k_rule,
                   "hyperedges": meta["hyperedges"], "pins": meta["pins"], "cap": meta["cap"],
                   "legacy_sha256": sha_file(gp), "rebuilt_digest": arrays_digest(arrays)}
        log("%-15s %s  k %d (rule %d)  E %s  pins %s  cap %s" % (ds, out[ds]["status"], k, k_rule, cm(meta["hyperedges"]), cm(meta["pins"]), meta["cap"]))
    return out


# --------------------------------------------------------------------------------------- canonical
def build_canonical(name, log=log):
    d = CanonicalDataset(name)
    N, ST, KN, NX = d.keysets(log=log)
    keys = {"STRUCT": ST, "KNN": KN}
    k = frozen_k(N)
    t = time.time()
    arrays, meta = build_hypergraph(N, keys, k, tag=name)
    out_dir = os.path.join(d.derived_dir, "hypergraph")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "H4_SK.npz")
    np.savez_compressed(fp, **arrays)
    meta.update({"dataset": name, "file": os.path.relpath(fp, REPO).replace("\\", "/"), "bytes": os.path.getsize(fp),
                 "file_sha256": sha_file(fp), "content_digest": arrays_digest(arrays),
                 "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1),
                 "inputs": {"DATASET_json_RECORD_SHA256": d.record_sha, "keys_npz_sha256": sha_file(d._keys_path()),
                            "STRUCT": int(len(ST)), "KNN": int(len(KN)), "graph_files_sha256": d.pins()["graph_files_sha256"]},
                 "builder": {"file": "src/l1_canonical/hypergraph.py", "frozen_rule_module": "scratchpad/_l1hu_build.py",
                             "frozen_rule_sha256": sha_file(os.path.join(REPO, "scratchpad", "_l1hu_build.py"))}})
    with io.open(fp[:-4] + ".json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, indent=1))
    log("  %s: cap=%s  %s hyperedges  %s pins  retention %.4f  w[%d,%d]  %.1f MB -> %s"
        % (name, meta["cap"], cm(meta["hyperedges"]), cm(meta["pins"]), meta["pin_retention_total"],
           meta["weight_min"], meta["weight_max"], meta["bytes"] / 1e6, meta["file"]))
    return meta


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "parity":
        res = parity(a[1:] or LEGACY_DS)
        p = os.path.join(REPO, "results", "L1_CANONICAL", "HYPERGRAPH_PARITY.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"RECORD": "L1_CANONICAL_HYPERGRAPH_PARITY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "what": "src/l1_canonical/hypergraph.build_hypergraph run on the LEGACY key sets reproduces the legacy H4_SPLIT_PRESERVE__SK artefacts",
                                "frozen_rule_sha256": sha_file(os.path.join(REPO, "scratchpad", "_l1hu_build.py")),
                                "ALL_IDENTICAL": all(v.get("status") == "IDENTICAL" for v in res.values()), "datasets": res}, indent=1))
        log("wrote", p)
    elif a and a[0] == "build":
        for name in a[1:]:
            build_canonical(name)
    else:
        print(__doc__)
