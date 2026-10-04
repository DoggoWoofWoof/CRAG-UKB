"""H4_SK.npz -> hMETIS .hgr (weighted nets, canonical hyperedge order) + the representation-independent
structure digest every other representation (hgr, monolithic net-list, H4_SK_STREAM_V1 shards) must reproduce.

STRUCTURE DIGEST (H4_SK_STRUCTURE_V1) = sha256 over the byte stream

    H4_SK_STRUCTURE_V1\\n
    N=<nodes> M=<hyperedges> P=<pins>\\n
    <e> <w_e> <p_1> <p_2> ... <p_|e|>\\n        for e = 0 .. M-1 in canonical hyperedge order, pins = canonical
                                               0-based positions sorted ascending (the incidence multiset of e)
    NODE_WEIGHTS unit N=<nodes>\\n              (vertex weights are unit by contract; fmt 1 files carry none)

so two representations are structurally identical iff their digests agree: same node count, same hyperedge count
and order, same pin count, same hyperedge weights, same incidence multisets, unit node weights.

    python src/l1_lowmem/hgr.py write <ds> [ds ...]     -> data/l1_lowmem/<ds>/H4_SK.hgr (+ .json with ORIGINAL_STRUCTURE_SHA256)
    python src/l1_lowmem/hgr.py check <ds> [ds ...]     re-derive the digest from the .hgr bytes and compare with the npz digest
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, ds_dir, log, pin, rj, sha_file, wj  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset  # noqa: E402

HGR_NAME = "H4_SK.hgr"


def load(ds):
    """the canonical H4_SK arrays, verified against their manifest (file sha + content digest) and DATASET.json."""
    d = CanonicalDataset(ds)
    npz = os.path.join(d.derived_dir, "hypergraph", "H4_SK.npz")
    meta = rj(npz[:-4] + ".json")
    if meta is None:
        raise RuntimeError("%s: no canonical hypergraph manifest" % ds)
    if meta["inputs"]["DATASET_json_RECORD_SHA256"] != d.record_sha:
        raise RuntimeError("%s: hypergraph built from another DATASET.json record" % ds)
    fsha = sha_file(npz)
    if fsha != meta["file_sha256"]:
        raise RuntimeError("%s: H4_SK.npz digest changed since its manifest" % ds)
    z = np.load(npz)
    eptr, eidx, ew = z["eptr"].astype(np.int64), z["eidx"].astype(np.int64), z["ew"].astype(np.int64)
    N, k = int(z["N"][0]), int(z["k"][0])
    if not (len(eptr) - 1 == meta["hyperedges"] and len(eidx) == meta["pins"] and N == meta["N"] == d.n_nodes and k == meta["k"]):
        raise RuntimeError("%s: npz counts disagree with the manifest" % ds)
    return {"d": d, "npz": npz, "npz_sha256": fsha, "meta": meta, "eptr": eptr, "eidx": eidx, "ew": ew, "N": N, "k": k,
            "M": int(len(eptr) - 1), "P": int(len(eidx))}


def digest_stream(N, M, P, nets_iter):
    """nets_iter yields (e, w, sorted_pins ndarray) in hyperedge order; returns the H4_SK_STRUCTURE_V1 sha256."""
    h = hashlib.sha256()
    h.update(b"H4_SK_STRUCTURE_V1\n")
    h.update(("N=%d M=%d P=%d\n" % (N, M, P)).encode())
    buf, n = [], 0
    for e, w, pins in nets_iter:
        buf.append("%d %d %s\n" % (e, w, " ".join(map(str, pins.tolist()))))
        n += 1
        if n % 20000 == 0:
            h.update("".join(buf).encode()); buf = []
    h.update("".join(buf).encode())
    h.update(("NODE_WEIGHTS unit N=%d\n" % N).encode())
    return h.hexdigest()


def structure_digest(eptr, eidx, ew, N):
    """digest straight from CSR arrays (pins sorted within each hyperedge)."""
    M, P = len(eptr) - 1, len(eidx)
    sizes = np.diff(eptr)
    hid = np.repeat(np.arange(M, dtype=np.int64), sizes)
    order = np.argsort(hid * np.int64(N) + eidx, kind="stable")   # (hyperedge, position) ascending
    sp = eidx[order]
    dup = int((np.diff(hid * np.int64(N) + sp) == 0).sum())      # a pin repeated inside one hyperedge
    if dup:
        raise RuntimeError("duplicate pin inside a hyperedge (%d) -- H4_SK never produces that" % dup)

    def it():
        for e in range(M):
            yield e, int(ew[e]), sp[eptr[e]:eptr[e + 1]]
    return digest_stream(N, M, P, it()), {"N": int(N), "M": int(M), "P": int(P), "weight_sum": int(ew.sum()),
                                          "weight_min": int(ew.min()), "weight_max": int(ew.max()),
                                          "size_min": int(sizes.min()), "size_max": int(sizes.max()), "node_weights": "unit",
                                          "isolated_nodes": int(N - len(np.unique(eidx)))}


def hgr_path(ds):
    return os.path.join(ds_dir(ds), HGR_NAME)


def write_hgr(ds):
    """hMETIS: header 'M N 1' (weighted nets, unit vertices); net e on line e+1 as '<w> <pin+1> ...' in the npz pin order."""
    t = time.time()
    H = load(ds)
    eptr, eidx, ew, N, M, P = H["eptr"], H["eidx"], H["ew"], H["N"], H["M"], H["P"]
    dig, counts = structure_digest(eptr, eidx, ew, N)
    out = hgr_path(ds)
    tmp = out + ".tmp"
    pins1 = eidx + 1
    with io.open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write("%d %d 1\n" % (M, N))
        buf = []
        for e in range(M):
            buf.append("%d %s\n" % (ew[e], " ".join(map(str, pins1[eptr[e]:eptr[e + 1]].tolist()))))
            if len(buf) >= 20000:
                f.write("".join(buf)); buf = []
        f.write("".join(buf))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    rec = {"RECORD": "H4_SK_HGR", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "format": "hMETIS, header 'M N 1' (net weights, unit vertex weights), 1-based pins, canonical hyperedge order, npz pin order",
           "source": {"npz": pin(H["npz"]), "content_digest": H["meta"]["content_digest"], "DATASET_json_RECORD_SHA256": H["d"].record_sha,
                      "nodes_jsonl_sha256(node_order)": H["d"].manifest["nodes"]["sha256"], "k": H["k"], "rule": H["meta"]["rule"],
                      "families": H["meta"]["families"], "famset": H["meta"]["famset"]},
           "ORIGINAL_STRUCTURE_SHA256": dig, "counts": counts, "hgr": pin(out), "seconds": round(time.time() - t, 1)}
    wj(out + ".json", rec)
    log("%-8s hgr written: N %d M %d P %d  %.1f MB  ORIGINAL_STRUCTURE %s  (%.1fs)" % (ds, N, M, P, rec["hgr"]["bytes"] / 1e6, dig[:16], time.time() - t))
    return rec


def iter_hgr(path):
    """stream a weighted hMETIS file: yields header (M, N, fmt) first, then (e, w, pins0 ndarray) per net."""
    with io.open(path, "r", encoding="ascii") as f:
        head = f.readline().split()
        M, N = int(head[0]), int(head[1])
        fmt = int(head[2]) if len(head) > 2 else 0
        yield ("HEADER", M, N, fmt)
        e = 0
        for line in f:
            if line.startswith("%"):
                continue
            tok = line.split()
            if fmt in (1, 11):
                w, pins = int(tok[0]), np.array(tok[1:], dtype=np.int64) - 1
            else:
                w, pins = 1, np.array(tok, dtype=np.int64) - 1
            yield (e, w, pins)
            e += 1
            if e == M:
                break


def digest_hgr(path):
    """re-derive H4_SK_STRUCTURE_V1 from the .hgr bytes (independent of the npz arrays)."""
    # the digest header needs P before the nets stream -> pass 1 counts pins, pass 2 digests (both streaming)
    P_total, M, N, fmt = 0, None, None, None
    for i, item in enumerate(iter_hgr(path)):
        if i == 0:
            _, M, N, fmt = item
        else:
            P_total += len(item[2])
    if fmt != 1:
        raise RuntimeError("expected fmt 1 (weighted nets, unit vertices), got %s" % fmt)
    g = iter_hgr(path)
    next(g)
    ws = []

    def it():
        for e, w, pins in g:
            ws.append(w)
            yield e, w, np.sort(pins)
    dig = digest_stream(N, M, P_total, it())
    return dig, {"N": N, "M": M, "P": P_total, "weight_sum": int(sum(ws)), "weight_min": int(min(ws)), "weight_max": int(max(ws))}


def digest_netl_lines(lines_iter, N, M):
    """H4_SK_STRUCTURE_V1 from node-centric net-list lines (fmt 1: 'net w net w ...', 1-based nets), streamed in
    canonical position order.  Memory O(P) here (pin pairs); also checks that every occurrence of a net carries one weight."""
    nodes, nets, ws = [], [], []
    n = 0
    for line in lines_iter:
        tok = line.split()
        if len(tok) % 2:
            raise RuntimeError("node %d: odd token count in a fmt-1 net-list line" % n)
        a = np.array(tok, dtype=np.int64).reshape(-1, 2) if tok else np.zeros((0, 2), np.int64)
        nets.append(a[:, 0] - 1); ws.append(a[:, 1]); nodes.append(np.full(len(a), n, np.int64))
        n += 1
    if n != N:
        raise RuntimeError("net-list has %d node lines, expected N=%d" % (n, N))
    nets, ws, nodes = (np.concatenate(x) if x else np.zeros(0, np.int64) for x in (nets, ws, nodes))
    P = int(len(nets))
    if P and (nets.min() < 0 or nets.max() >= M):
        raise RuntimeError("net id outside [1, M]")
    order = np.argsort(nets * np.int64(N) + nodes, kind="stable")
    nets, ws, nodes = nets[order], ws[order], nodes[order]
    cnt = np.bincount(nets, minlength=M)
    ptr = np.zeros(M + 1, np.int64); ptr[1:] = np.cumsum(cnt)
    wnet = np.zeros(M, np.int64)
    for e in range(M):
        seg = ws[ptr[e]:ptr[e + 1]]
        if len(seg) == 0:
            raise RuntimeError("net %d has no pins in the net-list (nets with zero pins cannot be represented node-centrically)" % e)
        if (seg != seg[0]).any():
            raise RuntimeError("net %d carries inconsistent weights across its pins" % e)
        wnet[e] = seg[0]

    def it():
        for e in range(M):
            yield e, int(wnet[e]), nodes[ptr[e]:ptr[e + 1]]
    return digest_stream(N, M, P, it()), {"N": N, "M": M, "P": P, "weight_sum": int(wnet.sum()), "weight_min": int(wnet.min()),
                                          "weight_max": int(wnet.max())}


def check(ds):
    rec = rj(hgr_path(ds) + ".json")
    if rec is None:
        raise RuntimeError("%s: write the hgr first" % ds)
    if sha_file(hgr_path(ds)) != rec["hgr"]["sha256"]:
        raise RuntimeError("%s: hgr bytes changed since its record" % ds)
    t = time.time()
    dig, counts = digest_hgr(hgr_path(ds))
    ok = dig == rec["ORIGINAL_STRUCTURE_SHA256"]
    log("%-8s hgr re-derived digest %s  == ORIGINAL %s  counts %s  (%.1fs)" % (ds, dig[:16], ok, counts, time.time() - t))
    return ok, dig, counts


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    cmd, names = a[0], a[1:]
    for ds in names:
        if cmd == "write":
            write_hgr(ds)
        elif cmd == "check":
            check(ds)
        else:
            raise SystemExit("unknown command %s" % cmd)
