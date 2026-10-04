"""H4_SK_STREAM_V1 -- the canonical H4_SK hypergraph as an ordered sequence of node-centric shards.

The mathematical hypergraph is never split: every shard holds the FREIGHT net-list lines of one contiguous range
of canonical positions [first, last], nets keep their GLOBAL ids (1-based, canonical hyperedge order) and carry
their weight on every occurrence (fmt 1), exactly the bytes the official KaHIP/FREIGHT two-pass converter
(tools/hmetis_to_freight_stream.cpp) writes for those nodes: per node the nets in increasing net id, single
spaces, '\\n' line ends, no node weight.  Concatenating the shards in order behind the header line 'N M 1'
reproduces the monolithic net-list byte for byte (reconstruction gate 1); the structure digest re-derived from
the shards equals ORIGINAL_STRUCTURE_SHA256 (reconstruction gate 2).

Conversion is the official converter's two-pass scheme with the O(pins) array replaced by a rescan per shard:
  pass 1   stream the .hgr once: degree per node, weight per net, pin count      -> pass1.json (+ pass1.npz), hashed
  pass 2   per shard: stream the .hgr again, keep only the pins of the shard's positions, write the shard
           to shard_NNNNN.netl.tmp -> flush -> fsync -> sha256 -> rename -> shard_NNNNN.complete (json) ->
           manifest updated.  Memory: O(N + M + pins in the shard).  A restart regenerates only the shard
           without a .complete marker; completed shards are verified by sha256 and left untouched.

    python src/l1_lowmem/stream.py convert <ds> [ds ...] [--shard_nodes N]
    python src/l1_lowmem/stream.py gate <ds> [ds ...]          reconstruction gates (digest; bytes vs official netl if present)
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, ds_dir, log, pin, rj, sha_file, wj  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402

FORMAT = "H4_SK_STREAM_V1"
SHARD_NODES_LOCAL = 5000      # local validation: several shards per dataset (N = 20k-118k); the production size (100k-1M) is measured later
BLOCK_LINES = 20000


def stream_dir(ds):
    p = os.path.join(ds_dir(ds), "stream")
    os.makedirs(p, exist_ok=True)
    return p


def shard_name(i):
    return "shard_%05d.netl" % i


def _fsync_write_text(path, text):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _hgr_blocks(path):
    """stream the .hgr in blocks: yields (e0, sizes[int64], weights[int64], pins0[int64]) -- flat pins, per-net sizes."""
    with io.open(path, "r", encoding="ascii") as f:
        f.readline()
        e0, lines = 0, []
        for line in f:
            if line.startswith("%"):
                continue
            lines.append(line)
            if len(lines) >= BLOCK_LINES:
                yield (e0,) + _parse_block(lines)
                e0 += len(lines); lines = []
        if lines:
            yield (e0,) + _parse_block(lines)


def _parse_block(lines):
    parts = [ln.split() for ln in lines]
    ntok = np.fromiter(map(len, parts), np.int64, len(parts))            # tokens per line (weight + pins)
    tok = np.array([x for p in parts for x in p], dtype=np.int64)
    if len(tok) != int(ntok.sum()) or (ntok < 2).any():
        raise RuntimeError("malformed hgr block (token counts)")
    ptr = np.zeros(len(lines) + 1, np.int64); ptr[1:] = np.cumsum(ntok)
    w = tok[ptr[:-1]]
    keep = np.ones(len(tok), bool); keep[ptr[:-1]] = False
    return ntok - 1, w, tok[keep] - 1


def pass1(ds, hgr_rec):
    sd = stream_dir(ds)
    p1 = os.path.join(sd, "pass1.json")
    old = rj(p1)
    if old and old["hgr_sha256"] == hgr_rec["hgr"]["sha256"] and os.path.exists(os.path.join(sd, "pass1.npz")) \
            and sha_file(os.path.join(sd, "pass1.npz")) == old["pass1_npz_sha256"]:
        log("%-8s pass 1 already complete (record %s)" % (ds, old["pass1_sha256"][:16]))
        return old
    t = time.time()
    M, N = hgr_rec["counts"]["M"], hgr_rec["counts"]["N"]
    degree = np.zeros(N, np.int64)
    netw = np.zeros(M, np.int64)
    P = 0
    for e0, sizes, w, pins in _hgr_blocks(HG.hgr_path(ds)):
        netw[e0:e0 + len(sizes)] = w
        degree += np.bincount(pins, minlength=N)
        P += int(len(pins))
    if not (P == hgr_rec["counts"]["P"] and int(netw.sum()) == hgr_rec["counts"]["weight_sum"]):
        raise RuntimeError("%s: pass 1 counts disagree with the hgr record" % ds)
    npz = os.path.join(sd, "pass1.npz")
    tmp = npz + ".tmp.npz"
    np.savez(tmp, degree=degree, netw=netw)
    os.replace(tmp, npz)
    rec = {"RECORD": "H4_SK_STREAM_V1_PASS1", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hgr_sha256": hgr_rec["hgr"]["sha256"], "N": N, "M": M, "P": P, "weight_sum": int(netw.sum()),
           "isolated_nodes": int((degree == 0).sum()), "degree_max": int(degree.max()),
           "pass1_npz_sha256": sha_file(npz), "seconds": round(time.time() - t, 1)}
    rec["pass1_sha256"] = HG.hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    _fsync_write_text(p1, json.dumps(rec, indent=1))
    log("%-8s pass 1: degree/netw recorded, P %d, isolated %d  (%.1fs)" % (ds, P, rec["isolated_nodes"], time.time() - t))
    return rec


def _write_shard(ds, i, a, b, degree, netw, hgr_path):
    """shard i = positions [a, b): rescan the hgr, keep the pins of [a, b), write the node lines."""
    t = time.time()
    off = np.zeros(b - a + 1, np.int64); off[1:] = np.cumsum(degree[a:b])
    npins = int(off[-1])
    nets = np.empty(npins, np.int64)
    cur = np.zeros(b - a, np.int64)
    for e0, sizes, w, pins in _hgr_blocks(hgr_path):
        m = (pins >= a) & (pins < b)
        if not m.any():
            continue
        hid = np.repeat(np.arange(e0, e0 + len(sizes), dtype=np.int64), sizes)[m]
        loc = pins[m] - a
        # nets arrive in increasing id (file order); within this block, stable order by local node keeps that order
        o = np.argsort(loc, kind="stable")
        loc, hid = loc[o], hid[o]
        cnt = np.bincount(loc, minlength=b - a)
        start = off[:-1] + cur
        dst = np.repeat(start, cnt) + (np.arange(len(loc)) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        nets[dst] = hid
        cur += cnt
    if not (cur == degree[a:b]).all():
        raise RuntimeError("shard %d: pin counts disagree with pass 1" % i)
    sd = stream_dir(ds)
    out = os.path.join(sd, shard_name(i))
    tmp = out + ".tmp"
    with io.open(tmp, "w", encoding="ascii", newline="\n") as f:
        buf = []
        for j in range(b - a):
            seg = nets[off[j]:off[j + 1]]
            if len(seg):
                pairs = np.empty(2 * len(seg), np.int64)
                pairs[0::2] = seg + 1
                pairs[1::2] = netw[seg]
                buf.append(" ".join(map(str, pairs.tolist())) + "\n")
            else:
                buf.append("\n")
            if len(buf) >= 5000:
                f.write("".join(buf)); buf = []
        f.write("".join(buf))
        f.flush(); os.fsync(f.fileno())
    sha = sha_file(tmp)
    os.replace(tmp, out)
    rec = {"index": i, "file": shard_name(i), "first_position": a, "last_position": b - 1, "nodes": b - a, "pins": npins,
           "bytes": os.path.getsize(out), "sha256": sha, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t, 1)}
    _fsync_write_text(out[:-5] + ".complete", json.dumps(rec, indent=1))
    return rec


def manifest_path(ds):
    return os.path.join(stream_dir(ds), "%s.json" % FORMAT)


def convert(ds, shard_nodes=SHARD_NODES_LOCAL):
    t = time.time()
    hgr_rec = rj(HG.hgr_path(ds) + ".json")
    if hgr_rec is None:
        hgr_rec = HG.write_hgr(ds)
    if sha_file(HG.hgr_path(ds)) != hgr_rec["hgr"]["sha256"]:
        raise RuntimeError("%s: hgr bytes changed since its record" % ds)
    N, M, P = hgr_rec["counts"]["N"], hgr_rec["counts"]["M"], hgr_rec["counts"]["P"]
    sd = stream_dir(ds)
    man = rj(manifest_path(ds))
    if man and man.get("COMPLETE") and man["hgr_sha256"] == hgr_rec["hgr"]["sha256"] and man["shard_nodes"] == shard_nodes:
        log("%-8s stream already complete: %d shards, manifest %s" % (ds, man["shard_count"], man["STREAM_MANIFEST_SHA256"][:16]))
        return man
    p1 = pass1(ds, hgr_rec)
    z = np.load(os.path.join(sd, "pass1.npz"))
    degree, netw = z["degree"], z["netw"]
    bounds = [(a, min(a + shard_nodes, N)) for a in range(0, N, shard_nodes)]
    shards, regenerated, kept = [], 0, 0
    for i, (a, b) in enumerate(bounds):
        cp = os.path.join(sd, shard_name(i)[:-5] + ".complete")
        c = rj(cp)
        f = os.path.join(sd, shard_name(i))
        if c and c["first_position"] == a and c["last_position"] == b - 1 and os.path.exists(f) and sha_file(f) == c["sha256"]:
            shards.append(c); kept += 1
            continue
        for stale in (f, f + ".tmp", cp):
            if os.path.exists(stale):
                os.remove(stale)
        c = _write_shard(ds, i, a, b, degree, netw, HG.hgr_path(ds))
        shards.append(c); regenerated += 1
        log("%-8s shard %3d/%d [%d, %d]  %d pins  %.1f MB  %s  (%.1fs)" % (ds, i + 1, len(bounds), a, b - 1, c["pins"], c["bytes"] / 1e6, c["sha256"][:12], c["seconds"]))
    tot_pins = sum(c["pins"] for c in shards)
    if tot_pins != P or sum(c["nodes"] for c in shards) != N:
        raise RuntimeError("%s: shards cover %d nodes / %d pins, expected %d / %d" % (ds, sum(c["nodes"] for c in shards), tot_pins, N, P))
    d = HG.load(ds)["d"]
    man = {"RECORD": FORMAT, "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "COMPLETE": True,
           "header": "%d %d 1" % (N, M), "N": N, "M": M, "P": P, "families": hgr_rec["source"]["families"], "famset": hgr_rec["source"]["famset"],
           "canonical_dataset_hash(DATASET_json_RECORD_SHA256)": d.record_sha, "node_order_hash(nodes_jsonl_sha256)": d.manifest["nodes"]["sha256"],
           "H4_SK_source": {"npz": hgr_rec["source"]["npz"], "content_digest": hgr_rec["source"]["content_digest"]},
           "hgr_sha256": hgr_rec["hgr"]["sha256"], "ORIGINAL_STRUCTURE_SHA256": hgr_rec["ORIGINAL_STRUCTURE_SHA256"],
           "shard_nodes": shard_nodes, "shard_count": len(shards), "shards": shards, "ordered_shard_sha256": [c["sha256"] for c in shards],
           "pass1": {k: p1[k] for k in ("pass1_sha256", "pass1_npz_sha256", "P", "isolated_nodes", "degree_max")},
           "pass2_strategy": "rescan the hgr once per shard; memory O(N + M + pins of the shard); completed shards never rewritten",
           "line_format": "fmt 1 net-list line per node: '<net> <w> <net> <w> ...' nets 1-based global ids ascending, no node weight",
           "this_run": {"shards_regenerated": regenerated, "shards_kept": kept, "seconds": round(time.time() - t, 1)}}
    man["STREAM_MANIFEST_SHA256"] = HG.hashlib.sha256(json.dumps({k: v for k, v in man.items() if k not in ("utc", "this_run")},
                                                                 sort_keys=True).encode()).hexdigest()
    _fsync_write_text(manifest_path(ds), json.dumps(man, indent=1))
    log("%-8s %s complete: %d shards (%d regenerated, %d kept), manifest %s  (%.1fs)" % (ds, FORMAT, len(shards), regenerated, kept,
                                                                                         man["STREAM_MANIFEST_SHA256"][:16], time.time() - t))
    return man


def iter_shard_lines(ds, man=None):
    man = man or rj(manifest_path(ds))
    for c in man["shards"]:
        f = os.path.join(stream_dir(ds), c["file"])
        with io.open(f, "r", encoding="ascii") as fh:
            for line in fh:
                yield line


def gate(ds, official_netl=None):
    """reconstruction gates.  1: shards -> structure digest == ORIGINAL.  2 (when an official monolithic net-list is
    present, data/l1_lowmem/<ds>/H4_SK.official.netl): header + concatenated shard bytes == official bytes."""
    t = time.time()
    man = rj(manifest_path(ds))
    if not (man and man.get("COMPLETE")):
        raise RuntimeError("%s: convert first" % ds)
    for c in man["shards"]:
        f = os.path.join(stream_dir(ds), c["file"])
        if sha_file(f) != c["sha256"]:
            raise RuntimeError("%s: shard %d bytes changed since its .complete record" % (ds, c["index"]))
    dig, counts = HG.digest_netl_lines(iter_shard_lines(ds, man), man["N"], man["M"])
    g1 = dig == man["ORIGINAL_STRUCTURE_SHA256"]
    res = {"RECORD": "H4_SK_STREAM_V1_RECONSTRUCTION_GATE", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "STREAM_MANIFEST_SHA256": man["STREAM_MANIFEST_SHA256"], "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"],
           "STREAM_STRUCTURE_SHA256": dig, "counts_from_shards": counts, "gate_digest": "PASS" if g1 else "FAIL",
           "counts_exact": bool(counts["N"] == man["N"] and counts["M"] == man["M"] and counts["P"] == man["P"])}
    official_netl = official_netl or os.path.join(ds_dir(ds), "H4_SK.official.netl")
    if os.path.exists(official_netl):
        h = HG.hashlib.sha256()
        h.update((man["header"] + "\n").encode())
        for c in man["shards"]:
            with open(os.path.join(stream_dir(ds), c["file"]), "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
        res["official_netl"] = pin(official_netl)
        res["header_plus_shards_sha256"] = h.hexdigest()
        res["gate_bytes"] = "PASS" if h.hexdigest() == res["official_netl"]["sha256"] else "FAIL"
    else:
        res["gate_bytes"] = "PENDING (no official monolithic net-list yet: needs the FREIGHT toolchain)"
    res["seconds"] = round(time.time() - t, 1)
    wj(os.path.join(stream_dir(ds), "RECONSTRUCTION_GATE.json"), res)
    log("%-8s reconstruction: digest %s (%s == ORIGINAL %s)  bytes-vs-official %s  (%.1fs)" % (ds, res["gate_digest"], dig[:16], g1, res["gate_bytes"], res["seconds"]))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    cmd = a[0]
    shard_nodes = SHARD_NODES_LOCAL
    if "--shard_nodes" in a:
        i = a.index("--shard_nodes"); shard_nodes = int(a[i + 1]); del a[i:i + 2]
    for ds in a[1:]:
        if cmd == "convert":
            convert(ds, shard_nodes)
        elif cmd == "gate":
            gate(ds)
        else:
            raise SystemExit("unknown command %s" % cmd)
