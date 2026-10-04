# -*- coding: utf-8 -*-
"""
VERIFY THE FREEBASE TREE
========================
Treats every artifact as invalid until counts and checksums establish otherwise.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/verify_freebase.py           # structure, counts, small-file digests, sampled CSR/name checks
    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/verify_freebase.py --full    # + every pinned byte hashed, every node shard read,
                                                                                        #   every CSR block order-checked, in/out degrees reconciled
Also callable from verify_canonical.py (verify_freebase(freeze, full, rep)), which is where
the result is written; standalone it writes data/final_canonical/freebase/VERIFICATION.json.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical"))
from freeze_canonical import FC, rj, sha_file, record_hash  # noqa: E402

FB = FC + "/freebase"
SMALL = 100 * 1024 * 1024
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
NAME_KINDS = ["ORIGINAL", "RECOVERED_ORIGINAL", "RECOVERED_EXTERNAL", "URI_SELF", "KEY_SEMANTIC", "INFERRED", "GENERATED_FLOOR"]
BARE_MID_RE = r"^[mg]\.[0-9a-z_]+$"
PLACEHOLDER_RE = r"^(Unnamed\b|Unknown\b|\s*$)"
CH = 100_000_000


def pinned(rep, entry, full, label):
    p = entry["file"]
    if not os.path.exists(p):
        rep.add(label, False, "missing %s" % p)
        return False
    b = os.path.getsize(p)
    if b != entry["bytes"]:
        rep.add(label, False, "%s bytes %d != pinned %d" % (p, b, entry["bytes"]))
        return False
    if "sha256" in entry and (full or b <= SMALL):
        h = sha_file(p)
        rep.add(label, h == entry["sha256"], None if h == entry["sha256"] else "%s sha drift" % p)
        return h == entry["sha256"]
    rep.add(label + " (bytes only)", True)
    return True


def walk_pins(rep, obj, full, label):
    """Every dict carrying file/bytes(/sha256) anywhere under obj is a pin."""
    if isinstance(obj, dict):
        if "file" in obj and "bytes" in obj:
            pinned(rep, obj, full, "%s %s" % (label, os.path.relpath(obj["file"], FB).replace(os.sep, "/")))
            return
        for k, v in obj.items():
            walk_pins(rep, v, full, label)
    elif isinstance(obj, list):
        for v in obj:
            walk_pins(rep, v, full, label)


def csr_block_check(indptr, other, rel, lo, hi, label, rep, n_nodes, n_rel, flag=None):
    """Blocks lo..hi: rows in range, (rel, other) ascending within every block."""
    a, b = int(indptr[lo]), int(indptr[hi])
    o = np.asarray(other[a:b]).astype(np.int64)
    r = np.asarray(rel[a:b]).astype(np.int64)
    ok = bool(o.size == 0 or (o.min() >= 0 and o.max() < n_nodes and r.min() >= 0 and r.max() < n_rel))
    if flag is not None:
        f = np.asarray(flag[a:b])
        ok = ok and bool(((f == 0) | (f == 1)).all())
    key = r * n_nodes + o
    if key.size < 2:
        return ok
    starts = (indptr[lo + 1:hi] - a).astype(np.int64)  # block boundaries inside the slice
    starts = starts[starts < key.size]                  # empty trailing blocks start at key.size
    inner = np.ones(key.size, dtype=bool)
    inner[0] = False
    inner[starts] = False  # comparisons across a boundary are not constrained
    d = np.diff(key)
    ordered = bool((d[inner[1:]] >= 0).all())
    return ok and ordered


def verify_freebase(freeze, full, rep):
    """freeze: the CANONICAL_FREEZE record (or None when run standalone). Adds checks to rep."""
    t0 = time.time()
    dp = FB + "/DATASET.json"
    if not os.path.exists(dp):
        rep.add("freebase DATASET.json present", False)
        return
    R = rj(dp)
    rep.add("freebase DATASET.json self-hash", record_hash(R) == R["RECORD_SHA256"] and record_hash(R, crlf=False) == R["RECORD_SHA256_LF"])
    if freeze is not None and "FREEBASE" in freeze:
        rep.add("freebase DATASET.json pinned by the freeze", sha_file(dp) == freeze["FREEBASE"]["DATASET.json"]["sha256"])
    N, E = int(R["n_nodes"]), int(R["n_edges"])
    # ---- every pinned file: bytes, sha (full or small) ---------------------------------------------------
    for section in ("nodes", "graph", "relations", "metadata", "bridge", "records", "names"):
        walk_pins(rep, R.get(section), full, "freebase pin")
    served = set()
    for r_, _, fs in os.walk(FB):
        for x in fs:
            served.add(os.path.join(r_, x).replace(os.sep, "/"))
    pins = set()

    def collect(o):
        if isinstance(o, dict):
            if "file" in o and "bytes" in o:
                pins.add(o["file"].replace(os.sep, "/"))
            else:
                for v in o.values():
                    collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
    collect(R)
    extra = sorted(served - pins - {FB + "/DATASET.json", FB + "/VERIFICATION.json"})
    rep.add("freebase no unpinned files in the tree", not extra, extra[:10])
    # ---- positions --------------------------------------------------------------------------------------
    uid = np.load(FB + "/nodes/node_uid.npy", mmap_mode="r")
    kind = np.load(FB + "/nodes/kind.npy", mmap_mode="r")
    nk = np.load(FB + "/nodes/name_kind.npy", mmap_mode="r")
    rep.add("freebase index arrays are N long", uid.shape == kind.shape == nk.shape == (N,), (uid.shape, kind.shape, nk.shape))
    inc = True
    prev = None
    for s in range(0, N, 20_000_000):
        u = np.asarray(uid[s:min(s + 20_000_000, N)])
        if prev is not None and not (u[0] > prev):
            inc = False
        if not (np.diff(u) > 0).all():
            inc = False
        prev = u[-1]
    rep.add("freebase node_uid strictly increasing (position == uid rank)", inc)
    kc = np.bincount(np.asarray(kind), minlength=len(KINDS))
    rep.add("freebase kind census == DATASET.json", {KINDS[i]: int(c) for i, c in enumerate(kc)} == R["nodes"]["by_kind"] and kc.sum() == N)
    nkc = np.bincount(np.asarray(nk), minlength=len(NAME_KINDS))
    rep.add("freebase name_kind census == DATASET.json", {NAME_KINDS[i]: int(c) for i, c in enumerate(nkc)} == R["names"]["census"] and nkc.sum() == N)
    # ---- node shards ----------------------------------------------------------------------------------------
    shards = R["nodes"]["shards"]
    which = range(len(shards)) if full else sorted(set([0, len(shards) // 2, len(shards) - 1]))
    bad = []
    census = np.zeros(len(NAME_KINDS), dtype=np.int64)
    rows_seen = 0
    for k in which:
        p = shards[k]["file"]
        if not os.path.exists(p):
            bad.append(p)
            continue
        pf = pq.ParquetFile(p)
        expected = 4_000_000 * k
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["position", "node_uid", "kind", "name", "name_kind", "floor_reason"])
            pos = t["position"].to_numpy()
            a = int(pos[0])
            ok = bool(a == expected and np.array_equal(pos, np.arange(a, a + t.num_rows, dtype=np.int32)))
            expected += t.num_rows
            ok = ok and np.array_equal(t["node_uid"].to_numpy(), np.asarray(uid[a:a + t.num_rows]))
            ok = ok and np.array_equal(t["kind"].to_numpy(), np.asarray(kind[a:a + t.num_rows]))
            tk = t["name_kind"].to_numpy()
            ok = ok and np.array_equal(tk, np.asarray(nk[a:a + t.num_rows]))
            names = t["name"]
            empty = pc.sum(pc.or_kleene(pc.is_null(names), pc.equal(names, ""))).as_py() or 0
            ok = ok and empty == 0
            # floor rows carry a reason, non-floor rows none
            fr = t["floor_reason"].to_numpy()
            ok = ok and bool(((tk == 6) == (fr > 0)).all())
            # generated floor names are never a bare MID or a placeholder
            gen = names.filter(pc.equal(t["name_kind"], 6))
            if len(gen):
                hit = pc.or_(pc.match_substring_regex(gen, BARE_MID_RE), pc.match_substring_regex(gen, PLACEHOLDER_RE))
                ok = ok and (pc.sum(hit).as_py() or 0) == 0
            census += np.bincount(tk, minlength=len(NAME_KINDS))
            if not ok:
                bad.append("%s rg %d" % (p, rg))
            rows_seen += t.num_rows
        if k % 10 == 0:
            print("  freebase shard %d checked (%.0fs)" % (k, time.time() - t0))
            sys.stdout.flush()
    rep.add("freebase node shards: positions contiguous, uid/kind/name_kind match the index, no empty names, no placeholder/bare-MID floor names (%s)"
            % ("all %d" % len(shards) if full else "sampled 3"), not bad, bad[:5])
    if full:
        rep.add("freebase name census recomputed from the shards", rows_seen == N and np.array_equal(census, nkc))
    # ---- relations --------------------------------------------------------------------------------------
    rel = pq.read_table(FB + "/relations/relations.parquet")
    n_rel = rel.num_rows
    rid = rel["rel_id"].to_numpy()
    rev = rel["reverse_rel_id"].to_numpy()
    names = rel["relation"].to_pylist()
    cnt = rel["canonical_edge_count"].to_numpy().astype(np.int64)
    ok = bool(np.array_equal(rid, np.arange(n_rel)) and names == sorted(names) and len(set(names)) == n_rel and cnt.sum() == E
              and ((rev == -1) | ((rev >= 0) & (rev < n_rel))).all() and n_rel == R["n_relations"])
    rep.add("freebase relations: rel_id == row, sorted unique strings, reverse ids in range, edge counts sum to E", ok)
    # ---- graph CSR --------------------------------------------------------------------------------------
    oi = np.load(FB + "/graph/out_indptr.npy", mmap_mode="r")
    ii = np.load(FB + "/graph/in_indptr.npy", mmap_mode="r")
    od = np.load(FB + "/graph/out_dst.npy", mmap_mode="r")
    orl = np.load(FB + "/graph/out_rel.npy", mmap_mode="r")
    ofl = np.load(FB + "/graph/out_flag.npy", mmap_mode="r")
    isr = np.load(FB + "/graph/in_src.npy", mmap_mode="r")
    irl = np.load(FB + "/graph/in_rel.npy", mmap_mode="r")
    rep.add("freebase CSR shapes", oi.shape == ii.shape == (N + 1,) and od.shape == orl.shape == ofl.shape == isr.shape == irl.shape == (E,))
    mono = True
    for arr in (oi, ii):
        if int(arr[0]) != 0 or int(arr[-1]) != E:
            mono = False
        for s in range(0, N + 1, 50_000_000):
            a = np.asarray(arr[s:min(s + 50_000_001, N + 1)])
            if not (np.diff(a) >= 0).all():
                mono = False
    rep.add("freebase indptr: start 0, end E, non-decreasing (out and in)", mono)
    outdeg = np.diff(np.asarray(oi)).astype(np.int32)   # max degree < 2**31; int32 halves the resident size (N = 302M)
    indeg = np.diff(np.asarray(ii)).astype(np.int32)
    rep.add("freebase degree records", int(outdeg.max()) == R["graph"]["records"]["out_degree"]["max"] and int(indeg.max()) == R["graph"]["records"]["in_degree"]["max"])
    if full:
        # in-degree from the out-CSR == in-CSR degrees; rel census from the out-CSR == relations; flag census == record
        # one CSR side at a time: the resident set is one int32[N] counter plus one int64[N] bincount temporary
        cnt_dst = np.zeros(N, dtype=np.int32)
        cnt_rel = np.zeros(n_rel, dtype=np.int64)
        flipped = 0
        for s in range(0, E, CH):
            e = min(s + CH, E)
            np.add(cnt_dst, np.bincount(np.asarray(od[s:e]), minlength=N), out=cnt_dst, casting="unsafe")
            cnt_rel += np.bincount(np.asarray(orl[s:e]), minlength=n_rel)
            flipped += int(np.count_nonzero(np.asarray(ofl[s:e])))
            print("  freebase out-CSR census %d / %d (%.0fs)" % (e, E, time.time() - t0))
            sys.stdout.flush()
        rep.add("freebase in-degree from out_dst == in_indptr degrees", np.array_equal(cnt_dst, indeg))
        rep.add("freebase relation census from out_rel == relations.canonical_edge_count", np.array_equal(cnt_rel, cnt))
        rep.add("freebase flipped census == DATASET.json", flipped == R["graph"]["flipped_edges"])
        del cnt_dst
        cnt_src = np.zeros(N, dtype=np.int32)
        cnt_rel_in = np.zeros(n_rel, dtype=np.int64)
        for s in range(0, E, CH):
            e = min(s + CH, E)
            np.add(cnt_src, np.bincount(np.asarray(isr[s:e]), minlength=N), out=cnt_src, casting="unsafe")
            cnt_rel_in += np.bincount(np.asarray(irl[s:e]), minlength=n_rel)
            print("  freebase in-CSR census %d / %d (%.0fs)" % (e, E, time.time() - t0))
            sys.stdout.flush()
        rep.add("freebase out-degree from in_src == out_indptr degrees", np.array_equal(cnt_src, outdeg))
        rep.add("freebase relation census from in_rel == relations.canonical_edge_count", np.array_equal(cnt_rel_in, cnt))
        del cnt_src
    # block order: every block (full) or a spread of blocks incl. the hubs
    if full:
        blocks = [(s, min(s + 2_000_000, N)) for s in range(0, N, 2_000_000)]
    else:
        hubs = [int(np.argmax(outdeg)), int(np.argmax(indeg))]
        blocks = [(s, min(s + 200_000, N)) for s in range(0, N, N // 12)] + [(h, h + 1) for h in hubs]
    ok_out = ok_in = True
    for bi, (lo, hi) in enumerate(blocks):
        ok_out = ok_out and csr_block_check(oi, od, orl, lo, hi, "out", rep, N, n_rel, ofl)
        ok_in = ok_in and csr_block_check(ii, isr, irl, lo, hi, "in", rep, N, n_rel)
        if full and bi % 20 == 0:
            print("  freebase CSR blocks %d / %d (%.0fs)" % (bi, len(blocks), time.time() - t0))
            sys.stdout.flush()
    rep.add("freebase out-CSR: endpoints/rel ids in range, flag in {0,1}, (rel, dst) ascending per node (%s)" % ("every block" if full else "sampled"), ok_out)
    rep.add("freebase in-CSR: endpoints/rel ids in range, (rel, src) ascending per node (%s)" % ("every block" if full else "sampled"), ok_in)
    # transposition spot check: out-edges of the nodes with the largest out-degree (bounded) appear in the in-CSR
    rng = np.random.RandomState(0)
    sample = rng.choice(N, 200, replace=False)
    ok_t = True
    for u in sample:
        a, b = int(oi[u]), int(oi[u + 1])
        if b - a > 2000:
            b = a + 2000
        for j in range(a, b):
            v, r = int(od[j]), int(orl[j])
            ia, ib = int(ii[v]), int(ii[v + 1])
            rr = np.asarray(irl[ia:ib]); ss = np.asarray(isr[ia:ib])
            if not ((rr == r) & (ss == u)).any():
                ok_t = False
                break
        if not ok_t:
            break
    rep.add("freebase out-edges of 200 random nodes found in the in-CSR", ok_t)
    del od, orl, ofl, isr, irl
    # ---- metadata ---------------------------------------------------------------------------------------
    for t, e in R["metadata"]["tables"].items():
        p = e["file"]
        if not os.path.exists(p):
            continue
        pf = pq.ParquetFile(p)
        ok = pf.metadata.num_rows == e["rows"] and pf.schema_arrow.names == e["columns"] and pf.schema_arrow.names[0] == "position"
        if ok and pf.metadata.num_rows:
            last = -1
            groups = range(pf.num_row_groups) if full else [0, pf.num_row_groups - 1]
            for rg in groups:
                pos = pf.read_row_group(rg, columns=["position"])["position"].to_numpy()
                if pos.min() < 0 or pos.max() >= N or not (np.diff(pos) >= 0).all() or (full and pos[0] < last):
                    ok = False
                    break
                last = pos[-1]
        rep.add("freebase metadata/%s rows/columns/position order%s" % (t, "" if full else " (sampled)"), ok)
    ti = np.load(FB + "/metadata/type_indptr.npy", mmap_mode="r")
    tv = np.load(FB + "/metadata/type_val.npy", mmap_mode="r")
    td = pq.read_table(FB + "/metadata/type_dict.parquet")
    trows = R["metadata"]["tables"]["type"]["rows"]
    ok = ti.shape == (N + 1,) and tv.shape == (trows,) and int(ti[0]) == 0 and int(ti[-1]) == trows and td.num_rows > 0
    ok = ok and np.array_equal(td["type_id"].to_numpy(), np.arange(td.num_rows))
    mx = 0
    for s in range(0, trows, CH):
        v = np.asarray(tv[s:min(s + CH, trows)])
        mx = max(mx, int(v.max())) if v.size else mx
    ok = ok and mx < td.num_rows
    rep.add("freebase type CSR: shapes, ends, type ids in the dictionary", ok)
    # ---- bridge -----------------------------------------------------------------------------------------
    wp = np.load(FB + "/bridge/webqsp_positions.npy")
    si = pq.read_table(FB + "/bridge/schema_index.parquet")
    n_w = R["bridge"]["build"]["webqsp_nodes"]
    ok = wp.shape == (n_w,) and int(wp.max()) < N and int(wp.min()) >= -1 and int((wp >= 0).sum()) == R["bridge"]["build"]["resolved_to_freebase_position"]
    sp = si["position"].to_numpy()
    ok = ok and bool((sp >= 0).all() and (sp < N).all()) and set(np.unique(np.asarray(kind)[sp]).tolist()) <= {2, 3, 4}
    rep.add("freebase bridge: shapes, ranges, schema_index points at SCHEMA_* nodes", ok)
    if os.path.exists(FC + "/webqsp/nodes.jsonl") and os.environ.get("PYTHONHASHSEED") == "0":
        okb = True
        n_checked = 0
        with io.open(FC + "/webqsp/nodes.jsonl", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 20000:
                    break
                mid = json.loads(line).get("source_mid")
                if mid is None:
                    okb = okb and wp[i] == -1
                    continue
                h = hash(mid.encode("utf-8"))
                p = int(np.searchsorted(uid, h))
                found = p < N and int(uid[p]) == h
                okb = okb and ((wp[i] == p) if found else (wp[i] == -1))
                n_checked += 1
        rep.add("freebase bridge re-derived for the first 20,000 webqsp nodes (%d with a MID)" % n_checked, okb)
    else:
        rep.add("freebase bridge re-derivation (skipped: PYTHONHASHSEED != 0 or no webqsp tree)", True)
    # ---- loader smoke -------------------------------------------------------------------------------------
    try:
        sys.path.insert(0, FC)
        from canonical import Freebase
        F = Freebase()
        p0 = int(np.argmax(outdeg > 0))
        dst, rl, fl = F.out_edges(p0)
        row = F.node(p0)
        nm = F.name(p0)
        rid_ = F.rel_id(names[0])
        hp = F.position(int(uid[p0]))
        ok = dst.size > 0 and dst.size == outdeg[p0] and isinstance(nm, str) and nm and row["position"] == p0 and rid_ == 0 and hp == p0
        if os.environ.get("PYTHONHASHSEED") == "0":
            ok = ok and F.position(row["node_id"]) == p0 if row["kind"] != 6 else ok
        rep.add("freebase loader smoke", ok)
    except Exception as ex:  # noqa: BLE001
        rep.add("freebase loader smoke", False, repr(ex))
    print("  freebase verified in %.0fs" % (time.time() - t0))
    sys.stdout.flush()


def main():
    from verify_canonical import Report, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rep = Report()
    freeze = rj(OUT) if os.path.exists(OUT) else None
    verify_freebase(freeze, a.full, rep)
    res = {"RECORD": "VERIFICATION", "dataset": "freebase", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "mode": "full" if a.full else "quick", "checks": len(rep.checks), "failed": rep.fail, "PASS": rep.fail == 0,
           "seconds": round(time.time() - t0, 1), "results": rep.checks}
    with io.open(FB + "/VERIFICATION.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("%s  %d checks, %d failed, %.0fs  -> %s" % ("PASS" if rep.fail == 0 else "FAIL", len(rep.checks), rep.fail, time.time() - t0, FB + "/VERIFICATION.json"))
    sys.exit(0 if rep.fail == 0 else 1)


if __name__ == "__main__":
    main()
