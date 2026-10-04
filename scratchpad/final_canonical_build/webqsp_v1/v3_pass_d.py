"""PASS D: reverse-relation canonicalisation, and the pre-registered reverse-pair checks.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_pass_d.py
                            [--parts N] [--consume] [--workers N] [--fold-workers N]

Freebase records a large fraction of its object properties twice, once in each direction, and says
so itself: type.property.reverse_property declares 5,780 directed pairs, frozen in PASS A. Storing
both directions is not extra information, it is the same fact asserted twice; it doubles degree,
doubles hypergraph work and double-counts support at retrieval time. This pass stores each such
fact once and keeps the discarded direction as metadata, so nothing about the source becomes
unrecoverable.

DIRECTION IS FREEBASE'S, NOT OURS. The master side is canonical. That is not an alphabetical or
incidental choice: type.property.master_property independently names a primary side for 5,759 of
the pairs and agrees with the reverse_property direction on 5,759 of 5,759, zero disagreements.

WHAT CANNOT BE FLIPPED. Flipping turns an object into a subject, and a literal cannot be a subject.
A reverse-side relation carrying a literal object is left in its original orientation and counted,
never silently flipped into a malformed triple and never silently dropped.

TWO PATHS, BECAUSE ONLY A QUARTER OF THE GRAPH IS AT RISK. Measured from the relation histogram:
381,948,022 reverse-side edges and 127,018,375 master-side edges, together 23.2% of the 2.19e9. The
other 76.8% belong to relations in no reverse pair, so no flip can move them and no flipped edge can
land on top of them. They take a different and much cheaper route:

  AT RISK   -> flipped where required, then hash-partitioned on a fingerprint of (src, rel, dst) so
               every copy of a triple lands in one part, then deduplicated inside the part on the
               FULL TRIPLE. The fingerprint places rows; it never decides equality, so a fingerprint
               collision cannot merge two distinct edges.
  UNCHANGED -> deduplicated IN PLACE, exactly, using the contiguity result this whole build rests
               on. Two identical triples share a subject; every triple of a subject lies in one
               contiguous run (REOPENED_SUBJECT_BLOCKS_N = 0) and no run straddles a member seam,
               so both copies are inside one run of one shard. Sorting a run makes them adjacent.
               No global partitioning is needed to be exact about them.

Routing the whole 2.19e9 through the partitioner would have been simpler to write and would not
have fitted: partitioning destroys the source ordering that lets PASS B store an edge in 5 bytes,
and 2.19e9 rows of effectively random 64-bit integers is roughly 46 GB of buckets against 32 GB of
free disk. The split is what makes the pass runnable, and it costs nothing in rigour because the
unchanged path is exact rather than sampled.
"""
import collections
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
PB = f"{V3}/pass_b"
PD = f"{V3}/pass_d"
BUCKETS = f"{PD}/_buckets"
EDGES = f"{PD}/edges"
FREEZE = f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json"
REPORT = f"{V3}/V3_PASS_D_CANONICAL_EDGES.json"
MAPFILE = f"{PD}/reverse_map.json"

K_LIT = 2
U64 = np.uint64
SCHEMA = pa.schema([("src", pa.int64()), ("rel", pa.int64()), ("dst", pa.int64()),
                    ("dst_kind", pa.int8()), ("flipped", pa.int8())])


def mix(x):
    x = x.astype(np.uint64, copy=True)
    x ^= x >> U64(30)
    x *= U64(0xBF58476D1CE4E5B9)
    x ^= x >> U64(27)
    x *= U64(0x94D049BB133111EB)
    x ^= x >> U64(31)
    return x


def fingerprint(s, r, d):
    h = mix(s.view(np.uint64))
    h = mix(h ^ r.view(np.uint64))
    h = mix(h ^ d.view(np.uint64))
    return h


def dedup_run(s, r, d, kd, fl):
    """Deduplicate one subject run on the full triple. Returns the kept rows, in (rel, dst) order."""
    if s.size < 2:
        return s, r, d, kd, fl, 0
    order = np.lexsort((d, r))
    s, r, d, kd, fl = s[order], r[order], d[order], kd[order], fl[order]
    same = np.zeros(s.size, dtype=bool)
    same[1:] = (r[1:] == r[:-1]) & (d[1:] == d[:-1])
    if not same.any():
        return s, r, d, kd, fl, 0
    keep = ~same
    return s[keep], r[keep], d[keep], kd[keep], fl[keep], int(same.sum())


class PartWriter:
    """Buffered parquet writer. Row groups of a few thousand rows would cost more in metadata than
    the rows do in data at this width."""

    def __init__(self, path, flush_rows=1 << 18):
        self.path, self.flush_rows = path, flush_rows
        self.buf, self.n, self.w = [], 0, None

    def add(self, s, r, d, kd, fl):
        if s.size == 0:
            return
        self.buf.append((s, r, d, kd, fl))
        self.n += s.size
        if self.n >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cols = [np.concatenate([b[i] for b in self.buf]) for i in range(5)]
        tb = pa.Table.from_arrays([pa.array(c) for c in cols], schema=SCHEMA)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, SCHEMA, compression="zstd")
        self.w.write_table(tb)
        self.buf, self.n = [], 0

    def close(self):
        self.flush()
        if self.w is not None:
            self.w.close()


def process_shard(job):
    """Split one PASS B edge shard into the at-risk and unchanged paths."""
    path, parts, tag, rev2mas, master_at_risk, consume = job
    revk = np.array(sorted(rev2mas), dtype=np.int64)
    revv = np.array([rev2mas[int(k)] for k in revk], dtype=np.int64)
    mask = np.array(sorted(master_at_risk), dtype=np.int64)

    bw = {}
    pt = PartWriter(f"{EDGES}/direct_{tag}.parquet")
    rows_in = at_risk = flipped = lit_blocked = pt_rows = pt_dups = 0
    per_rel_flip = collections.Counter()
    pend = None

    def emit_run(s, r, d, kd, fl):
        nonlocal pt_rows, pt_dups
        s, r, d, kd, fl, dups = dedup_run(s, r, d, kd, fl)
        pt.add(s, r, d, kd, fl)
        pt_rows += s.size
        pt_dups += dups

    try:
        for b in pq.ParquetFile(path).iter_batches(batch_size=1 << 21):
            s = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
            r = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
            d = b.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
            kd = b.column(3).to_numpy(zero_copy_only=False).astype(np.int8)
            rows_in += s.size
            fl = np.zeros(s.size, dtype=np.int8)

            # --- classify ---
            pos = np.searchsorted(revk, r)
            np.clip(pos, 0, max(revk.size - 1, 0), out=pos)
            isrev = (revk[pos] == r) if revk.size else np.zeros(s.size, bool)
            mp_ = np.searchsorted(mask, r)
            np.clip(mp_, 0, max(mask.size - 1, 0), out=mp_)
            ismas = (mask[mp_] == r) if mask.size else np.zeros(s.size, bool)
            # a reverse-side edge with a literal object cannot be flipped, so it cannot move and is
            # not at risk; it stays on the in-place path in its original orientation.
            blocked = isrev & (kd == K_LIT)
            lit_blocked += int(blocked.sum())
            risk = (isrev & ~blocked) | ismas

            # --- unchanged path: dedup within subject runs ---
            keep = ~risk
            if keep.any():
                ks, kr, kdd, kkd, kfl = s[keep], r[keep], d[keep], kd[keep], fl[keep]
                cuts = np.flatnonzero(ks[1:] != ks[:-1]) + 1 if ks.size > 1 else np.array([], int)
                starts = np.concatenate([[0], cuts])
                ends = np.concatenate([cuts, [ks.size]])
                for j in range(starts.size):
                    a, c = starts[j], ends[j]
                    seg = (ks[a:c], kr[a:c], kdd[a:c], kkd[a:c], kfl[a:c])
                    if pend is not None and pend[0][0] == seg[0][0]:
                        seg = tuple(np.concatenate([pend[i], seg[i]]) for i in range(5))
                        pend = None
                    elif pend is not None:
                        emit_run(*pend)
                        pend = None
                    if j == starts.size - 1:
                        # the last run may continue into the next batch, so it is held rather than
                        # emitted: splitting a subject run would hide a duplicate that spans it.
                        pend = seg
                    else:
                        emit_run(*seg)

            # --- at-risk path: flip, then scatter ---
            if risk.any():
                idx = np.flatnonzero(risk)
                fi = idx[isrev[idx] & ~blocked[idx]]
                if fi.size:
                    for u, c in zip(*np.unique(r[fi], return_counts=True)):
                        per_rel_flip[int(u)] += int(c)
                    ns, nd = d[fi].copy(), s[fi].copy()
                    s[fi], d[fi] = ns, nd
                    r[fi] = revv[pos[fi]]
                    kd[fi] = 0
                    fl[fi] = 1
                    flipped += fi.size
                rs, rr, rd, rkd, rfl = s[idx], r[idx], d[idx], kd[idx], fl[idx]
                at_risk += idx.size
                fp = fingerprint(rs, rr, rd)
                bkt = (fp & U64(parts - 1)).astype(np.int64)
                order = np.argsort(bkt, kind="stable")
                bounds = np.searchsorted(bkt[order], np.arange(parts + 1))
                for i in range(parts):
                    lo, hi = bounds[i], bounds[i + 1]
                    if lo == hi:
                        continue
                    sel = order[lo:hi]
                    w = bw.get(i)
                    if w is None:
                        w = PartWriter(f"{BUCKETS}/p{i:03d}_{tag}.parquet", flush_rows=1 << 15)
                        bw[i] = w
                    w.add(rs[sel], rr[sel], rd[sel], rkd[sel], rfl[sel])
        if pend is not None:
            emit_run(*pend)
    finally:
        for w in bw.values():
            w.close()
        pt.close()

    if pt_rows + pt_dups + at_risk != rows_in:
        raise SystemExit(f"{tag}: {rows_in} in, {pt_rows}+{pt_dups} direct + {at_risk} at risk")
    if consume:
        os.remove(path)
    return {"tag": tag, "rows": rows_in, "at_risk": at_risk, "direct_rows": pt_rows,
            "direct_dups": pt_dups, "flipped": flipped, "literal_blocked": lit_blocked,
            "per_rel_flip": {str(k): v for k, v in per_rel_flip.items()}}


def fold_part(job):
    """Deduplicate one at-risk bucket on the full triple."""
    i = job[0]
    files = sorted(glob.glob(f"{BUCKETS}/p{i:03d}_*.parquet"))
    if not files:
        return {"part": i, "rows": 0, "distinct": 0, "dup_rows": 0, "dup_from_flip": 0}
    S, R, D, KD, FL = [], [], [], [], []
    for f in files:
        t = pq.read_table(f)
        S.append(t.column(0).to_numpy(zero_copy_only=False))
        R.append(t.column(1).to_numpy(zero_copy_only=False))
        D.append(t.column(2).to_numpy(zero_copy_only=False))
        KD.append(t.column(3).to_numpy(zero_copy_only=False))
        FL.append(t.column(4).to_numpy(zero_copy_only=False))
    s = np.concatenate(S).astype(np.int64)
    r = np.concatenate(R).astype(np.int64)
    d = np.concatenate(D).astype(np.int64)
    kd = np.concatenate(KD).astype(np.int8)
    fl = np.concatenate(FL).astype(np.int8)
    del S, R, D, KD, FL
    n = s.size

    # Rebound one at a time, NOT as a tuple assignment: the tuple form builds all five reordered
    # arrays before rebinding any, so both copies are live and the fold peaks at twice the bucket.
    order = np.lexsort((d, r, s))
    s = s[order]
    r = r[order]
    d = d[order]
    kd = kd[order]
    fl = fl[order]
    del order

    same = np.zeros(n, dtype=bool)
    if n > 1:
        same[1:] = (s[1:] == s[:-1]) & (r[1:] == r[:-1]) & (d[1:] == d[:-1])
    keep = ~same
    nkeep = int(keep.sum())
    dup_rows = n - nkeep

    dup_from_flip = 0
    if dup_rows:
        di = np.flatnonzero(same)
        dup_from_flip = int((fl[di] != fl[di - 1]).sum())
        # The surviving row keeps flipped=0 when ANY copy of the fact was asserted directly, so the
        # flag stays a statement about the source rather than about which copy sorted first.
        grp = np.cumsum(keep) - 1
        anydirect = np.zeros(nkeep, dtype=bool)
        np.logical_or.at(anydirect, grp, fl == 0)
        fl2 = np.where(anydirect, 0, 1).astype(np.int8)
    else:
        fl2 = fl[keep]

    pq.write_table(pa.Table.from_arrays(
        [pa.array(s[keep]), pa.array(r[keep]), pa.array(d[keep]),
         pa.array(kd[keep]), pa.array(fl2)], schema=SCHEMA),
        f"{EDGES}/part_{i:03d}.parquet", compression="zstd")
    for f in files:
        os.remove(f)
    return {"part": i, "rows": n, "distinct": nkeep, "dup_rows": dup_rows,
            "dup_from_flip": dup_from_flip}


def cycles(pairs):
    """Cycles longer than the expected 2-cycle, and self-pairs. Either makes 'subject side'
    undefined, which is why the pre-registration named them."""
    nxt = dict(pairs)
    self_paired = sorted(a for a, b in pairs.items() if a == b)
    longc, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        path, cur, idx = [], start, {}
        while cur in nxt and cur not in idx:
            idx[cur] = len(path)
            path.append(cur)
            cur = nxt[cur]
        if cur in idx:
            cyc = path[idx[cur]:]
            if len(cyc) != 2 or nxt.get(cyc[1]) != cyc[0]:
                longc.append(cyc)
        seen.update(path)
    return self_paired, longc


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required so relation UIDs match PASS B's.")
    t0 = time.time()
    parts = 64
    if "--parts" in sys.argv:
        parts = int(sys.argv[sys.argv.index("--parts") + 1])
    if parts & (parts - 1):
        raise SystemExit("--parts must be a power of two: bucketing masks the low bits.")
    consume = "--consume" in sys.argv
    workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    fold_workers = 4
    if "--fold-workers" in sys.argv:
        fold_workers = int(sys.argv[sys.argv.index("--fold-workers") + 1])
    for p in (PD, BUCKETS, EDGES):
        os.makedirs(p, exist_ok=True)

    fr = json.load(open(FREEZE, encoding="utf-8"))
    pairs = fr["DECLARATIONS"]["REVERSE_PAIR_DIRECTED"]
    self_paired, longc = cycles(pairs)

    rt = pq.read_table(f"{PB}/relations_raw.parquet")
    uid = dict(zip(rt.column(1).to_pylist(), rt.column(0).to_pylist()))
    cnt = dict(zip(rt.column(1).to_pylist(), rt.column(2).to_pylist()))

    both = subj_only = obj_only = neither = 0
    neither_names, both_names = [], []
    rev2mas, master_at_risk, mapped, unmapped = {}, set(), 0, []
    for master, reverse in pairs.items():
        cm, cr = cnt.get(master, 0), cnt.get(reverse, 0)
        if cm and cr:
            both += 1
            both_names.append([master, cm, reverse, cr])
        elif cm:
            subj_only += 1
        elif cr:
            obj_only += 1
        else:
            neither += 1
            neither_names.append([master, reverse])
        if reverse in uid and master in uid:
            rev2mas[int(uid[reverse])] = int(uid[master])
            mapped += 1
            # A master-side edge is only at risk of colliding with a flipped edge when the reverse
            # side actually carries edges. When it does not, nothing can ever land on top of it and
            # it takes the cheap in-place path.
            if cr:
                master_at_risk.add(int(uid[master]))
        elif cr:
            # The reverse side carries edges but its master never occurs as a predicate anywhere.
            # Flipping into a relation the graph does not contain would invent one, so those edges
            # keep their own relation, and this is reported rather than silently resolved.
            unmapped.append([master, reverse, cr])
    census = {
        "REVERSE_PAIR_ASSERTIONS_N": len(pairs),
        "SUBJECT_SIDE_PREDICATES_N": sum(1 for m in pairs if cnt.get(m, 0)),
        "OBJECT_SIDE_PREDICATES_N": sum(1 for m in pairs.values() if cnt.get(m, 0)),
        "BOTH_SIDES_PRESENT_N": both,
        "NEITHER_SIDE_PRESENT_N": neither,
        "AMBIGUOUS_CYCLES_N": len(longc) + len(self_paired),
        "self_paired": self_paired,
        "long_cycles": longc[:20],
        "pairs_with_only_master_edges": subj_only,
        "pairs_with_only_reverse_edges": obj_only,
        "reverse_relations_mapped_to_a_master_uid": mapped,
        "master_relations_at_risk_of_collision": len(master_at_risk),
        "reverse_relations_with_edges_but_no_master_uid": unmapped[:20],
        "reverse_relations_with_edges_but_no_master_uid_n": len(unmapped),
        "examples_both_sides": both_names[:10],
        "examples_neither_side": neither_names[:10],
        "READING": {
            "on_the_raw_source": "NEITHER_SIDE here means Freebase declared the pair but asserted "
                                 "no edge on either side. That is a schema-only pair, not fact "
                                 "loss, and it is the prior question the pre-registration could "
                                 "not distinguish from the outside.",
            "on_an_anti_joined_variant": "NEITHER_SIDE there means the variant annihilated a pair "
                                         "the source did assert. That is the fact-loss alarm "
                                         "INVARIANT_2 was written for, and it is checked against "
                                         "IDIR separately.",
            "INVARIANT_1_here": "BOTH_SIDES_PRESENT_N is EXPECTED to be large on the raw source: "
                                "it measures the redundancy this pass removes. It is an alarm only "
                                "AFTER canonicalisation, where it must be zero.",
        },
    }
    with open(MAPFILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"reverse_uid_to_master_uid": {str(k): v for k, v in rev2mas.items()},
                   "census": census}, fh, indent=1)
    print(json.dumps({k: census[k] for k in (
        "REVERSE_PAIR_ASSERTIONS_N", "SUBJECT_SIDE_PREDICATES_N", "OBJECT_SIDE_PREDICATES_N",
        "BOTH_SIDES_PRESENT_N", "NEITHER_SIDE_PRESENT_N", "AMBIGUOUS_CYCLES_N",
        "reverse_relations_mapped_to_a_master_uid",
        "master_relations_at_risk_of_collision")}, indent=1), flush=True)

    shards = sorted(glob.glob(f"{PB}/edges/m*.parquet"))
    if not shards:
        raise SystemExit("no PASS B edge shards")
    if "--limit" in sys.argv:
        shards = shards[:int(sys.argv[sys.argv.index("--limit") + 1])]
    jobs = [(p, parts, os.path.basename(p)[:-8], rev2mas, master_at_risk, consume) for p in shards]
    print(f"canonicalising {len(jobs)} shards, {parts} at-risk buckets, {workers} workers",
          flush=True)
    res, done = [], 0
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(process_shard, jobs):
            res.append(r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}  t={time.time()-t0:.0f}s", flush=True)

    print(f"deduplicating {parts} at-risk buckets, {fold_workers} workers", flush=True)
    folds, done = [], 0
    with mp.Pool(fold_workers) as pool:
        for r in pool.imap_unordered(fold_part, [(i,) for i in range(parts)]):
            folds.append(r)
            done += 1
            if done % 16 == 0:
                print(f"  {done}/{parts}  t={time.time()-t0:.0f}s", flush=True)

    rows_in = sum(r["rows"] for r in res)
    at_risk = sum(r["at_risk"] for r in res)
    scattered = sum(f["rows"] for f in folds)
    risk_distinct = sum(f["distinct"] for f in folds)
    risk_dups = sum(f["dup_rows"] for f in folds)
    dup_flip = sum(f["dup_from_flip"] for f in folds)
    direct_rows = sum(r["direct_rows"] for r in res)
    direct_dups = sum(r["direct_dups"] for r in res)
    flipped = sum(r["flipped"] for r in res)
    lit_blocked = sum(r["literal_blocked"] for r in res)
    canonical = direct_rows + risk_distinct
    dups = direct_dups + risk_dups

    per_rel = collections.Counter()
    for r in res:
        for k, v in r["per_rel_flip"].items():
            per_rel[int(k)] += v
    u2name = dict(zip(rt.column(0).to_pylist(), rt.column(1).to_pylist()))

    mas2rev = {int(uid[m]): rv for m, rv in pairs.items() if m in uid}
    rel_uids = rt.column(0).to_pylist()
    revset = set(rev2mas)
    roles = ["REVERSE_CANONICALISED_AWAY" if u in revset
             else ("MASTER" if u in mas2rev else "UNPAIRED") for u in rel_uids]
    pq.write_table(pa.Table.from_arrays(
        [pa.array(rel_uids, type=pa.int64()),
         pa.array(rt.column(1).to_pylist(), type=pa.string()),
         pa.array([mas2rev.get(u) for u in rel_uids], type=pa.string()),
         pa.array(roles, type=pa.string()),
         pa.array(rt.column(2).to_pylist(), type=pa.int64())],
        names=["rel_uid", "relation", "reverse_relation", "role", "raw_edge_count"]),
        f"{PD}/relations.parquet", compression="zstd")

    doc = {
        "schema": "V3_PASS_D_CANONICAL_EDGES/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "at_risk_parts": parts,
        "freeze_hash": fr["FREEZE_HASH"],
        "PREREGISTERED_CHECKS": census,
        "CANONICALISATION": {
            "edges_in": rows_in,
            "at_risk_edges": at_risk,
            "at_risk_scattered": scattered,
            "scatter_balances": at_risk == scattered,
            "unchanged_edges_kept": direct_rows,
            "edges_flipped_to_master_side": flipped,
            "reverse_side_edges_with_literal_object_left_unflipped": lit_blocked,
            "literal_note": "a literal cannot be a subject, so these keep their original "
                            "orientation. They are counted rather than dropped or coerced.",
            "CANONICAL_EDGES": canonical,
            "duplicate_rows_removed": dups,
            "duplicates_in_at_risk_path": risk_dups,
            "duplicates_in_unchanged_path": direct_dups,
            "duplicates_where_both_directions_were_asserted": dup_flip,
            "redundancy_removed_pct": round(100.0 * dups / rows_in, 4) if rows_in else None,
            "arithmetic_balances": rows_in == canonical + dups,
            "method": {
                "at_risk": "hash-partitioned on a 64-bit fingerprint of (src, rel, dst), then "
                           "deduplicated on the FULL TRIPLE inside the part. The fingerprint places "
                           "rows; it never decides equality, so a collision cannot merge two "
                           "distinct edges.",
                "unchanged": "deduplicated in place within subject runs. Exact, because two "
                             "identical triples share a subject and every triple of a subject lies "
                             "in one contiguous run that does not straddle a member seam -- the "
                             "same contiguity result that removed the source-side external sort.",
            },
        },
        "TOP_FLIPPED_RELATIONS": [[u2name.get(u, str(u)), n] for u, n in per_rel.most_common(15)],
        "OUTPUT": {"edges": f"{EDGES}/part_*.parquet + {EDGES}/direct_*.parquet",
                   "relations": f"{PD}/relations.parquet"},
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"CANONICALISATION": doc["CANONICALISATION"],
                      "TOP_FLIPPED_RELATIONS": doc["TOP_FLIPPED_RELATIONS"]}, indent=1))


if __name__ == "__main__":
    main()
