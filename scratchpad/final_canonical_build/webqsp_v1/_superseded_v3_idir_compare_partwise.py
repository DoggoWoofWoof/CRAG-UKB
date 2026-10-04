"""STEP 5: project the raw-built graph into IDIR's FB+CVT-REV view and compare, at the frozen gate.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_idir_compare.py

IDIR's role was frozen early and has not changed: audit oracle, metadata accelerator, structural
validation reference. NOT the canonical graph. This is the pass that makes that role pay off -- an
independent party built a Freebase graph from the same dump with their own parser, and if two
independent parsers agree on ~134 million triples then neither is quietly dropping a relation
family, mis-splitting a literal or losing a CVT.

THE ORIENTATION IS MEASURED, NOT ASSUMED. V3_DESIGN_LOCK recorded that IDIR derives its reverse
map from the same type.property.reverse_property assertions we froze in PASS A, but that WHICH side
their anti-join retains "was not readable from the sources checked", making this comparison
load-bearing rather than confirmatory. So every IDIR triple is tested against our canonical set in
BOTH orientations, and which one matches is the empirical answer to that open question. Reporting
only forward agreement would turn a disagreement about orientation into a false coverage failure.

WHY THE COMPARISON IS EXACT. Both sides are reduced to the same 64-bit UID space -- IDIR's entity
ids resolve through backbone_entity2id to the same namespace-stripped MIDs we mint from -- and then
hash-partitioned on the same fingerprint PASS D used, so every copy of a triple lands in one part.
Inside a part, agreement is decided by comparing the FULL TRIPLE, never a hash of it. The
fingerprint chooses where to look; it never decides what matches.

THE GATE WAS FROZEN BEFORE THE NUMBER WAS KNOWN: >= 99.9% PASS, 99.0-99.9% INVESTIGATE,
< 99.0% STOP. It is applied here as written.
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ORACLE = "data/final_canonical/freebase_v3/_acquisition/idir/oracle"
PC = "data/final_canonical/freebase_v3/pass_c"
PD = "data/final_canonical/freebase_v3/pass_d"
WORK = "data/final_canonical/freebase_v3/idir_compare"
REPORT = "data/final_canonical/freebase_v3/V3_IDIR_ORACLE_COMPARISON.json"

GATE_PASS, GATE_INVESTIGATE = 99.9, 99.0
U64 = np.uint64
LITERAL, CVT_MEDIATOR, ENTITY_MID = 0, 1, 2


def mix(x):
    x = x.astype(np.uint64, copy=True)
    x ^= x >> U64(30)
    x *= U64(0xBF58476D1CE4E5B9)
    x ^= x >> U64(27)
    x *= U64(0x94D049BB133111EB)
    x ^= x >> U64(31)
    return x


def fingerprint(s, r, d):
    """Byte-identical to PASS D's. If these drift, matching triples land in different parts and the
    comparison reports a disagreement that does not exist."""
    h = mix(s.view(np.uint64))
    h = mix(h ^ r.view(np.uint64))
    h = mix(h ^ d.view(np.uint64))
    return h


def dot_form(slash):
    """IDIR names relations in slash form; the dump and therefore our relation UIDs use dot form."""
    return slash.strip("/").replace("/", ".")


def scatter_idir(job):
    """Map one slice of IDIR triples into our UID space and bucket it, forward and reversed."""
    lo, hi, parts, tag = job
    ent = np.load(f"{WORK}/idir_id_to_uid.npy", mmap_mode="r")
    rel = np.load(f"{WORK}/idir_rel_to_uid.npy", mmap_mode="r")
    relknown = np.load(f"{WORK}/idir_rel_known.npy", mmap_mode="r")
    entknown = np.load(f"{WORK}/idir_ent_known.npy", mmap_mode="r")
    pf = pq.ParquetFile(f"{ORACLE}/backbone_triples.parquet")
    writers = {}
    seen = 0
    unmapped_rel = unmapped_ent = 0
    try:
        for gi, b in enumerate(pf.iter_batches(batch_size=1 << 21)):
            if gi < lo or gi >= hi:
                continue
            s = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
            r = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
            o = b.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
            seen += s.size
            ok = entknown[s] & entknown[o] & relknown[r]
            unmapped_ent += int((~(entknown[s] & entknown[o])).sum())
            unmapped_rel += int((~relknown[r]).sum())
            if not ok.any():
                continue
            s, r, o = ent[s[ok]], rel[r[ok]], ent[o[ok]]
            for rev, (a, c) in enumerate(((s, o), (o, s))):
                fp = fingerprint(a, r, c)
                bkt = (fp & U64(parts - 1)).astype(np.int64)
                order = np.argsort(bkt, kind="stable")
                bounds = np.searchsorted(bkt[order], np.arange(parts + 1))
                for i in range(parts):
                    p0, p1 = bounds[i], bounds[i + 1]
                    if p0 == p1:
                        continue
                    sel = order[p0:p1]
                    tb = pa.Table.from_arrays(
                        [pa.array(a[sel]), pa.array(r[sel]), pa.array(c[sel])],
                        names=["s", "r", "o"])
                    key = (i, rev)
                    w = writers.get(key)
                    if w is None:
                        d = f"{WORK}/{'rev' if rev else 'fwd'}"
                        w = pq.ParquetWriter(f"{d}/p{i:03d}_{tag}.parquet", tb.schema,
                                             compression="zstd")
                        writers[key] = w
                    w.write_table(tb)
    finally:
        for w in writers.values():
            w.close()
    return {"tag": tag, "rows": seen, "rows_with_unmapped_entity": unmapped_ent,
            "rows_with_unmapped_relation": unmapped_rel}


def load_triples(paths):
    if not paths:
        return (np.zeros(0, np.int64),) * 3
    S, R, O = [], [], []
    for p in paths:
        t = pq.read_table(p)
        S.append(t.column(0).to_numpy(zero_copy_only=False))
        R.append(t.column(1).to_numpy(zero_copy_only=False))
        O.append(t.column(2).to_numpy(zero_copy_only=False))
    return (np.concatenate(S).astype(np.int64), np.concatenate(R).astype(np.int64),
            np.concatenate(O).astype(np.int64))


def in_set(qs, qr, qo, bs, br, bo):
    """Which of the q triples occur in the b triples. Union-sort, then adjacency on the full triple:
    exact, vectorised, and immune to fingerprint collisions because no hash is compared."""
    nq = qs.size
    if nq == 0 or bs.size == 0:
        return np.zeros(nq, dtype=bool)
    s = np.concatenate([qs, bs])
    r = np.concatenate([qr, br])
    o = np.concatenate([qo, bo])
    isq = np.zeros(s.size, dtype=bool)
    isq[:nq] = True
    order = np.lexsort((isq, o, r, s))       # a query row sorts after its base twin
    s, r, o, isq = s[order], r[order], o[order], isq[order]
    eqprev = np.zeros(s.size, dtype=bool)
    eqprev[1:] = (s[1:] == s[:-1]) & (r[1:] == r[:-1]) & (o[1:] == o[:-1])
    eqnext = np.zeros(s.size, dtype=bool)
    eqnext[:-1] = eqprev[1:]
    # a query row matches if any adjacent equal row is a base row; equal rows are contiguous, and
    # base rows sort first within a group, so "the previous equal row exists" suffices unless the
    # whole group is queries.
    grp = np.cumsum(~eqprev) - 1
    nb = np.zeros(int(grp[-1]) + 1, dtype=np.int64)
    np.add.at(nb, grp, ~isq)
    hit_sorted = (nb[grp] > 0) & isq
    del eqnext
    out = np.zeros(s.size, dtype=bool)
    out[order] = hit_sorted
    return out[:nq]


def compare_part(job):
    i, = job
    ours = load_triples(sorted(glob.glob(f"{PD}/edges/part_{i:03d}.parquet")))
    fwd = load_triples(sorted(glob.glob(f"{WORK}/fwd/p{i:03d}_*.parquet")))
    rev = load_triples(sorted(glob.glob(f"{WORK}/rev/p{i:03d}_*.parquet")))
    hf = in_set(*fwd, *ours)
    hr = in_set(*rev, *ours)
    return {"part": i, "ours": int(ours[0].size), "idir_fwd": int(fwd[0].size),
            "idir_rev": int(rev[0].size), "fwd_hits": int(hf.sum()), "rev_hits": int(hr.sum())}


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required so UIDs match the build's.")
    t0 = time.time()
    parts = len(glob.glob(f"{PD}/edges/part_*.parquet"))
    if not parts:
        raise SystemExit("PASS D edges not found; the comparison runs on the canonical graph")
    if parts & (parts - 1):
        raise SystemExit(f"{parts} PASS D parts is not a power of two")
    workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    for d in (WORK, f"{WORK}/fwd", f"{WORK}/rev"):
        os.makedirs(d, exist_ok=True)

    # ---- entity and relation dictionaries into our UID space ----
    print("mapping IDIR entity ids into the build UID space", flush=True)
    pf = pq.ParquetFile(f"{ORACLE}/backbone_entity2id.parquet")
    n_ent = pf.metadata.num_rows
    ent = np.zeros(n_ent + 1, dtype=np.int64)
    known = np.zeros(n_ent + 1, dtype=bool)
    maxid = -1
    for b in pf.iter_batches(batch_size=1 << 21):
        mids = b.column(0).to_pylist()
        ids = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        maxid = max(maxid, int(ids.max()))
        if maxid >= ent.size:
            grow = np.zeros(maxid + 1 - ent.size, dtype=np.int64)
            ent = np.concatenate([ent, grow])
            known = np.concatenate([known, np.zeros(grow.size, dtype=bool)])
        ent[ids] = np.fromiter((hash(m.encode()) for m in mids), dtype=np.int64, count=len(mids))
        known[ids] = True
    np.save(f"{WORK}/idir_id_to_uid.npy", ent)
    np.save(f"{WORK}/idir_ent_known.npy", known)

    rt = pq.read_table(f"{ORACLE}/backbone_relation2id.parquet")
    rel_names = [dot_form(x) for x in rt.column(0).to_pylist()]
    rel_ids = rt.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
    ours_rel = pq.read_table(f"{PD}/relations.parquet")
    rel_uid_by_name = dict(zip(ours_rel.column(1).to_pylist(), ours_rel.column(0).to_pylist()))
    role_by_name = dict(zip(ours_rel.column(1).to_pylist(), ours_rel.column(3).to_pylist()))
    rmap = np.zeros(int(rel_ids.max()) + 1, dtype=np.int64)
    rknown = np.zeros(rmap.size, dtype=bool)
    missing_rel = []
    for nm, rid in zip(rel_names, rel_ids):
        if nm in rel_uid_by_name:
            rmap[rid] = rel_uid_by_name[nm]
            rknown[rid] = True
        else:
            missing_rel.append(nm)
    np.save(f"{WORK}/idir_rel_to_uid.npy", rmap)
    np.save(f"{WORK}/idir_rel_known.npy", rknown)
    idir_rel_roles = {}
    for nm in rel_names:
        idir_rel_roles[role_by_name.get(nm, "ABSENT_FROM_BUILD")] = \
            idir_rel_roles.get(role_by_name.get(nm, "ABSENT_FROM_BUILD"), 0) + 1
    print(f"  {int(known.sum()):,} entities, {int(rknown.sum()):,}/{len(rel_names)} relations "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- scatter IDIR triples, both orientations ----
    ntb = pq.ParquetFile(f"{ORACLE}/backbone_triples.parquet")
    nbatch = (ntb.metadata.num_rows + (1 << 21) - 1) // (1 << 21)
    step = (nbatch + workers - 1) // workers
    jobs = [(i * step, min(nbatch, (i + 1) * step), parts, f"w{i:02d}")
            for i in range(workers) if i * step < nbatch]
    print(f"scattering {ntb.metadata.num_rows:,} IDIR triples into {parts} parts x 2 orientations",
          flush=True)
    sres = []
    with mp.Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(scatter_idir, jobs):
            sres.append(r)
            print(f"  {r['tag']} {r['rows']:,}", flush=True)

    # ---- compare ----
    print("comparing parts", flush=True)
    cres, done = [], 0
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(compare_part, [(i,) for i in range(parts)]):
            cres.append(r)
            done += 1
            if done % 32 == 0:
                print(f"  {done}/{parts}  t={time.time()-t0:.0f}s", flush=True)

    idir_rows = sum(r["rows"] for r in sres)
    mapped = sum(r["idir_fwd"] for r in cres)
    fwd_hits = sum(r["fwd_hits"] for r in cres)
    rev_hits = sum(r["rev_hits"] for r in cres)
    ours_total = sum(r["ours"] for r in cres)
    matched = fwd_hits + rev_hits
    cov = 100.0 * matched / mapped if mapped else 0.0
    verdict = ("PASS" if cov >= GATE_PASS else
               "INVESTIGATE" if cov >= GATE_INVESTIGATE else "STOP")
    orientation = ("SAME_AS_OURS_MASTER_SIDE" if fwd_hits > rev_hits * 10 else
                   "OPPOSITE_TO_OURS" if rev_hits > fwd_hits * 10 else "MIXED")

    doc = {
        "schema": "V3_IDIR_ORACLE_COMPARISON/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "parts": parts,
        "IDIR_SIDE": {
            "backbone_triples": idir_rows,
            "entities": int(known.sum()),
            "relations": len(rel_names),
            "relations_resolved_in_build": int(rknown.sum()),
            "relations_absent_from_build": missing_rel[:25],
            "relations_absent_from_build_n": len(missing_rel),
            "idir_relations_by_role_in_our_relations_table": idir_rel_roles,
            "triples_dropped_unmappable_entity": sum(r["rows_with_unmapped_entity"] for r in sres),
            "triples_dropped_unmappable_relation": sum(r["rows_with_unmapped_relation"]
                                                       for r in sres),
            "triples_compared": mapped,
        },
        "OUR_SIDE": {"canonical_edges_in_compared_parts": ours_total},
        "AGREEMENT": {
            "matched_same_orientation": fwd_hits,
            "matched_reversed_orientation": rev_hits,
            "matched_total": matched,
            "unmatched": mapped - matched,
            "COVERAGE_PCT": round(cov, 4),
            "RETAINED_ORIENTATION_MEASURED": orientation,
            "orientation_note": "V3_DESIGN_LOCK recorded that which side IDIR's anti-join keeps was "
                                "NOT confirmable from their scripts. This is the empirical answer: "
                                "a triple matching in the reversed direction is the same fact, "
                                "stored from the other end, and counts as agreement.",
        },
        "GATE": {
            "thresholds": {"PASS": f">= {GATE_PASS}%", "INVESTIGATE":
                           f"{GATE_INVESTIGATE}-{GATE_PASS}%", "STOP": f"< {GATE_INVESTIGATE}%"},
            "frozen_in": "V3_CANONICAL_CONTRACT_V2.json, before the comparison was run",
            "VERDICT": verdict,
        },
        "READING": "IDIR is an oracle, not a target. Our graph is deliberately LARGER than theirs -- "
                   "it keeps literals, schema nodes and external URIs that FB+CVT-REV excludes by "
                   "construction -- so coverage is asked in one direction only: does our graph "
                   "contain what an independent parser found? Edges of ours absent from IDIR are "
                   "expected and are not a defect.",
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({k: doc[k] for k in ("IDIR_SIDE", "AGREEMENT", "GATE")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
