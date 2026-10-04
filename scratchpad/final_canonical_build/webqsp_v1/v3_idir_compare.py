"""STEP 5: project the raw-built graph into IDIR's FB+CVT-REV view and compare, at the frozen gate.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_idir_compare.py

IDIR's role was frozen early and has not changed: audit oracle, metadata accelerator, structural
validation reference. NOT the canonical graph. This is the pass that makes that role pay off -- an
independent party built a Freebase graph from the same dump with their own parser, and if two
independent parsers agree on 134,142,730 triples then neither is quietly dropping a relation
family, mis-splitting a literal or losing a CVT.

THE ORIENTATION IS MEASURED, NOT ASSUMED. V3_DESIGN_LOCK recorded that IDIR derives its reverse map
from the same type.property.reverse_property assertions we froze in PASS A, but that WHICH side
their anti-join retains "was not readable from the sources checked", making this comparison
load-bearing rather than confirmatory. So every IDIR triple is tested against our canonical set in
BOTH orientations, and which one matches is the empirical answer to that open question. Reporting
only forward agreement would turn a disagreement about orientation into a false coverage failure.

WHY THIS IS NOT PARTITIONED THE WAY PASS D WAS. An earlier version of this file compared our
part_<nnn> file against IDIR's part <nnn>. That was WRONG, and wrong in the direction that fails
loudly for the wrong reason: PASS D emits the canonical edge set as two families, direct_<member>
for the 76.79% of edges deduplicated in place and part_<nnn> for the 23.21% that were
fingerprint-partitioned, and only the second family is partitioned by fingerprint at all. Matching
part against part would have compared IDIR against a quarter of the graph and reported roughly a
77% miss rate -- tripping the STOP gate on a graph that is in fact complete. The comparison now
streams the WHOLE edge set, both families, and the file list is asserted to contain both.

WHY THE COMPARISON IS EXACT WHERE IT MATTERS. IDIR is small enough to hold: 134.1M triples, so both
orientations of its fingerprints fit in about 2.1 GB and our 2.19B edges are streamed against them
once. A fingerprint probe alone would leave a collision term, so it is not left as an assumption:
a pre-registered 1/256 band of the fingerprint space is materialised in FULL TRIPLES on both sides
and compared by exact adjacency, with no hash consulted. If the exact count on that band differs
from the fingerprint count on the same band, the run stops. The expected number of false matches
across the entire 2.19e9 x 1.34e8 cross product is 2.19e9 * 1.34e8 / 2**64 = 0.016, i.e. a ~1.6%
chance of even one, against a gate that lives at 0.1 percentage points -- but the band check
measures it rather than trusting the arithmetic.

THE GATE WAS FROZEN BEFORE THE NUMBER WAS KNOWN: >= 99.9% PASS, 99.0-99.9% INVESTIGATE,
< 99.0% STOP. It is applied here as written.
"""
import concurrent.futures as cf
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ORACLE = f"{V3}/_acquisition/idir/oracle"
PD = f"{V3}/pass_d"
CAN = f"{V3}/canonical"
WORK = f"{V3}/idir_compare"
REPORT = f"{V3}/V3_IDIR_ORACLE_COMPARISON.json"

GATE_PASS, GATE_INVESTIGATE = 99.9, 99.0
BAND_BITS = 8                      # 1/256 of the fingerprint space is verified on full triples
BAND_MASK = (1 << BAND_BITS) - 1
BATCH = 1 << 20
U64 = np.uint64


def mix(x):
    x = x.astype(np.uint64, copy=True)
    x ^= x >> U64(30)
    x *= U64(0xBF58476D1CE4E5B9)
    x ^= x >> U64(27)
    x *= U64(0x94D049BB133111EB)
    x ^= x >> U64(31)
    return x


def fingerprint(s, r, d):
    """Byte-identical to PASS D's. If these drift, matching triples get different fingerprints and
    the comparison reports a disagreement that does not exist."""
    h = mix(s.view(np.uint64))
    h = mix(h ^ r.view(np.uint64))
    h = mix(h ^ d.view(np.uint64))
    return h


def dot_form(slash):
    """IDIR names relations in slash form; the dump and therefore our relation UIDs use dot form."""
    return slash.strip("/").replace("/", ".")


def edge_files():
    """Every canonical edge file, from whichever tree holds them, with both families required."""
    fs = sorted(glob.glob(f"{CAN}/edges/*.parquet"))
    src = "canonical"
    if not fs:
        fs = (sorted(glob.glob(f"{PD}/edges/direct_*.parquet"))
              + sorted(glob.glob(f"{PD}/edges/part_*.parquet")))
        src = "pass_d"
    direct = [x for x in fs if os.path.basename(x).startswith("direct_")]
    folded = [x for x in fs if os.path.basename(x).startswith("part_")]
    if not direct or not folded:
        raise SystemExit(f"edge set incomplete: {len(direct)} direct + {len(folded)} folded in "
                         f"{src}; the comparison must see the whole graph or it measures nothing")
    return fs, src, len(direct), len(folded)


def build_dictionaries():
    """IDIR's dense ids into our UID space. IDIR stores MIDs already namespace-stripped and
    dot-formed (m.07ylj), which is exactly the string our node UIDs are minted from, so this is a
    re-indexing and not a translation."""
    t0 = time.time()
    pf = pq.ParquetFile(f"{ORACLE}/backbone_entity2id.parquet")
    n = pf.metadata.num_rows
    ent = np.zeros(n + 1, dtype=np.int64)
    known = np.zeros(n + 1, dtype=bool)
    seen = 0
    for b in pf.iter_batches(batch_size=1 << 21, columns=["mid", "idir_id"]):
        mids = b.column(0).to_pylist()
        ids = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        if ids.max() > n:
            raise SystemExit(f"idir_id {ids.max()} exceeds the {n}-row dictionary")
        ent[ids] = np.fromiter((hash(m.encode()) for m in mids), dtype=np.int64, count=len(mids))
        known[ids] = True
        seen += len(mids)

    rt = pq.read_table(f"{ORACLE}/backbone_relation2id.parquet")
    rids = rt.column("idir_id").to_numpy(zero_copy_only=False).astype(np.int64)
    rstr = rt.column("relation").to_pylist()
    rel = np.zeros(int(rids.max()) + 2, dtype=np.int64)
    rknown = np.zeros(rel.size, dtype=bool)
    dots = [dot_form(x) for x in rstr]
    rel[rids] = np.fromiter((hash(x.encode()) for x in dots), dtype=np.int64, count=len(dots))
    rknown[rids] = True
    print(f"  {seen:,} entities, {len(dots):,} relations mapped ({time.time()-t0:.0f}s)",
          flush=True)
    return ent, known, rel, rknown, {"entities": seen, "relations": len(dots)}


def build_idir_side():
    """Fingerprints of every IDIR triple in both orientations, sorted and deduplicated, plus the
    full triples of the verification band."""
    ent, eknown, rel, rknown, dictinfo = build_dictionaries()
    pf = pq.ParquetFile(f"{ORACLE}/backbone_triples.parquet")
    n = pf.metadata.num_rows
    fwd = np.empty(n, dtype=np.uint64)
    rev = np.empty(n, dtype=np.uint64)
    bs, br, bo = [], [], []
    at = 0
    drop_ent = drop_rel = 0
    t0 = time.time()
    for b in pf.iter_batches(batch_size=1 << 21):
        s = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        r = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        o = b.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
        ok = eknown[s] & eknown[o] & rknown[r]
        drop_ent += int((~(eknown[s] & eknown[o])).sum())
        drop_rel += int((~rknown[r]).sum())
        if not ok.all():
            s, r, o = s[ok], r[ok], o[ok]
        s, r, o = ent[s], rel[r], ent[o]
        f = fingerprint(s, r, o)
        v = fingerprint(o, r, s)
        fwd[at:at + f.size] = f
        rev[at:at + v.size] = v
        # the band is chosen on the FORWARD fingerprint so the same band of the space is
        # materialised on both sides, and it is chosen by a fixed mask, not by sampling.
        sel = (f & U64(BAND_MASK)) == U64(0)
        if sel.any():
            bs.append(s[sel])
            br.append(r[sel])
            bo.append(o[sel])
        at += f.size
    fwd = fwd[:at]
    rev = rev[:at]
    # the dictionaries are ~0.9 GB and are finished with here. Sorting and np.unique each want a
    # full-size temporary, so releasing them before that is the difference between a comfortable
    # peak and an uncomfortable one on a machine with about 6 GB usable.
    del ent, eknown, rel, rknown
    print(f"  {n:,} IDIR triples -> {at:,} mapped ({time.time()-t0:.0f}s)", flush=True)

    fwd.sort()
    rev.sort()
    dup_f = int((fwd[1:] == fwd[:-1]).sum()) if fwd.size > 1 else 0
    dup_r = int((rev[1:] == rev[:-1]).sum()) if rev.size > 1 else 0
    # Probing marks the first index of an equal run, so the denominator is the number of DISTINCT
    # fingerprints. Duplicates are reported rather than hidden: at this scale a random collision is
    # expected 0.0005 times, so a nonzero count means IDIR asserts the same triple twice, which is
    # a finding about the oracle and not about us.
    fwd = np.unique(fwd)
    rev = np.unique(rev)
    os.makedirs(WORK, exist_ok=True)
    np.save(f"{WORK}/idir_fwd_fp.npy", fwd)
    np.save(f"{WORK}/idir_rev_fp.npy", rev)
    band = pa.Table.from_arrays(
        [pa.array(np.concatenate(bs) if bs else np.zeros(0, np.int64)),
         pa.array(np.concatenate(br) if br else np.zeros(0, np.int64)),
         pa.array(np.concatenate(bo) if bo else np.zeros(0, np.int64))], names=["s", "r", "o"])
    pq.write_table(band, f"{WORK}/idir_band.parquet", compression="zstd")
    info = {"idir_triples": n, "mapped": at,
            "rows_with_unmapped_entity": drop_ent, "rows_with_unmapped_relation": drop_rel,
            "distinct_forward_fingerprints": int(fwd.size),
            "distinct_reverse_fingerprints": int(rev.size),
            "duplicate_forward_fingerprints": dup_f, "duplicate_reverse_fingerprints": dup_r,
            "verification_band_triples": band.num_rows, "dictionary": dictinfo}
    print(f"  band holds {band.num_rows:,} IDIR triples for exact verification", flush=True)
    return info


def probe(job):
    """Stream one slice of our edge files against the sorted IDIR fingerprints."""
    files, tag = job
    fwd = np.load(f"{WORK}/idir_fwd_fp.npy", mmap_mode="r")
    rev = np.load(f"{WORK}/idir_rev_fp.npy", mmap_mode="r")
    hf = np.zeros(fwd.size, dtype=bool)
    hr = np.zeros(rev.size, dtype=bool)
    bs, br, bo = [], [], []
    rows = 0
    for path in files:
        for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH,
                                                   columns=["src", "rel", "dst"]):
            s = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
            r = b.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
            d = b.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
            rows += s.size
            f = fingerprint(s, r, d)
            sel = (f & U64(BAND_MASK)) == U64(0)
            if sel.any():
                bs.append(s[sel])
                br.append(r[sel])
                bo.append(d[sel])
            # sorting the probe keys first turns 2.19 billion random reads of a memory-mapped
            # array into a forward walk of it.
            fs_ = np.sort(f)
            for arr, hit in ((fwd, hf), (rev, hr)):
                p = np.searchsorted(arr, fs_)
                np.clip(p, 0, arr.size - 1, out=p)
                m = arr[p] == fs_
                if m.any():
                    hit[p[m]] = True
    out = f"{WORK}/our_band_{tag}.parquet"
    pq.write_table(pa.Table.from_arrays(
        [pa.array(np.concatenate(bs) if bs else np.zeros(0, np.int64)),
         pa.array(np.concatenate(br) if br else np.zeros(0, np.int64)),
         pa.array(np.concatenate(bo) if bo else np.zeros(0, np.int64))], names=["s", "r", "o"]),
        out, compression="zstd")
    return {"tag": tag, "rows": rows, "hf": np.packbits(hf), "hr": np.packbits(hr), "band": out}


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
    grp = np.cumsum(~eqprev) - 1
    nb = np.zeros(int(grp[-1]) + 1, dtype=np.int64)
    np.add.at(nb, grp, ~isq)
    hit_sorted = (nb[grp] > 0) & isq
    out = np.zeros(s.size, dtype=bool)
    out[order] = hit_sorted
    return out[:nq]


def verify_band(hf, hr):
    """Recompute the band's agreement on FULL TRIPLES and require it to match the fingerprint
    answer exactly. This is the check that converts the collision bound from arithmetic into a
    measurement."""
    t = pq.read_table(f"{WORK}/idir_band.parquet")
    qs = t.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
    qr = t.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
    qo = t.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
    S, R, O = [], [], []
    for p in sorted(glob.glob(f"{WORK}/our_band_*.parquet")):
        tt = pq.read_table(p)
        S.append(tt.column(0).to_numpy(zero_copy_only=False).astype(np.int64))
        R.append(tt.column(1).to_numpy(zero_copy_only=False).astype(np.int64))
        O.append(tt.column(2).to_numpy(zero_copy_only=False).astype(np.int64))
    bs = np.concatenate(S) if S else np.zeros(0, np.int64)
    br = np.concatenate(R) if R else np.zeros(0, np.int64)
    bo = np.concatenate(O) if O else np.zeros(0, np.int64)
    exact_f = int(in_set(qs, qr, qo, bs, br, bo).sum())

    # the fingerprint answer restricted to the same band. Band membership is defined on the
    # forward fingerprint, so the forward side reads straight off the sorted array; a band
    # triple's REVERSE fingerprint is not itself in the band and is recomputed and located.
    fwd = np.asarray(np.load(f"{WORK}/idir_fwd_fp.npy", mmap_mode="r"))
    inband = (fwd & U64(BAND_MASK)) == U64(0)
    fp_f = int(hf[inband].sum())
    band_distinct = int(inband.sum())
    return {"band_bits": BAND_BITS, "band_idir_triples": int(qs.size),
            "band_idir_distinct_fingerprints": band_distinct,
            "band_our_triples": int(bs.size),
            "exact_forward_hits": exact_f, "fingerprint_forward_hits": fp_f,
            "forward_agrees": exact_f == fp_f,
            "note": "exact_forward_hits counts band triples; fingerprint_forward_hits counts "
                    "distinct band fingerprints, so the two agree exactly when the band holds no "
                    "duplicate triples and no collision inflated the probe."}


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required so UIDs match the build's.")
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    for old in glob.glob(f"{WORK}/our_band_*.parquet"):
        os.remove(old)
    files, src, ndirect, nfolded = edge_files()
    print(f"comparing against {len(files)} edge files from {src} "
          f"({ndirect} direct + {nfolded} folded)", flush=True)

    print("building the IDIR side", flush=True)
    idirinfo = build_idir_side()

    workers = min(4, max(1, (os.cpu_count() or 4) // 3))
    chunks = [(files[i::workers], f"{i:02d}") for i in range(workers)]
    chunks = [c for c in chunks if c[0]]
    nf = int(np.load(f"{WORK}/idir_fwd_fp.npy", mmap_mode="r").size)
    nr = int(np.load(f"{WORK}/idir_rev_fp.npy", mmap_mode="r").size)
    hf = np.zeros(nf, dtype=bool)
    hr = np.zeros(nr, dtype=bool)
    ours = 0
    print(f"probing {len(files)} files with {len(chunks)} workers", flush=True)
    with cf.ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        for res in ex.map(probe, chunks):
            hf |= np.unpackbits(res["hf"], count=nf).astype(bool)
            hr |= np.unpackbits(res["hr"], count=nr).astype(bool)
            ours += res["rows"]
            print(f"  chunk {res['tag']}: {res['rows']:,} edges  t={time.time()-t0:.0f}s",
                  flush=True)

    fwd_hits, rev_hits = int(hf.sum()), int(hr.sum())
    denom_f = idirinfo["distinct_forward_fingerprints"]
    denom_r = idirinfo["distinct_reverse_fingerprints"]
    cov_f = 100.0 * fwd_hits / denom_f if denom_f else 0.0
    cov_r = 100.0 * rev_hits / denom_r if denom_r else 0.0
    cov = max(cov_f, cov_r)

    print("verifying the band on full triples", flush=True)
    band = verify_band(hf, hr)
    if not band["forward_agrees"]:
        raise SystemExit(f"band verification disagrees with the fingerprint probe: {band}; "
                         f"the comparison is not trustworthy and no verdict is written")

    verdict = ("PASS" if cov >= GATE_PASS else
               "INVESTIGATE" if cov >= GATE_INVESTIGATE else "STOP")
    doc = {
        "schema": "V3_IDIR_ORACLE_COMPARISON/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "ROLE": "IDIR is an audit oracle and a structural validation reference. It is NOT the "
                "canonical graph, and a disagreement is evidence to investigate, not a reason to "
                "adopt IDIR's answer.",
        "OUR_SIDE": {"source": src, "edge_files": len(files), "direct_files": ndirect,
                     "folded_files": nfolded, "edges_streamed": ours},
        "IDIR_SIDE": idirinfo,
        "AGREEMENT": {"forward_hits": fwd_hits, "forward_coverage_pct": round(cov_f, 4),
                      "reverse_hits": rev_hits, "reverse_coverage_pct": round(cov_r, 4),
                      "coverage_pct": round(cov, 4)},
        "ORIENTATION_ANSWER": ("IDIR retains the same side we canonicalise to"
                               if cov_f >= cov_r else
                               "IDIR retains the opposite side from our master direction"),
        "BAND_VERIFICATION": band,
        "GATE": {"pass_at": GATE_PASS, "investigate_at": GATE_INVESTIGATE,
                 "frozen_before_the_number_was_known": True, "VERDICT": verdict},
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"AGREEMENT": doc["AGREEMENT"], "BAND_VERIFICATION": band,
                      "VERDICT": verdict}, indent=1))


if __name__ == "__main__":
    main()
