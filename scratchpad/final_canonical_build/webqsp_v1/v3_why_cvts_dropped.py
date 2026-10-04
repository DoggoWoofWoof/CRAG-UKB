"""What actually removed the 488,337 CVT nodes: the literal filter, or the reverse anti-join?

    python scratchpad/final_canonical_build/webqsp_v1/v3_why_cvts_dropped.py

Check 6 established that IDIR still holds metadata for 488,280 of the 488,337 missing CVTs -- it
knows the objects exist and has their types. So they were not out of scope; something in the
pipeline removed them. Two candidates, with very different consequences for CRAG:

  LITERAL_FILTER   FBDataDump.sh keeps only triples whose subject AND object match ^/m/|^/g/. A CVT
                   whose arguments are dates, numbers or strings loses every edge and falls out of
                   the graph. This is exactly the literal deficit the V3 design predicted. Only the
                   raw dump restores it.

  ANTI_JOIN        FB3.sh removes every triple whose predicate sits on the object side of a
                   reverse_property assertion. A CVT reachable only by such predicates is orphaned.
                   Fixable without the raw dump, by building from FB+CVT+REV instead.

DISCRIMINATOR: FB+CVT+REV is the same corpus WITHOUT the anti-join but WITH the same literal filter.
If the missing CVTs have edges there, the anti-join removed them. If they are missing there too, the
literal filter did. The two variants differ in exactly one step, which is what makes this clean.

Extracts FB+CVT+REV's entity2id and triple files (7.85 GB) using the FINDING_4 true-size rule.
"""
import json, os, struct, time, zipfile, zlib
import numpy as np
import pyarrow.parquet as pq
from pyarrow import csv as pacsv

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
ZIP = f"{ROOT}/idirlab-freebases.zip"
DST = f"{ROOT}/extracted"
OFF = f"{ROOT}/recovered_offsets.json"
IDIR = f"{DST}/idirlab-freebases"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_WHY_CVTS_DROPPED.json"

NEED = ["idirlab-freebases/FB+CVT+REV/entity2id.txt",
        "idirlab-freebases/FB+CVT+REV/train.txt",
        "idirlab-freebases/FB+CVT+REV/test.txt",
        "idirlab-freebases/FB+CVT+REV/valid.txt"]


def inflate(fh, info, out_path):
    fh.seek(info.header_offset)
    h = fh.read(30)
    if h[:4] != b"PK\x03\x04":
        raise ValueError("no local header")
    nl, el = struct.unpack("<HH", h[26:30])
    fh.seek(info.header_offset + 30 + nl + el)
    d = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    left, produced, crc = info.compress_size, 0, 0
    with open(out_path, "wb") as sink:
        while left > 0:
            b = fh.read(min(1 << 22, left))
            if not b:
                break
            left -= len(b)
            c = d.decompress(b) if d else b
            if c:
                produced += len(c)
                crc = zlib.crc32(c, crc)
                sink.write(c)
        if d:
            c = d.flush()
            if c:
                produced += len(c)
                crc = zlib.crc32(c, crc)
                sink.write(c)
    return produced, crc == info.CRC


def main():
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    infos = {i.filename: i for i in z.infolist()}
    for n, o in json.load(open(OFF, encoding="utf-8")).items():
        infos[n].header_offset = o

    ext = []
    with open(ZIP, "rb") as fh:
        for w in NEED:
            out = os.path.join(DST, w)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                ext.append({"name": w, "bytes": os.path.getsize(out), "reused": True})
                print(f"[skip] {w} already present", flush=True)
                continue
            n, ok = inflate(fh, infos[w], out)
            ext.append({"name": w, "bytes": n, "crc_match": ok})
            print(f"[{'ok' if ok else 'FAIL'}] {w} {n/1e9:.2f} GB crc={ok} t={time.time()-t0:.0f}s",
                  flush=True)
            if not ok:
                raise SystemExit("CRC mismatch; refusing to measure on unverified bytes")

    t = pq.read_table(f"{PROBE}/v1_mid_node_kind.parquet")
    kind = dict(zip(t.column("mid").to_pylist(), t.column("node_kind").to_pylist()))
    cvt = {m for m, k in kind.items() if k == "CVT_MEDIATOR"}
    want = {"/" + m.replace(".", "/", 1): m for m in kind}

    # who is in FB+CVT-REV (the backbone) -- recompute rather than trust a cached set
    def graph_members(variant):
        found, max_id = {}, 0
        with open(f"{IDIR}/{variant}/entity2id.txt", encoding="utf-8") as fh:
            for line in fh:
                c = line.rfind(",")
                if c < 0:
                    continue
                i = int(line[c + 1:])
                if i > max_id:
                    max_id = i
                m = want.get(line[:c])
                if m is not None:
                    found[m] = i
        seen = np.zeros(max_id + 1, dtype=bool)
        n_trip = 0
        for split in ("train", "test", "valid"):
            tb = pacsv.read_csv(f"{IDIR}/{variant}/{split}.txt",
                                read_options=pacsv.ReadOptions(autogenerate_column_names=True),
                                convert_options=pacsv.ConvertOptions(
                                    column_types={"f0": "int32", "f1": "int32", "f2": "int32"}))
            s = tb.column(0).to_numpy()
            o = tb.column(2).to_numpy()
            n_trip += len(s)
            seen[s] = True
            seen[o] = True
            del tb, s, o
        out = {m for m, i in found.items() if seen[i]}
        print(f"[{variant}] entity2id ids={max_id+1:,} triples={n_trip:,} "
              f"probe-in-graph={len(out):,} t={time.time()-t0:.0f}s", flush=True)
        return out, n_trip, max_id + 1

    minus, n_minus, ids_minus = graph_members("FB+CVT-REV")
    plus, n_plus, ids_plus = graph_members("FB+CVT+REV")

    missing_minus = cvt - minus
    rescued = missing_minus & plus          # present without the anti-join -> anti-join removed them
    still_gone = missing_minus - plus       # absent from both -> the literal filter removed them
    n_missing = len(missing_minus)

    verdict = ("ANTI_JOIN" if len(rescued) > 0.6 * n_missing else
               "LITERAL_FILTER" if len(still_gone) > 0.6 * n_missing else
               "MIXED")
    doc = {
        "schema": "V3_WHY_CVTS_DROPPED/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extracted": ext,
        "discriminator": "FB+CVT+REV differs from FB+CVT-REV in exactly one pipeline step, the "
                         "reverse anti-join. Both carry the same subject-and-object-must-be-an-MID "
                         "literal filter.",
        "decision_rule_fixed_before_measurement": "ANTI_JOIN if over 60% of the missing CVTs have "
                                                  "edges in FB+CVT+REV; LITERAL_FILTER if over 60% "
                                                  "are missing there too; otherwise MIXED.",
        "v1_cvt_nodes": len(cvt),
        "cvt_in_FB+CVT-REV": len(cvt & minus),
        "cvt_missing_from_FB+CVT-REV": n_missing,
        "of_those_present_in_FB+CVT+REV": len(rescued),
        "of_those_present_in_FB+CVT+REV_pct": round(100 * len(rescued) / n_missing, 3) if n_missing else None,
        "of_those_absent_from_both": len(still_gone),
        "of_those_absent_from_both_pct": round(100 * len(still_gone) / n_missing, 3) if n_missing else None,
        "VERDICT": verdict,
        "cvt_coverage_if_built_on_FB+CVT+REV": {
            "cvt_present": len(cvt & (minus | plus)),
            "cvt_present_pct": round(100 * len(cvt & (minus | plus)) / len(cvt), 3),
            "compare_backbone_pct": round(100 * len(cvt & minus) / len(cvt), 3),
        },
        "variant_stats": {
            "FB+CVT-REV": {"entity2id_ids": ids_minus, "triples": n_minus},
            "FB+CVT+REV": {"entity2id_ids": ids_plus, "triples": n_plus},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: v for k, v in doc.items()
                      if k not in ("extracted", "discriminator",
                                   "decision_rule_fixed_before_measurement")}, indent=1))


if __name__ == "__main__":
    main()
