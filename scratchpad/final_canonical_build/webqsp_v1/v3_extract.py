"""Selective extraction, plus two archive findings recorded before anything is unpacked.

    python scratchpad/final_canonical_build/webqsp_v1/v3_extract.py

Extracts only what checks 2-7 need. Skips the URI conversion maps (6.95 GB) and the three variants
CRAG is not building on (~7.7 GB).

Two things the full manifest showed that the top-30 listing did not:

  1. There is NO reverse-properties file anywhere in the archive. The /type/property/reverse_property
     assertions that FB3.sh consumes are an input to their pipeline, not an output of it. Check 5
     therefore cannot be run as specified from this package alone.
  2. Both +REV variants are inconsistent with their own published triple counts, while both -REV
     variants are consistent. Sizes are recorded below so the claim is checkable.
"""
import json, os, time, zipfile

ZIP = "data/final_canonical/freebase_v3/_acquisition/idir/idirlab-freebases.zip"
DST = "data/final_canonical/freebase_v3/_acquisition/idir/extracted"
REC = "data/final_canonical/freebase_v3/V3_EXTRACTION_RECORD.json"
ACQ = "data/final_canonical/freebase_v3/V3_ACQUISITION_RECORD.json"

WANT = [
    # the backbone CRAG is evaluating
    "idirlab-freebases/FB+CVT-REV/train.txt",
    "idirlab-freebases/FB+CVT-REV/test.txt",
    "idirlab-freebases/FB+CVT-REV/valid.txt",
    "idirlab-freebases/FB+CVT-REV/entity2id.txt",
    "idirlab-freebases/FB+CVT-REV/relation2id.txt",
    # every variant's relation vocabulary: the -REV/+REV set difference is the only empirical
    # handle on the reverse split available without the missing mapping file
    "idirlab-freebases/FB+CVT+REV/relation2id.txt",
    "idirlab-freebases/FB-CVT-REV/relation2id.txt",
    "idirlab-freebases/FB-CVT+REV/relation2id.txt",
    # metadata for rendering, classification and provenance (checks 3, 6, 7)
    "idirlab-freebases/Metadata/object_names.csv",
    "idirlab-freebases/Metadata/object_types.csv",
    "idirlab-freebases/Metadata/object_ids.csv",
    "idirlab-freebases/Metadata/entities_id_label.csv",
    "idirlab-freebases/Metadata/properties_id_label.csv",
    "idirlab-freebases/Metadata/types_id_label.csv",
    "idirlab-freebases/Metadata/domains_id_label.csv",
    "idirlab-freebases/TypeSystem/freebase_endtypes.csv",
]

SKIP_REASON = {
    "Metadata/uri_original2simplified.json": "3.48 GB URI conversion map; not needed for checks 2-7",
    "Metadata/uri_simplified2original.json": "3.48 GB URI conversion map; not needed for checks 2-7",
    "FB+CVT+REV/{train,test,valid,entity2id}": "not the backbone; only its relation2id is needed",
    "FB-CVT-REV/{train,test,valid,entity2id}": "not the backbone; only its relation2id is needed",
    "FB-CVT+REV/{train,test,valid,entity2id}": "not the backbone; only its relation2id is needed",
}

PUBLISHED = {
    "FB+CVT-REV": 134213735,
    "FB-CVT-REV": 125124274,
    "FB+CVT+REV": 244112599,
    "FB-CVT+REV": 238981274,
}


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    sizes = {i.filename: i.file_size for i in z.infolist()}

    # ---- finding 2, computed before extraction ----
    consistency = {}
    for v, n in PUBLISHED.items():
        b = sum(sizes.get(f"idirlab-freebases/{v}/{p}.txt", 0)
                for p in ("train", "test", "valid"))
        consistency[v] = {
            "published_triples": n,
            "triple_file_bytes": b,
            "bytes_per_triple": round(b / n, 2),
            "train_bytes": sizes.get(f"idirlab-freebases/{v}/train.txt", 0),
            "test_bytes": sizes.get(f"idirlab-freebases/{v}/test.txt", 0),
            "valid_bytes": sizes.get(f"idirlab-freebases/{v}/valid.txt", 0),
        }
    for v, c in consistency.items():
        c["plausible"] = c["bytes_per_triple"] >= 10.0
    print(json.dumps(consistency, indent=1), flush=True)

    missing = [w for w in WANT if w not in sizes]
    got = []
    for w in WANT:
        if w not in sizes:
            continue
        z.extract(w, DST)
        got.append({"name": w, "bytes": sizes[w]})
        print(f"[extract] {w} {sizes[w]/1e6:.1f} MB t={time.time()-t0:.0f}s", flush=True)

    doc = {
        "schema": "V3_EXTRACTION_RECORD/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": ZIP,
        "archive_sha256": json.load(open(ACQ, encoding="utf-8"))["sha256_delivered"],
        "destination": DST,
        "extracted": got,
        "extracted_n": len(got),
        "extracted_bytes": sum(g["bytes"] for g in got),
        "extracted_gb": round(sum(g["bytes"] for g in got) / 1e9, 2),
        "requested_but_absent": missing,
        "deliberately_skipped": SKIP_REASON,

        "FINDING_1_NO_REVERSE_PROPERTY_FILE": {
            "statement": "the archive contains no reverse-properties table and no "
                         "/type/property/reverse_property mapping in any form. The 35 real files "
                         "are four variants x (train/test/valid/entity2id/relation2id), seven "
                         "Metadata files, two URI maps and one TypeSystem file.",
            "consequence": "check 5 CANNOT be run as preregistered from this package. "
                           "REVERSE_PAIR_ASSERTIONS_N, OBJECT_SIDE_PREDICATES_N and "
                           "SUBJECT_SIDE_PREDICATES_N all require the mapping, which is an INPUT to "
                           "IDIR's pipeline rather than an output of it.",
            "what_can_still_be_done": "the set difference between each variant's relation2id and "
                                      "its +REV counterpart identifies the removed predicates "
                                      "empirically. relation2id files are small and appear "
                                      "complete, so this works even where the triple files do not.",
            "what_cannot": "NEITHER_SIDE_PRESENT_N. Detecting a pair whose BOTH halves were "
                           "annihilated requires knowing the pair existed, and only the mapping "
                           "says that. This is the first concrete thing IDIR is missing that CRAG "
                           "needs, and it bears directly on the raw-mirror decision.",
            "cheap_path_if_needed": "the reverse_property assertions are a few thousand triples. "
                                    "Recovering them means streaming the 31.3 GB dump with a "
                                    "predicate filter -- the download is the cost, the extraction "
                                    "is trivial. No decompression to disk required.",
        },

        "FINDING_2_PLUS_REV_VARIANTS_INCONSISTENT": {
            "statement": "both -REV variants are size-consistent with their published triple "
                         "counts; both +REV variants are not, by more than an order of magnitude.",
            "per_variant": consistency,
            "reading": "at roughly 20 bytes per id-encoded triple, FB+CVT-REV and FB-CVT-REV land "
                       "at 20.9 and 19.8 bytes/triple. FB+CVT+REV lands at 4.4 and FB-CVT+REV at "
                       "2.1, which no plausible encoding supports. FB-CVT+REV's train.txt is 11.5 "
                       "MB against test and valid at 239 MB each -- train an order of magnitude "
                       "SMALLER than its own test split.",
            "impact_on_us": "none for the backbone. FB+CVT-REV, the variant CRAG is evaluating, is "
                            "one of the two consistent ones. The +REV variants are needed only for "
                            "their relation2id vocabularies, which are separate small files and "
                            "appear intact (0.20 MB vs 0.12 MB, a 1.67x ratio matching the "
                            "published 4,425 vs 2,641 relation counts).",
            "not_asserted": "why the +REV triple files are short. Truncation, a different split "
                            "policy and a packaging error are all consistent with the sizes alone. "
                            "Do not guess; it does not block the backbone.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REC)
    print(json.dumps({k: doc[k] for k in ("extracted_n", "extracted_gb", "requested_but_absent")},
                     indent=1))


if __name__ == "__main__":
    main()
