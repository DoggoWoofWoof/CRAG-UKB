# -*- coding: utf-8 -*-
"""
FINAL VALIDATION OF THE FREEBASE INFERENCE LAYER
================================================
Everything that must be true before CRAG_FREEBASE_INFERENCE_V1 can be frozen, established by
measurement rather than by trusting the artifacts that are being frozen.

The admissibility sidecar is positional -- it carries a reject_reason and no node_uid -- so
"the sidecar lines up with the overlay" is an assumption, not an observation. The strongest
answer is to not depend on the alignment at all: check A measures it, and check B then
re-derives every one of the 64,038,024 decisions from the overlay's own inferred_name using
the identical head rule and compares row-for-row against what is stored. If B passes, the
sidecar is reproducible from the overlay and the alignment is proved rather than assumed.

CHECKS
  A  STRUCTURE      overlay and sidecar have the same parts, in the same order, with the
                    same per-part row counts
  B  REPRODUCIBLE   the stored reject_reason is exactly what the published rule produces
  C  NO_BOOKKEEPING no row that survives as ADMISSIBLE has a bookkeeping relation head
  D  UID_UNIQUE     node_uid is unique across the overlay -- one inferred name per node
  E  FLOOR_INTACT   the inference layer never lands on a node that already has an ORIGINAL
                    name. This is the standing rule that the baseline is a floor the new pass
                    may not degrade, checked as a set disjointness rather than asserted.
  F  NAME_SANITY    admissible names are non-empty, carry no U+FFFD replacement character,
                    and are never the frozen floor string dressed up as an upgrade
  G  ACCOUNTING     per-source totals reconcile exactly with the frozen admissibility record

A note on one thing that is NOT a defect: names contain " — " (U+2014 between spaces) as a
template boundary, and a console that is not UTF-8 renders that as a question mark or a
replacement glyph. F therefore tests the actual code points, not what a terminal prints.
"""
import glob
import io
import json
import os
import re
import sys
import time
from collections import Counter

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np                      # noqa: E402
import pyarrow.parquet as pq            # noqa: E402
import pyarrow.compute as pc            # noqa: E402

FB = "data/final_canonical/freebase_v3"
INF = FB + "/inference_overlay_v1"
ADM = FB + "/inference_admissibility_v1"
NAM = FB + "/inferred_name_v1"
REC = "scratchpad/fb/admissibility.json"
OUT = "scratchpad/fb/validate_inference.json"

# identical to build_admissibility.py -- reproduced, not imported, so a drift shows up as a
# check-B failure instead of silently agreeing with itself
BOOK_HEAD = {"last referenced by", "freeq", "permission", "permissions", "creator", "created by",
             "timestamp", "id", "guid", "key", "namespace", "attribution", "provenance",
             "last referenced", "referenced by", "write permission", "read permission"}
SPLIT = re.compile(r" of | — | in | on | for | as | about | at ")
ADMISSIBLE, REJECT_BOOKKEEPING = 0, 1
FLOOR = "Unnamed Freebase entity"


def head_of(v):
    v = v or ""
    m = SPLIT.search(v)
    return (v[:m.start()] if m else v).strip().lower()


def main():
    t0 = time.time()
    frozen = json.load(io.open(REC, encoding="utf-8"))
    inf = sorted(glob.glob(INF + "/*.parquet"))
    adm = sorted(glob.glob(ADM + "/*.parquet"))
    R = {"RECORD": "CRAG_FREEBASE_INFERENCE_VALIDATION_V1",
         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "checks": {}}

    # ---- A  structure -----------------------------------------------------------------
    a = {"overlay_parts": len(inf), "sidecar_parts": len(adm)}
    a["same_basenames"] = ([os.path.basename(x) for x in inf]
                           == [os.path.basename(x) for x in adm])
    rows_i = [pq.ParquetFile(x).metadata.num_rows for x in inf]
    rows_a = [pq.ParquetFile(x).metadata.num_rows for x in adm]
    a["row_counts_match"] = rows_i == rows_a
    a["total_rows"] = int(sum(rows_i))
    a["matches_frozen_total"] = (a["total_rows"] == frozen["TOTAL_ROWS"])
    a["PASS"] = bool(a["same_basenames"] and a["row_counts_match"] and a["matches_frozen_total"]
                     and len(inf) == len(adm))
    R["checks"]["A_STRUCTURE"] = a
    print("A STRUCTURE      parts=%d rows=%s basenames=%s counts=%s %s"
          % (len(inf), format(a["total_rows"], ","), a["same_basenames"],
             a["row_counts_match"], "PASS" if a["PASS"] else "FAIL"), flush=True)
    if not a["PASS"]:
        json.dump(R, io.open(OUT, "w", encoding="utf-8"), indent=1)
        sys.exit("structure failed; nothing downstream is meaningful")

    # ---- B/C/D/F  one pass over the layer ---------------------------------------------
    mismatch = 0
    adm_bookkeeping = 0
    empty_name = 0
    fffd = 0
    floor_named = 0
    rej = 0
    by_src = Counter()
    rows_by_src = Counter()
    uids = []
    ex_mismatch = []
    for i, (fi, fa) in enumerate(zip(inf, adm)):
        t = pq.read_table(fi, columns=["node_uid", "inferred_name", "inference_source"])
        stored = pq.read_table(fa, columns=["reject_reason"])["reject_reason"].to_numpy(
            zero_copy_only=False).astype(np.int8)
        uids.append(t["node_uid"].to_numpy(zero_copy_only=False).astype(np.int64))
        nm = t["inferred_name"].combine_chunks()
        src = pc.cast(t["inference_source"], "string").combine_chunks().to_numpy(
            zero_copy_only=False)
        dd = nm.dictionary_encode()
        dic = dd.dictionary.to_pylist()
        idx = dd.indices.to_numpy(zero_copy_only=False)

        bad = np.zeros(len(dic), bool)
        blank = np.zeros(len(dic), bool)
        repl = np.zeros(len(dic), bool)
        isfloor = np.zeros(len(dic), bool)
        for j, v in enumerate(dic):
            v = v or ""
            bad[j] = head_of(v) in BOOK_HEAD
            blank[j] = not v.strip()
            repl[j] = "�" in v
            isfloor[j] = v.strip() == FLOOR
        recomputed = np.where(bad[idx], REJECT_BOOKKEEPING, ADMISSIBLE).astype(np.int8)

        d = np.nonzero(recomputed != stored)[0]
        mismatch += int(d.size)
        for k in d[:3]:
            if len(ex_mismatch) < 5:
                ex_mismatch.append({"part": os.path.basename(fi), "row": int(k),
                                    "name": dic[idx[k]][:120],
                                    "stored": int(stored[k]), "recomputed": int(recomputed[k])})
        ok = stored == ADMISSIBLE
        adm_bookkeeping += int((ok & bad[idx]).sum())
        empty_name += int((ok & blank[idx]).sum())
        fffd += int((ok & repl[idx]).sum())
        floor_named += int((ok & isfloor[idx]).sum())
        m = stored != ADMISSIBLE
        rej += int(m.sum())
        for s in np.unique(src):
            sm = src == s
            rows_by_src[s] += int(sm.sum())
            by_src[s] += int((m & sm).sum())
        print("  [%2d/%d] rows=%s mism=%d adm_book=%d  %.0fs"
              % (i + 1, len(inf), format(int(stored.size), ","), mismatch,
                 adm_bookkeeping, time.time() - t0), flush=True)

    uids = np.concatenate(uids)
    R["checks"]["B_REPRODUCIBLE"] = {
        "rows_compared": int(uids.size), "mismatches": mismatch,
        "examples": ex_mismatch,
        "means": ("the stored sidecar is exactly what the published rule produces from the "
                  "overlay, so its positional alignment is proved, not assumed"),
        "PASS": mismatch == 0}
    R["checks"]["C_NO_BOOKKEEPING"] = {
        "admissible_rows_with_bookkeeping_head": adm_bookkeeping,
        "PASS": adm_bookkeeping == 0}
    print("B REPRODUCIBLE   mismatches=%d %s" % (mismatch, "PASS" if mismatch == 0 else "FAIL"),
          flush=True)
    print("C NO_BOOKKEEPING admissible_with_bookkeeping_head=%d %s"
          % (adm_bookkeeping, "PASS" if adm_bookkeeping == 0 else "FAIL"), flush=True)

    su = np.sort(uids)
    dup = int((np.diff(su) == 0).sum())
    R["checks"]["D_UID_UNIQUE"] = {"n": int(su.size), "duplicate_node_uids": dup,
                                   "PASS": dup == 0}
    print("D UID_UNIQUE     n=%s duplicates=%d %s"
          % (format(int(su.size), ","), dup, "PASS" if dup == 0 else "FAIL"), flush=True)

    R["checks"]["F_NAME_SANITY"] = {
        "admissible_empty": empty_name, "admissible_with_U+FFFD": fffd,
        "admissible_equal_to_floor": floor_named, "floor_string": FLOOR,
        "note": ("names legitimately contain U+2014 as a template boundary; a non-UTF-8 "
                 "console renders that as a replacement glyph, so this counts code points"),
        "PASS": bool(empty_name == 0 and fffd == 0 and floor_named == 0)}
    print("F NAME_SANITY    empty=%d fffd=%d equal_to_floor=%d %s"
          % (empty_name, fffd, floor_named,
             "PASS" if R["checks"]["F_NAME_SANITY"]["PASS"] else "FAIL"), flush=True)

    # ---- E  the floor is not degraded --------------------------------------------------
    orig = []
    n_named = 0
    for f in sorted(glob.glob(NAM + "/*.parquet")):
        t = pq.read_table(f, columns=["node_uid", "is_original_name"])
        b = t["is_original_name"].to_numpy(zero_copy_only=False).astype(bool)
        n_named += int(t.num_rows)
        if b.any():
            orig.append(t["node_uid"].to_numpy(zero_copy_only=False).astype(np.int64)[b])
    orig = np.concatenate(orig) if orig else np.zeros(0, np.int64)
    p = np.searchsorted(su, orig)
    hit = (p < su.size) & (su[np.minimum(p, su.size - 1)] == orig)
    n_overwrite = int(hit.sum())
    R["checks"]["E_FLOOR_INTACT"] = {
        "name_layer_rows": n_named, "original_named_nodes": int(orig.size),
        "overlay_rows_landing_on_an_original_name": n_overwrite,
        "means": ("inference only ever names a node that had no original name, so the frozen "
                  "baseline is a floor the layer cannot degrade"),
        "PASS": n_overwrite == 0}
    print("E FLOOR_INTACT   original_named=%s overlay_overwrites=%d %s"
          % (format(int(orig.size), ","), n_overwrite,
             "PASS" if n_overwrite == 0 else "FAIL"), flush=True)

    # ---- G  accounting -----------------------------------------------------------------
    g = {"rejected_total": rej, "frozen_rejected_total":
         frozen["REJECT_BOOKKEEPING_RELATION_NAME_N"],
         "rows_by_source": {k: int(v) for k, v in rows_by_src.items()},
         "rejected_by_source": {k: int(v) for k, v in by_src.items()}}
    g["rows_by_source_match"] = (g["rows_by_source"] == frozen["ROWS_BY_SOURCE"])
    g["rejected_by_source_match"] = (g["rejected_by_source"] == frozen["REJECTED_BY_SOURCE"])
    g["PASS"] = bool(g["rows_by_source_match"] and g["rejected_by_source_match"]
                     and rej == frozen["REJECT_BOOKKEEPING_RELATION_NAME_N"])
    R["checks"]["G_ACCOUNTING"] = g
    print("G ACCOUNTING     rejected=%s rows_by_src=%s rejected_by_src=%s %s"
          % (format(rej, ","), g["rows_by_source_match"], g["rejected_by_source_match"],
             "PASS" if g["PASS"] else "FAIL"), flush=True)

    R["ADMISSIBLE_N"] = int(uids.size) - rej
    R["REJECTED_N"] = rej
    R["ALL_PASS"] = bool(all(v["PASS"] for v in R["checks"].values()))
    R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nALL_PASS =", R["ALL_PASS"], " %.1fs -> %s" % (R["elapsed_s"], OUT))


if __name__ == "__main__":
    main()
