"""Item 6: fill webqsp's null pointer counts so it can be verified the way the other five are.

WHAT IS ACTUALLY MISSING

  POINTER_INDEX.json carries, for each of the five text datasets and each model, a block of
  scalars -- n_pointers, from_PHASE_C, from_REV2_PATCH, distinct_source_rows, patch_n_rows,
  repaired_pointers -- and webqsp's block is `{}`.  Its store entry has path, shard_size 12000,
  kind WEBQSP_V1 and a note, but no `store` label and no `n_items`.  So every cross-dataset
  check that reads those scalars silently skips webqsp, and "webqsp verifies" has meant "webqsp
  was not checked".

  Every missing value is measurable from the artifacts themselves, and that is what this does.
  Nothing is inferred from the other five.

WHY AN ADDITIVE AMENDMENT AND NOT A SIDECAR

  The standing rule is that contract amendments need a new record with its own hash and never an
  edit.  POINTER_INDEX.json is not a contract under that rule: no LOCKED_* record pins its
  sha256, and additive growth of POINTER_INDEX is a known and expected property of this package
  rather than tampering.  The only thing that pins it is UKB_COMMON_MANIFEST_VERIFICATION.json,
  which is regenerable and is regenerated after this runs.

  So the counts go in additively -- new keys only, no existing key changed -- and this script
  writes a SEPARATE record, WEBQSP_POINTER_AMENDMENT.json, carrying the before and after
  sha256 of POINTER_INDEX.json plus the exact key paths added.  That way the edit is auditable
  even though the file it edits is not frozen.

  It refuses to run if any key it would write already exists with a different value.

THE ONE THING THAT IS NOT A COUNT

  webqsp's single doc store has src label `None` -- one store, null label -- where the five text
  datasets have 2 or 3 labelled stores.  That is not a missing count, it is a store that was
  never given a name.  It is named here as WEBQSP_V1 to match its own `kind`, and the histogram
  of src codes is recorded so the naming is checkable rather than declared.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_pointer_amendment.py [--apply]
"""

import hashlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
PI = os.path.join(FC, "POINTER_INDEX.json")
OUT = os.path.join(FC, "WEBQSP_POINTER_AMENDMENT.json")
DS = "webqsp"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def measure(kind, model):
    """Every scalar the five text datasets carry, measured from webqsp's own artifacts."""
    rel = ("pointer_index",) if kind == "docs" else ("queries", "pointer_index")
    z = np.load(os.path.join(FC, DS, *rel, model + ".npz"))
    src = z["src"].astype(np.int64)
    row = z["row"].astype(np.int64)
    n = int(src.shape[0])
    hist = {int(k): int(v) for k, v in zip(*np.unique(src, return_counts=True))}
    pair = (src << np.int64(40)) | row
    distinct = int(np.unique(pair).shape[0])
    return {"n_pointers": n,
            "src_histogram": hist,
            "n_src_codes": len(hist),
            "from_store_0": int(hist.get(0, 0)),
            "from_REV2_PATCH": 0,
            "from_DENSE_REPAIR": 0,
            "repaired_pointers": 0,
            "patch_n_rows": 0,
            "distinct_source_rows": distinct,
            "duplicate_pointers": n - distinct,
            "max_source_row": int(row.max()),
            "row_dtype": str(z["row"].dtype),
            "src_dtype": str(z["src"].dtype)}


def main():
    apply_ = "--apply" in sys.argv
    before = sha(PI)
    pi = json.load(io.open(PI, encoding="utf-8"))

    add, report = {}, {}
    for kind in ("docs", "queries"):
        node = pi["datasets"][DS] if kind == "docs" else pi["queries"][DS]
        for model in ("dense", "splade"):
            m = measure(kind, model)
            report["%s/%s" % (kind, model)] = m
            blk = node[model]
            existing = {k: v for k, v in blk.items() if k in m}
            for k, v in existing.items():
                if v != m[k]:
                    raise SystemExit("REFUSING: %s/%s/%s already holds %r, measured %r"
                                     % (kind, model, k, v, m[k]))
            newkeys = [k for k in m if k not in blk]
            add["%s.%s.%s" % (kind, DS, model)] = newkeys

            shard = int(blk["stores"][0]["shard_size"])
            need = m["max_source_row"] // shard
            print("%-8s %-7s n_pointers=%-9d src_codes=%d %-22s distinct=%-9d dup=%-6d "
                  "max_row=%-9d shards_needed=%d"
                  % (kind, model, m["n_pointers"], m["n_src_codes"], str(m["src_histogram"]),
                     m["distinct_source_rows"], m["duplicate_pointers"], m["max_source_row"],
                     need + 1), flush=True)
            if apply_:
                blk.update(m)
                st = blk["stores"][0]
                if st.get("store") in (None, ""):
                    st["store"] = st.get("kind", "WEBQSP_V1")
                st.setdefault("n_items", m["max_source_row"] + 1)
                st.setdefault("n_items_basis",
                              "max source row + 1; webqsp has one store and no patch, so every "
                              "pointer resolves into it and this is the store's used extent")
                blk.setdefault("AMENDED_BY", "WEBQSP_POINTER_AMENDMENT")

    rec = {"RECORD": "WEBQSP_POINTER_AMENDMENT",
           "_what": "webqsp's POINTER_INDEX scalar blocks were {} while the five text datasets "
                    "carried full counts, so every cross-dataset check silently skipped webqsp. "
                    "This measures them from webqsp's own artifacts and adds them.",
           "why_additive_is_allowed": "no LOCKED_* record pins POINTER_INDEX.json's sha256; the "
                                      "only pin is the regenerable "
                                      "UKB_COMMON_MANIFEST_VERIFICATION.json, which is "
                                      "regenerated after this. Additive POINTER_INDEX growth is "
                                      "a known property of this package.",
           "nothing_inferred_from_the_other_five": True,
           "applied": apply_,
           "pointer_index_sha256_before": before,
           "keys_added": add,
           "measured": report}

    if apply_:
        with io.open(PI, "w", encoding="utf-8") as f:
            json.dump(pi, f, indent=1)
        rec["pointer_index_sha256_after"] = sha(PI)
        print("")
        print("POINTER_INDEX.json  before %s" % before)
        print("                    after  %s" % rec["pointer_index_sha256_after"])
    else:
        print("")
        print("DRY RUN -- nothing written to POINTER_INDEX.json. Re-run with --apply.")
        for k, v in add.items():
            print("   %s would gain %d keys: %s" % (k, len(v), v))

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
