"""Is every damaged dense row covered by a repair redirect? The webqsp case decides it.

WHAT IS ESTABLISHED

  The repair is REGISTERED and EFFECTIVE where it is wired. At the canonical positions whose
  pointer src is DENSE_REPAIR, the resolver returns the patch bytes exactly -- max abs delta
  0.00000000 over 256 sampled musique rows, 256/256 byte-identical.

  Wired DENSE_REPAIR positions, counted from the pointer indices themselves:
    musique  docs 1,473                     squad  queries 2,220
    metaqa   queries 2,709 + 1,665 + 3,351  hotpotqa docs 47,732 / queries 4,851
    2wiki    docs 63,138 / queries 1,601
  total 128,740.

  webqsp is the exception: its dense doc pointer has ONE src covering all 2,592,894 rows,
  and _dense_repair_patch/webqsp__docs (12,814 rows) is referenced by nothing.

WHY THE EARLIER "LIVE_ROWS_CLEAN" DID NOT ANSWER THIS

  It sampled the patch's dense_rows.json values as canonical positions. They are not: that
  file records rows in the SOURCE STORE's addressing (contiguous, 54,911+ for musique) while
  the pointer index maps canonical position -> patch array index as an unsorted scatter
  (positions 56, 99, 214 ... -> patch rows 894, 187, 558 ...). So it read 2,048 arbitrary
  healthy rows and pronounced them healthy. True, and beside the point.

THE ACTUAL INVARIANT

  For every dataset:  { rows whose vector is all-zero or non-finite }
                        SUBSET OF
                      { canonical positions redirected to a repair store }

  Damage is all-zero rows, and that is measured rather than assumed: reading musique's raw
  PHASE_C bytes at the positions the pointer redirects AWAY from gives 1,471 all-zero of
  1,473, norm mean 0.0008.

  So walk every live resolved row of a store, count the damaged ones, and check they are
  exactly the redirected set. A dataset with damaged rows and no redirect covering them is
  serving zero vectors, which retrieve nothing and score 0 against every query.

  webqsp docs is 2,592,894 x 1536 fp16 = 7.96 GB and the six datasets together are ~43 GB, so
  stores are walked in shard order with progress, largest question first.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/repair_coverage_check.py [dataset ...]
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "REPAIR_COVERAGE_CHECK.json")
sys.path.insert(0, FC)
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

BATCH = 8192
# webqsp first: it is the only open question. Then the rest, cheapest last.
ORDER = [("webqsp", "docs"), ("webqsp", "queries"),
         ("musique", "docs"), ("squad", "queries"), ("metaqa", "queries"),
         ("metaqa", "docs"), ("musique", "queries"), ("squad", "docs"),
         ("hotpotqa", "queries"), ("2wiki", "queries"),
         ("hotpotqa", "docs"), ("2wiki", "docs")]


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def repair_positions(ds, kind, pi):
    """Canonical positions the pointer index redirects into a repair/patch store."""
    sec = "datasets" if kind == "docs" else "queries"
    node = ((pi.get(sec) or {}).get(ds) or {}).get("dense") or {}
    st = node.get("stores") or []
    patchy = {i for i, s in enumerate(st)
              if (s.get("store") or "") in ("DENSE_REPAIR", "REV2_PATCH")}
    rel = ("pointer_index",) if kind == "docs" else ("queries", "pointer_index")
    z = np.load(os.path.join(FC, ds, *rel, "dense.npz"))
    src = z["src"]
    if not patchy:
        return set(), src.shape[0], [s.get("store") for s in st]
    mask = np.isin(src, list(patchy))
    return set(np.where(mask)[0].tolist()), src.shape[0], [s.get("store") for s in st]


def scan(ds, kind, n):
    """Damaged canonical positions in the LIVE resolved view of a store."""
    E = CanonicalEmbeddings(ds, "dense", kind=kind, root=FC)
    assert len(E) == n, "resolver length %d != pointer length %d" % (len(E), n)
    bad_zero, bad_nan, nonunit = [], [], 0
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        v = np.asarray(E.gather(list(range(s, e))), dtype=np.float32)
        nm = np.linalg.norm(v, axis=1)
        z = np.where(nm < 1e-6)[0]
        f = np.where(~np.isfinite(nm))[0]
        bad_zero.extend((s + z).tolist())
        bad_nan.extend((s + f).tolist())
        nonunit += int(((np.abs(nm - 1.0) > 2e-2) & (nm >= 1e-6) & np.isfinite(nm)).sum())
        if (s // BATCH) % 40 == 0 or e == n:
            print("     %9d / %9d rows   zero=%d nan=%d nonunit=%d"
                  % (e, n, len(bad_zero), len(bad_nan), nonunit), flush=True)
    return bad_zero, bad_nan, nonunit


def main():
    pi = rj(os.path.join(FC, "POINTER_INDEX.json"))
    want = sys.argv[1:]
    todo = [(d, k) for d, k in ORDER if not want or d in want]

    print("patch stores on disk vs wired positions")
    pdir = os.path.join(FC, "_dense_repair_patch")
    on_disk = {}
    for d in sorted(os.listdir(pdir)):
        rp = os.path.join(pdir, d, "dense_rows.json")
        if os.path.isfile(rp) and "__p" not in d:
            on_disk[d] = len(rj(rp))
    print("  %s" % json.dumps(on_disk))
    print("  total rows in the ten merged patch stores: %d" % sum(on_disk.values()))

    res, fails = {}, []
    for ds, kind in todo:
        rep, n, labels = repair_positions(ds, kind, pi)
        print("")
        print("=" * 96)
        print("%s %s   n=%d   stores=%s   redirected_to_patch=%d"
              % (ds, kind, n, labels, len(rep)))
        print("=" * 96)
        bz, bn, nu = scan(ds, kind, n)
        dmg = set(bz) | set(bn)
        uncovered = sorted(dmg - rep)
        ok = not uncovered
        v = "COVERED" if ok else "UNCOVERED_DAMAGE"
        if not ok:
            fails.append("%s.%s: %d damaged rows with no repair redirect"
                         % (ds, kind, len(uncovered)))
        print("   damaged(zero=%d nan=%d)  redirected=%d  uncovered=%d  nonunit=%d  -> %s"
              % (len(bz), len(bn), len(rep), len(uncovered), nu, v))
        if uncovered[:10]:
            print("   first uncovered canonical positions: %s" % uncovered[:10])
        res["%s.%s" % (ds, kind)] = {
            "rows": n, "store_labels": labels, "redirected_to_patch": len(rep),
            "damaged_zero": len(bz), "damaged_nonfinite": len(bn),
            "uncovered_damaged": len(uncovered),
            "uncovered_first_positions": uncovered[:50],
            "non_unit_norm_rows": nu, "verdict": v}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"RECORD": "REPAIR_COVERAGE_CHECK",
                       "_what": "tests the invariant that every all-zero or non-finite dense row "
                                "is a canonical position the pointer index redirects into a "
                                "repair store. MEASUREMENT ONLY -- nothing was rebuilt.",
                       "damage_signature": "all-zero rows; measured from musique's raw PHASE_C "
                                           "bytes at redirected positions (1,471 of 1,473 "
                                           "all-zero, norm mean 0.0008)",
                       "patch_stores_on_disk": on_disk,
                       "results": res, "failures": fails,
                       "complete": len(res) == len(todo)}, f, indent=1)

    print("")
    print("=" * 96)
    print("INVARIANT  every damaged dense row is covered by a repair redirect: %s"
          % ("YES" if not fails else "NO"))
    for x in fails:
        print("   %s" % x)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
