# -*- coding: utf-8 -*-
"""
INFERENCE VALIDATION, PART 2: the two checks part 1 could not settle
====================================================================

E  FLOOR_INTACT, redone against the right layer and with a positive control
    Part 1 asked whether any inferred node already had an ORIGINAL name by reading
    is_original_name from inferred_name_v1. That column is False on all 69,243,435 rows,
    because inferred_name_v1 IS the inferred tier -- so the check returned zero for the same
    reason a query against an empty table returns zero. Under this project's standing
    ZERO_JOIN_INVARIANT a clean zero from a large source is not evidence until a positive
    control shows the join reaches the population at all, so that result is discarded rather
    than counted as a pass.

    The original names live in overlay_v1 -- CRAG_FREEBASE_RESOLUTION_OVERLAY_V1, frozen,
    301,977,131 rows, one per node in the canonical universe, carrying is_original_name.
    The check is now:

        POSITIVE CONTROL   every one of the 64,038,024 inferred node_uids must be FOUND in
                           the frozen overlay. If the join lands on nothing, nothing below
                           it means anything.
        THE CLAIM          of the nodes that join, exactly zero may already carry an
                           original name.

    Both must hold. The control is what makes the zero admissible as evidence.

F  MOJIBAKE, enumerated rather than merely counted
    110 admissible generated names contain U+FFFD. These are not a console artifact: the
    check counts code points, and the damage is inherited from neighbour titles that were
    already corrupt in the source -- Cyrillic and CJK titles replaced wholesale, plus
    Latin-1 mojibake like "Sp?ka" for "Spolka" and "? la Mode" for "a la Mode".

    They are enumerated in full, with the fraction of the IDENTITY segment -- the part before
    the template boundary, which is the part that is supposed to carry the specific meaning --
    that is replacement characters. That fraction is what decides whether a row is an
    unreadable name or a readable name with one bad glyph, and the numbers are written down
    before any rule is proposed.

Nothing here edits a frozen artifact. The admissibility sidecar stays exactly as it is; per
the contract rule an amendment is a NEW record with its own hash, never an edit.
"""
import glob
import io
import json
import os
import re
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np                      # noqa: E402
import pyarrow.parquet as pq            # noqa: E402
import pyarrow.compute as pc            # noqa: E402

FB = "data/final_canonical/freebase_v3"
INF = FB + "/inference_overlay_v1"
ADM = FB + "/inference_admissibility_v1"
FROZEN_OVERLAY = FB + "/overlay_v1"
OUT = "scratchpad/fb/validate_inference2.json"
SPLIT = re.compile(r" of | — | in | on | for | as | about | at ")
REPL = "�"


def main():
    t0 = time.time()
    R = {"RECORD": "CRAG_FREEBASE_INFERENCE_VALIDATION_V1_PART2",
         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}

    inf_files = sorted(glob.glob(INF + "/*.parquet"))
    adm_files = sorted(glob.glob(ADM + "/*.parquet"))

    # ---------- collect inferred uids, and enumerate the mojibake rows in the same pass ----
    uids = []
    moj = []
    n_adm = 0
    for fi, fa in zip(inf_files, adm_files):
        t = pq.read_table(fi, columns=["node_uid", "inferred_name", "inference_source",
                                       "inference_rule"])
        u = t["node_uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        uids.append(u)
        r = pq.read_table(fa, columns=["reject_reason"])["reject_reason"].to_numpy(
            zero_copy_only=False)
        ok = r == 0
        n_adm += int(ok.sum())
        nm = t["inferred_name"].combine_chunks()
        dd = nm.dictionary_encode()
        dic = dd.dictionary.to_pylist()
        idx = dd.indices.to_numpy(zero_copy_only=False)
        badname = np.zeros(len(dic), bool)
        for j, v in enumerate(dic):
            badname[j] = bool(v) and REPL in v
        hit = np.nonzero(ok & badname[idx])[0]
        if hit.size:
            src = pc.cast(t["inference_source"], "string").to_pylist()
            rul = pc.cast(t["inference_rule"], "string").to_pylist()
            for k in hit:
                v = dic[idx[k]]
                m = SPLIT.search(v)
                ident = v[:m.start()] if m else v
                nrep = ident.count(REPL)
                moj.append({"node_uid": int(u[k]), "name": v,
                            "identity_segment": ident,
                            "identity_len": len(ident),
                            "identity_replacement_chars": nrep,
                            "identity_replacement_frac": (round(nrep / len(ident), 4)
                                                          if ident else None),
                            "inference_source": src[k], "inference_rule": rul[k]})
        del t, nm, dd
    uids = np.concatenate(uids)
    su = np.sort(uids)
    print("inferred rows=%s admissible=%s mojibake_rows=%d  %.0fs"
          % (format(int(su.size), ","), format(n_adm, ","), len(moj), time.time() - t0),
          flush=True)

    # ---------- E: join against the FROZEN name layer, with a positive control -------------
    ov = sorted(glob.glob(FROZEN_OVERLAY + "/*.parquet"))
    n_rows = n_joined = n_joined_original = n_original_total = 0
    ex = []
    for i, f in enumerate(ov):
        t = pq.read_table(f, columns=["node_uid", "is_original_name"])
        u = t["node_uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        b = t["is_original_name"].to_numpy(zero_copy_only=False).astype(bool)
        n_rows += int(u.size)
        n_original_total += int(b.sum())
        p = np.searchsorted(su, u)
        j = (p < su.size) & (su[np.minimum(p, su.size - 1)] == u)
        n_joined += int(j.sum())
        k = j & b
        n_joined_original += int(k.sum())
        if k.any() and len(ex) < 5:
            for q in np.nonzero(k)[0][:5 - len(ex)]:
                ex.append(int(u[q]))
        if (i + 1) % 40 == 0 or i + 1 == len(ov):
            print("  overlay [%3d/%d] rows=%s joined=%s joined_with_original=%d  %.0fs"
                  % (i + 1, len(ov), format(n_rows, ","), format(n_joined, ","),
                     n_joined_original, time.time() - t0), flush=True)

    control_ok = n_joined == int(su.size)
    R["checks"]["E_FLOOR_INTACT"] = {
        "frozen_overlay": FROZEN_OVERLAY,
        "frozen_overlay_rows": n_rows,
        "frozen_overlay_original_named_nodes": n_original_total,
        "inferred_nodes": int(su.size),
        "POSITIVE_CONTROL_inferred_nodes_found_in_overlay": n_joined,
        "POSITIVE_CONTROL_PASS": control_ok,
        "inferred_nodes_that_already_had_an_original_name": n_joined_original,
        "examples": ex,
        "why_the_control": ("a zero here is only evidence if the join reaches the "
                            "population; part 1's zero came from an empty column and is "
                            "discarded"),
        "PASS": bool(control_ok and n_joined_original == 0)}
    e = R["checks"]["E_FLOOR_INTACT"]
    print("\nE FLOOR_INTACT  control %s/%s (%s)  already_named=%d  %s"
          % (format(n_joined, ","), format(int(su.size), ","),
             "PASS" if control_ok else "FAIL", n_joined_original,
             "PASS" if e["PASS"] else "FAIL"), flush=True)

    # ---------- F: the mojibake population, described before any rule is proposed ----------
    fr = [m["identity_replacement_frac"] for m in moj
          if m["identity_replacement_frac"] is not None]
    fr_sorted = sorted(fr)
    R["checks"]["F_MOJIBAKE"] = {
        "admissible_rows_with_U+FFFD": len(moj),
        "as_fraction_of_admissible": (round(len(moj) / float(n_adm), 12) if n_adm else None),
        "identity_replacement_frac": {
            "min": fr_sorted[0] if fr_sorted else None,
            "median": fr_sorted[len(fr_sorted) // 2] if fr_sorted else None,
            "max": fr_sorted[-1] if fr_sorted else None,
            "n_ge_0.5": int(sum(1 for x in fr if x >= 0.5)),
            "n_eq_1.0": int(sum(1 for x in fr if x >= 0.999)),
            "n_lt_0.5": int(sum(1 for x in fr if x < 0.5))},
        "by_source": {},
        "rows": moj,
        "note": ("the damage is inherited from neighbour titles that were already corrupt "
                 "upstream; the template itself is intact"),
        "PASS": None}
    for m in moj:
        R["checks"]["F_MOJIBAKE"]["by_source"][m["inference_source"]] = \
            R["checks"]["F_MOJIBAKE"]["by_source"].get(m["inference_source"], 0) + 1
    d = R["checks"]["F_MOJIBAKE"]["identity_replacement_frac"]
    print("F MOJIBAKE      n=%d  identity replacement frac: min=%s median=%s max=%s  "
          "fully_unreadable=%d  partly=%d"
          % (len(moj), d["min"], d["median"], d["max"], d["n_eq_1.0"], d["n_lt_0.5"]),
          flush=True)

    R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nwrote", OUT, " %.1fs" % R["elapsed_s"])


if __name__ == "__main__":
    main()
