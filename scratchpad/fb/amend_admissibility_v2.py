# -*- coding: utf-8 -*-
"""
INFERENCE ADMISSIBILITY V2 -- AMENDMENT (a new record, never an edit)
====================================================================
Validation check F found 110 admissible generated names containing U+FFFD. This adds ONE new
reject reason to cover the subset of those that are not names at all, and records it as a
separate artifact with its own hash.

WHY AN AMENDMENT AND NOT A REBUILD
    inference_admissibility_v1 is frozen and row-aligned to inference_overlay_v1 across
    64,038,024 rows. This decision changes 36 of them. Regenerating 64M rows to flip 36 would
    replace a verified artifact with an unverified one for no gain, and the project's contract
    rule is explicit: an amendment is a NEW record with its own hash, never an in-place edit.
    So v1 stays exactly as it is, and this record is an OVERLAY applied on top of it:

        effective_reject_reason(node) = v2_amendment.get(node) or v1[row]

REJECT REASON 2 = UNREADABLE_IDENTITY_PAYLOAD
    The identity segment -- the text before the first template boundary, which is the part
    carrying the specific meaning -- contains no readable character once U+FFFD and whitespace
    are removed.

    This is a PREDICATE, not a tuned threshold, and it follows directly from the governing
    criterion that the inference layer may only upgrade weak structural names into MORE
    SPECIFIC semantic ones. A name whose specific part is entirely replacement characters --
    "############# - notable for Musical Album" -- is not more specific than the structural
    floor it would replace; it is less. Rejecting it returns the node to its frozen display
    name, so the baseline remains the floor and this can never push a node below it.

WHY NOT A FRACTION THRESHOLD
    The obvious rule, "reject when the identity is more than half replacement characters",
    gets a real case wrong. This name is 76% replacement characters:

        "################################ Kak dela? - notable for Musical Album"

    and yet "Kak dela?" (in Cyrillic) survives and names the recording. A 0.5 cutoff destroys
    it; the readability predicate keeps it. The measured split is 36 unreadable / 74 readable,
    against 39 that a 0.5 cutoff would have taken -- so the threshold is not merely inelegant,
    it is wrong on specific rows.

WHAT IS DELIBERATELY NOT REJECTED
    74 rows keep a bad glyph inside an otherwise readable name: "Sp##ka" for Spolka, "Priv#e"
    for Privee, "Pokey # la Mode". These still identify their subject and are still upgrades.
    The damage is inherited from neighbour titles that were ALREADY corrupt in the source; the
    template is intact and the inference did not introduce it. Discarding readable names to
    tidy up an encoding problem the source handed us would lose real coverage.

THIS RE-DERIVES RATHER THAN TRUSTS
    The 36 rows are recomputed here from the frozen parquets, not copied from the validation
    JSON, so the amendment stands on the source it amends.
"""
import glob
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FB = "data/final_canonical/freebase_v3"
INF = FB + "/inference_overlay_v1"
ADM = FB + "/inference_admissibility_v1"
OUT = FB + "/inference_admissibility_v2_amendment.json"
SPLIT = re.compile(r" of | — | in | on | for | as | about | at ")   # identical to the gate
REPL = "�"


def identity_segment(name):
    m = SPLIT.search(name)
    return name[:m.start()] if m else name


def unreadable(ident):
    """No readable character survives once replacement chars and whitespace are removed."""
    return not ident.replace(REPL, "").strip()


def main():
    t0 = time.time()
    inf_files = sorted(glob.glob(INF + "/*.parquet"))
    adm_files = sorted(glob.glob(ADM + "/*.parquet"))
    if len(inf_files) != len(adm_files):
        raise SystemExit("overlay/sidecar part counts differ")

    rejects, keeps = [], []
    n_rows = n_adm = 0
    for fi, fa in zip(inf_files, adm_files):
        t = pq.read_table(fi, columns=["node_uid", "inferred_name", "inference_source",
                                       "inference_rule"])
        r = pq.read_table(fa, columns=["reject_reason"])["reject_reason"].to_numpy(
            zero_copy_only=False)
        if r.size != t.num_rows:
            raise SystemExit("row counts differ within a part: %s" % fi)
        n_rows += int(t.num_rows)
        ok = r == 0
        n_adm += int(ok.sum())

        nm = t["inferred_name"].combine_chunks().dictionary_encode()
        dic = nm.dictionary.to_pylist()
        idx = nm.indices.to_numpy(zero_copy_only=False)
        bad = np.zeros(len(dic), bool)
        for j, v in enumerate(dic):
            bad[j] = bool(v) and REPL in v
        hit = np.nonzero(ok & bad[idx])[0]
        if hit.size:
            u = t["node_uid"].to_numpy(zero_copy_only=False).astype(np.int64)
            src = pc.cast(t["inference_source"], "string").to_pylist()
            rul = pc.cast(t["inference_rule"], "string").to_pylist()
            for k in hit:
                v = dic[idx[k]]
                ident = identity_segment(v)
                rec = {"node_uid": int(u[k]), "inferred_name": v,
                       "identity_segment": ident,
                       "identity_replacement_chars": ident.count(REPL),
                       "identity_len": len(ident),
                       "inference_source": src[k], "inference_rule": rul[k]}
                (rejects if unreadable(ident) else keeps).append(rec)
        del t, nm

    rejects.sort(key=lambda x: x["node_uid"])
    keeps.sort(key=lambda x: x["node_uid"])
    print("scanned %s rows, %s admissible; mojibake %d = %d unreadable + %d readable  %.0fs"
          % (format(n_rows, ","), format(n_adm, ","), len(rejects) + len(keeps),
             len(rejects), len(keeps), time.time() - t0), flush=True)

    rec = {
        "RECORD": "CRAG_FREEBASE_INFERENCE_ADMISSIBILITY_V2_AMENDMENT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "FROZEN",
        "AMENDS": {"record": "CRAG_FREEBASE_INFERENCE_ADMISSIBILITY_V1", "path": ADM,
                   "v1_is_unchanged": True,
                   "how_to_apply": ("an OVERLAY, not a replacement: "
                                    "effective_reject_reason(node) = "
                                    "v2_amendment.get(node_uid) or v1[row]"),
                   "why_not_a_rebuild": (
                       "v1 is frozen, row-aligned across 64,038,024 rows and verified "
                       "reproducible. This changes 36 of them. Regenerating 64M rows to flip "
                       "36 would swap a verified artifact for an unverified one, and the "
                       "contract rule is that an amendment is a new record with its own hash, "
                       "never an in-place edit.")},
        "NEW_REJECT_REASON": {
            "code": 2, "name": "UNREADABLE_IDENTITY_PAYLOAD",
            "predicate": ("the identity segment -- the text before the first template boundary "
                          "-- contains no readable character once U+FFFD and whitespace are "
                          "removed"),
            "identity_boundary_regex": SPLIT.pattern,
            "why": ("the inference layer may only upgrade weak structural names into MORE "
                    "specific semantic names. A name whose specific part is entirely "
                    "replacement characters is less specific than the structural floor it "
                    "would replace. Rejection returns the node to its frozen display name, so "
                    "the baseline stays the floor and this can never push a node below it."),
            "why_not_a_fraction_threshold": (
                "a 0.5 cutoff would take 39 rows and is wrong on specific ones: a name that is "
                "76% replacement characters still ends in a readable Cyrillic title that names "
                "the recording. Readability is the property that matters, so it is tested "
                "directly instead of approximated by a ratio."),
            "existing_reasons_unchanged": {
                "0": "admissible", "1": "bookkeeping relation name (v1)"}},
        "SCALE": {
            "rows_rejected_by_this_amendment": len(rejects),
            "admissible_rows_in_v1": n_adm,
            "fraction_of_admissible": (len(rejects) / float(n_adm)) if n_adm else None,
            "honest_reading": ("negligible in aggregate -- 36 rows in 62 million. This is a "
                               "correctness rule, not a coverage lever, and is recorded "
                               "because it is right, not because it moves a number.")},
        "REJECTED_ROWS": rejects,
        "MOJIBAKE_ROWS_DELIBERATELY_KEPT": keeps,
        "PROVENANCE_OF_THE_DAMAGE": (
            "inherited from neighbour titles already corrupt in the source -- Cyrillic and CJK "
            "titles replaced wholesale, plus Latin-1 mojibake. The template is intact and the "
            "inference did not introduce the damage."),
        "scanned_rows": n_rows, "elapsed_s": round(time.time() - t0, 1)}

    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    body = io.open(OUT, "rb").read()
    rec["RECORD_SHA256"] = hashlib.sha256(body).hexdigest()
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("rejects=%d keeps=%d  sha256(pre-stamp)=%s" % (len(rejects), len(keeps),
                                                         rec["RECORD_SHA256"]))
    print("wrote %s  %.1fs" % (OUT, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
