# -*- coding: utf-8 -*-
"""
SPLADE_POOLING_FINDING
======================
Two SPLADE code paths wrote into this package, and they do not compute the same function.

    canonical_encode.py:211            log1p(relu(logits)).max(dim=1)              NO mask
    transfer/encode_rev2_delta.py:76   (log1p(relu(logits)) * attn_mask).max(dim=1)   masked

The first pools over PADDING positions as well as real ones. Padding logits are not zero, and
relu/log1p keeps whatever is positive, so a row picks up dimensions contributed by tokens that
are not in its own text. Worse, HuggingFace pads to the longest text IN THE BATCH, so how much
contamination a row receives depends on which other texts happened to share its shard batch.
A PHASE_C splade vector is therefore not a function of its text alone.

WHY THIS IS MEASURED AND NOT FIXED
    The package is being FROZEN, not rebuilt. Re-encoding 11.4M documents and 872k queries
    would invalidate every experiment already run against these vectors, and the standing
    instruction for the text datasets is VERIFY, not REBUILD. What a freeze owes its consumer
    is an accurate description, so the property is measured, quantified and published here --
    including the part that is genuinely inconsistent.

WHAT IS AND IS NOT CONSISTENT
    Consistent  every QUERY splade vector and 99.2% of DOCUMENT splade vectors come from the
                unmasked path, so query and document sides are scored in the same space.
    NOT         87,857 document rows (hotpotqa 94, 2wiki 87,763) were written by the REV2
                patch and ARE masked. Those are rows whose rev1 text was empty; the honest
                comparison for them is not "masked vs unmasked" but "masked vs no text at all".

WHAT THIS SCRIPT PROVES
    That masking is exactly what encoding a text ALONE means -- not an alternative convention.
    Every text is encoded three ways: in a padded batch without masking, in the same batch with
    masking, and by itself in a batch of one. If masked == solo and unmasked != solo, then the
    unmasked path is contamination rather than a different-but-equal choice.
"""
import collections
import io
import json
import os
import sys
import time

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
OUT = ROOT + "/SPLADE_POOLING_FINDING.json"
NAME = "naver/splade-cocondenser-ensembledistil"
SAMPLE = {"musique": 16, "squad": 16, "2wiki": 16, "hotpotqa": 16, "metaqa": 16}
SEED = 20260908


def sample_texts(ds, k):
    """A deterministic sample, taken by RESERVOIR so it is not just the head of the file.

    The head of nodes.jsonl is sorted by content hash, which correlates with nothing, but it is
    still one contiguous region. Batch contamination depends on the SPREAD of lengths in a
    batch, so a sample that under-represents long documents would understate the effect.
    """
    rs = np.random.RandomState(SEED)
    keep, seen = [], 0
    with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
        for ln in f:
            t = (json.loads(ln).get("text") or "").strip()
            if not t:
                continue
            seen += 1
            if len(keep) < k:
                keep.append(t)
            else:
                j = rs.randint(0, seen)
                if j < k:
                    keep[j] = t
    return keep, seen


def main():
    t0 = time.time()
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(NAME)
    mdl = AutoModelForMaskedLM.from_pretrained(NAME).eval()

    per = {}
    allcos_u, allcos_m = [], []
    for ds, k in SAMPLE.items():
        texts, seen = sample_texts(ds, k)
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            w = torch.log1p(torch.relu(mdl(**enc).logits))
        unmasked = w.max(dim=1).values.numpy()
        masked = (w * enc["attention_mask"].unsqueeze(-1)).max(dim=1).values.numpy()
        solo = []
        for t in texts:
            e = tok([t], return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                solo.append(
                    torch.log1p(torch.relu(mdl(**e).logits)).max(dim=1).values.numpy()[0])
        S = np.stack(solo)

        def cos(A, B):
            a = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
            b = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
            return (a * b).sum(1)

        cu, cm = cos(S, unmasked), cos(S, masked)
        allcos_u += cu.tolist()
        allcos_m += cm.tolist()
        tl = enc["attention_mask"].sum(1).numpy()
        per[ds] = {
            "n_sampled": len(texts), "n_nonempty_texts_in_dataset": seen,
            "token_len_min": int(tl.min()), "token_len_max": int(tl.max()),
            "solo_vs_masked_max_abs_diff": float(np.abs(S - masked).max()),
            "solo_vs_unmasked_max_abs_diff": float(np.abs(S - unmasked).max()),
            "cosine_solo_vs_masked_min": float(cm.min()),
            "cosine_solo_vs_unmasked_min": float(cu.min()),
            "cosine_solo_vs_unmasked_mean": float(cu.mean()),
            "nnz_solo": int((S > 0).sum()), "nnz_masked": int((masked > 0).sum()),
            "nnz_unmasked": int((unmasked > 0).sum())}
        print("  %-9s solo~masked %.2e   solo~unmasked %.2e  cos(unmasked) min %.6f  %.0fs"
              % (ds, per[ds]["solo_vs_masked_max_abs_diff"],
                 per[ds]["solo_vs_unmasked_max_abs_diff"],
                 per[ds]["cosine_solo_vs_unmasked_min"], time.time() - t0), flush=True)

    # store composition, READ from the pointer index rather than assumed
    man = json.load(io.open(ROOT + "/POINTER_INDEX.json", encoding="utf-8"))
    comp = {}
    for sect, sub in (("datasets", "docs"), ("queries", "queries")):
        for ds, mm in man[sect].items():
            spec = mm.get("splade") if isinstance(mm, dict) else None
            if not isinstance(spec, dict):
                continue
            paths = [s.get("path") for s in spec.get("stores", []) if isinstance(s, dict)]
            d = ROOT + ("/%s/pointer_index" % ds if sub == "docs"
                        else "/%s/queries/pointer_index" % ds)
            src = np.load(os.path.join(d, "splade.npz"))["src"]
            c = collections.Counter(src.tolist())
            rows = {}
            for i, p in enumerate(paths):
                rows[p] = {"n_rows": int(c.get(i, 0)),
                           "pooling": ("MASKED" if "_rev2_encoder_patch" in (p or "")
                                       else "UNMASKED")}
            comp.setdefault(sub, {})[ds] = rows

    def tot(sub, pool):
        return sum(v["n_rows"] for ds in comp[sub].values() for v in ds.values()
                   if v["pooling"] == pool)

    nu, nm = tot("docs", "UNMASKED"), tot("docs", "MASKED")
    qu, qm = tot("queries", "UNMASKED"), tot("queries", "MASKED")
    pct = 100.0 * nu / max(nu + nm, 1)

    rec = {
        "RECORD": "SPLADE_POOLING_FINDING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "MEASURED_AND_PUBLISHED_NOT_FIXED",
        "FINDING": (
            "the splade vectors in this package were pooled over PADDING positions as well as "
            "real ones. Because HuggingFace pads to the longest text in the batch, a PHASE_C "
            "splade vector is a function of its text AND of the other texts in its shard "
            "batch, not of its text alone."),
        "PROOF": (
            "each sampled text was encoded three ways: unmasked in a padded batch, masked in "
            "the same batch, and alone in a batch of one. masked reproduces solo to float "
            "noise; unmasked does not. That is what makes this contamination rather than a "
            "different-but-equal pooling convention."),
        "code_paths": {
            "UNMASKED": {"file": "src/experiments/canonical_encode.py", "line": 211,
                         "expr": "log1p(relu(logits)).max(dim=1)"},
            "MASKED": {"file": "transfer/encode_rev2_delta.py", "line": 76,
                       "expr": "(log1p(relu(logits)) * attention_mask.unsqueeze(-1)).max(dim=1)"}},
        "per_dataset_measurement": per,
        "aggregate": {
            "cosine_solo_vs_unmasked_min": min(allcos_u),
            "cosine_solo_vs_unmasked_mean": float(np.mean(allcos_u)),
            "cosine_solo_vs_masked_min": min(allcos_m),
            "n_texts_measured": len(allcos_u),
            "sample_seed": SEED, "sampling": "reservoir over all non-empty node texts"},
        "store_composition": comp,
        "totals": {"doc_rows_unmasked": nu, "doc_rows_masked": nm,
                   "query_rows_unmasked": qu, "query_rows_masked": qm,
                   "pct_doc_rows_unmasked": round(pct, 4)},
        "CONSISTENCY": (
            "every query vector and %.2f%% of document vectors use the unmasked path, so the "
            "query and document sides are scored in the SAME space for almost the whole "
            "package. The exception is %s document rows written by the REV2 text patch, which "
            "are masked. Those rows had EMPTY text in rev1, so for them the real comparison is "
            "masked vs no text at all, not masked vs unmasked."
            % (pct, format(nm, ","))),
        "WHY_NOT_FIXED": (
            "freezing, not rebuilding. Re-encoding 11.4M documents and 872k queries would "
            "invalidate every experiment already run against these vectors, and the standing "
            "instruction for the text datasets is VERIFY not REBUILD. A freeze owes its "
            "consumer an accurate description, which is what this record is."),
        "WEBQSP_DECISION": (
            "dataset 6 is encoded with the MASKED path. It has no legacy splade rows to stay "
            "consistent with, its queries and documents are both built here so they match each "
            "other, and the unmasked path is not even reproducible for new data -- its output "
            "depends on batch composition. Deliberately reproducing a batch-dependent defect "
            "in a corpus that does not have it would be indefensible. The difference from the "
            "other five is recorded rather than hidden."),
        "CONSUMER_GUIDANCE": (
            "do not compare raw splade scores ACROSS datasets, and do not treat a splade "
            "vector as a reproducible function of its text. WITHIN a dataset ranking is "
            "consistent because both sides share the convention, and rank-based fusion (RRF) "
            "is unaffected by the score-scale part of this.")}
    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\ndocs    unmasked %s  masked %s  (%.4f%% unmasked)"
          % (format(nu, ","), format(nm, ","), pct))
    print("queries unmasked %s  masked %s" % (format(qu, ","), format(qm, ",")))
    print("cosine(solo, unmasked) min %.6f mean %.6f over %d texts"
          % (min(allcos_u), float(np.mean(allcos_u)), len(allcos_u)))
    print("wrote %s  %.1fs" % (OUT, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
