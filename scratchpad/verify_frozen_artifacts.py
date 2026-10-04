# -*- coding: utf-8 -*-
"""
Re-hash every artifact the two freeze records declare, and compare against what they recorded.

WHY
    A freeze record's own RECORD_SHA256 proves the record has not changed. It proves nothing
    about the files the record describes -- those can drift underneath it, and nothing in the
    package notices. The standing invariant here is "treat every artifact as invalid until
    counts/checksums establish otherwise", and until now that had been established once, at
    freeze time, and never re-established.

    This was written because HANDOFF.md was found to have diverged from the digest LOCKED_5_OF_5
    recorded for it. That divergence is expected and benign -- the document had to be rewritten
    when dataset 6 landed -- but "expected and benign" is a claim, and a claim about integrity
    needs the whole population measured, not the one file that prompted the question.

OUTPUT
    HANDOFF divergences into three buckets that mean different things:
      MATCH      byte-identical to what was frozen.
      DOCS_DRIFT a file whose job is to DESCRIBE the package (HANDOFF.md), changed after the
                 freeze. Expected when the description is updated; never data.
      ADDITIVE   a package-level file that dataset 6 was written into, where removing the
                 webqsp entries reproduces the frozen bytes EXACTLY. That is proof the five's
                 entries did not move, so it clears -- but only because the test ran. An
                 artifact is never cleared by being named in an exemption list.
      DIVERGED   anything else. A data artifact that no longer hashes to its recorded value is
                 a real integrity failure and the verdict fails on it.
      MISSING    declared and not on disk.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
# reconfigure in place. Assigning a NEW TextIOWrapper over sys.stdout.buffer leaves the
# wrapper it replaced unreferenced; its __del__ closes the shared buffer, and the next
# print anywhere -- including in a module that imported this one -- dies with "I/O
# operation on closed file". reconfigure creates no second object.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
OUT = ROOT + "/FROZEN_ARTIFACT_RECHECK.json"

# files whose CONTENT is a description of the package rather than package data. A change here
# is a documentation update, not corruption -- but it is still reported, never hidden.
DOC_ARTIFACTS = {"data/final_canonical/HANDOFF.md"}

# the additive test, imported so there is exactly one implementation of it
_spec = importlib.util.spec_from_file_location("_explain",
                                               "scratchpad/explain_diverged_artifacts.py")
_EX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_EX)


def additive_reconstruction(path, recorded_sha256):
    """Does removing the webqsp entries reproduce the frozen bytes? Returns the convention
    that did it, or None. A None here is not proof of corruption -- it just means this
    artifact does not get cleared by this route."""
    try:
        cur = json.load(io.open(path, encoding="utf-8"))
    except Exception:
        return None
    stripped = _EX.strip_webqsp(copy.deepcopy(cur))
    if json.dumps(stripped, sort_keys=True) == json.dumps(cur, sort_keys=True):
        return None                      # nothing webqsp-shaped in it; not this explanation
    for tag, digest, _n in _EX.serialisations(stripped):
        if digest == recorded_sha256:
            return tag
    return None


def sha(path, bufsize=1 << 22):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def collect(rec, out, origin):
    """Pull every {path, bytes, sha256} triple out of a freeze record, wherever it sits."""
    if isinstance(rec, dict):
        if "path" in rec and "sha256" in rec and isinstance(rec.get("path"), str):
            out.append((origin, rec["path"], rec.get("bytes"), rec["sha256"]))
            return
        for v in rec.values():
            collect(v, out, origin)
    elif isinstance(rec, list):
        for v in rec:
            collect(v, out, origin)


def main():
    t0 = time.time()
    decl = []
    for name in ("LOCKED_5_OF_5_BENCHMARK_SUBSTRATES", "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES"):
        p = "%s/%s.json" % (ROOT, name)
        collect(json.load(io.open(p, encoding="utf-8")), decl, name)

    # a path can be declared by both records; check it once, remember both claimants
    by_path = {}
    for origin, path, nbytes, digest in decl:
        e = by_path.setdefault(path, {"declared_by": [], "sha256": digest, "bytes": nbytes})
        e["declared_by"].append(origin)
        if e["sha256"] != digest:
            e["CONFLICT"] = "the two records declare different digests for this path"

    buckets = {"MATCH": [], "DOCS_DRIFT": [], "ADDITIVE": [], "DIVERGED": [], "MISSING": []}
    detail, checked_bytes = {}, 0
    for i, (path, e) in enumerate(sorted(by_path.items()), 1):
        if not os.path.exists(path):
            buckets["MISSING"].append(path)
            detail[path] = {"status": "MISSING", "declared_by": e["declared_by"]}
            continue
        got, n = sha(path)
        checked_bytes += n
        if got == e["sha256"]:
            buckets["MATCH"].append(path)
            detail[path] = {"status": "MATCH", "bytes": n}
        else:
            conv = None
            if path in DOC_ARTIFACTS:
                b = "DOCS_DRIFT"
            else:
                conv = additive_reconstruction(path, e["sha256"])
                b = "ADDITIVE" if conv else "DIVERGED"
            buckets[b].append(path)
            detail[path] = {"status": b, "declared_by": e["declared_by"],
                            "recorded_sha256": e["sha256"], "recorded_bytes": e["bytes"],
                            "actual_sha256": got, "actual_bytes": n}
            if conv:
                detail[path]["additive_proof"] = (
                    "removing the webqsp entries and re-serialising as [%s] reproduces the "
                    "frozen digest exactly, so no byte of the five's entries changed" % conv)
        if i % 25 == 0:
            print("  %d/%d  %.1f GB" % (i, len(by_path), checked_bytes / 1e9), flush=True)

    fails = []
    for p in buckets["DIVERGED"]:
        fails.append("data artifact no longer matches its frozen digest: %s" % p)
    for p in buckets["MISSING"]:
        fails.append("declared artifact is missing from disk: %s" % p)
    for p, e in by_path.items():
        if "CONFLICT" in e:
            fails.append("the two freeze records disagree about %s" % p)

    rec = {
        "RECORD": "FROZEN_ARTIFACT_RECHECK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "every artifact declared by LOCKED_5_OF_5 and LOCKED_6_OF_6, re-hashed from disk and "
            "compared against the digest recorded at freeze time. The freeze records' own "
            "RECORD_SHA256 proves the RECORDS are intact; this proves the FILES are."),
        "n_declared_paths": len(by_path),
        "bytes_hashed": checked_bytes,
        "counts": {k: len(v) for k, v in buckets.items()},
        "ADDITIVE_MEANING": (
            "POINTER_INDEX.json and ID_BRIDGE.json are package-level files that dataset 6 had "
            "to be written into, so their 5/5 digests necessarily moved. That is the obvious "
            "explanation and it is not accepted on those grounds: the gate strips the webqsp "
            "entries, re-serialises, and requires the frozen digest to come back byte for "
            "byte. It does, for both. Anything that fails that test is reported as DIVERGED "
            "however plausible its story."),
        "DOCS_DRIFT_MEANING": (
            "HANDOFF.md is hashed by LOCKED_5_OF_5 as a package-level artifact. It had to be "
            "rewritten when dataset 6 landed -- it is the document that TELLS a reader there are "
            "six datasets -- so its digest necessarily moved. Freezing a hash of a document that "
            "is expected to be revised was a design mistake in freeze_five: the description of a "
            "package and the package's data should not be sealed by the same mechanism, because "
            "one of them has to change whenever the other is extended. LOCKED_6_OF_6 does not "
            "repeat the mistake -- it does not hash HANDOFF.md. The 5/5 record is NOT edited to "
            "paper over this; the divergence is recorded here instead."),
        "buckets": buckets,
        "detail": detail,
        "failures": fails,
        "VERDICT": "ALL_PASS" if not fails else "FAIL",
        "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\n%d declared paths, %.2f GB hashed" % (len(by_path), checked_bytes / 1e9))
    for k in ("MATCH", "DOCS_DRIFT", "ADDITIVE", "DIVERGED", "MISSING"):
        print("  %-11s %d" % (k, len(buckets[k])))
        for p in buckets[k][:6] if k != "MATCH" else []:
            print("      %s" % p)
    print("\nVERDICT %s  (%d failures)  %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:20]:
        print("  FAIL %s" % f)
    print("wrote %s" % OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
