# -*- coding: utf-8 -*-
"""Verify the canonical package against what the frozen records claim about it.

The standing rule on this package is "treat every artifact as invalid until counts and
checksums establish otherwise". ukb_manifest.py deliberately COPIES hashes rather than
recomputing them, so it cannot detect drift on its own. This is the tool that recomputes.

Five independent checks, each runnable alone (--all runs every one):

  --records   the LOCKED_* records (5/5, 6/6, FAMILY_COMPLETION_V1 and, once it exists,
              FAMILY_COMPLETION_V2) still hash to their own recorded digests, under the
              repo convention: json.dumps(indent=1) with RECORD_SHA256/RECORD_SHA256_LF
              popped, hashed as CRLF bytes for RECORD_SHA256 and as LF for RECORD_SHA256_LF.
              A record that no longer self-hashes means the freeze was edited, which is the
              one thing that must never happen -- amendments get a NEW record.
  --sizes     every declared artifact exists and is the declared byte length. Cheap, catches
              truncation and deletion. Runs by default.
  --graphs    sha256 of every graph family .npz (frozen and graph2) plus the graph2 manifest
              digests. ~1.1 GB, seconds. A family in both trees is keyed by the precedence
              rule: frozen wins unless the graph2 manifest declares supersedes.<family>
              (today: musique/knn), in which case the frozen copy is still hashed against its
              pin but reported as <family>@frozen_superseded.
  --builders  does the builder each dataset RECORDS still exist on disk at that sha256?
              Reproducibility, not integrity: hotpotqa records build_kb.py at a46d745a...
              and the on-disk file is e723192e..., the revision that built 2wiki. hotpotqa's
              artifacts are valid and verified; its builder revision is simply gone.
  --full      sha256 of all 206 declared artifacts, ~29 GB. This is the real gate.

THE THREE KNOWN DIVERGENCES, AND WHY THEY ARE NOT FAILURES HERE
---------------------------------------------------------------
Three artifacts declared by LOCKED_5_OF_5 no longer match their recorded digests. All three
were already adjudicated, and this tool defers to those records rather than re-deciding:

  POINTER_INDEX.json, ID_BRIDGE.json -> ADDITIVE, per FROZEN_ARTIFACT_RECHECK.json and
      DIVERGED_ARTIFACT_EXPLANATION.json (VERDICT: EXPLAINED_ADDITIVE). "webqsp was added"
      was NOT accepted as an explanation on plausibility: the gate strips the webqsp entries,
      re-serialises under the original convention, and requires the frozen digest back byte
      for byte. It came back for both. Note POINTER_INDEX is SMALLER on disk than its frozen
      size -- the freeze serialised it at indent=2 and the current file is indent=1, so a
      size decrease here is a formatting change, not a loss, and the byte reconstruction is
      what actually settles it.
  HANDOFF.md -> DOCS_DRIFT. LOCKED_5_OF_5 hashed the document that describes the package, so
      it necessarily moved when dataset six landed. FROZEN_ARTIFACT_RECHECK calls that a
      design mistake in freeze_five and LOCKED_6_OF_6 deliberately does not repeat it. The
      5/5 record was NOT edited to paper over it.

So a size or hash mismatch on those three is reported as EXPECTED_DRIFT with a pointer to the
adjudicating record. Anything else is a real failure and fails the run.
"""
import hashlib
import io
import json
import os
import sys
import time

ROOT = "data/final_canonical"
DS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]
RECORD_FILES = ["LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
                "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
                "LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json"]
# FAMILY_COMPLETION_V2 (the musique kNN supersession) joins the chain once it exists on disk.
# It BUILDS_ON V1, so check_records verifies V1 -> V2 the same way it verifies 5/5 -> 6/6 -> V1.
FAMILY_V2 = "LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"
if os.path.exists("%s/%s" % (ROOT, FAMILY_V2)):
    RECORD_FILES.append(FAMILY_V2)


def latest_family_record():
    """The family-completion record that governs graph2 today: V2 when present, else V1."""
    return RECORD_FILES[3] if len(RECORD_FILES) > 3 else RECORD_FILES[2]

EXPECTED_DRIFT = {
    "data/final_canonical/POINTER_INDEX.json":
        "ADDITIVE: webqsp appended as dataset six. Adjudicated by "
        "DIVERGED_ARTIFACT_EXPLANATION.json -- stripping webqsp and re-serialising at the "
        "frozen convention (indent=2, CRLF) reproduces the 26,788 frozen bytes exactly. The "
        "on-disk file is indent=1, which is why it is SMALLER, not larger.",
    "data/final_canonical/ID_BRIDGE.json":
        "ADDITIVE: webqsp appended as dataset six. Adjudicated by "
        "DIVERGED_ARTIFACT_EXPLANATION.json -- stripping webqsp reproduces the 4,032 frozen "
        "bytes exactly (indent=1, CRLF).",
    "data/final_canonical/HANDOFF.md":
        "DOCS_DRIFT: LOCKED_5_OF_5 hashed the document that describes the package, so it had "
        "to move when dataset six landed. Adjudicated by FROZEN_ARTIFACT_RECHECK.json, which "
        "calls the doc-in-the-data-freeze a design mistake; LOCKED_6_OF_6 does not hash it.",
    "data/final_canonical/RETRIEVAL_CACHE.json":
        "ADDITIVE: webqsp, hotpotqa and 2wiki appended when their caches were built on "
        "2026-09-10, after every freeze, and NOT_MATERIALISED emptied. Adjudicated by "
        "RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json -- stripping the three post-freeze entries "
        "and re-serialising at the frozen convention (indent=1, ensure_ascii=False, CRLF) "
        "reproduces the 9,580 frozen bytes and the pinned "
        "ae2b5c0f981fa86e0a3f177fe3b331802e8a3036d7b7cf4c4c811d61fd37dcfb exactly, so no byte "
        "of the five's entries changed. The recipe is carried INSIDE the file as "
        "_FROZEN_ORIGINAL, so the reproduction needs nothing external. Note this supersedes "
        "FROZEN_ARTIFACT_RECHECK.json, which still lists this file as MATCH at 9,580 bytes -- "
        "that verdict predates the append.",
}
ADJUDICATORS = ["data/final_canonical/FROZEN_ARTIFACT_RECHECK.json",
                "data/final_canonical/DIVERGED_ARTIFACT_EXPLANATION.json",
                "data/final_canonical/RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json"]


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def sha_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def record_hash(rec, crlf=True):
    """The repo convention, reproduced exactly -- see scratchpad/freeze_six.py."""
    d = dict(rec)
    d.pop("RECORD_SHA256", None)
    d.pop("RECORD_SHA256_LF", None)
    txt = json.dumps(d, indent=1)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def declared_index():
    """Every hash-pinned artifact -> (bytes, sha256, pinned_by).

    Returns (index, five, six, fam) where fam is the GOVERNING family-completion record (V2
    when present, else V1). V1's added families are indexed under FAMILY_V1 and V2's under
    FAMILY_V2; V2 carries V1's entries too, so the union is the same set plus the supersession.
    """
    five, six, fam1 = (rj("%s/%s" % (ROOT, f)) for f in RECORD_FILES[:3])
    fam2 = rj("%s/%s" % (ROOT, FAMILY_V2)) if len(RECORD_FILES) > 3 else None
    d = {}

    def put(p, b, s, src):
        d[p.replace("\\", "/")] = (b, s, src)

    for ds, v in five["datasets"].items():
        for a in v["artifacts"]:
            put(a["path"], a["bytes"], a["sha256"], "LOCKED_5_OF_5:" + ds)
    for _, v in five["package_level_artifacts"].items():
        put(v["path"], v["bytes"], v["sha256"], "LOCKED_5_OF_5:pkg")
    for a in six["webqsp"]["artifacts"]:
        put(a["path"], a["bytes"], a["sha256"], "LOCKED_6_OF_6:webqsp")
    for ds, v in fam1["datasets"].items():
        for f, a in (v.get("added") or {}).items():
            put(ROOT + "/" + a["file"], a["npz_bytes"], a["npz_sha256"], "FAMILY_V1:" + ds)
    if fam2:
        for ds, v in fam2["datasets"].items():
            for f, a in (v.get("added") or {}).items():
                p = ROOT + "/" + a["file"]
                if p.replace("\\", "/") not in d:
                    put(p, a["npz_bytes"], a["npz_sha256"], "FAMILY_V2:" + ds)
    return d, five, six, (fam2 or fam1)


# ---------------------------------------------------------------------------
def check_records():
    res = {"check": "records", "records": {}, "PASS": True}
    for f in RECORD_FILES:
        rec = rj("%s/%s" % (ROOT, f))
        got_crlf, got_lf = record_hash(rec), record_hash(rec, crlf=False)
        want_crlf = rec.get("RECORD_SHA256")
        want_lf = rec.get("RECORD_SHA256_LF")
        ok = (want_crlf == got_crlf) or (want_lf is not None and want_lf == got_lf)
        # LOCKED_5_OF_5 predates the LF field, so CRLF alone is its whole contract
        res["records"][f] = {
            "self_hash_ok": ok,
            "recorded_crlf": want_crlf, "computed_crlf": got_crlf,
            "recorded_lf": want_lf, "computed_lf": got_lf,
        }
        if not ok:
            res["PASS"] = False
    # each record must still pin its predecessor at the digest it recorded. BUILDS_ON stores
    # that under `recorded_sha256`, not `RECORD_SHA256` -- the latter is the record's own hash.
    chain = []
    for name in RECORD_FILES[1:]:
        rec = rj("%s/%s" % (ROOT, name))
        bo = rec.get("BUILDS_ON") or {}
        base, want = bo.get("record"), bo.get("recorded_sha256")
        entry = {"record": name, "builds_on": base, "pinned_sha256": want,
                 "base_path": bo.get("path")}
        r2 = rj(bo["path"]) if bo.get("path") and os.path.exists(bo["path"]) else None
        if r2 is None:
            entry["hash_matches"] = False
            entry["why"] = "the base record named in BUILDS_ON is not on disk"
        else:
            entry["actual_sha256"] = r2.get("RECORD_SHA256")
            # compare against the base's own self-hash AND against a recomputation, so a base
            # whose stored digest was edited to match cannot slip through
            entry["recomputed_base_sha256"] = record_hash(r2)
            entry["hash_matches"] = (want == r2.get("RECORD_SHA256") == entry["recomputed_base_sha256"])
        if not entry["hash_matches"]:
            res["PASS"] = False
        chain.append(entry)
    res["chain"] = chain
    res["adjudicating_records_present"] = {p: os.path.exists(p) for p in ADJUDICATORS}
    return res


def check_sizes(decl):
    res = {"check": "sizes", "n_declared": len(decl), "missing": [], "wrong": [],
           "expected_drift": [], "ok": 0}
    for p, (b, s, src) in sorted(decl.items()):
        if not os.path.exists(p):
            res["missing"].append({"path": p, "pinned_by": src})
            continue
        n = os.path.getsize(p)
        if b is not None and n != b:
            rec = {"path": p, "declared": b, "disk": n, "pinned_by": src}
            if p in EXPECTED_DRIFT:
                rec["reason"] = EXPECTED_DRIFT[p]
                res["expected_drift"].append(rec)
            else:
                res["wrong"].append(rec)
        else:
            res["ok"] += 1
    res["PASS"] = not res["missing"] and not res["wrong"]
    return res


def check_graphs(decl):
    """Hash every family .npz in both trees, and apply the precedence rule to the keys.

    A family present in BOTH trees is one of two things. If the graph2 manifest declares
    `supersedes.<family>`, graph2 serves it: the frozen copy is still hashed against its
    LOCKED_5_OF_5 pin but recorded under `<family>@frozen_superseded`. Without that
    declaration the frozen copy serves it and the graph2 copy is recorded under
    `<family>@graph2_shadowed` -- and flagged, because an undeclared duplicate is a mistake
    waiting to be misread, even though nothing is wrong with either file's bytes.
    """
    res = {"check": "graphs", "datasets": {}, "PASS": True, "supersessions": {},
           "undeclared_duplicates": []}
    for ds in DS:
        e = {"families": {}}
        for tree in ("graph", "graph2"):
            mp = "%s/%s/%s/GRAPH_MANIFEST.json" % (ROOT, ds, tree)
            if not os.path.exists(mp):
                continue
            man = rj(mp)
            for fam, v in man["families"].items():
                if not v.get("present"):
                    continue
                fp = "%s/%s/%s" % (ROOT, ds, v["file"])
                rec = {"tree": tree, "path": fp, "n_edges": v.get("n_edges"),
                       "exists": os.path.exists(fp)}
                if rec["exists"]:
                    rec["bytes"] = os.path.getsize(fp)
                    rec["sha256"] = sha_file(fp)
                    want = v.get("npz_sha256") or (decl.get(fp) or (None, None, None))[1]
                    rec["expected_sha256"] = want
                    rec["sha_ok"] = (want is None) or (want == rec["sha256"])
                    rec["sha_source"] = ("graph2 manifest" if v.get("npz_sha256")
                                         else ("frozen record" if want else "UNPINNED"))
                else:
                    rec["sha_ok"] = False
                if not rec["sha_ok"]:
                    res["PASS"] = False
                key = fam
                if tree == "graph2" and fam in e["families"]:
                    dec = (man.get("supersedes") or {}).get(fam)
                    if dec:
                        frozen_rec = e["families"].pop(fam)
                        frozen_rec["superseded_by"] = "graph2/%s.npz" % fam
                        frozen_rec["frozen_sha_matches_declaration"] = (
                            frozen_rec.get("sha256") == dec.get("frozen_sha256"))
                        if not frozen_rec["frozen_sha_matches_declaration"]:
                            res["PASS"] = False
                        e["families"][fam + "@frozen_superseded"] = frozen_rec
                        rec["supersedes_frozen"] = {"frozen_file": dec.get("frozen_file"),
                                                    "frozen_sha256": dec.get("frozen_sha256"),
                                                    "reason": dec.get("reason")}
                        res["supersessions"].setdefault(ds, []).append(fam)
                    else:
                        key = fam + "@graph2_shadowed"
                        rec["shadowed_by_frozen"] = True
                        res["undeclared_duplicates"].append("%s/%s" % (ds, fam))
                e["families"][key] = rec
        res["datasets"][ds] = e
    # the GOVERNING family-completion record pins each graph2 manifest by digest
    fam_rec = rj("%s/%s" % (ROOT, latest_family_record()))
    res["graph2_manifests_pinned_by"] = latest_family_record()
    g2 = {}
    for ds, v in fam_rec["datasets"].items():
        want = v.get("graph2_manifest_sha256")
        if not want:
            continue
        mp = "%s/%s/graph2/GRAPH_MANIFEST.json" % (ROOT, ds)
        got = sha_file(mp) if os.path.exists(mp) else None
        g2[ds] = {"path": mp, "recorded": want, "computed": got, "ok": got == want}
        if got != want:
            res["PASS"] = False
    res["graph2_manifests"] = g2
    return res


def check_builders():
    """Does the builder that each dataset RECORDS still exist on disk at that sha256?

    This is a reproducibility check, not an integrity check. A dataset whose builder has
    moved on is still valid -- its artifacts are hash-pinned and verified by --full -- but it
    can no longer be REBUILT byte-for-byte from this tree, and that is worth knowing before
    anyone plans a rebuild. Known state at the time this check was written:

        build.py     26455c9f...  matches squad, musique, metaqa
        build_kb.py  e723192e...  matches 2wiki; hotpotqa recorded a46d745a..., an earlier
                                  revision that is no longer on disk

    So hotpotqa is the one dataset whose builder revision is gone.
    """
    res = {"check": "builders", "datasets": {}, "PASS": True, "reproducible": [],
           "not_reproducible": []}
    for ds in DS:
        bi_p = "%s/%s/build_info.json" % (ROOT, ds)
        if not os.path.exists(bi_p):
            res["datasets"][ds] = {"build_info": None,
                                   "note": "no build_info.json -- this dataset was not built "
                                           "by build.py/build_kb.py (webqsp)"}
            res["not_reproducible"].append(ds)
            continue
        bi = rj(bi_p)
        b, want = bi.get("builder"), bi.get("builder_sha256")
        got = sha_file(b) if b and os.path.exists(b) else None
        ok = (got is not None and got == want)
        res["datasets"][ds] = {"builder": b, "recorded_sha256": want, "on_disk_sha256": got,
                               "on_disk_matches_recorded": ok,
                               "git_commit": bi.get("git_commit")}
        (res["reproducible"] if ok else res["not_reproducible"]).append(ds)
    # this check reports, it does not fail the run: a moved builder is a fact about the repo,
    # not damage to the package
    res["MEANING"] = ("PASS is always True. A dataset in not_reproducible has valid, verified "
                      "artifacts; what it lacks is the exact builder revision needed to "
                      "regenerate them byte-for-byte.")
    return res


def check_full(decl):
    res = {"check": "full", "n_declared": len(decl), "bad": [], "expected_drift": [],
           "unpinned": [], "ok": 0, "bytes_read": 0}
    t0 = time.time()
    items = sorted(decl.items())
    for i, (p, (b, s, src)) in enumerate(items, 1):
        if not os.path.exists(p):
            res["bad"].append({"path": p, "why": "missing", "pinned_by": src})
            continue
        if s is None:
            res["unpinned"].append(p)
            continue
        got = sha_file(p)
        res["bytes_read"] += os.path.getsize(p)
        if got != s:
            rec = {"path": p, "declared": s, "computed": got, "pinned_by": src,
                   "declared_bytes": b, "disk_bytes": os.path.getsize(p)}
            if p in EXPECTED_DRIFT:
                rec["reason"] = EXPECTED_DRIFT[p]
                res["expected_drift"].append(rec)
            else:
                res["bad"].append(rec)
        else:
            res["ok"] += 1
        if i % 20 == 0 or i == len(items):
            print("    %d/%d  %.1f GB read  %.0fs" % (i, len(items), res["bytes_read"] / 1e9,
                                                      time.time() - t0), flush=True)
    res["PASS"] = not res["bad"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------
def main():
    args = set(sys.argv[1:])
    out_path = None
    for a in list(args):
        if a.startswith("--out="):
            out_path = a.split("=", 1)[1]
            args.discard(a)
    want = {a.lstrip("-") for a in args} or {"records", "sizes", "graphs", "builders"}
    if "all" in want:
        want = {"records", "sizes", "graphs", "builders", "full"}

    decl, five, six, fam = declared_index()
    report = {"RECORD": "UKB_COMMON_MANIFEST_VERIFICATION",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "checks_run": sorted(want), "results": {}}

    order = [("records", check_records, ()), ("sizes", check_sizes, (decl,)),
             ("graphs", check_graphs, (decl,)), ("builders", check_builders, ()),
             ("full", check_full, (decl,))]
    allpass = True
    for name, fn, a in order:
        if name not in want:
            continue
        print("[%s] running..." % name, flush=True)
        r = fn(*a)
        report["results"][name] = r
        allpass &= bool(r.get("PASS"))
        print("[%s] PASS=%s" % (name, r.get("PASS")), flush=True)
        if name == "records":
            for f, v in r["records"].items():
                print("    %-46s self_hash_ok=%s" % (f[:46], v["self_hash_ok"]))
            for c in r["chain"]:
                print("    %-46s builds_on=%s match=%s"
                      % (c["record"][:46], c.get("builds_on"), c.get("hash_matches")))
        if name == "sizes":
            print("    ok=%d missing=%d wrong=%d expected_drift=%d"
                  % (r["ok"], len(r["missing"]), len(r["wrong"]), len(r["expected_drift"])))
            for x in r["missing"] + r["wrong"]:
                print("    !! %s" % x)
            for x in r["expected_drift"]:
                print("    ~~ %s  declared=%d disk=%d  (%s)"
                      % (x["path"], x["declared"], x["disk"], x["reason"]))
        if name == "graphs":
            for ds, e in r["datasets"].items():
                fams = ", ".join("%s:%s(%s,%s)" % (k, v["tree"], v["n_edges"],
                                                   "OK" if v["sha_ok"] else "BAD")
                                 for k, v in sorted(e["families"].items()))
                print("    %-9s %s" % (ds, fams))
            for ds, v in r["graph2_manifests"].items():
                print("    graph2 manifest %-9s ok=%s" % (ds, v["ok"]))
        if name == "builders":
            for ds, v in r["datasets"].items():
                print("    %-9s %-46s match=%s"
                      % (ds, str(v.get("builder"))[:46], v.get("on_disk_matches_recorded")))
            print("    reproducible from this tree: %s" % ", ".join(r["reproducible"]))
            print("    NOT reproducible (builder revision gone): %s"
                  % ", ".join(r["not_reproducible"]))
        if name == "full":
            print("    ok=%d bad=%d expected_drift=%d unpinned=%d  %.1f GB in %.0fs"
                  % (r["ok"], len(r["bad"]), len(r["expected_drift"]), len(r["unpinned"]),
                     r["bytes_read"] / 1e9, r["seconds"]))
            for x in r["bad"]:
                print("    !! %s" % x)

    report["PASS"] = allpass
    print("\nOVERALL PASS=%s" % allpass)
    if out_path:
        with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, indent=1, ensure_ascii=False))
        print("wrote %s" % out_path)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
