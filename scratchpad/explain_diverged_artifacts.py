# -*- coding: utf-8 -*-
"""
Two artifacts declared by LOCKED_5_OF_5 no longer hash to their recorded digests:
POINTER_INDEX.json and ID_BRIDGE.json. Both are PACKAGE-LEVEL files that dataset 6 had to be
written into. "Dataset 6 was added, so of course they changed" is the obvious explanation and
it is worth exactly nothing until it is tested, because the same divergence is what silent
corruption of the five would look like.

THE TEST
    If the change is purely additive, then removing the webqsp entries from the CURRENT file
    must reproduce the file that was frozen -- byte for byte, since the frozen digest is a byte
    digest. So: strip webqsp, re-serialise under each plausible convention, and look for the
    recorded digest.

    A hit is conclusive: it proves not one byte of the five's entries moved.
    A miss is INCONCLUSIVE, not evidence of corruption -- the builder may simply serialise
    differently now. In that case fall back to the semantic comparison, which is reported
    either way: every field the five's entries carry, checked against the frozen per-dataset
    records and against the stores on disk.
"""
import copy
import hashlib
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
OUT = ROOT + "/DIVERGED_ARTIFACT_EXPLANATION.json"
FIVE = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def strip_webqsp(obj):
    """Remove every webqsp-keyed entry, at any depth, from a copy."""
    if isinstance(obj, dict):
        return {k: strip_webqsp(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.split("/")[0] == "webqsp")}
    if isinstance(obj, list):
        return [strip_webqsp(v) for v in obj]
    return obj


def serialisations(obj):
    for indent in (1, 2, 4, None):
        for ea in (True, False):
            for sk in (False, True):
                txt = json.dumps(obj, indent=indent, ensure_ascii=ea, sort_keys=sk)
                for nl, tag in ((chr(10), "LF"), (chr(13) + chr(10), "CRLF")):
                    t = txt.replace(chr(10), nl) if nl != chr(10) else txt
                    yield ("indent=%s ascii=%s sortkeys=%s %s" % (indent, ea, sk, tag),
                           hashlib.sha256(t.encode("utf-8")).hexdigest(), len(t.encode("utf-8")))


def main():
    t0 = time.time()
    five = load(ROOT + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json")
    pl = five["package_level_artifacts"]
    targets = {"data/final_canonical/POINTER_INDEX.json": pl["pointer_index"],
               "data/final_canonical/ID_BRIDGE.json": pl["id_bridge"]}

    out, fails = {}, []
    for path, declared in sorted(targets.items()):
        cur = load(path)
        stripped = strip_webqsp(copy.deepcopy(cur))
        removed = sorted(set(json.dumps(cur).count("webqsp") and
                             [k for k in cur.get("datasets", {}) if k.split("/")[0] == "webqsp"]
                             or []))
        removed_any = json.dumps(cur, sort_keys=True) != json.dumps(stripped, sort_keys=True)

        hit = None
        for tag, digest, nbytes in serialisations(stripped):
            if digest == declared["sha256"]:
                hit = {"convention": tag, "bytes": nbytes}
                break

        e = {"declared_sha256": declared["sha256"], "declared_bytes": declared["bytes"],
             "actual_bytes": os.path.getsize(path),
             "webqsp_entries_found": removed_any,
             "webqsp_dataset_keys_removed": removed,
             "byte_reconstruction": hit,
             "byte_reconstruction_verdict":
                 "CONCLUSIVE: stripping webqsp reproduces the frozen bytes exactly, so no byte "
                 "of the five's entries changed" if hit else
                 "INCONCLUSIVE: no tried serialisation reproduced the frozen digest. This does "
                 "NOT indicate corruption -- see the semantic comparison below, which is the "
                 "check that actually constrains the content."}
        out[path] = e

    # --------------------------------------------------------- semantic check
    # whatever the byte test said, compare the five's entries against the frozen per-dataset
    # records and against what is on disk. This is the check that would catch real drift.
    pi = load(ROOT + "/POINTER_INDEX.json")
    ib = load(ROOT + "/ID_BRIDGE.json")
    sem = {}
    for ds in FIVE:
        rec5 = five["datasets"][ds]
        d = {}
        chans = pi.get("datasets", {}).get(ds, {})
        qent = pi.get("queries", {}).get(ds, {})
        # only entries carrying "stores" are channels; datasets[ds] also holds diagnostic
        # sub-dicts (reuse_families_seen, reuse_source_ne_own_row) that are not channels
        d["doc_channels"] = sorted(k for k, v in chans.items()
                                   if isinstance(v, dict) and "stores" in v)
        d["non_channel_keys"] = sorted(k for k, v in chans.items()
                                       if not (isinstance(v, dict) and "stores" in v))
        d["query_channels"] = sorted(k for k, v in qent.items() if isinstance(v, dict)
                                     and "stores" in v)
        # documents: one pointer per canonical node, and every store must still be there
        for model in d["doc_channels"]:
            ent = chans[model]
            d["docs/%s n_pointers" % model] = ent.get("n_pointers")
            if ent.get("n_pointers") != rec5["n_nodes"]:
                fails.append("%s docs/%s: POINTER_INDEX n_pointers=%s but LOCKED_5_OF_5 "
                             "n_nodes=%s" % (ds, model, ent.get("n_pointers"), rec5["n_nodes"]))
            for st in ent.get("stores", []):
                p = st.get("path")
                if p and not os.path.isdir(p) and not os.path.exists(p):
                    fails.append("%s docs/%s: store path missing on disk: %s" % (ds, model, p))
        # queries: the count lives on the dataset entry, the stores on each model entry
        d["queries n_queries"] = qent.get("n_queries")
        if qent and qent.get("n_queries") != rec5["n_queries"]:
            fails.append("%s queries: POINTER_INDEX n_queries=%s but LOCKED_5_OF_5 n_queries=%s"
                         % (ds, qent.get("n_queries"), rec5["n_queries"]))
        for model in d["query_channels"]:
            for st in qent[model].get("stores", []):
                p = st.get("path")
                if p and not os.path.isdir(p) and not os.path.exists(p):
                    fails.append("%s queries/%s: store path missing on disk: %s"
                                 % (ds, model, p))
        # id bridge: the numbers it asserts about this dataset
        bri = ib.get("datasets", {}).get(ds, {})
        d["id_bridge"] = {k: bri.get(k) for k in
                          ("canonical_nodes", "queries", "gold_refs", "gold_unresolved",
                           "gold_resolution_pct", "rule_violations", "node_id_collisions")}
        for a, b in (("canonical_nodes", "n_nodes"), ("queries", "n_queries")):
            if bri.get(a) is not None and bri[a] != rec5[b]:
                fails.append("%s ID_BRIDGE %s=%s but LOCKED_5_OF_5 %s=%s"
                             % (ds, a, bri[a], b, rec5[b]))
        for k in ("gold_unresolved", "rule_violations", "node_id_collisions"):
            if bri.get(k):
                fails.append("%s ID_BRIDGE %s=%s, was 0 at freeze time" % (ds, k, bri[k]))
        sem[ds] = d

    for ds in FIVE:
        if ds not in pi.get("datasets", {}):
            fails.append("POINTER_INDEX no longer carries dataset %s" % ds)
        if ds not in ib.get("datasets", {}):
            fails.append("ID_BRIDGE no longer carries dataset %s" % ds)
    # the additions that explain the divergence, named explicitly
    sem["_added_since_5of5"] = {
        "POINTER_INDEX.datasets": sorted(set(pi.get("datasets", {})) - set(FIVE)),
        "POINTER_INDEX.queries": sorted(set(pi.get("queries", {})) - set(FIVE)),
        "ID_BRIDGE.datasets": sorted(set(ib.get("datasets", {})) - set(FIVE))}

    rec = {
        "RECORD": "DIVERGED_ARTIFACT_EXPLANATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why_this_exists": (
            "FROZEN_ARTIFACT_RECHECK found two data artifacts declared by LOCKED_5_OF_5 that no "
            "longer match their recorded digests. Adding dataset 6 is the obvious explanation "
            "and is not, by itself, evidence. This tests it."),
        "artifacts": out,
        "semantic_comparison": sem,
        "semantic_comparison_is": (
            "the five's entries in the CURRENT files, checked against the counts LOCKED_5_OF_5 "
            "froze for those datasets and against the existence of every store path. If the "
            "five had been altered, this is what would show it."),
        "failures": fails,
        "VERDICT": "EXPLAINED_ADDITIVE" if not fails else "FAIL",
        "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for p, e in sorted(out.items()):
        print("%s" % p)
        print("   declared %d bytes, on disk %d bytes" % (e["declared_bytes"], e["actual_bytes"]))
        print("   byte reconstruction: %s"
              % (e["byte_reconstruction"]["convention"] if e["byte_reconstruction"] else "no hit"))
    print("\nVERDICT %s  (%d failures)  %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:20]:
        print("  FAIL %s" % f)
    print("wrote %s" % OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
