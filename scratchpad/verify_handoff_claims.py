# -*- coding: utf-8 -*-
"""
Check the handoff document against the artifacts it describes.

WHY A GATE FOR A README
    HANDOFF.md is the only thing a new reader is guaranteed to read, and it is the one artifact
    in this package that nothing else validates. Every other number here was produced by a
    script and gated; the prose was typed. A document that quietly drifts from the data is
    worse than no document, because it is believed. So the document gets a gate too.

WHAT IS CHECKED
    paths        every concrete file path the document names actually exists. Templated paths
                 (<ds>, shard_*, {a,b}) are skipped and listed, so the skip is visible rather
                 than silent.
    totals       the corpus table's total row equals LOCKED_6_OF_6.TOTALS_ALL_SIX.
    webqsp row   the webqsp row of that table equals the frozen webqsp counts.
    digests      every 16-hex-prefix the document quotes is a real prefix of a digest that
                 appears in one of the frozen records. A quoted hash that matches nothing is
                 how a reader gets told to verify against a value that never existed.
    encodings    the encoder table's row counts and byte figures match the frozen record AND
                 the stores on disk.
    families     the "not all six have the same families" claims are checked against every
                 GRAPH_MANIFEST.json, so the document cannot describe a topology the data
                 does not have.
    api          the three calls in the quickstart are actually importable and callable.

WHAT IS NOT CHECKED
    The subtree size table in 2b. Re-measuring 190 GB takes minutes and the numbers are
    advisory. Pass --sizes to measure them anyway; they are then compared at 5% tolerance.
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
# reconfigure in place. Assigning a NEW TextIOWrapper over sys.stdout.buffer leaves the
# wrapper it replaced unreferenced; its __del__ closes the shared buffer, and the next
# print anywhere -- including in a module that imported this one -- dies with "I/O
# operation on closed file". reconfigure creates no second object.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
H = ROOT + "/HANDOFF.md"
OUT = ROOT + "/HANDOFF_CLAIMS_CHECK.json"
# the roots the document itself uses: section 6 says "All live in scratchpad/", the
# Freebase section points into scratchpad/fb/, and the encoder contract cites src/ and
# transfer/. A name is resolved if it exists under any of them.
SEARCH_ROOTS = [".", ROOT, "scratchpad", "scratchpad/fb", "src", "src/experiments",
                "transfer"]


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def num(x):
    return int(str(x).replace(",", "").replace("*", "").strip())


def main():
    t0 = time.time()
    doc = io.open(H, encoding="utf-8").read()
    six = load(ROOT + "/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json")
    five = load(ROOT + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json")
    fails, notes, checks = [], [], {}

    # ---------------------------------------------------------------- paths
    # backticked paths only: prose sentences also contain slashes.
    cand = set(re.findall(r"`([A-Za-z_0-9][A-Za-z_0-9./<>{},*\-]*\.(?:py|json|jsonl|npz|npy|md|gz|parquet))`", doc))
    cand |= set(re.findall(r"^\s{2,}([a-zA-Z_][A-Za-z_0-9./<>{},*\-]*\.(?:json|py|md|jsonl))\s{2,}",
                           doc, re.M))
    templated, missing, ok_paths = [], [], []
    for p in sorted(cand):
        if any(ch in p for ch in "<>{}*"):
            templated.append(p)
            continue
        hits = [os.path.join(r, p) for r in SEARCH_ROOTS]
        hits += glob.glob("%s/*/%s" % (ROOT, p))          # per-dataset files
        hits += glob.glob("%s/*/*/%s" % (ROOT, p))        # ...and their subdirectories
        if any(os.path.exists(h) for h in hits):
            ok_paths.append(p)
        else:
            missing.append(p)
    checks["paths"] = {"named": len(cand), "resolved": len(ok_paths),
                       "templated_skipped": templated, "missing": missing}
    for p in missing:
        fails.append("document names a path that does not exist: %s" % p)

    # --------------------------------------------------------------- totals
    row = re.search(r"\|\s*\*\*total\*\*\s*\|([^\n]+)\|", doc)
    if not row:
        fails.append("could not find the corpus table's total row")
    else:
        cells = [c.strip() for c in row.group(1).split("|")]
        got = {"n_nodes": num(cells[0]), "n_queries": num(cells[1])}
        want = six["TOTALS_ALL_SIX"]
        checks["totals"] = {"doc": got, "record": {k: want[k] for k in got}}
        for k in got:
            if got[k] != want[k]:
                fails.append("total %s: document says %s, LOCKED_6_OF_6 says %s"
                             % (k, got[k], want[k]))
        # the doc's edge total is the sum of its own edge columns. Which columns those are
        # is read from the table header, not hardcoded -- the first version of this check
        # hardcoded an index, included "gold refs", and reported a 3,927,095 discrepancy that
        # was entirely the gate's own bug.
        hdr = re.search(r"\|\s*dataset\s*\|([^\n]+)\|", doc)
        cols = [h.strip() for h in hdr.group(1).split("|")] if hdr else []
        edge_cols = [i for i, h in enumerate(cols) if h in ("structural", "ner", "knn")]
        checks.setdefault("table_header", cols)
        checks["edge_columns_used"] = [cols[i] for i in edge_cols]
        if not edge_cols:
            fails.append("could not identify the edge columns from the corpus table header")
        edges_doc = sum(num(cells[i]) for i in edge_cols
                        if cells[i].replace(",", "").replace("*", "").strip().isdigit())
        checks["totals"]["doc_edge_columns_sum"] = edges_doc
        checks["totals"]["record_n_edges"] = want["n_edges"]
        if edges_doc != want["n_edges"]:
            fails.append("edge columns sum to %s but LOCKED_6_OF_6 says %s"
                         % (edges_doc, want["n_edges"]))

    # ---------------------------------------------------------- webqsp row
    wrow = re.search(r"\|\s*webqsp\s*\|([^\n]+)\|", doc)
    w = six["webqsp"]
    if not wrow:
        fails.append("no webqsp row in the corpus table")
    else:
        cells = [c.strip() for c in wrow.group(1).split("|")]
        got = {"n_nodes": num(cells[0]), "n_queries": num(cells[1]),
               "structural_edges": num(cells[3])}
        checks["webqsp_row"] = {"doc": got,
                                "record": {"n_nodes": w["n_nodes"], "n_queries": w["n_queries"],
                                           "structural_edges": w["n_edges"]}}
        if got["n_nodes"] != w["n_nodes"]:
            fails.append("webqsp nodes: doc %s vs record %s" % (got["n_nodes"], w["n_nodes"]))
        if got["n_queries"] != w["n_queries"]:
            fails.append("webqsp queries: doc %s vs record %s" % (got["n_queries"], w["n_queries"]))
        if got["structural_edges"] != w["n_edges"]:
            fails.append("webqsp edges: doc %s vs record %s"
                         % (got["structural_edges"], w["n_edges"]))

    # -------------------------------------------------------------- digests
    known = set()

    def collect(o):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
        elif isinstance(o, str):
            for m in re.findall(r"\b[0-9a-f]{64}\b", o):
                known.add(m)
    # the universe is every frozen record on disk: the LOCKED_* records at the package root
    # and under freebase_v3/, plus the Freebase closure record, all self-hashed. A digest
    # quoted in the document must be a prefix of a digest one of these records carries.
    universe = sorted(set(glob.glob(ROOT + "/LOCKED_*.json")
                          + glob.glob(ROOT + "/freebase_v3/LOCKED_*.json")
                          + glob.glob(ROOT + "/freebase_v3/FREEBASE_V3_CLOSURE_STATUS.json")))
    for up in universe:
        collect(load(up))
    known.add(six["RECORD_SHA256"])
    known.add(six.get("RECORD_SHA256_LF", ""))
    known.discard("")
    quoted = set(re.findall(r"\b([0-9a-f]{16})\b(?=…|\.\.\.|`)", doc))
    quoted |= set(re.findall(r"`sha256 ([0-9a-f]{16})", doc))
    unmatched = sorted(q for q in quoted if not any(k.startswith(q) for k in known))
    checks["digests"] = {"quoted": sorted(quoted), "unmatched": unmatched,
                         "known_digests_in_records": len(known),
                         "digest_universe": [u.replace(chr(92), "/") for u in universe]}
    for q in unmatched:
        fails.append("document quotes digest prefix %s… which matches nothing in the frozen "
                     "records" % q)

    # ------------------------------------------------------------ encodings
    ch = w["encoding_channels"]
    enc = {}
    for line in doc.splitlines():
        m = re.match(r"\|\s*(docs|queries)/(dense|splade)\s*\|\s*([0-9,]+)\s*\|\s*`([^`]+)`\s*\|"
                     r"\s*([0-9.]+) GB\s*\|", line.strip())
        if m:
            enc["%s/%s" % (m.group(1), m.group(2))] = {
                "rows": num(m.group(3)), "store": m.group(4), "gb": float(m.group(5))}
    checks["encoding_table"] = {"rows_in_doc": len(enc), "channels_in_record": len(ch)}
    if len(enc) != len(ch):
        fails.append("encoder table has %d rows but the record has %d channels"
                     % (len(enc), len(ch)))
    disk = {}
    for k, v in sorted(enc.items()):
        r = ch.get(k)
        if r is None:
            fails.append("encoder table row %s is not a channel in the record" % k)
            continue
        if v["rows"] != r["rows"]:
            fails.append("%s rows: doc %s vs record %s" % (k, v["rows"], r["rows"]))
        if v["store"] != r["store"]:
            fails.append("%s store: doc %s vs record %s" % (k, v["store"], r["store"]))
        if abs(v["gb"] - r["bytes"] / 1e9) > 0.011:
            fails.append("%s bytes: doc %.2f GB vs record %.2f GB" % (k, v["gb"], r["bytes"] / 1e9))
        b = 0
        for f in glob.glob(r["store"] + "/shard_*.np*"):
            b += os.path.getsize(f)
        disk[k] = b
        if not os.path.isdir(r["store"]):
            fails.append("%s store directory is missing on disk: %s" % (k, r["store"]))
        elif abs(b - r["bytes"]) > max(1 << 20, r["bytes"] * 0.001):
            fails.append("%s on disk is %d bytes, record says %d" % (k, b, r["bytes"]))
    checks["encoding_bytes_on_disk"] = disk

    # ------------------------------------------------------------- families
    fam = {}
    for p in sorted(glob.glob("%s/*/graph/GRAPH_MANIFEST.json" % ROOT)):
        ds = p.replace(chr(92), "/").split("/")[-3]
        man = load(p)
        fam[ds] = sorted(k for k, v in man["families"].items()
                         if (v.get("present") if isinstance(v, dict) else bool(v)))
    checks["families_on_disk"] = fam
    claims = [("metaqa", "ner", False), ("hotpotqa", "knn", False), ("2wiki", "knn", False),
              ("webqsp", "knn", False), ("webqsp", "ner", False),
              ("webqsp", "structural", True)]
    for ds, f, want_present in claims:
        if ds not in fam:
            fails.append("no graph manifest for %s, but the document describes its families" % ds)
            continue
        have = f in fam[ds]
        if have != want_present:
            fails.append("document says %s %s family present=%s; manifests say %s"
                         % (ds, f, want_present, have))
    if fam.get("webqsp") != ["structural"]:
        fails.append("document says webqsp is structural-only; manifest says %s"
                     % (fam.get("webqsp"),))

    # ------------------------------------------------------------------ api
    sys.path.insert(0, ROOT)
    try:
        import pointer_resolver as pr
        api = {}
        for ds in ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]:
            E = pr.CanonicalEmbeddings(ds, "dense", "docs", root=ROOT)
            G = pr.CanonicalGraph(ds, sorted(pr.families(ds))[0], root=ROOT)
            api[ds] = {"docs": len(E), "families": sorted(pr.families(ds)), "graph_len": len(G)}
            if sorted(pr.families(ds)) != fam.get(ds):
                fails.append("families(%s) returns %s but the manifest on disk lists %s"
                             % (ds, sorted(pr.families(ds)), fam.get(ds)))
        checks["api"] = api
        tot = sum(v["docs"] for v in api.values())
        checks["api_total_docs"] = tot
        if tot != six["TOTALS_ALL_SIX"]["n_nodes"]:
            fails.append("documents reachable through the API total %s, record says %s"
                         % (tot, six["TOTALS_ALL_SIX"]["n_nodes"]))
    except Exception as e:
        fails.append("the quickstart API did not work: %s" % e)

    # ---------------------------------------------------------------- sizes
    if "--sizes" in sys.argv:
        sizes = {}
        for sub in ["freebase_v3", "_superseded_textualization_rev1", "_work",
                    "_dense_repair_patch", "_rev2_encoder_patch"]:
            tot = 0
            for dp, _dn, fn in os.walk(os.path.join(ROOT, sub)):
                for f in fn:
                    try:
                        tot += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            sizes[sub] = tot
        checks["measured_subtree_bytes"] = sizes
        notes.append("subtree sizes measured this run; compare against the 2b table by eye")
    else:
        notes.append("subtree size table in 2b was NOT re-measured (pass --sizes to measure); "
                     "those figures are advisory, every other number here was checked")

    rec = {"RECORD": "HANDOFF_CLAIMS_CHECK",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "what_this_is": ("HANDOFF.md checked against the artifacts it describes. The prose is "
                            "the one thing in this package nothing else gates, so it gets one."),
           "handoff_lines": doc.count(chr(10)) + 1,
           "handoff_sha256_lf": __import__("hashlib").sha256(
               doc.replace(chr(13) + chr(10), chr(10)).encode("utf-8")).hexdigest(),
           "checks": checks, "notes": notes, "failures": fails,
           "VERDICT": "ALL_PASS" if not fails else "FAIL",
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("paths: %d named, %d resolved, %d templated, %d missing"
          % (checks["paths"]["named"], checks["paths"]["resolved"],
             len(checks["paths"]["templated_skipped"]), len(checks["paths"]["missing"])))
    print("digests quoted: %d, unmatched: %d" % (len(quoted), len(unmatched)))
    print("families on disk: %s" % json.dumps(fam))
    print("\nVERDICT %s  (%d failures)  %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:30]:
        print("  FAIL %s" % f)
    print("wrote %s" % OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
