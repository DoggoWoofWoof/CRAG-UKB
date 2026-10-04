# -*- coding: utf-8 -*-
"""
Check data/final_canonical/HANDOFF.md against the records it describes.

HANDOFF.md is the one thing a new reader is guaranteed to read and the one file in the package
that no digest pins (it is prose, written after the freeze). Every number it quotes therefore
gets checked against CANONICAL_FREEZE.json, the six DATASET.json records, the graph manifests,
VERIFICATION.json and the materialisation log -- and the quickstart is actually executed.

    paths        every backticked concrete path exists (templated ones are listed, not skipped silently)
    table        section 1: per-dataset nodes / queries / structural / ner / knn and the total row
    splits       the split table: every "name (rows)" cell and the evaluate-on column
    families     the direction letters (d/u) of every family against GRAPH_MANIFEST.json
    limitations  every KNOWN_LIMITATIONS entry of the freeze appears verbatim
    digests      every 16-hex prefix quoted is a prefix of a digest carried by a record in the universe
    verify       the VERIFICATION.json verdict / counts quoted are the file's
    freed        the "GB freed" figure equals the materialisation log's deletion total
    api          the quickstart calls work on all six datasets
    lanes        section 6 (when the freeze carries QUERY_LANES): per dataset the three lane sizes, and that the
                 metaqa QUALITY_LOCKED lane is stated to be the test split
    builders     section 10 (when the freeze carries BUILDERS): the snapshot record is named and every LOST builder
                 revision is stated with its digest prefix
    freebase     section 11 (when data/final_canonical/freebase/DATASET.json exists): node / edge / relation counts,
                 the name_kind table, the floor-by-rule table, the freebase VERIFICATION.json verdict, and the
                 Freebase loader smoke (position <-> uid, names, out/in edges)

Writes data/final_canonical/_history/logs/HANDOFF_CLAIMS_CHECK.json; exit 1 on any failure.
"""
import glob
import hashlib
import io
import json
import os
import re
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FC = "data/final_canonical"
HIST = FC + "/_history"
H = FC + "/HANDOFF.md"
OUT = HIST + "/logs/HANDOFF_CLAIMS_CHECK.json"
DATASETS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
SEARCH_ROOTS = [".", FC, HIST, HIST + "/records", HIST + "/logs", HIST + "/docs", HIST + "/pre_v3",
                "src/dataset_canonical", "src/dataset_canonical/freebase", "scratchpad", "scratchpad/fb4",
                FC + "/freebase_v3", FC + "/freebase", FC + "/freebase/records"]


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def num(x):
    return int(str(x).replace(",", "").replace("*", "").strip())


def main():
    t0 = time.time()
    doc = io.open(H, encoding="utf-8").read()
    fz = load(FC + "/CANONICAL_FREEZE.json")
    D = {ds: load("%s/%s/DATASET.json" % (FC, ds)) for ds in DATASETS}
    fails, checks = [], {}

    # ---------------------------------------------------------------- paths
    cand = set(re.findall(r"`([A-Za-z_0-9][A-Za-z_0-9./<>{},*\-]*\.(?:py|json|jsonl|npz|npy|md|gz|txt))`", doc))
    templated, missing, ok_paths = [], [], []
    for p in sorted(cand):
        if any(ch in p for ch in "<>{}*"):
            templated.append(p)
            continue
        hits = [os.path.join(r, p) for r in SEARCH_ROOTS]
        hits += glob.glob("%s/*/%s" % (FC, p)) + glob.glob("%s/*/*/%s" % (FC, p))
        (ok_paths if any(os.path.exists(h) for h in hits) else missing).append(p)
    checks["paths"] = {"named": len(cand), "resolved": len(ok_paths), "templated": templated, "missing": missing}
    for p in missing:
        fails.append("document names a path that does not exist: %s" % p)

    # ---------------------------------------------------------------- table
    tab = {}
    for ds in DATASETS:
        m = re.search(r"^\|\s*%s\s*\|([^\n]+)\|\s*$" % re.escape(ds), doc, re.M)
        if not m:
            fails.append("section 1 table has no row for %s" % ds)
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        got = {"n_nodes": num(cells[0]), "n_queries": num(cells[1]),
               "structural": num(cells[2]), "ner": num(cells[3]), "knn": num(cells[4])}
        want = fz["DATASETS"][ds]
        exp = {"n_nodes": want["n_nodes"], "n_queries": want["n_queries"]}
        exp.update({f: want["graph"]["n_edges"].get(f) for f in ("structural", "ner", "knn")})
        tab[ds] = {"doc": got, "record": exp}
        for k in got:
            if got[k] != exp[k]:
                fails.append("table %s %s: document %s, freeze %s" % (ds, k, got[k], exp[k]))
    m = re.search(r"^\|\s*\*\*total\*\*\s*\|([^\n]+)\|\s*$", doc, re.M)
    if not m:
        fails.append("section 1 table has no total row")
    else:
        cells = [c.strip() for c in m.group(1).split("|")]
        got = {"nodes": num(cells[0]), "queries": num(cells[1]),
               "edges": num(cells[2]) + num(cells[3]) + num(cells[4])}
        T = fz["TOTALS"]
        tab["total"] = {"doc": got, "record": {"nodes": T["nodes"], "queries": T["queries"], "edges": T["edges_served"]}}
        for k, kk in (("nodes", "nodes"), ("queries", "queries"), ("edges", "edges_served")):
            if got[k] != T[kk]:
                fails.append("total %s: document %s, freeze %s" % (k, got[k], T[kk]))
    checks["table"] = tab

    # --------------------------------------------------------------- splits
    sp = {}
    for ds in DATASETS:
        m = re.search(r"^\|\s*%s\s*\|([^|]+)\|([^|]+)\|\s*$" % re.escape(ds), doc, re.M)
        if not m:
            fails.append("split table has no row for %s" % ds)
            continue
        cells = re.findall(r"(\w+)\s*\(([0-9,]+)\)", m.group(1))
        got = [(a, num(b)) for a, b in cells]
        q = D[ds]["queries"]
        exp = [(s, q["by_split"][s]["n"]) for s in q["splits_in_row_order"]]
        ev = m.group(2).strip().strip("`")
        sp[ds] = {"doc": got, "record": exp, "eval_doc": ev, "eval_record": fz["EVAL_SPLITS"][ds]["split"]}
        if got != exp:
            fails.append("splits %s: document %s, DATASET.json %s" % (ds, got, exp))
        if ev != fz["EVAL_SPLITS"][ds]["split"]:
            fails.append("evaluate-on %s: document %r, freeze %r" % (ds, ev, fz["EVAL_SPLITS"][ds]["split"]))
    checks["splits"] = sp

    # ------------------------------------------------------------- families
    fam = {}
    for ds in DATASETS:
        gm = load("%s/%s/graph/GRAPH_MANIFEST.json" % (FC, ds))
        if gm.get("RECORD") != "CANONICAL_GRAPH":
            fails.append("%s graph manifest is not CANONICAL_GRAPH" % ds)
        fam[ds] = {k: ("d" if v["directed"] else "u") for k, v in gm["families"].items() if v.get("present")}
    rows = re.findall(r"^\|\s*(\w+)\s*\|\s*([du]) [^|]*\|\s*([du]) [^|]*\|\s*([du]) [^|]*\|\s*$", doc, re.M)
    fdoc = {r[0]: {"structural": r[1], "ner": r[2], "knn": r[3]} for r in rows}
    checks["families"] = {"doc": fdoc, "manifests": fam}
    for ds in DATASETS:
        if ds not in fdoc:
            fails.append("family table has no row for %s" % ds)
        elif fdoc[ds] != fam[ds]:
            fails.append("family directions %s: document %s, manifest %s" % (ds, fdoc[ds], fam[ds]))

    # ----------------------------------------------------------- limitations
    lim_missing = [l for l in fz["KNOWN_LIMITATIONS"] if l not in doc]
    checks["limitations"] = {"in_freeze": len(fz["KNOWN_LIMITATIONS"]), "missing_from_doc": lim_missing}
    for l in lim_missing:
        fails.append("KNOWN_LIMITATIONS entry not in the document verbatim: %s" % l[:80])
    res_missing = [l for l in fz["RESOLVED_SINCE_THE_EARLIER_FREEZES"] if l not in doc]
    for l in res_missing:
        fails.append("RESOLVED_SINCE entry not in the document verbatim: %s" % l[:80])
    if fz["CORE_INVARIANT"] not in doc:
        fails.append("CORE_INVARIANT is not quoted verbatim")

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
            for x in re.findall(r"\b[0-9a-f]{64}\b", o):
                known.add(x)
    universe = sorted(set([FC + "/CANONICAL_FREEZE.json", HIST + "/INDEX.json", FC + "/VERIFICATION.json"]
                          + ["%s/%s/DATASET.json" % (FC, ds) for ds in DATASETS]
                          + glob.glob(HIST + "/records/*.json")
                          + glob.glob(FC + "/freebase_v3/LOCKED_*.json")
                          + glob.glob(FC + "/freebase_v3/FREEBASE_V3_CLOSURE_STATUS*.json")   # V1 (2026-09-11) and V2 (terminal, 2026-09-13)
                          + glob.glob(HIST + "/logs/FREEBASE_V3_SUBLAYER_DELETION.json")
                          + glob.glob(FC + "/freebase_v3/NAME_HIERARCHY_CONTRACT_V1.json")
                          + glob.glob(FC + "/freebase/DATASET.json") + glob.glob(FC + "/freebase/VERIFICATION.json")
                          + glob.glob(FC + "/freebase/records/*.json")))
    for u in universe:
        if os.path.exists(u):
            collect(load(u))
    quoted = set(re.findall(r"\b([0-9a-f]{16})\b(?=…|\.\.\.|`)", doc))
    unmatched = sorted(q for q in quoted if not any(k.startswith(q) for k in known))
    checks["digests"] = {"quoted": sorted(quoted), "unmatched": unmatched, "known": len(known),
                         "universe_files": len(universe)}
    for q in unmatched:
        fails.append("document quotes digest prefix %s… which no record carries" % q)

    # --------------------------------------------------------------- verify
    V = load(FC + "/VERIFICATION.json") if os.path.exists(FC + "/VERIFICATION.json") else None
    if V is None:
        fails.append("VERIFICATION.json is missing")
    else:
        m = re.search(r"([0-9,]+) checks, ([0-9,]+) failures", doc)
        got = (num(m.group(1)), num(m.group(2))) if m else None
        exp = (V["checks"], V["failed"])
        checks["verify"] = {"doc": got, "record": exp, "pass": V["PASS"], "mode": V.get("mode")}
        if got != exp:
            fails.append("verification counts: document %s, VERIFICATION.json %s" % (got, exp))
        if ("PASS" if V["PASS"] else "FAIL") not in doc:
            fails.append("VERIFICATION verdict is not quoted")
        if V.get("mode") != "full":
            fails.append("VERIFICATION.json is not a --full run")

    # ---------------------------------------------------------------- freed
    m = re.search(r"\*\*deleted\*\* — ([0-9.]+) GB deleted", doc)
    freed = fz["LINEAGE"]["consolidation_and_materialisation"]["deleted"]
    exp_gb = round(freed["total_bytes"] / 1e9, 2)
    checks["freed"] = {"doc_gb": float(m.group(1)) if m else None, "record_gb": exp_gb}
    if not m or abs(float(m.group(1)) - exp_gb) > 0.06:
        fails.append("freed GB: document %s, log %s" % (m.group(1) if m else None, exp_gb))

    # ------------------------------------------------------------------ api
    sys.path.insert(0, FC)
    try:
        import numpy as np
        import canonical
        api = {}
        for ds in DATASETS:
            d = canonical.Dataset(ds)
            E = d.embeddings("dense", "docs")
            S = d.embeddings("splade", "docs")
            fams = d.families()
            G = d.graph(fams[0])
            v = E.read([0, min(5, d.n_nodes - 1)])
            s = S.read([0])
            nb = G.neighbors(0)
            api[ds] = {"n_nodes": d.n_nodes, "n_queries": d.n_queries, "families": fams, "dense_read": list(v.shape),
                       "splade_read": list(s.shape), "graph_len": len(G), "neighbors_0": int(nb.size)}
            if d.n_nodes != D[ds]["n_nodes"] or E.n_rows != D[ds]["n_nodes"] or S.n_rows != D[ds]["n_nodes"]:
                fails.append("api %s: n_nodes disagree with DATASET.json" % ds)
            if d.n_queries != D[ds]["n_queries"]:
                fails.append("api %s: n_queries disagree with DATASET.json" % ds)
            if v.shape != (2, 1536) or v.dtype != np.float16 or s.shape != (1, 30522):
                fails.append("api %s: read shapes/dtypes wrong" % ds)
            if fams != fz["DATASETS"][ds]["graph"]["families_present"]:
                fails.append("api %s: families %s vs freeze %s" % (ds, fams, fz["DATASETS"][ds]["graph"]["families_present"]))
        ids, scores = canonical.Dataset("webqsp").cache("dense")
        api["webqsp_cache"] = [list(ids.shape), str(ids.dtype), str(scores.dtype)]
        if list(ids.shape) != fz["DATASETS"]["webqsp"]["retrieval_cache"]["dense"]["shape"]:
            fails.append("api: webqsp cache shape %s vs freeze" % (ids.shape,))
        checks["api"] = api
    except Exception as e:
        fails.append("the quickstart API did not work: %r" % e)

    # ---------------------------------------------------------------- lanes
    QL = fz.get("QUERY_LANES")
    if QL:
        sec6 = doc[doc.index("## 6."):doc.index("## 7.")] if "## 6." in doc and "## 7." in doc else ""
        lanes = {}
        for ds, ln in QL["census"].items():
            if not ln:
                continue
            m = re.search(r"^\|\s*%s\s*\|(.*)$" % re.escape(ds), sec6, re.M)
            row = m.group(1) if m else ""
            want = [ln[k]["n"] for k in ("LEGACY_CONTINUITY", "QUALITY_LOCKED", "SCALE_ALL")]
            nums = [num(x) for x in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", row)]
            ok = m is not None and all(w in nums for w in want) and (ds != "metaqa" or "test" in row)
            lanes[ds] = {"doc_row": row.strip(), "record": want, "ok": ok}
            if not ok:
                fails.append("lanes %s: section 6 row %r does not state %s%s" % (ds, row.strip(), want, " (and 'test')" if ds == "metaqa" else ""))
        checks["lanes"] = lanes

    # ------------------------------------------------------------- builders
    BL = fz.get("BUILDERS")
    if BL:
        sec10 = doc[doc.index("## 10."):doc.index("## 11.")] if "## 10." in doc and "## 11." in doc else ""
        b = {"index_named": "_history/builders/BUILDERS.json" in sec10, "lost": {}}
        if not b["index_named"]:
            fails.append("builders: section 10 does not name _history/builders/BUILDERS.json")
        for ds, e in BL["build_info_revisions"].items():
            if e and e.get("status") == "LOST":
                pref = (e.get("builder_sha256") or "")[:8]
                b["lost"][ds] = {"sha_prefix": pref, "stated": pref in sec10}
                if pref not in sec10:
                    fails.append("builders: the lost %s builder revision %s... is not stated in section 10" % (ds, pref))
        checks["builders"] = b

    # ------------------------------------------------------------- freebase
    FBD = FC + "/freebase/DATASET.json"
    if os.path.exists(FBD):
        fb = load(FBD)
        fbc = {}
        sec = doc[doc.index("## 11."):doc.index("## 12.")] if "## 11." in doc and "## 12." in doc else ""
        if not sec:
            fails.append("freebase: section 11 not found")
        for key, label in (("n_nodes", "nodes"), ("n_edges", "edges"), ("n_relations", "relations")):
            want = "{:,}".format(fb[key])
            fbc[key] = {"doc_has": want in sec, "record": fb[key]}
            if want not in sec:
                fails.append("freebase: section 11 does not state %s = %s" % (label, want))
        fzf = fz.get("FREEBASE")
        if not fzf:
            fails.append("freebase: CANONICAL_FREEZE.json has no FREEBASE section although freebase/DATASET.json exists")
        elif fzf["DATASET.json"]["sha256"] != hashlib.sha256(open(FBD, "rb").read()).hexdigest():
            fails.append("freebase: DATASET.json is not the file the freeze pinned")
        kinds = {}
        for k, n in fb["names"]["census"].items():
            m = re.search(r"^\|\s*%s\s*\|\s*([0-9,]+)\s*\|" % re.escape(k), sec, re.M)
            kinds[k] = {"doc": num(m.group(1)) if m else None, "record": n}
            if not m or num(m.group(1)) != n:
                fails.append("freebase name_kind %s: document %s, DATASET.json %s" % (k, m.group(1) if m else None, n))
        fbc["name_kinds"] = kinds
        rules = {}
        for k, n in (fb["names"].get("floor_by_rule") or {}).items():
            m = re.search(r"^\|\s*%s\s*\|\s*([0-9,]+)\s*\|" % re.escape(k), sec, re.M)
            rules[k] = {"doc": num(m.group(1)) if m else None, "record": n}
            if not m or num(m.group(1)) != n:
                fails.append("freebase floor rule %s: document %s, DATASET.json %s" % (k, m.group(1) if m else None, n))
        fbc["floor_rules"] = rules
        FBV = FC + "/freebase/VERIFICATION.json"
        if os.path.exists(FBV):
            VF = load(FBV)
            m = re.search(r"freebase[^\n]*?([0-9,]+) checks, ([0-9,]+) failures", sec)
            got = (num(m.group(1)), num(m.group(2))) if m else None
            fbc["verify"] = {"doc": got, "record": (VF["checks"], VF["failed"]), "pass": VF["PASS"], "mode": VF.get("mode")}
            if got != (VF["checks"], VF["failed"]):
                fails.append("freebase verification counts: document %s, freebase/VERIFICATION.json %s" % (got, (VF["checks"], VF["failed"])))
            if VF.get("mode") != "full" or not VF["PASS"]:
                fails.append("freebase/VERIFICATION.json is not a passing --full run")
        else:
            fails.append("freebase/VERIFICATION.json is missing")
        try:
            sys.path.insert(0, FC)
            import numpy as np
            import canonical
            f = canonical.Freebase()
            uid = f.node_uid[12345]
            assert f.position(int(uid)) == 12345
            nm = f.name(12345)
            assert isinstance(nm, str) and nm
            dst, rel, flag = f.out_edges(12345)
            src, rel2 = f.in_edges(12345)
            fbc["api"] = {"n_nodes": int(f.n_nodes), "n_edges": int(f.n_edges), "name_12345": nm, "out_12345": int(len(dst)), "in_12345": int(len(src)),
                          "kind_12345": f.node(12345)["kind_name"]}
            if f.n_nodes != fb["n_nodes"] or f.n_edges != fb["n_edges"]:
                fails.append("freebase api: n_nodes/n_edges disagree with DATASET.json")
        except Exception as e:
            fails.append("the Freebase loader did not work: %r" % e)
        checks["freebase"] = fbc

    rec = {"RECORD": "HANDOFF_CLAIMS_CHECK",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "what": "HANDOFF.md checked against the freeze, the DATASET.json records, the graph manifests, VERIFICATION.json "
                   "and the materialisation log; the quickstart executed on all six datasets",
           "handoff_lines": doc.count("\n") + 1,
           "handoff_sha256_lf": hashlib.sha256(doc.replace("\r\n", "\n").encode("utf-8")).hexdigest(),
           "checks": checks, "failures": fails, "VERDICT": "ALL_PASS" if not fails else "FAIL",
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("paths %d named / %d resolved / %d templated / %d missing" % (
        checks["paths"]["named"], checks["paths"]["resolved"], len(templated), len(missing)))
    print("digests quoted %d, unmatched %d" % (len(quoted), len(unmatched)))
    print("VERDICT %s (%d failures) %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:40]:
        print("  FAIL", f)
    print("wrote", OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
