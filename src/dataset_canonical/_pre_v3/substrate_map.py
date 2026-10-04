"""ONE authoritative path per (dataset, role) -- derived from the records, then enforced.

The danger this exists to remove: the physical layout does not match the logical one, so a
plausible-looking path can be the wrong substrate.

  2wiki      docs resolve from data/canonical/2wiki_universe/, queries from data/canonical/2wiki/
             -- the same dataset, two trees, and the tree named plainly is the SUPERSEDED one
  metaqa     docs from data/canonical/metaqa/, queries from THREE trees, metaqa_{1,2,3}hop
  webqsp     resolves to data/canonical/webqsp_rog_v1/; data/canonical/webqsp is a DIFFERENT
             node universe (1,316,466 nodes, no shared ids) and is now an empty shell

Every one of those is discoverable by globbing data/canonical/* and picking the obvious name,
which is why this module reads POINTER_INDEX, the graph manifests and RETRIEVAL_CACHE instead
of the directory listing. A path is authoritative because a record resolves to it, never
because of what it is called.

Two things are then true and checkable, rather than documented and hoped for:
  * authoritative(ds, role) returns the resolving path(s), and each exists and is non-empty
  * assert_not_superseded(p) raises on any path in the superseded registry

    PYTHONHASHSEED=0 python src/dataset_canonical/substrate_map.py
    PYTHONHASHSEED=0 python src/dataset_canonical/substrate_map.py --scan-code
"""
import argparse
import collections
import datetime
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = "data/final_canonical"
DS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]
FAMILIES = ["structural", "ner", "knn"]

ROLES = ["documents", "queries", "graph", "dense_docs", "dense_queries",
         "splade_docs", "splade_queries", "retrieval_cache"]

# Paths nothing may resolve, each with the successor that replaced it. A path is listed here
# only when a record proves the successor, never on a naming hunch.
#
# data/ukb_storage is deliberately NOT in this list. It is not a dead canonical substrate --
# it is the LEGACY STACK, live for every L1/L2/L3 experiment and explicitly not to be deleted.
# Listing it here made 141 of 167 "violations" be ordinary working code and buried the 26 that
# matter. It gets its own registry below, and its own verdict.
SUPERSEDED = [
    ("data/canonical/webqsp",
     "data/canonical/webqsp_rog_v1",
     "A DIFFERENT node universe: 1,316,466 Phase-C nodes against the corpus's 2,592,894, "
     "sharing no ids. POINTER_INDEX resolves webqsp entirely into webqsp_rog_v1. Emptied by "
     "the cleanup; only bare directories remained."),
    ("data/canonical/2wiki/encodings/dense/docs",
     "data/canonical/2wiki_universe/encodings/dense/docs",
     "docs side of the superseded 398,354-node tree. The corpus has 5,989,847 nodes and no "
     "pointer resolves here."),
    ("data/canonical/2wiki/encodings/splade/docs",
     "data/canonical/2wiki_universe/encodings/splade/docs",
     "as above, SPLADE channel."),
    ("data/canonical/2wiki/documents.jsonl",
     "data/final_canonical/2wiki/nodes.jsonl",
     "the 398,354-node Phase-C corpus, superseded by the 5,989,847-node rebuild."),
    ("data/canonical/2wiki/graph_structural.tsv",
     "data/canonical/2wiki_universe/graph_structural.tsv",
     "398,354-node graph. 2wiki/graph/GRAPH_MANIFEST.json cites the 2wiki_universe TSV as "
     "source_tsv for the frozen family, not this one."),
    ("data/canonical/2wiki/graph_ner.tsv",
     "data/canonical/2wiki_universe/graph_ner.tsv",
     "as above, NER family."),
    ("data/canonical/2wiki/graph_knn.tsv",
     None,
     "398,354-node kNN. The corpus's kNN family is graph2/knn.npz over the 5,989,847-node "
     "universe; this TSV is not its source and nothing cites it."),
    ("data/canonical/2wiki/partition_map_C.json",
     None,
     "a partition map over the 398,354-node tree. Partitioning has not been run on "
     "canonical_v1, and a map keyed to a dead node space would silently mis-scope every query."),
    ("data/final_canonical/_superseded_textualization_rev1",
     "data/final_canonical/<ds>/nodes.jsonl",
     "TEXTUALIZATION_REV 1 for hotpotqa and 2wiki. REV 2 is the substrate."),
    ("data/final_canonical/2wiki/_superseded_context_union_398354",
     "data/final_canonical/2wiki/nodes.jsonl",
     "the 398,354-node context union."),
]

# A different stack, not a dead one. A reference means "this file is on the legacy substrate",
# which is a fact worth reporting and not a defect to fix.
OTHER_STACK = [
    ("data/ukb_storage",
     "the LEGACY substrate, live for the L1/L2/L3 experiment stack and NOT to be deleted. "
     "Never the canonical_v1 answer to any role here, so code reading it is on the other "
     "stack -- which is the mixing hazard to watch once canonical_v1 acquires consumers."),
]


def rj(p):
    with io.open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


def _stores(node):
    return [s["path"].replace("\\", "/") for s in node.get("stores", []) if s.get("path")]


def graph2_supersedes(ds):
    """Families the graph2 manifest explicitly declares as superseding the frozen copy.

    The precedence rule (pointer_resolver_v2): frozen wins for a family both trees declare,
    UNLESS graph2/GRAPH_MANIFEST.json carries `supersedes.<family>`. Only such a declaration
    is returned; a family merely present in graph2 is not.
    """
    mp = os.path.join(ROOT, FC, ds, "graph2", "GRAPH_MANIFEST.json")
    if not os.path.isfile(mp):
        return {}
    man = rj("%s/%s/graph2/GRAPH_MANIFEST.json" % (FC, ds))
    return {f: d for f, d in (man.get("supersedes") or {}).items()
            if (man.get("families") or {}).get(f, {}).get("present")}


def serving_tree(ds, fam):
    """Which tree pointer_resolver_v2 hands out for (ds, fam): "graph", "graph2" or None."""
    sup = graph2_supersedes(ds)
    for sub in (("graph2", "graph") if fam in sup else ("graph", "graph2")):
        if os.path.isfile(os.path.join(ROOT, FC, ds, sub, "%s.npz" % fam)):
            return sub
    return None


def build():
    pi = rj(FC + "/POINTER_INDEX.json")
    rc = rj(FC + "/RETRIEVAL_CACHE.json")

    # retrieval caches by dataset, read out of the record rather than by globbing
    cache = collections.defaultdict(list)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "file" and isinstance(v, str):
                    p = v.replace("\\", "/")
                    m = re.search(r"final_canonical/([^/]+)/retrieval_cache/", p)
                    if m:
                        cache[m.group(1)].append(p)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(rc)

    out = {}
    for ds in DS:
        d = pi["datasets"][ds]
        q = pi["queries"][ds]
        graphs = {}
        for fam in FAMILIES:
            sub = serving_tree(ds, fam)
            if sub is None:
                continue
            graphs[fam] = "%s/%s/%s/%s.npz" % (FC, ds, sub, fam)
        qdir = os.path.join(ROOT, FC, ds, "queries")
        out[ds] = {
            "documents": ["%s/%s/nodes.jsonl" % (FC, ds)],
            "queries": sorted("%s/%s/queries/%s" % (FC, ds, f) for f in os.listdir(qdir)
                              if f.endswith(".jsonl")),
            "graph": graphs,
            "dense_docs": _stores(d["dense"]),
            "dense_queries": _stores(q.get("dense", {})),
            "splade_docs": _stores(d["splade"]),
            "splade_queries": _stores(q.get("splade", {})),
            "retrieval_cache": sorted(cache.get(ds, [])),
        }
    return out


def superseded_frozen_graphs():
    """{ds: {fam: {frozen_path, frozen_sha256, sha256_on_disk, byte_identical, reason}}} for every
    frozen family a graph2 manifest supersedes. Not a role: the 8-role shape of build() is what
    every consumer iterates, so this lives beside it. The frozen file must still be on disk and
    byte-identical to the declaration -- a supersession describes, it never repairs."""
    import hashlib
    out = {}
    for ds in DS:
        for fam, dcl in graph2_supersedes(ds).items():
            if serving_tree(ds, fam) != "graph2":
                continue
            fp = "%s/%s/graph/%s.npz" % (FC, ds, fam)
            ap = os.path.join(ROOT, fp)
            h = None
            if os.path.isfile(ap):
                hh = hashlib.sha256()
                with open(ap, "rb") as f:
                    for c in iter(lambda: f.read(1 << 22), b""):
                        hh.update(c)
                h = hh.hexdigest()
            out.setdefault(ds, {})[fam] = {
                "frozen_path": fp, "frozen_sha256": dcl.get("frozen_sha256"),
                "sha256_on_disk": h, "byte_identical": h is not None and h == dcl.get("frozen_sha256"),
                "still_pinned_by": dcl.get("pinned_by"), "reason": dcl.get("reason"),
                "declared_in": "%s/%s/graph2/GRAPH_MANIFEST.json#supersedes.%s" % (FC, ds, fam)}
    return out


def check(m):
    """Every authoritative path must exist and be non-empty. An empty directory is a failure,
    not a pass -- data/canonical/webqsp survived the cleanup as exactly that."""
    bad = []
    for ds, roles in m.items():
        for role, v in roles.items():
            for p in (list(v.values()) if isinstance(v, dict) else v):
                ap = os.path.join(ROOT, p)
                if os.path.isfile(ap):
                    if os.path.getsize(ap) == 0:
                        bad.append((ds, role, p, "file is empty"))
                elif os.path.isdir(ap):
                    if not [f for f in os.listdir(ap) if not f.startswith(".")]:
                        bad.append((ds, role, p, "directory is empty"))
                else:
                    bad.append((ds, role, p, "does not exist"))
    return bad


def assert_not_superseded(path):
    """Raise if `path` is, or is inside, a superseded substrate."""
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    for dead, succ, why in SUPERSEDED:
        if p == dead or p.startswith(dead.rstrip("/") + "/"):
            raise ValueError(
                "SUPERSEDED SUBSTRATE: %s\n  matched rule: %s\n  use instead: %s\n  %s"
                % (path, dead, succ or "(nothing -- this artifact has no successor)", why))
    return path


def authoritative(dataset, role, _cache={}):
    """The path(s) a dataset's role actually resolves to."""
    if not _cache:
        _cache.update(build())
    if dataset not in _cache:
        raise KeyError("unknown dataset %r; known: %s" % (dataset, ", ".join(DS)))
    if role not in ROLES:
        raise KeyError("unknown role %r; known: %s" % (role, ", ".join(ROLES)))
    return _cache[dataset][role]


def scan_code():
    """The proof, not the promise: can any experiment resolve to a superseded substrate?

    A dead path is harmless only while nothing reads it, and WHERE it is named decides whether
    that is true:

      src/         experiment and pipeline code. A reference here is a VIOLATION -- this is
                   what "an experiment silently resolves to the wrong substrate" looks like.
      scratchpad/  build and audit code. Nearly every reference here is the BUILDER OF THE
                   ARTIFACT ITSELF -- c3_2wiki_graph.py's line 3 names
                   data/canonical/2wiki/graph_structural.tsv because it WRITES it. Deleting the
                   builder of a superseded artifact would delete the record of how the
                   substrate came to be, so these are reported as HISTORICAL, not as defects.

    Scans .py and .json, excluding this package -- it must name every dead path to forbid it.
    """
    hits = collections.defaultdict(list)
    other = collections.defaultdict(list)
    scanned = 0
    for base in ("src", "scratchpad"):
        for r, dirs, files in os.walk(os.path.join(ROOT, base)):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in files:
                if not f.endswith((".py", ".json")):
                    continue
                rel = os.path.relpath(os.path.join(r, f), ROOT).replace("\\", "/")
                if rel.startswith("src/dataset_canonical/"):
                    continue
                scanned += 1
                try:
                    txt = io.open(os.path.join(r, f), encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for dead, succ, _w in SUPERSEDED:
                    mo = re.search(re.escape(dead) + r"(?![\w])", txt)
                    if mo:
                        hits[dead].append({
                            "file": rel,
                            "line": txt.count("\n", 0, mo.start()) + 1,
                            "successor": succ,
                            "verdict": "VIOLATION" if base == "src" else "HISTORICAL_BUILDER",
                        })
                for legacy, _w in OTHER_STACK:
                    mo = re.search(re.escape(legacy) + r"(?![\w])", txt)
                    if mo:
                        other[legacy].append(rel)
    return scanned, hits, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-code", action="store_true")
    ap.add_argument("--out", default=FC + "/CANONICAL_SUBSTRATE_MAP.json")
    a = ap.parse_args()

    m = build()
    print("%-9s %-16s %s" % ("dataset", "role", "authoritative path(s)"))
    for ds in DS:
        for role in ROLES:
            v = m[ds][role]
            ps = ["%s=%s" % (k, x) for k, x in v.items()] if isinstance(v, dict) else list(v)
            if not ps:
                print("  %-9s %-16s (none: slot absent for this dataset)" % (ds, role))
            for i, p in enumerate(ps):
                print("  %-9s %-16s %s" % (ds if i == 0 else "", role if i == 0 else "", p))
        print()

    bad = check(m)
    sup = superseded_frozen_graphs()
    for ds, fams in sup.items():
        for fam, v in fams.items():
            print("  %-9s %-16s frozen %s superseded by graph2 (%s)"
                  % (ds, "graph", v["frozen_path"],
                     "byte-identical" if v["byte_identical"] else "FROZEN BYTES MOVED"))
            if not v["byte_identical"]:
                bad.append((ds, "graph", v["frozen_path"],
                            "superseded frozen copy is missing or not byte-identical to the "
                            "declaration in graph2/GRAPH_MANIFEST.json"))
    print("AUTHORITATIVE PATHS: %d datasets x %d roles -- %d broken"
          % (len(DS), len(ROLES), len(bad)))
    for ds, role, p, why in bad:
        print("  !! %-9s %-16s %-58s %s" % (ds, role, p, why))

    scanned, hits, other = 0, {}, {}
    viol = []
    if a.scan_code:
        scanned, hits, other = scan_code()
        viol = [e for v in hits.values() for e in v if e["verdict"] == "VIOLATION"]
        hist = [e for v in hits.values() for e in v if e["verdict"] == "HISTORICAL_BUILDER"]
        print("\nCODE SCAN: %d files under src/ and scratchpad/" % scanned)
        print("  VIOLATIONS (src/, an experiment could resolve here): %d" % len(viol))
        for e in viol[:20]:
            print("     !! %s:%d" % (e["file"], e["line"]))
        print("  HISTORICAL (scratchpad/, the builder of the artifact): %d" % len(hist))
        for dead, v in sorted(hits.items(), key=lambda kv: -len(kv[1])):
            print("       %-56s %d file(s)" % (dead, len(v)))
        for legacy, v in other.items():
            insrc = [f for f in v if f.startswith("src/")]
            print("  OTHER STACK %s: %d files (%d in src/) -- on the legacy substrate by "
                  "design, not a defect" % (legacy, len(v), len(insrc)))

    rec = {
        "RECORD": "CANONICAL_SUBSTRATE_MAP",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_what": ("the authoritative path per (dataset, role), derived from POINTER_INDEX, the "
                  "graph manifests and RETRIEVAL_CACHE -- never from directory names."),
        "_how_to_use": ("substrate_map.authoritative(dataset, role) to resolve; "
                        "substrate_map.assert_not_superseded(path) to refuse a dead one."),
        "roles": ROLES,
        "datasets": m,
        "graph_precedence": ("frozen graph/ wins for a family both trees declare UNLESS the "
                             "graph2 manifest carries an explicit supersedes.<family> block "
                             "(pointer_resolver_v2.family_source); the frozen copy then stays "
                             "on disk, pinned and byte-identical, and is listed below"),
        "graph_superseded_frozen": sup,
        "authoritative_paths_broken": [{"dataset": d, "role": r, "path": p, "why": w}
                                       for d, r, p, w in bad],
        "superseded": [{"path": p, "successor": s, "why": w} for p, s, w in SUPERSEDED],
        "other_stack": [{"path": p, "why": w} for p, w in OTHER_STACK],
        "code_scan": {
            "ran": a.scan_code,
            "files_scanned": scanned,
            "violations_in_src": viol,
            "n_violations_in_src": len(viol),
            "references": dict(hits),
            "other_stack_references": {k: v for k, v in other.items()},
            "_verdict_rule": ("a reference in src/ is a VIOLATION (experiment code could "
                              "resolve to a dead substrate); a reference in scratchpad/ is "
                              "HISTORICAL -- that file is normally the builder that WROTE the "
                              "superseded artifact, and is the record of how it was made."),
        },
        "NO_EXPERIMENT_CAN_RESOLVE_TO_A_SUPERSEDED_SUBSTRATE": (not viol) if a.scan_code else None,
        "PASS": (not bad) and not viol,
    }
    with io.open(os.path.join(ROOT, a.out), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("\nwrote %s" % a.out)
    ok = (not bad) and not viol
    print("")
    print("SUBSTRATE AUTHORITY: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
