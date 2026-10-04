"""One unambiguous name per substrate, and an honest account of the misleading physical ones.

"give proper final names not confusing ones."

The names are confusing for one structural reason: a physical tree under data/canonical/ is
named after the BUILD that produced it, not after the dataset or the role it serves.  Three
consequences, all real, all currently live:

  tree name != dataset name   data/canonical/2wiki_universe holds 2wiki's documents.
  tree name != role           data/canonical/metaqa holds metaqa's DOCUMENTS ONLY; its queries
                              are in three other trees.
  one tree, two lifetimes     data/canonical/2wiki is the corpse of the superseded 398,354-doc
                              tree with 2wiki's LIVE query vectors still inside it.  Half of
                              that directory was deleted as dead and half of it is load-bearing.

So a bare tree name cannot be trusted, and the fix is not to pick better strings -- it is that
no consumer should ever type a tree name at all.  The only name an experiment uses is the
LOGICAL name "<dataset>.<role>", resolved through substrate_map.authoritative().  That is the
name this module makes canonical, and it checks two properties that make it safe to rely on:

  TOTAL     every one of the 6 x 8 logical names resolves to something that exists, or is
            explicitly ABSENT with a reason.  No silent None.
  COVERING  every physical tree on disk is claimed by at least one logical name.  An unclaimed
            tree is either dead (delete it) or a substrate nobody can reach (worse).

WHY NOT JUST RENAME THE DIRECTORIES

Renaming is cheap in bytes -- a same-volume directory rename is a metadata operation, not a
copy of 33.71 GB.  It is expensive in RECORDS.  POINTER_INDEX.json stores absolute store paths
and is hash-pinned by LOCKED_6_OF_6; the standing contract is that an amendment gets a new
record with its own hash and never an edit.  So a rename is not "mv plus sed", it is: mv, write
POINTER_INDEX_V2 pinning the predecessor by recorded_sha256, re-verify all 41 stores, and
update every record and builder that names the old path.  cost_of_physical_rename below counts
those referrers from disk rather than estimating them.

WHY NOT SYMLINKS OR NTFS JUNCTIONS

A data/canonical/_by_role/2wiki/docs junction pointing at 2wiki_universe would read nicely and
costs nothing.  It is still the wrong answer: it creates a SECOND path to the same bytes, so
two experiments can cite different names for one substrate and a diff of their configs shows a
difference that is not one.  The user's stated fear is that "later experiments silently mix
these" -- two names for one substrate is a mixing hazard, not a cure for one.  One name, and it
is the logical name.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/canonical_names.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substrate_map as S  # noqa: E402

ROOT = S.ROOT if hasattr(S, "ROOT") else os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAN = os.path.join(ROOT, "data", "canonical")
OUT = os.path.join(ROOT, "data", "final_canonical", "CANONICAL_NAMES.json")

DATASETS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]

# What each physical tree really holds, and what its name wrongly implies.  "" = the name is
# accurate and needs no warning.
TREE_TRUTH = {
    "squad": ("squad documents + queries", ""),
    "musique": ("musique documents + queries", ""),
    "hotpotqa": ("hotpotqa documents + queries", ""),
    "metaqa": ("metaqa DOCUMENTS ONLY (43,234 KB entities)",
               "the name implies the whole metaqa substrate; the queries are in the three "
               "metaqa_*hop trees"),
    "metaqa_1hop": ("metaqa QUERIES, 1-hop split", "looks like a separate dataset; it is one "
                    "query split of metaqa"),
    "metaqa_2hop": ("metaqa QUERIES, 2-hop split", "looks like a separate dataset; it is one "
                    "query split of metaqa"),
    "metaqa_3hop": ("metaqa QUERIES, 3-hop split", "looks like a separate dataset; it is one "
                    "query split of metaqa"),
    "2wiki": ("2wiki QUERIES ONLY (192,606 vectors)",
              "THE WORST NAME IN THE PACKAGE. It is the surviving half of the superseded "
              "398,354-doc tree: the doc side was deleted as dead, the query side is live. "
              "The name says '2wiki' but 2wiki's documents are in 2wiki_universe"),
    "2wiki_universe": ("2wiki DOCUMENTS (5,989,847 vectors, 33.71 GB)",
                       "'_universe' reads like an optional extra or a superset; it is the "
                       "actual 2wiki document corpus"),
    "webqsp_rog_v1": ("webqsp documents + queries, the RoG WebQSP+CWQ MID subgraph",
                      ""),
}

# The logical name is what a consumer writes.  Nothing else is a name.
def logical(ds, role):
    return "%s.%s" % (ds, role)


def norm(p):
    return p.replace("\\", "/")


def tree_of(p):
    p = norm(str(p))
    return p.split("data/canonical/")[1].split("/")[0] if "data/canonical/" in p else None


def referrers(tree):
    """Records and code that name this tree by its literal path -- the rename bill."""
    pat = "canonical/%s/" % tree
    out = {}
    for label, where, inc in (("records", "data/final_canonical", "*.json"),
                              ("src", "src", "*.py"),
                              ("scratchpad", "scratchpad", "*.py")):
        try:
            r = subprocess.run(["grep", "-rl", "--include=" + inc, pat, where],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            out[label] = sorted(norm(x) for x in r.stdout.split() if x.strip())
        except Exception:
            out[label] = []
    return out


def flatten(v):
    """authoritative() returns a str, a list, or -- for the graph role -- a DICT of
    family -> path.  Wrapping a dict in [v] and str()ing it yields "{'structural': ...}",
    which exists as no file, so all six graph roles reported a phantom missing path while
    every one of the 18 families was in fact present and correctly routed between graph/
    and graph2/.  The substrate was right and the reporter was wrong."""
    if v is None:
        return []
    if isinstance(v, dict):
        out = []
        for x in v.values():
            out.extend(flatten(x))
        return out
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            out.extend(flatten(x))
        return out
    return [v] if v else []


def main():
    print("=" * 98)
    print("THE ONLY NAMES A CONSUMER MAY USE  --  logical name -> physical path")
    print("=" * 98)
    table, missing = {}, []
    for ds in DATASETS:
        for role in S.ROLES:
            try:
                v = S.authoritative(ds, role)
            except Exception as e:
                v = None
                err = str(e)
            else:
                err = None
            paths = [norm(str(x)) for x in flatten(v)]
            exists = [p for p in paths if os.path.exists(os.path.join(ROOT, p))]
            gone = [p for p in paths if p not in exists]
            table[logical(ds, role)] = {"paths": paths, "exists": len(exists),
                                        "missing": gone, "error": err}
            if gone:
                missing.append((logical(ds, role), gone))
    w = max(len(k) for k in table)
    for k in sorted(table, key=lambda x: (DATASETS.index(x.split(".")[0]), x)):
        e = table[k]
        if not e["paths"]:
            print("  %-*s  ABSENT" % (w, k))
        else:
            trees = sorted({t for t in (tree_of(p) for p in e["paths"]) if t})
            tag = "" if not e["missing"] else "   *** %d MISSING ***" % len(e["missing"])
            print("  %-*s  %-34s %s%s" % (w, k, ",".join(trees) or "-",
                                          e["paths"][0][:46], tag))
    print("\n  logical names with a missing path: %d" % len(missing))

    print("\n" + "=" * 98)
    print("WHAT EACH PHYSICAL TREE REALLY HOLDS  --  and why its name misleads")
    print("=" * 98)
    on_disk = sorted(os.listdir(CAN)) if os.path.isdir(CAN) else []
    claimed = {}
    for k, e in table.items():
        for p in e["paths"]:
            t = tree_of(p)
            if t:
                claimed.setdefault(t, set()).add(k)
    orphans, undoc = [], []
    for t in on_disk:
        holds, warn = TREE_TRUTH.get(t, ("UNDOCUMENTED", "not in TREE_TRUTH"))
        if t not in TREE_TRUTH:
            undoc.append(t)
        by = sorted(claimed.get(t, []))
        if not by:
            orphans.append(t)
        print("\n  data/canonical/%s" % t)
        print("    holds       %s" % holds)
        print("    claimed by  %s" % (", ".join(by) if by else "*** NO LOGICAL NAME ***"))
        if warn:
            print("    MISLEADING  %s" % warn)
    print("\n  trees on disk: %d   unclaimed: %s   undocumented: %s"
          % (len(on_disk), orphans or "none", undoc or "none"))

    print("\n" + "=" * 98)
    print("COST OF A PHYSICAL RENAME  --  counted, not estimated")
    print("=" * 98)
    proposed = {"2wiki": "2wiki_queries_phasec398",
                "2wiki_universe": "2wiki_docs",
                "metaqa": "metaqa_docs",
                "metaqa_1hop": "metaqa_queries_1hop",
                "metaqa_2hop": "metaqa_queries_2hop",
                "metaqa_3hop": "metaqa_queries_3hop"}
    print("  %-18s -> %-26s %8s %7s %10s" % ("tree", "proposed", "records", "src/.py", "scratch"))
    bill = {}
    tot = [0, 0, 0]
    for t in on_disk:
        r = referrers(t)
        bill[t] = {"proposed": proposed.get(t), "referrers": r,
                   "counts": {k: len(v) for k, v in r.items()}}
        if t in proposed:
            tot[0] += len(r["records"]); tot[1] += len(r["src"]); tot[2] += len(r["scratchpad"])
        print("  %-18s -> %-26s %8d %7d %10d" % (t, proposed.get(t, "(name is fine)"),
                                                 len(r["records"]), len(r["src"]),
                                                 len(r["scratchpad"])))
    print("\n  renaming the 6 misleading trees would touch %d records, %d files in src/, "
          "%d in scratchpad/," % tuple(tot))
    print("  and require a NEW POINTER_INDEX record pinning the old one by recorded_sha256")
    print("  (the frozen one may not be edited), plus a full 41-store re-resolution.")
    print("  Bytes moved: 0 -- same-volume renames are metadata only.")

    rec = {
        "RECORD": "CANONICAL_NAMES_V1",
        "generated_by": "src/dataset_canonical/canonical_names.py",
        "rule": "A consumer names a substrate ONLY as '<dataset>.<role>' and resolves it with "
                "substrate_map.authoritative(). A physical tree name under data/canonical/ is "
                "a build artifact and is not a name.",
        "roles": S.ROLES,
        "logical_to_physical": table,
        "logical_names_with_missing_paths": dict(missing),
        "tree_reality": {t: {"holds": TREE_TRUTH.get(t, ("UNDOCUMENTED", ""))[0],
                             "misleading_because": TREE_TRUTH.get(t, ("", ""))[1],
                             "claimed_by": sorted(claimed.get(t, []))} for t in on_disk},
        "unclaimed_trees": orphans,
        "undocumented_trees": undoc,
        "cost_of_physical_rename": bill,
        "why_not_symlinks_or_junctions":
            "An alias creates a second path to one substrate, so two experiment configs can "
            "differ in name while pointing at identical bytes. That is the mixing hazard the "
            "logical name exists to remove, not a cure for it.",
        "rename_contract":
            "POINTER_INDEX.json stores absolute store paths and is hash-pinned by "
            "LOCKED_6_OF_6. A rename requires a NEW POINTER_INDEX record whose BUILDS_ON pins "
            "the predecessor under recorded_sha256. The frozen record is never edited.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote %s" % norm(os.path.relpath(OUT, ROOT)))
    ok = not missing and not orphans and not undoc
    print("NAMES: %s" % ("PASS" if ok else "ATTENTION"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
