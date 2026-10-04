"""Which stack does the code actually read? -- the machine-checkable form of the
"have we replaced the old files with the new ones" question.

The honest answer is no, and the reason is a count, not an opinion: the legacy substrate has
dozens of readers in src/ and canonical_v1 has none outside this tooling package. That makes
"delete the old preprocessing code" a breaking change, and "canonical_v1 is finished but
unwired" a fact rather than a hedge.

Patterns are printed alongside the counts, so a reader can re-run the grep and disagree with
the number rather than with the claim.

    PYTHONHASHSEED=0 python src/dataset_canonical/stack_census.py
    PYTHONHASHSEED=0 python src/dataset_canonical/stack_census.py --out=data/final_canonical/STACK_CENSUS.json
"""
import argparse
import collections
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This package consumes NEITHER stack -- it is the tooling that describes them, and it names
# both stacks' paths in its own patterns and prose. So it is excluded from every probe, not
# just the canonical one: counted in, the census reports its own existence as progress on one
# side and as legacy debt on the other. This file alone moved four counts the first time it
# was run against itself.
SELF = "src/dataset_canonical/"

PROBES = [
    ("legacy", "ukb_storage_path", r"ukb_storage",
     "reads or writes the legacy substrate data/ukb_storage/<ds>_clean/"),
    ("legacy", "imports_src_pipeline",
     r"^\s*(?:from|import)\s+(?:src\.)?pipeline\b|^\s*from\s+\.{1,2}pipeline\b",
     "imports the legacy preprocessing package src/pipeline"),
    ("legacy", "uses_standardizer", r"\bstandardizer\b",
     "depends on src/pipeline/standardizer.py"),
    ("legacy", "uses_ukb_results", r"\bukb_results\b",
     "depends on src/pipeline/ukb_results.py, the UKB result-path authority"),
    ("canonical_v1", "reads_final_canonical", r"final_canonical|pointer_resolver",
     "reads the canonical_v1 substrate or resolves through its pointer index"),
]


def census():
    rx = [(name, re.compile(pat, re.M)) for _s, name, pat, _w in PROBES]
    hits = collections.defaultdict(set)
    n_py = n_self = 0
    for base, dirs, files in os.walk(os.path.join(ROOT, "src")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            n_py += 1
            rel = os.path.relpath(os.path.join(base, f), ROOT).replace("\\", "/")
            if rel.startswith(SELF):
                n_self += 1
            try:
                txt = open(os.path.join(base, f), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for name, r in rx:
                if r.search(txt):
                    hits[name].add(rel)

    out = {"n_py_under_src": n_py, "n_py_in_this_package": n_self, "probes": {}}
    for stack, name, pat, why in PROBES:
        files = sorted(hits[name])
        own = [p for p in files if p.startswith(SELF)]
        con = [p for p in files if not p.startswith(SELF)]
        out["probes"][name] = {
            "stack": stack,
            "pattern": pat,
            "means": why,
            "n_consumers": len(con),
            "n_matches_including_this_package": len(files),
            "in_this_package": own,
            "consumers": con if len(con) <= 40 else con[:40] + ["... %d more" % (len(con) - 40)],
        }

    cv = out["probes"]["reads_final_canonical"]["n_consumers"]
    legacy = max(out["probes"][n]["n_consumers"] for s, n, _p, _w in PROBES if s == "legacy")
    out["VERDICT"] = {
        "canonical_v1_downstream_consumers": cv,
        "legacy_substrate_readers": legacy,
        "old_code_replaced_by_new": False,
        "why": ("canonical_v1 is complete, frozen and verified, and %d files in src/ read it "
                "outside its own tooling package. %d read the legacy substrate. Deleting "
                "src/pipeline or data/ukb_storage today breaks the L1/L2/L3 stack, so cleanup "
                "here means retiring DATA that has a named successor -- not removing live "
                "code. Wiring canonical_v1 up is the partitioning step and everything after "
                "it." % (cv, legacy)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    out = census()
    out["RECORD"] = "CANONICAL_V1_STACK_CENSUS"
    out["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("%d .py files under src/, %d of them in %s (excluded from every count)\n"
          % (out["n_py_under_src"], out["n_py_in_this_package"], SELF))
    for stack, name, _pat, why in PROBES:
        pr = out["probes"][name]
        extra = "  (+%d here, excluded)" % len(pr["in_this_package"]) if pr["in_this_package"] else ""
        print("  %-12s %-22s %3d%s" % (stack, name, pr["n_consumers"], extra))
        print("               %s" % why)

    v = out["VERDICT"]
    print("\nOLD CODE REPLACED BY NEW: %s" % v["old_code_replaced_by_new"])
    print("  canonical_v1 downstream consumers in src/: %d" % v["canonical_v1_downstream_consumers"])
    print("  legacy substrate readers in src/:          %d" % v["legacy_substrate_readers"])

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
