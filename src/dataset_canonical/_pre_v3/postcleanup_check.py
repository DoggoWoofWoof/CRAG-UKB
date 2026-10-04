"""Prove the cleanup did not destroy anything live -- by RESOLVING, not by re-hashing.

The hash verification in verify_manifest.py checks the 206 DECLARED artifacts. That is not
sufficient after a deletion: the encoding stores under data/canonical/<tree>/encodings/ are
tens of thousands of shards, and a declared-artifact walk would not notice one of them
missing. The only honest test is to resolve real rows through the pointer index and read the
bytes back -- for docs AND queries, on all six datasets, including the repair-patch redirects
that hold the only good copies of the 142,633 damaged rows.

A zero vector is a FAILURE here, not a pass: reading a damaged shard directly returns zeros,
so "no exception" alone would hide exactly the breakage this check exists to catch.

    PYTHONHASHSEED=0 python src/dataset_canonical/postcleanup_check.py
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
sys.path.insert(0, FC)

import pointer_resolver as pr          # noqa: E402
import pointer_resolver_v2 as pr2      # noqa: E402

DS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]
FAMILIES = ["structural", "ner", "knn"]
N_SAMPLE = 64


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def stores_from(pi):
    """Every declared store, read from the "path" key ONLY.

    "store" next to it is a LABEL (PHASE_C, REV2_PATCH, DENSE_REPAIR), not a location -- a
    crawler that treats any string under a store-ish key as a path reports 37 phantom missing
    files and buries the real result.
    """
    out = []
    def walk(o, where):
        if isinstance(o, dict):
            if isinstance(o.get("path"), str):
                out.append({"path": o["path"], "label": o.get("store"), "where": where,
                            "flat": bool(o.get("flat")), "n_items": o.get("n_items"),
                            "shard_size": o.get("shard_size")})
            for k, v in o.items():
                if k != "path":
                    walk(v, where + "." + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (where, i))
    walk(pi, "")
    return out


def check_stores():
    """Existence AND completeness: a store missing one shard is the failure mode a deletion
    actually produces, and no manifest hash would notice it."""
    st = stores_from(rj(os.path.join(FC, "POINTER_INDEX.json")))
    bad = []
    for e in st:
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(ROOT, e["path"])
        if e["flat"]:
            e["exists"] = os.path.isfile(p)
            if not e["exists"]:
                bad.append(dict(e, why="flat array missing"))
            continue
        e["exists"] = os.path.isdir(p)
        if not e["exists"]:
            bad.append(dict(e, why="shard dir missing"))
            continue
        # a sharded store must hold ceil(n_items / shard_size) shards, all of them
        n, ss = e.get("n_items"), e.get("shard_size")
        if not n or not ss:
            continue
        want = (n + ss - 1) // ss
        got = len([f for f in os.listdir(p) if not f.startswith(".")])
        e["shards_expected"], e["shards_found"] = want, got
        if got < want:
            bad.append(dict(e, why="%d of %d shards present" % (got, want)))
    return st, bad


def check_embeddings():
    """Resolve real rows and insist the bytes come back non-zero."""
    rows = []
    for ds in DS:
        for kind in ("docs", "queries"):
            rec = {"dataset": ds, "kind": kind}
            try:
                emb = pr.CanonicalEmbeddings(ds, "dense", kind=kind, root=FC)
                n = len(emb)
                idx = sorted(set(np.linspace(0, n - 1, min(N_SAMPLE, n)).astype(int).tolist()))
                X = emb.gather(idx)
                X = np.asarray(X, dtype=np.float32)
                norms = np.linalg.norm(X, axis=1)
                rec.update(n_rows=n, n_sampled=len(idx), dim=int(X.shape[1]),
                           min_norm=float(norms.min()), max_norm=float(norms.max()),
                           n_zero_rows=int((norms == 0).sum()))
                rec["ok"] = rec["n_zero_rows"] == 0 and rec["min_norm"] > 0.5
            except Exception as e:                       # noqa: BLE001
                rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
            rows.append(rec)
    return rows


def check_graphs():
    """Every family must still load and report its edge count, from graph/ or graph2/."""
    rows = []
    for ds in DS:
        fams = pr2.families(ds, root=FC)
        for fam in FAMILIES:
            rec = {"dataset": ds, "family": fam, "listed": fam in fams}
            try:
                g = pr2.CanonicalGraph(ds, fam, root=FC)
                rec.update(source=pr2.family_source(ds, fam, root=FC),
                           n_edges=len(g), ok=len(g) > 0)
            except Exception as e:                       # noqa: BLE001
                rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
            rows.append(rec)
    return rows


def main():
    out = {"RECORD": "POST_CLEANUP_RESOLUTION_CHECK"}

    st, missing = check_stores()
    nsh = sum(1 for e in st if e.get("shards_expected"))
    print("POINTER_INDEX declares %d stores (%d sharded, %d flat arrays); %d broken"
          % (len(st), nsh, sum(1 for e in st if e["flat"]), len(missing)))
    for m in missing[:20]:
        print("  !! %-34s %s" % (m["why"], m["path"]))
    out["stores"] = {"n_declared": len(st), "n_broken": len(missing),
                     "broken": missing, "declared": st}

    emb = check_embeddings()
    print("\n%-10s %-8s %10s %6s %9s %9s %6s" %
          ("dataset", "kind", "rows", "samp", "min|v|", "max|v|", "ok"))
    for r in emb:
        if "error" in r:
            print("%-10s %-8s  FAILED  %s" % (r["dataset"], r["kind"], r["error"]))
        else:
            print("%-10s %-8s %10d %6d %9.4f %9.4f %6s" %
                  (r["dataset"], r["kind"], r["n_rows"], r["n_sampled"],
                   r["min_norm"], r["max_norm"], r["ok"]))
    out["embeddings"] = emb

    gr = check_graphs()
    print()
    for ds in DS:
        parts = ["%s:%s(%s)" % (r["family"], r.get("source"), r.get("n_edges", "FAIL"))
                 for r in gr if r["dataset"] == ds]
        print("  %-10s %s" % (ds, "  ".join(parts)))
    out["graphs"] = gr

    bad = (list(missing) + [r for r in emb if not r.get("ok")]
           + [r for r in gr if not r.get("ok")])
    out["PASS"] = not bad
    print("\nPOST-CLEANUP RESOLUTION CHECK: PASS=%s  (%d failures)" % (not bad, len(bad)))
    with io.open(os.path.join(FC, "POST_CLEANUP_CHECK.json"), "w",
                 encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
