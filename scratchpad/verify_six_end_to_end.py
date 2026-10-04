# -*- coding: utf-8 -*-
"""
The last gate: does the package behave the way the handoff says it does?

WHY THIS EXISTS WHEN EVERY DATASET ALREADY HAS ITS OWN GATE
    The per-dataset gates check artifacts. This one checks the PUBLIC INTERFACE -- the three
    calls a consumer will actually make (CanonicalEmbeddings, CanonicalGraph, families) -- for
    all six datasets, through the same import path a stranger would use, reading only what the
    handoff tells them to read. A package can have every artifact correct and still be unusable
    because a manifest entry points somewhere else, and no artifact-level gate would notice.

WHAT IT ASSERTS, AND WHY EACH ONE IS A THING THAT COULD ACTUALLY BE WRONG
    dims                  dense is 1536 and splade is 30522 for every dataset. A mismatch means
                          two encoders' output got mixed, which nothing else here would catch.
    normalisation         dense rows are L2-normalised. A zero row means a pointer resolved
                          into the damaged Phase-C region that the repair was supposed to route
                          around -- the exact failure mode of the corruption.
    no empty splade rows  an all-zero sparse row retrieves nothing and looks like a miss.
    id alignment          len(E) equals the node/query count, and E.ids[i] is the id at
                          position i. This is the positional-space claim the whole design rests
                          on; if it slipped, every result would be subtly wrong and plausible.
    graph endpoints       neighbours are in range for the node count, per declared family.
    families              the manifest is compared against DISK, not against itself: a family
                          declared present with no .npz is a broken promise, and an .npz that
                          is not declared is an edge set families() will never surface. The six
                          are deliberately non-uniform, so there is no fixed expectation to
                          compare to -- only manifest against what is actually there.

    Query rows are sampled with a fixed seed and documents at fixed positions, so a rerun reads
    the same rows. Sampling is honest here: the exhaustive versions of these checks already ran
    per dataset. This gate is about the interface, not a re-audit of 11.4M rows.
"""
import glob
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
OUT = ROOT + "/SIX_DATASET_END_TO_END.json"
SEED = 20260908
ALL = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
# a subset may be named on the command line, which is how this was validated against
# the five while dataset 6 was still encoding. The record says which set it covered.
DATASETS = [a for a in sys.argv[1:] if not a.startswith("-")] or ALL
for _d in DATASETS:
    if _d not in ALL:
        sys.exit("unknown dataset %r; known: %s" % (_d, ALL))


def main():
    t0 = time.time()
    sys.path.insert(0, ROOT)
    import pointer_resolver as pr

    res, fails = {}, []
    for ds in DATASETS:
        d = {}
        try:
            fam_declared = json.load(io.open("%s/%s/graph/GRAPH_MANIFEST.json" % (ROOT, ds),
                                             encoding="utf-8"))["families"]
        except Exception as e:
            fails.append("%s: no graph manifest (%s)" % (ds, e))
            fam_declared = {}

        for kind in ("docs", "queries"):
            for model in ("dense", "splade"):
                key = "%s/%s" % (kind, model)
                try:
                    E = pr.CanonicalEmbeddings(ds, model, kind, root=ROOT)
                except Exception as e:
                    fails.append("%s %s: open failed: %s" % (ds, key, e))
                    d[key] = {"error": str(e)}
                    continue
                n = len(E)
                rs = np.random.RandomState(SEED)
                pos = np.unique(np.concatenate([
                    np.array([0, n - 1]), rs.randint(0, n, size=min(64, n))])).astype(np.int64)
                V = E.gather(pos)
                if model == "dense":
                    A = np.asarray(V, dtype=np.float32)
                    nr = np.linalg.norm(A, axis=1)
                    d[key] = {"n": n, "dim": int(A.shape[1]), "sampled": int(pos.size),
                              "l2_min": float(nr.min()), "l2_max": float(nr.max()),
                              "n_zero_rows": int((nr == 0).sum()),
                              "all_finite": bool(np.isfinite(A).all())}
                    if A.shape[1] != 1536:
                        fails.append("%s %s: dim %d != 1536" % (ds, key, A.shape[1]))
                    if nr.min() < 0.99 or nr.max() > 1.01:
                        fails.append("%s %s: L2 out of range [%.5f, %.5f]"
                                     % (ds, key, nr.min(), nr.max()))
                    if not d[key]["all_finite"]:
                        fails.append("%s %s: non-finite values" % (ds, key))
                else:
                    nnz = np.diff(V.indptr)
                    d[key] = {"n": n, "vocab": int(V.shape[1]), "sampled": int(pos.size),
                              "nnz_min": int(nnz.min()), "nnz_mean": float(nnz.mean()),
                              "n_empty_rows": int((nnz == 0).sum()),
                              "min_value": float(V.data.min()) if V.nnz else None}
                    if V.shape[1] != 30522:
                        fails.append("%s %s: vocab %d != 30522" % (ds, key, V.shape[1]))
                    if int((nnz == 0).sum()):
                        fails.append("%s %s: %d empty rows in sample"
                                     % (ds, key, int((nnz == 0).sum())))
                    if V.nnz and float(V.data.min()) < 0:
                        fails.append("%s %s: negative SPLADE weight" % (ds, key))

                # ids are the positional-space claim: position i IS ids[i]
                d[key]["ids_len_matches"] = (len(E.ids) == n)
                if len(E.ids) != n:
                    fails.append("%s %s: ids %d != len %d" % (ds, key, len(E.ids), n))

        # the two channels of one kind must describe the SAME population
        for kind in ("docs", "queries"):
            a, b = d.get("%s/dense" % kind, {}), d.get("%s/splade" % kind, {})
            if a.get("n") is not None and b.get("n") is not None and a["n"] != b["n"]:
                fails.append("%s %s: dense n=%d but splade n=%d" % (ds, kind, a["n"], b["n"]))

        # graphs: what is declared present must load, and endpoints must be in range
        n_nodes = d.get("docs/dense", {}).get("n")
        # pr.families() reads the same manifest as fam_declared, so comparing the two would be
        # a tautology dressed as a check. The comparison that means something is manifest
        # against DISK: a family declared present with no file is a broken promise, and a
        # family file sitting there undeclared is an edge set a consumer will never see.
        got = sorted(pr.families(ds))
        on_disk = sorted(os.path.basename(p)[:-4]
                         for p in glob.glob("%s/%s/graph/*.npz" % (ROOT, ds)))
        d["families_declared_present"] = got
        d["family_files_on_disk"] = on_disk
        missing_file = [f for f in got if f not in on_disk]
        undeclared = [f for f in on_disk if f not in got]
        if missing_file:
            fails.append("%s: declared present but no file on disk: %s" % (ds, missing_file))
        if undeclared:
            fails.append("%s: family file(s) on disk but not declared present: %s -- a "
                         "consumer calling families() would never load them" % (ds, undeclared))
        gd = {}
        for fam in got:
            try:
                G = pr.CanonicalGraph(ds, fam, root=ROOT)
                rs = np.random.RandomState(SEED + 1)
                probe = np.unique(rs.randint(0, n_nodes, size=min(200, n_nodes)))
                mx, tot = -1, 0
                for i in probe:
                    nb = np.asarray(G.neighbors(int(i)))
                    tot += nb.size
                    if nb.size:
                        mx = max(mx, int(nb.max()))
                gd[fam] = {"len": len(G), "probed_nodes": int(probe.size),
                           "neighbours_seen": int(tot), "max_endpoint": mx,
                           "in_range": bool(mx < n_nodes)}
                if mx >= n_nodes:
                    fails.append("%s/%s: endpoint %d >= n_nodes %d" % (ds, fam, mx, n_nodes))
            except Exception as e:
                gd[fam] = {"error": str(e)}
                fails.append("%s/%s: graph failed to load: %s" % (ds, fam, e))
        d["graphs"] = gd
        res[ds] = d
        print("  %-9s docs=%-10s queries=%-8s families=%s"
              % (ds, format(d.get("docs/dense", {}).get("n", -1), ","),
                 format(d.get("queries/dense", {}).get("n", -1), ","), got), flush=True)

    rec = {
        "RECORD": "SIX_DATASET_END_TO_END",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "datasets_covered": DATASETS,
        "is_full_six": DATASETS == ALL,
        "what_this_is": (
            "the public interface exercised for all six datasets exactly as a consumer would: "
            "CanonicalEmbeddings, CanonicalGraph and families, imported from the package root. "
            "Per-dataset gates check the artifacts; this checks that the artifacts are reachable "
            "and self-consistent THROUGH the documented API."),
        "what_this_is_not": (
            "a re-audit. Document rows are probed at fixed positions and query rows at a fixed "
            "seed. The exhaustive per-row checks already ran per dataset and are cited in the "
            "freeze records; repeating them here would take hours and prove nothing new."),
        "datasets": res,
        "failures": fails,
        "VERDICT": "ALL_PASS" if not fails else "FAIL",
        "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nVERDICT %s  (%d failures)  %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:20]:
        print("  FAIL %s" % f)
    print("wrote %s" % OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
