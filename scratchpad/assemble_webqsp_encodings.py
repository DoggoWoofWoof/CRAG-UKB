# -*- coding: utf-8 -*-
"""
Assemble the WebQSP encoder output into the canonical store layout and wire up the pointer
index.

WHY THE MODAL PARTS BECOME THE SHARDS DIRECTLY
    The five frozen datasets shard at 40,000 rows. The Modal parts are 12,000. Re-sharding
    would mean writing a second full copy of a 5.5 GB dense matrix on a volume with ~13 GB
    free, to change a number that cannot affect a single vector: shard_size is carried PER
    STORE in POINTER_INDEX.json and the resolver reads it from there
    (_Store.read: shard = row // self.shard_size). So the parts are renamed into place and the
    store declares 12,000. The difference is recorded rather than hidden, and it is a storage
    detail, not an encoding one.

WHAT IS VERIFIED BEFORE ANYTHING IS MOVED
    Each part carries rows.json, the encoder rows it covers in array order. Concatenated in
    part order those must be exactly 0..N-1 -- not a permutation of it, not a superset, exactly
    that sequence -- because shard k offset j is claimed to be row 12000k + j. A part that is
    short, out of order, duplicated or missing breaks that identity, and every downstream shape
    would still agree. So it is checked before the first rename, and the run refuses rather
    than assembling something plausible.

THE POINTER INDEX
    docs     row[i] = encoder_row_of_node[i]  -- the many-to-one map, 2,592,894 -> 1,791,533
    queries  row[i] = i                       -- identity; query order IS query_ids.json
    src      all zeros: one store per channel, no patches (webqsp has no REV2 history and no
             damaged shards to redirect around)
"""
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
D = ROOT + "/webqsp"
SRC = D + "/_enc"                                   # where `modal volume get` landed
DEST = "data/canonical/webqsp_rog_v1/encodings"     # NOT data/canonical/webqsp -- see below
CHUNK = 12000
OUT = D + "/ENCODING_ASSEMBLY.json"

CHANNELS = [("dense", "docs", "dense", "dense.npy"),
            ("splade", "docs", "splade", "splade.npz"),
            ("dense", "queries", "dense_q", "dense.npy"),
            ("splade", "queries", "splade_q", "splade.npz")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def expected_n(kind):
    p = D + ("/encoder_inputs.jsonl" if kind == "docs" else "/query_inputs.jsonl")
    return sum(1 for _ in io.open(p, encoding="utf-8"))


def main():
    t0 = time.time()
    rec = {"RECORD": "WEBQSP_ENCODING_ASSEMBLY",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "shard_size": CHUNK,
           "shard_size_note": (
               "12,000 not 40,000. The Modal parts ARE the shards; re-sharding would write a "
               "second 5.5 GB copy on a volume with ~13 GB free to change a number the "
               "resolver reads per-store from POINTER_INDEX.json and that cannot affect a "
               "vector."),
           "store_root": DEST,
           "store_root_note": (
               "webqsp_rog_v1, NOT data/canonical/webqsp. That tree is a DIFFERENT node "
               "universe (1,316,466 nodes) whose dense docs channel is also one of the ten "
               "damaged ones. Writing here would silently mix two corpora that share a name."),
           "channels": {}}

    for model, kind, tag, fname in CHANNELS:
        parts = sorted(glob.glob("%s/%s__p*" % (SRC, tag)))
        if not parts:
            sys.exit("no parts found for %s (%s/%s__p*) -- did `modal volume get` run?"
                     % (tag, SRC, tag))
        n_exp = expected_n(kind)

        # ---- verify every part is COMPLETE before moving anything -------------------------
        # The coverage check below reads only rows.json. A part left by an interrupted pull can
        # have rows.json and no payload yet, which passes coverage and then dies in the middle
        # of the move loop -- half the shards relocated, half not, and the source tree no longer
        # reconstructible without re-downloading. So existence is checked across all parts
        # first, when the cost of stopping is zero.
        missing = [os.path.basename(p) for p in parts
                   if not (os.path.exists(os.path.join(p, "rows.json"))
                           and os.path.exists(os.path.join(p, fname)))]
        if missing:
            sys.exit("%s: %d part(s) are incomplete (missing rows.json or %s), first few: %s "
                     "-- rerun scratchpad/pull_webqsp_encodings.py, it is resumable"
                     % (tag, len(missing), fname, missing[:5]))
        stray = sorted(glob.glob("%s/%s__p*/*.part" % (SRC, tag)))
        if stray:
            sys.exit("%s: %d unfinished .part file(s) present, first: %s -- the pull did not "
                     "finish" % (tag, len(stray), stray[0]))

        # ---- verify coverage BEFORE moving anything ---------------------------------------
        seen, sizes = [], []
        for k, p in enumerate(parts):
            rows = json.load(io.open(os.path.join(p, "rows.json"), encoding="utf-8"))
            seen.extend(rows)
            sizes.append(len(rows))
            if len(rows) != CHUNK and k != len(parts) - 1:
                sys.exit("%s part %s has %d rows; only the LAST part may be short"
                         % (tag, os.path.basename(p), len(rows)))
        if seen != list(range(n_exp)):
            bad = next((i for i, v in enumerate(seen) if v != i), len(seen))
            sys.exit("%s: concatenated rows are not 0..%d (length %d, first divergence at %d) "
                     "-- shard k offset j would not be row %d*k+j"
                     % (tag, n_exp - 1, len(seen), bad, CHUNK))

        # ---- move into the canonical shard layout -----------------------------------------
        od = "%s/%s/%s" % (DEST, model, kind)
        os.makedirs(od, exist_ok=True)
        files, nrows, nbytes = [], 0, 0
        for k, p in enumerate(parts):
            ext = "npy" if model == "dense" else "npz"
            dst = "%s/shard_%05d.%s" % (od, k, ext)
            src_f = os.path.join(p, fname)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src_f, dst)
            json.dump(json.load(io.open(os.path.join(p, "rows.json"), encoding="utf-8")),
                      io.open("%s/ids_%05d.json" % (od, k), "w", encoding="utf-8"))
            nbytes += os.path.getsize(dst)
            files.append({"shard": k, "rows": sizes[k], "bytes": os.path.getsize(dst)})
            nrows += sizes[k]

        # ---- confirm what landed, by reading it back --------------------------------------
        last = "%s/shard_%05d.%s" % (od, len(parts) - 1, "npy" if model == "dense" else "npz")
        if model == "dense":
            a = np.load(last, mmap_mode="r")
            shape, extra = list(a.shape), {}
            nr = np.linalg.norm(np.asarray(a[:256], dtype=np.float32), axis=1)
            extra = {"l2_min_first256": float(nr.min()), "l2_max_first256": float(nr.max())}
            if not (nr.min() > 0.99 and nr.max() < 1.01):
                sys.exit("%s: last shard is not L2-normalised [%.4f, %.4f]"
                         % (tag, nr.min(), nr.max()))
            if shape[1] != 1536:
                sys.exit("%s: dense dim is %d, contract says 1536" % (tag, shape[1]))
        else:
            z = np.load(last)
            shape = [int(z["shape"][0]), int(z["shape"][1])]
            extra = {"nnz_last_shard": int(z["data"].size)}
            if shape[1] != 30522:
                sys.exit("%s: splade vocab is %d, contract says 30522" % (tag, shape[1]))

        rec["channels"]["%s/%s" % (kind, model)] = {
            "store": od, "n_parts": len(parts), "n_rows": nrows, "expected_rows": n_exp,
            "bytes": nbytes, "last_shard_shape": shape, **extra}
        print("  %-16s %3d shards  %s rows  %.2f GB  %s  %.0fs"
              % ("%s/%s" % (kind, model), len(parts), format(nrows, ","), nbytes / 1e9,
                 shape, time.time() - t0), flush=True)

    # ---- pointer index ---------------------------------------------------------------------
    m = np.load(D + "/encoder_row_of_node.npz")
    node_row = m[list(m.keys())[0]].astype(np.int32)
    nq = expected_n("queries")
    os.makedirs(D + "/pointer_index", exist_ok=True)
    os.makedirs(D + "/queries/pointer_index", exist_ok=True)
    for model in ("dense", "splade"):
        np.savez(D + "/pointer_index/%s.npz" % model,
                 src=np.zeros(node_row.size, dtype=np.int8), row=node_row)
        np.savez(D + "/queries/pointer_index/%s.npz" % model,
                 src=np.zeros(nq, dtype=np.int8),
                 row=np.arange(nq, dtype=np.int32))

    man_p = ROOT + "/POINTER_INDEX.json"
    man = json.load(io.open(man_p, encoding="utf-8"))
    for sect, kind in (("datasets", "docs"), ("queries", "queries")):
        for model in ("dense", "splade"):
            man.setdefault(sect, {}).setdefault("webqsp", {})[model] = {
                "stores": [{"path": "%s/%s/%s" % (DEST, model, kind),
                            "shard_size": CHUNK,
                            "kind": "WEBQSP_V1",
                            "note": ("shard_size is 12,000 here, not the 40,000 the five text "
                                     "datasets use; the resolver reads it from this record")}]}
    json.dump(man, io.open(man_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---- round trip through the resolver ----------------------------------------------------
    sys.path.insert(0, ROOT)
    import importlib
    pr = importlib.import_module("pointer_resolver")
    importlib.reload(pr)
    checks = {}
    for kind in ("docs", "queries"):
        for model in ("dense", "splade"):
            E = pr.CanonicalEmbeddings("webqsp", model, kind, root=ROOT)
            n = len(E)
            probe = E.gather(np.array([0, 1, n // 2, n - 1]))
            shp = (list(probe.shape) if model == "dense" else list(probe.shape))
            checks["%s/%s" % (kind, model)] = {"len": n, "probe_shape": shp}
            print("  resolver %-16s len=%s probe=%s"
                  % ("%s/%s" % (kind, model), format(n, ","), shp), flush=True)

    # the many-to-one map is the point of the dedup: two nodes sharing a text MUST share a
    # vector. If they do not, the map and the store disagree and the dedup is a lie.
    uniq, first_idx, counts = np.unique(node_row, return_index=True, return_counts=True)
    shared = uniq[counts > 1]
    tie = 0
    if shared.size:
        E = pr.CanonicalEmbeddings("webqsp", "dense", "docs", root=ROOT)
        for r in shared[:5]:
            nodes = np.nonzero(node_row == r)[0][:2]
            v = E.gather(nodes)
            if np.array_equal(np.asarray(v[0]), np.asarray(v[1])):
                tie += 1
    checks["shared_text_nodes_share_a_vector"] = {
        "pairs_checked": int(min(shared.size, 5)), "identical": tie,
        "n_texts_used_by_more_than_one_node": int(shared.size)}
    print("  shared-text nodes share a vector: %d/%d pairs identical"
          % (tie, min(shared.size, 5)))
    if shared.size and tie != min(shared.size, 5):
        sys.exit("a node pair sharing a text does not share a vector -- the dedup map and the "
                 "store disagree")

    rec["resolver_round_trip"] = checks
    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nwrote %s  %.1fs" % (OUT, rec["elapsed_s"]))
    print("updated %s" % man_p)


if __name__ == "__main__":
    main()
