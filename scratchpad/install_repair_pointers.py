"""
INSTALL THE DENSE REPAIR INTO THE POINTER INDEX
===============================================
Redirects every canonical pointer that currently resolves to a corrupted Phase-C dense row so it
resolves to the repaired vector instead.

This is deliberately NON-DESTRUCTIVE. The corrupted Phase-C shards are left exactly as they are;
canonical_v1 simply stops pointing at the damaged rows. That keeps the repair reversible and keeps
this pass from rewriting artifacts other work may depend on. The consequence has to be stated
rather than buried: anything that reads data/canonical/<tree>/encodings/dense/ shard files
DIRECTLY, instead of going through the pointer index, still reads the corrupt rows.

The manifest gains an explicit per-channel STORES list so the resolver stays data-driven:
src is an index into that list, and each entry says where the vectors live and how to address
them. Splade channels get no repair store because SPLADE scanned clean everywhere.
"""
import glob, io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"
REPAIR = os.path.join(ROOT, "_dense_repair_patch")
DOC_TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
            "hotpotqa": "hotpotqa", "squad": "squad"}


def repair_rows(tree, kind):
    """ascending Phase-C rows repaired for this channel, or None if there is no repair."""
    d = os.path.join(REPAIR, "%s__%s" % (tree, kind))
    p = os.path.join(d, "dense_rows.json")
    if not os.path.exists(p):
        return None, None
    return np.asarray(json.load(io.open(p, encoding="utf-8")), dtype=np.int64), d


def install(man):
    changed = {}
    for ds, tree in DOC_TREE.items():
        # ---------------- documents ----------------
        rr, rd = repair_rows(tree, "docs")
        spec = man["datasets"][ds]
        for model in ("dense", "splade"):
            stores = [{"store": "PHASE_C", "tree": tree,
                       "path": "data/canonical/%s/encodings/%s/docs" % (tree, model),
                       "shard_size": spec[model]["phase_c_shard_size"],
                       "n_items": spec[model]["phase_c_n_items"]}]
            ext = "npy" if model == "dense" else "npz"
            pp = "%s/_rev2_encoder_patch/%s/%s.%s" % (ROOT, ds, model, ext)
            if os.path.exists(pp):
                stores.append({"store": "REV2_PATCH", "path": pp, "flat": True,
                               "n_items": spec[model]["patch_n_rows"]})
            p = "%s/%s/pointer_index/%s.npz" % (ROOT, ds, model)
            z = np.load(p)
            src, row = z["src"].copy(), z["row"].copy()
            n_fix = 0
            if model == "dense" and rr is not None:
                code = len(stores)
                stores.append({"store": "DENSE_REPAIR", "flat": True,
                               "path": os.path.join(rd, "dense.npy").replace("\\", "/"),
                               "n_items": int(rr.size),
                               "repairs_rows_of": "PHASE_C"})
                m = src == 0
                pos = np.searchsorted(rr, row)
                hit = m & (pos < rr.size) & (rr[np.minimum(pos, rr.size - 1)] == row)
                n_fix = int(hit.sum())
                src[hit] = code
                row[hit] = pos[hit].astype(np.int32)
                np.savez(p, src=src, row=row)
            spec[model]["stores"] = stores
            spec[model]["repaired_pointers"] = n_fix
            changed["%s/docs/%s" % (ds, model)] = n_fix

        # ---------------- queries ----------------
        q = man["queries"][ds]
        for model in ("dense", "splade"):
            trees = q["trees"]
            sizes = q[model]["shard_size_per_tree"]
            counts = q[model]["n_items_per_tree"]
            stores = [{"store": "PHASE_C", "tree": t,
                       "path": "data/canonical/%s/encodings/%s/queries" % (t, model),
                       "shard_size": sizes[i], "n_items": counts[i]}
                      for i, t in enumerate(trees)]
            p = "%s/%s/queries/pointer_index/%s.npz" % (ROOT, ds, model)
            z = np.load(p)
            src, row = z["src"].copy(), z["row"].copy()
            n_fix = 0
            if model == "dense":
                for i, t in enumerate(trees):
                    qr, qd = repair_rows(t, "queries")
                    if qr is None:
                        continue
                    code = len(stores)
                    stores.append({"store": "DENSE_REPAIR", "flat": True,
                                   "path": os.path.join(qd, "dense.npy").replace("\\", "/"),
                                   "n_items": int(qr.size), "repairs_tree": t,
                                   "repairs_rows_of": "PHASE_C"})
                    m = src == i
                    pos = np.searchsorted(qr, row)
                    hit = m & (pos < qr.size) & (qr[np.minimum(pos, qr.size - 1)] == row)
                    n_fix += int(hit.sum())
                    src[hit] = code
                    row[hit] = pos[hit].astype(np.int32)
                np.savez(p, src=src, row=row)
            q[model]["stores"] = stores
            q[model]["repaired_pointers"] = n_fix
            changed["%s/queries/%s" % (ds, model)] = n_fix
    return changed


if __name__ == "__main__":
    t0 = time.time()
    p = os.path.join(ROOT, "POINTER_INDEX.json")
    man = json.load(io.open(p, encoding="utf-8"))
    if man.get("dense_repair_installed"):
        sys.exit("repair already installed; rerunning would double-redirect")
    ch = install(man)
    man["dense_repair_installed"] = True
    man["dense_repair"] = {
        "why": ("Phase-C dense shards carry 142,633 rows lost to storage-layer corruption: "
                "contiguous all-zero runs, 4096-byte aligned, ending on filesystem extent "
                "boundaries, tearing vectors mid-row at both edges. SPLADE is unaffected."),
        "evidence": "scratchpad/encoding_integrity.json",
        "non_destructive": ("the corrupt Phase-C shards are untouched; canonical_v1 simply stops "
                            "pointing at them. ANY CONSUMER READING data/canonical/<tree>/"
                            "encodings/dense/ SHARDS DIRECTLY STILL READS THE CORRUPT ROWS."),
        "repaired_pointers_by_channel": ch,
        "total_repaired_pointers": int(sum(ch.values()))}
    man["resolver"]["STORES"] = ("src indexes the per-channel stores list. A store with flat=true "
                                 "is a single array addressed by row; otherwise "
                                 "shard = row // shard_size, offset = row % shard_size.")
    json.dump(man, io.open(p, "w", encoding="utf-8"), indent=2)
    for k, v in sorted(ch.items()):
        if v:
            print("  %-28s redirected %s" % (k, format(v, ",")))
    print("\ntotal redirected pointers %s  %.1fs"
          % (format(sum(ch.values()), ","), time.time() - t0))
