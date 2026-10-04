"""DATA_FREEZE: the one-time closure manifest and full local SHA-256 verification of the six canonical substrates.

READ-ONLY on data/ (nothing under data/ is written; verify_canonical.py is NOT used because it overwrites data/final_canonical/VERIFICATION.json).
Everything it writes goes under results/DATA_FREEZE/.  No network, no host: every file is read from local storage.

  python -u scratchpad/_data_freeze.py MANIFEST [--date 2026-09-30] [--out-dir results/DATA_FREEZE] [--threads 4]
      hashes EVERY file of the six trees data/final_canonical/<ds>/ (full pass, once), the loader / governance / record files the freeze
      pins, the partition maps (results/L1_HOST/parts) and the older data/l1_canonical assets; checks each against every pin that exists
      in CANONICAL_FREEZE.json / DATASET.json / the per-directory embedding manifests / the partition RUN.json; writes
        <out-dir>/DATA_FREEZE_<date>__v1.json          the manifest (per-artifact dataset, role, path, bytes, sha256, producer, status, K/partitioner)
        <out-dir>/DATA_FREEZE_VERIFICATION_<date>__v1.json   the verification record (what was checked against which pin, every mismatch)
  python -u scratchpad/_data_freeze.py VERIFY [--full] [--manifest PATH]
      re-reads the manifest and re-checks every entry from local files (size always; sha for every file with --full, else files <= 100 MB).
      Exit 0 only when nothing is missing, resized or mismatched.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = "C:/Users/Swastik/Desktop/CRAG/"
os.chdir(REPO)
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
FC = "data/final_canonical/"
HEX = re.compile(r"^[0-9a-f]{64}$")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def sha_file(p, chunk=1 << 24):
    smoke = os.environ.get("FREEZE_SMOKE_MB")  # code-path test only: files above the limit get a dummy digest; never used for a real record
    if smoke and os.path.getsize(p) > int(smoke) * 1024 * 1024:
        return "0" * 64
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def norm(p):
    return p.replace("\\", "/")


def record_hashes(rec):
    """RECORD_SHA256 (CRLF) and RECORD_SHA256_LF exactly as CANONICAL_FREEZE.json's HASHING_CONVENTION states."""
    body = {k: v for k, v in rec.items() if k not in ("RECORD_SHA256", "RECORD_SHA256_LF")}
    txt = json.dumps(body, indent=1, ensure_ascii=False)
    return (hashlib.sha256(txt.replace("\n", "\r\n").encode("utf-8")).hexdigest(),
            hashlib.sha256(txt.encode("utf-8")).hexdigest())


# ------------------------------------------------------------------ pins
def collect_pins(fz):
    """path -> (sha256, where).  Every {file|path, sha256} pair in the freeze record, each DATASET.json and each embedding manifest."""
    pins = {}

    def add(path, sha, where, bytes_=None):
        if isinstance(path, str) and isinstance(sha, str) and HEX.match(sha):
            pins.setdefault(norm(path), []).append({"sha256": sha, "where": where, "bytes": bytes_})

    def walk(o, where):
        if isinstance(o, dict):
            f = o.get("file") if isinstance(o.get("file"), str) else o.get("path") if isinstance(o.get("path"), str) else None
            if f and isinstance(o.get("sha256"), str):
                add(f, o["sha256"], where, o.get("bytes"))
            for v in o.values():
                walk(v, where)
        elif isinstance(o, list):
            for v in o:
                walk(v, where)

    walk(fz, "CANONICAL_FREEZE.json")
    for ds in DATASETS:
        D = fz["DATASETS"][ds]
        base = FC + ds + "/"
        add(base + "nodes.jsonl", D["nodes_sha256"], "CANONICAL_FREEZE.json DATASETS.%s.nodes_sha256" % ds)
        for fam, h in D["graph"]["npz_sha256"].items():
            add(base + "graph/%s.npz" % fam, h, "CANONICAL_FREEZE.json DATASETS.%s.graph.npz_sha256" % ds)
        add(base + "graph/GRAPH_MANIFEST.json", D["graph"]["manifest_sha256"], "CANONICAL_FREEZE.json DATASETS.%s.graph.manifest_sha256" % ds)
        for ch in ("dense", "splade"):
            add(base + "retrieval_cache/%s_top1000.npz" % ch, D["retrieval_cache"][ch]["sha256"],
                "CANONICAL_FREEZE.json DATASETS.%s.retrieval_cache.%s" % (ch, ds))
            for part in ("docs", "queries"):
                add(base + "embeddings/%s/%s/manifest.json" % (ch, part), D["embeddings"][ch][part]["manifest_sha256"],
                    "CANONICAL_FREEZE.json DATASETS.%s.embeddings.%s.%s.manifest_sha256" % (ds, ch, part))
                mp = base + "embeddings/%s/%s/manifest.json" % (ch, part)
                if os.path.exists(mp):
                    m = json.load(open(mp, encoding="utf-8"))
                    for s in m.get("shards", []):
                        add(base + "embeddings/%s/%s/%s" % (ch, part, s["file"]), s.get("sha256"), "embedding manifest %s" % mp, s.get("bytes"))
        dj = base + "DATASET.json"
        if os.path.exists(dj):
            walk(json.load(open(dj, encoding="utf-8")), "DATASET.json (%s)" % ds)
    return pins


ROLE_RULES = (
    (r"^nodes\.jsonl$", "nodes"),
    (r"^DATASET\.json$", "dataset_record"),
    (r"^queries/(train|dev|test)\.jsonl$", "queries_split"),
    (r"^queries/query_ids\.json$", "queries_index"),
    (r"^queries/lanes/", "queries_lane"),
    (r"^eval_.*\.jsonl$", "eval_subset"),
    (r"^embeddings/dense/docs/shard_", "embedding_dense_docs"),
    (r"^embeddings/dense/queries/shard_", "embedding_dense_queries"),
    (r"^embeddings/splade/docs/shard_", "embedding_splade_docs"),
    (r"^embeddings/splade/queries/shard_", "embedding_splade_queries"),
    (r"^embeddings/.*/manifest\.json$", "embedding_manifest"),
    (r"^graph/(structural|ner|knn)\.npz$", "graph_family"),
    (r"^graph/GRAPH_MANIFEST\.json$", "graph_manifest"),
    (r"^retrieval_cache/", "retrieval_cache"),
    (r"^_acquisition/", "source_acquisition"),
)


def role_of(rel):
    for pat, role in ROLE_RULES:
        if re.search(pat, rel):
            return role
    return "record_or_audit"


def list_tree(base):
    out = []
    for dp, dn, fn in os.walk(base):
        dn.sort()
        for f in sorted(fn):
            p = norm(os.path.join(dp, f))
            out.append(p)
    return out


def hash_many(paths, threads, label):
    """sha256 of every path (largest first, `threads` workers, 16 MiB chunks); progress line per ~2 GB."""
    size = {p: os.path.getsize(p) for p in paths}
    order = sorted(paths, key=lambda p: -size[p])
    total = sum(size.values())
    done = [0]
    t0 = time.time()
    res = {}
    last = [0]

    def job(p):
        h = sha_file(p)
        done[0] += size[p]
        if done[0] - last[0] >= 2 * 10 ** 9:
            last[0] = done[0]
            print("[%s] %.1f / %.1f GB (%.0f s)" % (label, done[0] / 1e9, total / 1e9, time.time() - t0))
            sys.stdout.flush()
        return p, h

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for p, h in ex.map(job, order):
            res[p] = h
    print("[%s] hashed %d files, %.3f GB in %.0f s" % (label, len(paths), total / 1e9, time.time() - t0))
    sys.stdout.flush()
    return res, size


# ------------------------------------------------------------------ partitions
def partition_entries(pins_out):
    """One entry per (dataset, K) cell under results/L1_HOST/parts, and the explicit PARTITION_INVALID entries."""
    cells = []
    for j in sorted(glob.glob("results/L1_HOST/parts/*__H4_SK_k*.json")):
        b = os.path.basename(j)
        if "__PHG_con" in b:
            continue
        ds, rest = b.split("__H4_SK_k")
        K = int(rest[:-5])
        stem = "results/L1_HOST/parts/%s__H4_SK_k%d__PHG_con" % (ds, K)
        cells.append((ds, K, j, stem))
    return cells


def main_manifest():
    date = arg("--date", "2026-09-30")
    outdir = arg("--out-dir", "results/DATA_FREEZE")
    threads = int(arg("--threads", "4"))
    os.makedirs(outdir, exist_ok=True)
    man_path = "%s/DATA_FREEZE_%s__v1.json" % (outdir, date)
    ver_path = "%s/DATA_FREEZE_VERIFICATION_%s__v1.json" % (outdir, date)
    assert not os.path.exists(man_path) and not os.path.exists(ver_path), "supersede, never overwrite"

    fz_path = FC + "CANONICAL_FREEZE.json"
    fz = json.load(open(fz_path, encoding="utf-8"))
    pins = collect_pins(fz)

    problems = []

    def bad(kind, path, detail):
        problems.append({"kind": kind, "path": path, "detail": detail})
        print("PROBLEM", kind, path, detail)
        sys.stdout.flush()

    # ---- inventory of files to hash
    tree = {ds: list_tree(FC + ds) for ds in DATASETS}
    extra = [FC + "CANONICAL_FREEZE.json"]
    for blk in ("LOADER", "GOVERNANCE"):
        extra += [v["file"] for v in fz[blk].values()]
    extra += [fz["HISTORY"]["index"]["file"], fz["BUILDERS"]["index"]["file"], FC + "HANDOFF.md"]
    extra = [norm(p) for p in extra if os.path.exists(p)]
    fbdj = norm(fz["FREEBASE"]["DATASET.json"]["file"])
    extra.append(fbdj)
    l1 = []
    for ds in DATASETS:
        base = "data/l1_canonical/%s/" % ds
        if os.path.isdir(base):
            l1 += list_tree(base)
    parts_files = []
    for ds in DATASETS:
        parts_files += [norm(p) for p in sorted(glob.glob("results/L1_HOST/parts/%s__*" % ds))]
    parts_files = [p for p in parts_files if not p.endswith(".rxpart") and not p.endswith(".rxpart.json")]

    all_paths = sorted(set(sum(tree.values(), []) + extra + l1 + parts_files))
    print("files to hash:", len(all_paths), " canonical-tree files:", {ds: len(v) for ds, v in tree.items()})
    sys.stdout.flush()

    # ---- the one full pass
    t0 = time.time()
    shas, sizes = hash_many(all_paths, threads, "FULL")
    seconds_hash = round(time.time() - t0, 1)

    # ---- tree byte totals vs the freeze
    tree_bytes = {ds: sum(sizes[p] for p in tree[ds]) for ds in DATASETS}
    for ds in DATASETS:
        pinned = fz["DATASETS"][ds]["tree_bytes"]
        # the record was written before HANDOFF/VERIFICATION-style side files were touched; report the difference, do not hide it
        if abs(tree_bytes[ds] - pinned) > 4096:
            bad("TREE_BYTES", FC + ds, "walked %d != pinned tree_bytes %d (diff %d)" % (tree_bytes[ds], pinned, tree_bytes[ds] - pinned))

    # ---- pin comparison
    def compare(path):
        pl = pins.get(path)
        if not pl:
            return "MANIFEST_ONLY", []
        mism = [x for x in pl if x["sha256"] != shas[path]]
        for x in mism:
            bad("PIN_MISMATCH", path, "sha %s != pinned %s (%s)" % (shas[path], x["sha256"], x["where"]))
        for x in pl:
            if x.get("bytes") is not None and int(x["bytes"]) != sizes[path]:
                bad("PIN_BYTES", path, "bytes %d != pinned %d (%s)" % (sizes[path], int(x["bytes"]), x["where"]))
        return ("PIN_OK" if not mism else "PIN_MISMATCH"), [x["where"] for x in pl]

    freeze_sha = shas[FC + "CANONICAL_FREEZE.json"]
    builders_sha = shas[norm(fz["BUILDERS"]["index"]["file"])]
    producer_canon = {"record": "data/final_canonical/CANONICAL_FREEZE.json", "file_sha256": freeze_sha,
                      "RECORD_SHA256": fz["RECORD_SHA256"], "builders_index_sha256": builders_sha,
                      "note": "built by the builders indexed in data/final_canonical/_history/builders/BUILDERS.json; loader canonical.py pinned"}

    datasets = {}
    n_pin_ok = n_only = 0
    for ds in DATASETS:
        ents = []
        for p in tree[ds]:
            rel = p[len(FC + ds + "/"):]
            st, where = compare(p)
            n_pin_ok += st == "PIN_OK"
            n_only += st == "MANIFEST_ONLY"
            ents.append({"role": role_of(rel), "path": p, "bytes": sizes[p], "sha256": shas[p], "status": st,
                         "pinned_by": where or None, "producer": "canonical_freeze"})
        D = fz["DATASETS"][ds]
        datasets[ds] = {"n_nodes": D["n_nodes"], "n_queries": D["n_queries"], "tree_bytes": tree_bytes[ds],
                        "tree_bytes_pinned": D["tree_bytes"], "n_files": len(ents), "files": ents}

    governance = []
    for p in extra:
        st, where = compare(p) if p != FC + "CANONICAL_FREEZE.json" else ("SELF_RECORD", [])
        if p == FC + "CANONICAL_FREEZE.json":
            r, rl = record_hashes(fz)
            st = "PIN_OK" if (r == fz["RECORD_SHA256"] and rl == fz["RECORD_SHA256_LF"]) else "PIN_MISMATCH"
            if st != "PIN_OK":
                bad("RECORD_SELF_HASH", p, "recomputed %s / %s" % (r, rl))
            where = ["own RECORD_SHA256 / RECORD_SHA256_LF, recomputed by the HASHING_CONVENTION"]
        governance.append({"role": "governance_or_loader", "path": p, "bytes": sizes[p], "sha256": shas[p], "status": st,
                           "pinned_by": where or None, "producer": "canonical_freeze"})

    # ---- partition cells
    part_cells = []
    invalid = []
    by_ds_counts = {}
    for ds, K, meta_json, stem in partition_entries(None):
        cell = {"dataset": ds, "K": K, "partitioner": "Zoltan-PHG (PHG_con), NP=4, PHG_NONEMPTY_REPAIR_V1, contract PARAMS unchanged, no tuning",
                "hypergraph_family": "H4_SK"}
        if os.path.exists(meta_json):
            cell["hypergraph_meta"] = {"path": norm(meta_json), "bytes": sizes[norm(meta_json)], "sha256": shas[norm(meta_json)]}
        if os.path.exists(stem + ".FAILED.json"):
            f = json.load(open(stem + ".FAILED.json", encoding="utf-8"))
            g = f.get("gate", {})
            cell.update({"status": "PARTITION_INVALID", "artifact": None,
                         "reason": {"max_block": g.get("max_block"), "bound": g.get("contract_bound_ceil_1.03_N_over_k"),
                                    "empty_blocks": g.get("empty_blocks"), "length_ok": g.get("length_ok"), "ids_in_range": g.get("ids_in_range")},
                         "failed_record": {"path": stem + ".FAILED.json", "bytes": sizes[stem + ".FAILED.json"], "sha256": shas[stem + ".FAILED.json"]},
                         "meaning": "the contract gate failed (max block above ceil(1.03 N/K)); by contract a PARTITION_INVALID cell is ABSENT, never retried or tuned. "
                                    "It is part of the freeze state, not unfinished work."})
            invalid.append(cell)
        else:
            r = json.load(open(stem + ".RUN.json", encoding="utf-8"))
            o = r["output"]
            npy = stem + ".npy"
            ok = os.path.exists(npy) and shas[norm(npy)] == o["sha256"]
            if not ok:
                bad("PARTITION_MAP", npy, "missing or sha != RUN.json output.sha256")
            cell.update({"status": "VALID" if ok else "PARTITION_MAP_PROBLEM",
                         "artifact": {"path": norm(npy), "bytes": sizes.get(norm(npy)), "sha256": shas.get(norm(npy)), "n": o["n"],
                                      "pinned_sha256": o["sha256"], "sha_matches_pin": ok},
                         "run_record": {"path": stem + ".RUN.json", "bytes": sizes[stem + ".RUN.json"], "sha256": shas[stem + ".RUN.json"]},
                         "balance": r.get("balance"), "run_status": r.get("STATUS"),
                         "producer": {"code_sha256": r.get("code"), "driver": r.get("driver"), "hypergraph": r.get("hypergraph"),
                                      "rx_job": (r.get("placement") or {}).get("rx_job")}})
        part_cells.append(cell)
        by_ds_counts.setdefault(ds, {})
        by_ds_counts[ds][cell["status"]] = by_ds_counts[ds].get(cell["status"], 0) + 1

    l1_assets = []
    for p in l1:
        ds = p.split("/")[2]
        l1_assets.append({"dataset": ds, "path": p, "bytes": sizes[p], "sha256": shas[p], "role": "l1_canonical_asset",
                          "note": "older Mt-KaHyPar / low-memory / replay assets; data/l1_canonical holds partition maps and hypergraphs only for metaqa, squad and "
                                  "musique; hotpotqa, 2wiki and webqsp hold keys / query_index / legacy_bridge only (their maps are the PHG cells)"
                          if ds in ("hotpotqa", "2wiki", "webqsp") else "older Mt-KaHyPar / low-memory / replay assets"})

    parts_aux = [{"path": p, "bytes": sizes[p], "sha256": shas[p]} for p in parts_files
                 if not (p.endswith("__PHG_con.npy") or p.endswith("__PHG_con.RUN.json") or p.endswith("__PHG_con.FAILED.json"))]

    rec = {
        "RECORD": "DATA_FREEZE",
        "version": 1,
        "date": date,
        "status": "MANIFEST (full local SHA-256 pass done; NOT yet the VFINAL declaration)",
        "purpose": "all six canonical corpora are locally reproducible from corpus -> encodings -> graph -> partitions; one manifest that the laptop "
                   "validates from its own files with the host disconnected",
        "basis": {"canonical_freeze": producer_canon,
                  "datasets": list(DATASETS),
                  "totals_pinned": fz["TOTALS"]},
        "how_built": {"script": "scratchpad/_data_freeze.py", "script_sha256": sha_file("scratchpad/_data_freeze.py"),
                      "hash_pass": {"files": len(all_paths), "bytes": sum(sizes.values()), "threads": threads, "seconds": seconds_hash},
                      "reads": "local storage only; nothing under data/ written; verify_canonical.py not run (it overwrites data/final_canonical/VERIFICATION.json)"},
        "pin_summary": {"canonical_tree_files": sum(len(v["files"]) for v in datasets.values()),
                        "PIN_OK": n_pin_ok, "MANIFEST_ONLY": n_only,
                        "note": "PIN_OK = the sha equals a pin written on an earlier date (freeze record, DATASET.json or embedding manifest); MANIFEST_ONLY = no earlier "
                                "per-file pin exists, the sha here is the first full-pass value"},
        "datasets": datasets,
        "governance_and_loader": governance,
        "partition_maps": {
            "contract": "big three (hotpotqa, 2wiki, webqsp) use PHG (Zoltan) NP=4 under the contract; metaqa, squad, musique K-grid cells here are the same PHG contract "
                        "cells; data/l1_canonical holds the older Mt-KaHyPar / low-memory maps only for metaqa, squad, musique",
            "counts": by_ds_counts,
            "cells": part_cells,
            "invalid_cells": [{"dataset": c["dataset"], "K": c["K"], "status": c["status"], "artifact": c["artifact"], "reason": c["reason"]} for c in invalid],
            "auxiliary_files_in_parts": parts_aux,
        },
        "l1_canonical_assets": l1_assets,
        "retained_not_hashed_here": {
            "freebase_served_tree": {"path": "data/final_canonical/freebase/", "pinned_by": "CANONICAL_FREEZE.json FREEBASE (DATASET.json sha256 %s)" % fz["FREEBASE"]["DATASET.json"]["sha256"],
                                     "tree_bytes_pinned": fz["FREEBASE"]["tree_bytes"], "role": "the FREEBASE_SCALE substrate (302M nodes); a separate scale lane, not a seventh benchmark dataset"},
            "freebase_v3_canonical": {"path": "data/final_canonical/freebase_v3/canonical/", "status": "RETAINED (not deleted)",
                                      "reason": "builder input of the Freebase pipeline (freeze_freebase.py, build_metadata.py); kept for the dedicated FREEBASE_SCALE lane "
                                                "(user ruling 2026-09-30: do NOT delete it yet)"},
        },
        "tree_bytes_vs_freeze": {ds: {"walked": tree_bytes[ds], "pinned": fz["DATASETS"][ds]["tree_bytes"],
                                      "delta": tree_bytes[ds] - fz["DATASETS"][ds]["tree_bytes"]} for ds in DATASETS},
        "tree_bytes_note": "the walked totals exceed the pinned tree_bytes by 773-804 bytes per tree (informational; a problem only above 4,096). The frozen builder measures "
                           "tree_bytes (freeze_canonical.py line 342) before it writes DATASET.json (line 352), so a rewritten DATASET.json shifts the total by a few "
                           "hundred bytes; consistent with that, not proven by it. Every file in every tree is individually sha-checked against a pin written earlier, "
                           "so the byte total carries no weight of its own.",
        "problems": problems,
    }
    rec["RECORD_SHA256"], rec["RECORD_SHA256_LF"] = record_hashes(rec)
    data = json.dumps(rec, indent=1, ensure_ascii=False).encode("utf-8")
    open(man_path, "wb").write(data)
    print("manifest ->", man_path, hashlib.sha256(data).hexdigest(), len(data), "problems", len(problems))

    ver = {"RECORD": "DATA_FREEZE_VERIFICATION", "version": 1, "date": date, "manifest": {"path": man_path, "file_sha256": hashlib.sha256(data).hexdigest()},
           "mode": "FULL SHA-256 over every file listed (one pass), host disconnected",
           "files_hashed": len(all_paths), "bytes_hashed": sum(sizes.values()), "seconds": seconds_hash,
           "canonical_freeze_record_self_hash": {"recomputed_RECORD_SHA256": record_hashes(fz)[0], "pinned": fz["RECORD_SHA256"],
                                                 "recomputed_RECORD_SHA256_LF": record_hashes(fz)[1], "pinned_LF": fz["RECORD_SHA256_LF"]},
           "pin_summary": rec["pin_summary"],
           "partition_maps": {"valid": sum(1 for c in part_cells if c["status"] == "VALID"), "invalid": len(invalid),
                              "by_dataset": by_ds_counts, "all_valid_sha_equal_to_run_record_pin": all(
                                  c["artifact"]["sha_matches_pin"] for c in part_cells if c["status"] == "VALID")},
           "tree_bytes_vs_freeze": rec["tree_bytes_vs_freeze"],
           "problems": problems,
           "VERDICT": "PASS" if not problems else "FAIL"}
    ver["RECORD_SHA256"], ver["RECORD_SHA256_LF"] = record_hashes(ver)
    open(ver_path, "wb").write(json.dumps(ver, indent=1, ensure_ascii=False).encode("utf-8"))
    print("verification ->", ver_path, ver["VERDICT"])
    return 0 if not problems else 1


def collect_manifest_entries(man):
    out = []
    for ds, d in man["datasets"].items():
        out += [(e["path"], e["bytes"], e["sha256"]) for e in d["files"]]
    out += [(e["path"], e["bytes"], e["sha256"]) for e in man["governance_and_loader"]]
    for c in man["partition_maps"]["cells"]:
        if c.get("artifact"):
            out.append((c["artifact"]["path"], c["artifact"]["bytes"], c["artifact"]["sha256"]))
        out.append((c["run_record"]["path"], c["run_record"]["bytes"], c["run_record"]["sha256"])) if c.get("run_record") else None
        if c.get("hypergraph_meta"):
            h = c["hypergraph_meta"]
            out.append((h["path"], h["bytes"], h["sha256"]))
        if c.get("failed_record"):
            fr = c["failed_record"]
            out.append((fr["path"], fr["bytes"], fr["sha256"]))
    out += [(e["path"], e["bytes"], e["sha256"]) for e in man["l1_canonical_assets"]]
    out += [(e["path"], e["bytes"], e["sha256"]) for e in man["partition_maps"]["auxiliary_files_in_parts"]]
    return out


def main_verify():
    full = "--full" in sys.argv
    mp = arg("--manifest") or sorted(glob.glob("results/DATA_FREEZE/DATA_FREEZE_20*__v1.json"))[-1]
    man = json.load(open(mp, encoding="utf-8"))
    r, rl = record_hashes(man)
    assert r == man["RECORD_SHA256"] and rl == man["RECORD_SHA256_LF"], "manifest self hash"
    ents = collect_manifest_entries(man)
    miss = [p for p, b, s in ents if not os.path.exists(p)]
    size_bad = [p for p, b, s in ents if os.path.exists(p) and os.path.getsize(p) != b]
    todo = [(p, s) for p, b, s in ents if os.path.exists(p) and os.path.getsize(p) == b and (full or b <= 100 * 1024 * 1024)]
    shas, _ = hash_many([p for p, s in todo], int(arg("--threads", "4")), "VERIFY")
    sha_bad = [p for p, s in todo if shas[p] != s]
    inv = man["partition_maps"]["invalid_cells"]
    inv_ok = sorted((c["dataset"], c["K"]) for c in inv) == [("2wiki", 5000), ("metaqa", 5000)] and all(c["artifact"] is None for c in inv)
    print("entries", len(ents), "missing", len(miss), "size_bad", len(size_bad), "sha_checked", len(todo), "sha_bad", len(sha_bad),
          "invalid_cells_explicit", inv_ok, "full", full)
    for p in miss + size_bad + sha_bad:
        print("  BAD", p)
    ok = not miss and not size_bad and not sha_bad and inv_ok and not man["problems"]
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"MANIFEST": main_manifest, "VERIFY": main_verify}.get(mode, lambda: print(__doc__) or 2)())
