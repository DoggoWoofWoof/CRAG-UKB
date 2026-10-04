"""
VERIFY THE CANONICAL FREEZE
===========================
Treats every artifact as invalid until counts and checksums establish otherwise.

    python src/dataset_canonical/verify_canonical.py            # structure, counts, small-file digests, loader smoke
    python src/dataset_canonical/verify_canonical.py --full     # + every pinned byte hashed, per-row digest chains
                                                                #   recomputed, every dense row scanned for damage
    python src/dataset_canonical/verify_canonical.py --dataset 2wiki [--full]
    python src/dataset_canonical/verify_canonical.py --dataset freebase [--full]   # the seventh tree alone
    python src/dataset_canonical/verify_canonical.py --full --skip-freebase        # the six only

Writes data/final_canonical/VERIFICATION.json (a report, overwritten each run) and exits non-zero
on any failure.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical"))
from freeze_canonical import (FC, HIST, OUT, DATASETS, SPLITS, KEEP_ROOT, rj, sha_file, record_hash, count_lines)  # noqa: E402

SMALL = 100 * 1024 * 1024
DIM = 1536
VOCAB = 30522
NORM_TOL = 0.01


class Report(object):
    def __init__(self):
        self.checks = []
        self.fail = 0

    def add(self, name, ok, detail=None):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.fail += 1
            print("  FAIL %s  %s" % (name, detail if detail is not None else ""))
        sys.stdout.flush()


def pinned(rep, entry, full, label):
    p = entry["file"]
    if not os.path.exists(p):
        rep.add(label, False, "missing %s" % p)
        return
    b = os.path.getsize(p)
    if b != entry["bytes"]:
        rep.add(label, False, "%s bytes %d != pinned %d" % (p, b, entry["bytes"]))
        return
    if "sha256" in entry and (full or b <= SMALL):
        h = sha_file(p)
        rep.add(label, h == entry["sha256"], None if h == entry["sha256"] else "%s sha drift" % p)
    else:
        rep.add(label + " (bytes only)", True)


def row_digest_chain(model, d, n_shards):
    out = []
    for k in range(n_shards):
        if model == "dense":
            a = np.ascontiguousarray(np.load(os.path.join(d, "shard_%05d.npy" % k), mmap_mode="r"))
            for i in range(a.shape[0]):
                out.append(hashlib.sha1(a[i].tobytes()).digest())
        else:
            z = np.load(os.path.join(d, "shard_%05d.npz" % k))
            indptr, ind, dat = z["indptr"].astype(np.int64), z["indices"], z["data"]
            z.close()
            for i in range(indptr.size - 1):
                a, b = int(indptr[i]), int(indptr[i + 1])
                h = hashlib.sha1(ind[a:b].tobytes())
                h.update(dat[a:b].tobytes())
                out.append(h.digest())
    return hashlib.sha256(b"".join(out)).hexdigest(), len(out)


def damaged_rows(d, n_shards):
    bad = 0
    for k in range(n_shards):
        mm = np.load(os.path.join(d, "shard_%05d.npy" % k), mmap_mode="r")
        for i in range(0, mm.shape[0], 20000):
            v = np.asarray(mm[i:i + 20000], dtype=np.float32)
            b = ~np.isfinite(v).all(axis=1)
            n = np.sqrt((v * v).sum(axis=1))
            b |= (n == 0.0) | (np.abs(n - 1.0) > NORM_TOL)
            bad += int(b.sum())
        del mm
    return bad


def verify_dataset(ds, freeze, full, rep):
    t0 = time.time()
    d = FC + "/" + ds
    pin = freeze["DATASETS"][ds]
    dp = d + "/DATASET.json"
    if not os.path.exists(dp):
        rep.add("%s DATASET.json present" % ds, False)
        return
    R = rj(dp)
    rep.add("%s DATASET.json self-hash" % ds, record_hash(R) == R["RECORD_SHA256"] and record_hash(R, crlf=False) == R["RECORD_SHA256_LF"])
    rep.add("%s DATASET.json pinned by the freeze" % ds, sha_file(dp) == pin["DATASET.json"]["sha256"])
    # layout: nothing superseded left behind
    for gone in ("pointer_index", "reuse_map", "queries/pointer_index", "graph2", "encoder_row_of_node.npz", "_enc", "status.json"):
        rep.add("%s has no %s" % (ds, gone), not os.path.exists(os.path.join(d, gone)))
    # nodes
    n = count_lines(d + "/nodes.jsonl")
    rep.add("%s nodes.jsonl lines == n_nodes" % ds, n == R["n_nodes"] == pin["n_nodes"], "%d vs %d" % (n, R["n_nodes"]))
    pinned(rep, R["nodes"], full, "%s nodes.jsonl digest" % ds)
    # queries
    qids = rj(d + "/queries/query_ids.json")
    cat = []
    for sp in SPLITS[ds]:
        p = d + "/queries/%s.jsonl" % sp
        k = 0
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                cat.append(json.loads(ln)["query_id"])
                k += 1
        rep.add("%s queries/%s.jsonl count" % (ds, sp), k == R["queries"]["by_split"][sp]["n"] == pin["queries_by_split"][sp])
        pinned(rep, R["queries"]["by_split"][sp], full, "%s queries/%s.jsonl digest" % (ds, sp))
    rep.add("%s query_ids == split concatenation" % ds, cat == qids and len(qids) == R["n_queries"] == pin["n_queries"])
    pinned(rep, R["queries"]["query_ids"], full, "%s query_ids.json digest" % ds)
    for name, e in (R["queries"].get("lanes") or {}).items():
        pinned(rep, e, full, "%s lanes/%s" % (ds, name))
    for name, e in (R["queries"].get("eval_subsets") or {}).items():
        pinned(rep, e, full, "%s %s" % (ds, name))
    # embeddings
    for model in ("dense", "splade"):
        for kind in ("docs", "queries"):
            e = R["embeddings"][model][kind]
            sd = d + "/embeddings/%s/%s" % (model, kind)
            label = "%s %s/%s" % (ds, model, kind)
            if not os.path.exists(sd + "/manifest.json"):
                rep.add(label + " manifest", False, "missing")
                continue
            m = rj(sd + "/manifest.json")
            rep.add(label + " manifest pinned", sha_file(sd + "/manifest.json") == e["manifest"]["sha256"] == pin["embeddings"][model][kind]["manifest_sha256"])
            expect = n if kind == "docs" else len(qids)
            rep.add(label + " n_rows", m["n_rows"] == e["n_rows"] == expect and m["n_shards"] == e["n_shards"] and m["shard_size"] == 40000)
            files = sorted(os.listdir(sd))
            rep.add(label + " directory == manifest", files == sorted([s["file"] for s in m["shards"]] + ["manifest.json"]))
            rows = 0
            for s in m["shards"]:
                p = os.path.join(sd, s["file"])
                ok = os.path.exists(p) and os.path.getsize(p) == s["bytes"]
                if ok and full:
                    ok = sha_file(p) == s["sha256"]
                if not ok:
                    rep.add(label + " shard " + s["file"], False, "bytes/sha drift or missing")
                rows += s["rows"]
            rep.add(label + " shards present%s" % ("" if full else " (bytes only)"), rows == m["n_rows"])
            chain_pin = m["materialised"]["proof"]["new_store_row_digest_chain_sha256"]
            rep.add(label + " chain pinned by DATASET.json/freeze", chain_pin == e["row_digest_chain_sha256"] == pin["embeddings"][model][kind]["row_digest_chain_sha256"])
            if full:
                got, cnt = row_digest_chain(model, sd, m["n_shards"])
                rep.add(label + " row-digest chain recomputed", got == chain_pin and cnt == m["n_rows"], "%s vs %s" % (got[:16], chain_pin[:16]))
                if model == "dense":
                    bad = damaged_rows(sd, m["n_shards"])
                    rep.add(label + " dense integrity (0 damaged rows)", bad == 0, "%d damaged" % bad)
    # graph
    gm = rj(d + "/graph/GRAPH_MANIFEST.json")
    rep.add("%s graph manifest is CANONICAL_GRAPH and pinned" % ds, gm.get("RECORD") == "CANONICAL_GRAPH" and gm["n_nodes"] == n
            and sha_file(d + "/graph/GRAPH_MANIFEST.json") == R["graph"]["manifest"]["sha256"] == pin["graph"]["manifest_sha256"])
    for fam, e in R["graph"]["families"].items():
        if not e["present"]:
            rep.add("%s graph/%s absent as recorded" % (ds, fam), not os.path.exists(d + "/graph/%s.npz" % fam))
            continue
        p = d + "/graph/%s.npz" % fam
        pinned(rep, {"file": p, "bytes": e["bytes"], "sha256": e["sha256"]}, full, "%s graph/%s digest" % (ds, fam))
        if os.path.exists(p):
            z = np.load(p)
            src, dst = z["src"], z["dst"]
            ne = int(src.size)
            mx = int(max(src.max(), dst.max())) if ne else -1
            z.close()
            rep.add("%s graph/%s edges + endpoints" % (ds, fam), ne == e["n_edges"] == pin["graph"]["n_edges"][fam] and mx < n, "%d edges, max %d" % (ne, mx))
    # caches
    for model, e in R["retrieval_cache"].items():
        pinned(rep, e, full, "%s cache %s digest" % (ds, model))
        if os.path.exists(e["file"]):
            z = np.load(e["file"])
            ids = z["ids"]
            ok = list(ids.shape) == e["shape"] == pin["retrieval_cache"][model]["shape"] and ids.shape[0] == len(qids) and int(ids.max()) < n and int(ids.min()) >= 0
            z.close()
            rep.add("%s cache %s shape/index space" % (ds, model), ok)
        if "meta" in e:
            pinned(rep, e["meta"], full, "%s cache %s meta" % (ds, model))
    # records and kept dirs
    for name, e in R["records"].items():
        pinned(rep, e, full, "%s record %s" % (ds, name))
    for name, e in R["kept_dirs"].items():
        for x in e["inventory"]:
            pinned(rep, x, full, "%s %s/%s" % (ds, name, os.path.basename(x["file"])))
    # nothing unpinned at the dataset root
    extra = [f for f in os.listdir(d) if f not in R["records"] and f not in R["kept_dirs"] and f not in
             ("nodes.jsonl", "DATASET.json", "embeddings", "graph", "queries", "retrieval_cache") and not (f.startswith("eval_") and f in (R["queries"].get("eval_subsets") or {}))]
    rep.add("%s no unpinned files at the dataset root" % ds, not extra, extra)
    # loader smoke
    try:
        sys.path.insert(0, FC)
        from canonical import Dataset
        D = Dataset(ds)
        E = D.embeddings("dense", "docs")
        S = D.embeddings("splade", "queries")
        v = E.read([0, n - 1, n // 2])
        s = S.read([0, len(qids) - 1])
        G = D.graph(D.families()[0])
        nb = G.neighbors(int(np.argmax(G.degree() > 0)))
        ids, _ = D.cache("dense")
        rep.add("%s loader smoke" % ds, v.shape == (3, DIM) and v.dtype == np.float16 and s.shape == (2, VOCAB) and nb.size > 0 and ids.shape[0] == len(qids))
    except Exception as ex:  # noqa: BLE001
        rep.add("%s loader smoke" % ds, False, repr(ex))
    print("  %s verified in %.0fs" % (ds, time.time() - t0))


def verify_root(freeze, full, rep):
    rep.add("CANONICAL_FREEZE.json self-hash (CRLF)", record_hash(freeze) == freeze["RECORD_SHA256"])
    rep.add("CANONICAL_FREEZE.json self-hash (LF)", record_hash(freeze, crlf=False) == freeze["RECORD_SHA256_LF"])
    rep.add("no data/canonical", not os.path.exists("data/canonical"))
    rep.add("no data/_retired_v3", not os.path.exists("data/_retired_v3"))
    extra = [f for f in os.listdir(FC) if f not in KEEP_ROOT and f not in DATASETS and f not in ("freebase", "freebase_v3", "_history", "__pycache__")]
    rep.add("root holds only the freeze, loader, handoff, governance, six trees, freebase, freebase_v3, _history", not extra, extra)
    if freeze.get("FREEBASE"):
        rep.add("freebase/DATASET.json present and pinned", os.path.exists(FC + "/freebase/DATASET.json")
                and sha_file(FC + "/freebase/DATASET.json") == freeze["FREEBASE"]["DATASET.json"]["sha256"])
    for name, e in freeze["GOVERNANCE"].items():
        pinned(rep, e, full, "governance %s" % name)
    for name, e in freeze["LOADER"].items():
        pinned(rep, e, full, "loader %s" % name)
    hi = HIST + "/INDEX.json"
    if os.path.exists(hi):
        H = rj(hi)
        rep.add("_history/INDEX.json self-hash + pinned", record_hash(H) == H["RECORD_SHA256"] and sha_file(hi) == freeze["HISTORY"]["index"]["sha256"])
        if full:
            bad = [x["file"] for x in H["inventory"] if not (os.path.exists(x["file"]) and os.path.getsize(x["file"]) == x["bytes"] and sha_file(x["file"]) == x["sha256"])]
            rep.add("_history files all present and unchanged (%d)" % len(H["inventory"]), not bad, bad[:10])
        else:
            bad = [x["file"] for x in H["inventory"] if not (os.path.exists(x["file"]) and os.path.getsize(x["file"]) == x["bytes"])]
            rep.add("_history files all present (bytes)", not bad, bad[:10])
    else:
        rep.add("_history/INDEX.json present", False)
    # the claims-check log is not digest-pinned (it is rewritten by the gate itself); it must be present, passing, and
    # made on the HANDOFF.md that is served now
    cl = HIST + "/logs/HANDOFF_CLAIMS_CHECK.json"
    hd = FC + "/HANDOFF.md"
    if os.path.exists(cl) and os.path.exists(hd):
        C = rj(cl)
        h = hashlib.sha256(io.open(hd, encoding="utf-8").read().replace("\r\n", "\n").encode("utf-8")).hexdigest()
        ok = C.get("VERDICT") == "ALL_PASS" and C.get("handoff_sha256_lf") == h
        rep.add("HANDOFF claims check: present, ALL_PASS, made on the served HANDOFF.md", ok,
                None if ok else {"verdict": C.get("VERDICT"), "checked_handoff": (C.get("handoff_sha256_lf") or "")[:16], "served_handoff": h[:16]})
    else:
        rep.add("HANDOFF claims check: present, ALL_PASS, made on the served HANDOFF.md", False, "missing %s" % (cl if not os.path.exists(cl) else hd))
    if freeze.get("BUILDERS"):
        bi = HIST + "/builders/BUILDERS.json"
        B = rj(bi) if os.path.exists(bi) else None
        rep.add("_history/builders/BUILDERS.json self-hash + pinned", B is not None and record_hash(B) == B["RECORD_SHA256"]
                and sha_file(bi) == freeze["BUILDERS"]["index"]["sha256"] and B["files"] == freeze["BUILDERS"]["files"])
        if B is not None:
            if full:
                bad = [x["archived"] for x in B["inventory"] if not (os.path.exists(x["archived"]) and os.path.getsize(x["archived"]) == x["bytes"] and sha_file(x["archived"]) == x["sha256"])]
            else:
                bad = [x["archived"] for x in B["inventory"] if not (os.path.exists(x["archived"]) and os.path.getsize(x["archived"]) == x["bytes"])]
            rep.add("archived builder sources all present%s (%d)" % (" and unchanged" if full else "", len(B["inventory"])), not bad, bad[:10])
            lost = sorted(ds for ds, e in freeze["BUILDERS"]["build_info_revisions"].items() if e and e.get("status") == "LOST")
            rep.add("every build_info builder revision is archived or declared LOST in the freeze",
                    all((e or {}).get("status") in ("ARCHIVED", "LOST") or (e or {}).get("build_info") is None
                        for e in freeze["BUILDERS"]["build_info_revisions"].values()), lost)
    SL = (freeze.get("FREEBASE") or {}).get("source_layer")
    if SL:
        # the Freebase source layer after the 2026-09-13 deletion: terminal closure record, deletion log, protected sources
        v2p = SL["closure_status_v2"]["file"]
        V2 = rj(v2p) if os.path.exists(v2p) else None
        rep.add("freebase_v3 terminal closure record (V2) present, pinned, self-hash ok",
                V2 is not None and sha_file(v2p) == SL["closure_status_v2"]["sha256"] and record_hash(V2) == V2["RECORD_SHA256"]
                and record_hash(V2, crlf=False) == V2["RECORD_SHA256_LF"])
        if V2 is not None:
            b = V2["BUILDS_ON"]
            rep.add("closure V2 builds on the V1 record that is on disk (bytes unchanged)",
                    os.path.exists(b["file"]) and sha_file(b["file"]) == b["sha256_of_bytes"])
            rep.add("closure V2 cites the served freebase/DATASET.json that is pinned by this freeze",
                    V2["SERVED_AS"]["DATASET.json"]["sha256_of_bytes"] == freeze["FREEBASE"]["DATASET.json"]["sha256"])
        dlp = SL["sublayer_deletion_log"]["file"]
        DL = rj(dlp) if os.path.exists(dlp) else None
        rep.add("freebase_v3 sub-layer deletion log present, pinned, self-hash ok, status APPLIED with no mismatch",
                DL is not None and sha_file(dlp) == SL["sublayer_deletion_log"]["sha256"] and record_hash(DL) == DL["RECORD_SHA256"]
                and DL["status"] == "APPLIED" and not DL["mismatched_not_deleted"] and not DL["missing_at_apply"]
                and DL["deleted"] == DL["files_total"])
        if DL is not None:
            gone = [f["file"] for f in DL["files"] if os.path.exists(f["file"])]
            rep.add("every deleted sub-layer file is gone (%d)" % DL["files_total"], not gone, gone[:10])
            left = []
            for L in DL["layers"]:
                d = FC + "/freebase_v3/" + L
                if os.path.isdir(d):
                    left += [os.path.join(r, x) for r, _, fs in os.walk(d) for x in fs if not x.lower().endswith(".json")]
            rep.add("only records (*.json) remain under the deleted sub-layers", not left, left[:10])
            stays = [FC + "/freebase_v3/_acquisition/raw/freebase-rdf-latest.gz", FC + "/freebase_v3/_acquisition/idir/idirlab-freebases.zip",
                     FC + "/freebase_v3/canonical/CANONICAL_MANIFEST.json", FC + "/freebase_v3/overlay_v1/OVERLAY_MANIFEST.json"]
            miss = [p for p in stays if not os.path.exists(p)]
            rep.add("the protected sources stay (raw mirror, IDIR zip, canonical manifest, overlay manifest)", not miss, miss)
    if freeze.get("QUERY_LANES"):
        for ds, ln in freeze["QUERY_LANES"]["census"].items():
            if not ln:
                rep.add("%s has no lane files (as recorded)" % ds, not os.path.isdir(FC + "/%s/queries/lanes" % ds))
                continue
            for lane, e in ln.items():
                f = FC + "/" + e["file"]
                n = 0
                by = {}
                if os.path.exists(f):
                    with io.open(f, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                n += 1
                                sp = json.loads(line).get("split")
                                by[sp] = by.get(sp, 0) + 1
                rep.add("%s lane %s: %d rows, split composition as recorded" % (ds, lane, e["n"]), os.path.exists(f) and n == e["n"] and by == e["by_split"],
                        {"rows": n, "by_split": by})
        m = freeze["QUERY_LANES"]["census"].get("metaqa") or {}
        rep.add("metaqa QUALITY_LOCKED is the test split and EVAL_SPLITS names dev",
                (m.get("QUALITY_LOCKED") or {}).get("by_split") == {"test": 39093} and freeze["EVAL_SPLITS"]["metaqa"]["split"] == "dev")
    tot = freeze["TOTALS"]
    rep.add("totals recomputed from DATASETS", tot["nodes"] == sum(v["n_nodes"] for v in freeze["DATASETS"].values())
            and tot["queries"] == sum(v["n_queries"] for v in freeze["DATASETS"].values())
            and tot["edges_served"] == sum(sum(v["graph"]["n_edges"].values()) for v in freeze["DATASETS"].values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--skip-freebase", action="store_true", help="the six only (the freebase checks read ~60 GB in --full mode)")
    a = ap.parse_args()
    t0 = time.time()
    rep = Report()
    freeze = rj(OUT)
    verify_root(freeze, a.full, rep)
    for ds in ([a.dataset] if a.dataset else DATASETS):
        if ds == "freebase":
            continue
        verify_dataset(ds, freeze, a.full, rep)
    if freeze.get("FREEBASE") and (a.dataset in (None, "freebase")) and not a.skip_freebase:
        sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical", "freebase"))
        from verify_freebase import verify_freebase
        verify_freebase(freeze, a.full, rep)
    res = {"RECORD": "VERIFICATION", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": "full" if a.full else "quick",
           "datasets": ([a.dataset] if a.dataset else DATASETS + (["freebase"] if freeze.get("FREEBASE") and not a.skip_freebase else [])),
           "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
           "checks": len(rep.checks), "failed": rep.fail, "PASS": rep.fail == 0, "seconds": round(time.time() - t0, 1),
           "results": rep.checks}
    with io.open(FC + "/VERIFICATION.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("%s  %d checks, %d failed, %.0fs  -> %s" % ("PASS" if rep.fail == 0 else "FAIL", len(rep.checks), rep.fail, time.time() - t0, FC + "/VERIFICATION.json"))
    sys.exit(0 if rep.fail == 0 else 1)


if __name__ == "__main__":
    main()
