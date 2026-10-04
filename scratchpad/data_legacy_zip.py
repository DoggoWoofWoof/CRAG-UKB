"""LEGACY DATA ZIP (user 2026-09-30: "zip the files that are referenced by older scripts at least and save up space", then "continue and zip").

Lossless archive of what only LEGACY (pre-canonical) code names: the old BGE / ColBERT / BM25 / FAISS stack of data/ukb_storage, the old
SOTA baseline stack (ukb_storage/_sota), the pre-acquisition downloads (data/raw) and the old encoder checkpoints (repo checkpoints/).
A reference scan (scratchpad/data_refscan.py, 2026-09-30) found no L1 / L3 / encoder / freeze module that reads them.  NOT in scope (PROTECT
is asserted on every file): the frozen substrate (data/final_canonical, l1_canonical, l1_lowmem), data/original, data/processed,
data/_family_v1, ukb_storage/*/gte_qwen + graph.pt + ner_edges + splade_doc_embs + partition_map, ukb_storage/*/results, _head_cache,
_review, results/, scratchpad/, src/.  (L2 artifacts are deleted, not zipped: scratchpad/l2_cleanup.py.)
Protocol per unit (a zip of <= ~1 GB of source): stream the files into <zip>.partial while hashing each source (sha256) -> close ->
rename -> read every member back and compare its sha256 and size -> re-stat the sources (size + mtime unchanged since the plan) ->
ONLY THEN delete the sources (a file another process holds open is kept and recorded) -> remove the directories that became empty.
The ledger (results/DATA_CLEANUP/LEGACY_ZIP_LEDGER__2026-09-30.jsonl, append-only) carries per file: path, bytes, sha256, mtime, unit, zip
member; per unit: zip path, bytes, sha256, status.  RESTORE puts a unit's members back at their original paths (never over an existing file),
each verified against the ledger.  Free space is guarded: a unit runs only with free >= its estimated zip + RESERVE, and aborts (partial
removed, sources untouched) when free space drops below FLOOR while writing.  All file access uses the \\\\?\\ extended-length form
(_sota/external/repos/*/.venv paths exceed MAX_PATH).

Usage: python scratchpad/data_legacy_zip.py PLAN                 -> prints the units (writes nothing)
       python scratchpad/data_legacy_zip.py RUN [group ...]      -> zips, verifies, deletes the sources of each verified unit
       python scratchpad/data_legacy_zip.py RESTORE <unit ...>   -> extracts the unit's members back under the repo
"""
import glob
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile

REPO = "C:/Users/Swastik/Desktop/CRAG"
DATA = REPO + "/data"
OUTZ = REPO + "/archive/legacy_data_zip"
LED_DIR = REPO + "/results/DATA_CLEANUP"
LEDGER = LED_DIR + "/LEGACY_ZIP_LEDGER__2026-09-30.jsonl"
RESERVE = 600 * 2 ** 20                 # free space kept after a unit (estimated zip size subtracted first)
FLOOR = 350 * 2 ** 20                   # free space below which a running unit aborts
UNIT_MAX = 1 << 30                      # uncompressed bytes per zip unit (a single larger file is its own unit)
CH = 8 << 20
# (group, selectors relative to the repo, deflate level, conservative zipped / source ratio used for the free-space guard)
GROUPS = [
    ("sota", ["data/ukb_storage/_sota"], 6, 0.5),
    ("raw", ["data/raw"], 6, 0.5),
    ("bm25", ["data/ukb_storage/*/bm25.pkl"], 6, 0.5),
    ("hpr_std", ["data/ukb_storage/*_hpr_clean", "data/ukb_storage/*_std_clean"], 6, 0.8),
    ("colbert", ["data/ukb_storage/metaqa/colbert_token_embs.pkl", "data/ukb_storage/metaqa/colbert_ukb"], 6, 0.9),
    ("checkpoints", ["data/ukb_storage/*/checkpoints"], 3, 1.0),
    ("cache", ["data/ukb_storage/*/cache"], 3, 1.0),
    ("ft_bge", ["data/ukb_storage/*/ft_bge"], 3, 1.0),
    ("faiss", ["data/ukb_storage/*/nodes.index", "data/ukb_storage/hotpotqa_clean/centroids.index"], 3, 1.0),
    ("bge_large", ["data/ukb_storage/*/bge_large"], 3, 1.0),
    ("old_models", ["checkpoints/metaqa", "checkpoints/2wiki", "checkpoints/musique", "checkpoints/squad"], 3, 1.0),
]
PROTECT = ("data/final_canonical", "data/l1_canonical", "data/l1_lowmem", "data/original", "data/processed", "data/_family_v1",
           "data/l2_corpus", "results", "scratchpad", "src", ".git", "archive")
PROTECT_PARTS = ("gte_qwen", "graph.pt", "splade_doc_embs.pkl", "partition_map.json", "_head_cache", "_review")


def lp(p):
    """extended-length form for every open / stat / remove on Windows."""
    p = os.path.abspath(p).replace("/", "\\")
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def free():
    return shutil.disk_usage(REPO).free


def sha_file(p):
    h = hashlib.sha256()
    with open(lp(p), "rb") as f:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p.replace("/", os.sep), REPO.replace("/", os.sep)).replace("\\", "/")


def files_under(root):
    """regular files below root (a file root yields itself); symlinks / junctions are never followed or archived."""
    if os.path.isfile(lp(root)):
        yield root.replace("\\", "/")
        return
    stack = [root.replace("\\", "/")]
    while stack:
        d = stack.pop()
        with os.scandir(lp(d)) as it:
            ents = sorted(it, key=lambda e: e.name)
        for e in reversed(ents):
            q = d.rstrip("/") + "/" + e.name
            if e.is_symlink() or getattr(e, "is_junction", lambda: False)():
                print("  SKIP link: " + rel(q), flush=True)
            elif e.is_dir(follow_symlinks=False):
                stack.append(q)
            elif e.is_file(follow_symlinks=False):
                yield q


def check_protected(p):
    r = rel(p)
    assert not any(r == x or r.startswith(x + "/") for x in PROTECT), "protected path selected: " + r
    parts = r.split("/")
    if len(parts) > 3 and parts[0] == "data" and parts[1] == "ukb_storage" and not parts[2].endswith(("_hpr_clean", "_std_clean")):   # live inputs sit directly in the dataset dir; the same names inside
        assert parts[2] not in PROTECT_PARTS and parts[3] not in PROTECT_PARTS, "protected name selected: " + r   # bge_large/, *_hpr_clean/... have no live reader (2026-09-30 scan)


def plan():
    """[(unit id, group, level, ratio, [(path, size, mtime_ns, floor_dir)])] in group order."""
    units = []
    for g, sels, lvl, ratio in GROUPS:
        fl = []
        for s in sels:
            for root in sorted(glob.glob(REPO + "/" + s)):
                root = root.replace("\\", "/")
                floor = os.path.dirname(root)
                for p in files_under(root):
                    check_protected(p)
                    st = os.stat(lp(p))
                    fl.append((p, st.st_size, st.st_mtime_ns, floor))
        cur, cur_b, n = [], 0, 0
        for f in fl:
            if cur and (cur_b + f[1] > UNIT_MAX):
                n += 1
                units.append(("%s_%03d" % (g, n), g, lvl, ratio, cur))
                cur, cur_b = [], 0
            cur.append(f)
            cur_b += f[1]
        if cur:
            n += 1
            units.append(("%s_%03d" % (g, n), g, lvl, ratio, cur))
    return units


def ledger(o):
    os.makedirs(LED_DIR, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(o, sort_keys=True) + "\n")


def build_unit(uid, lvl, files):
    os.makedirs(OUTZ, exist_ok=True)
    zp = "%s/%s.zip" % (OUTZ, uid)
    assert not os.path.exists(zp), "exists: " + zp
    part = zp + ".partial"
    meta = []
    zf = zipfile.ZipFile(lp(part), "w", zipfile.ZIP_DEFLATED, compresslevel=lvl, allowZip64=True)
    try:
        for p, size, mt, _ in files:
            arc = rel(p)
            zi = zipfile.ZipInfo.from_file(lp(p), arc, strict_timestamps=False)
            zi.compress_type = zipfile.ZIP_DEFLATED
            h = hashlib.sha256()
            n = 0
            with open(lp(p), "rb") as src, zf.open(zi, "w", force_zip64=True) as dst:
                for b in iter(lambda: src.read(CH), b""):
                    h.update(b)
                    dst.write(b)
                    n += len(b)
                    if free() < FLOOR:
                        raise OSError("free space below the floor while writing %s" % uid)
            assert n == size, "size changed while reading " + p
            meta.append({"path": arc, "bytes": size, "sha256": h.hexdigest(), "mtime_ns": mt})
    except BaseException:
        zf.close()
        os.remove(lp(part))
        raise
    zf.close()
    os.replace(lp(part), lp(zp))
    return zp, meta


def verify_unit(zp, meta):
    with zipfile.ZipFile(lp(zp)) as z:
        assert z.testzip() is None, "CRC failure in " + zp
        names = {i.filename for i in z.infolist()}
        assert names == {m["path"] for m in meta}, "member set differs"
        for m in meta:
            h = hashlib.sha256()
            n = 0
            with z.open(m["path"]) as f:
                for b in iter(lambda: f.read(CH), b""):
                    h.update(b)
                    n += len(b)
            assert n == m["bytes"] and h.hexdigest() == m["sha256"], "readback mismatch: " + m["path"]


def remove_file(p):
    try:
        os.remove(lp(p))
    except PermissionError:                              # read-only attribute (git pack files, venv wheels): clear it, retry once
        os.chmod(lp(p), stat.S_IWRITE)
        os.remove(lp(p))


def archived_paths():
    out = set()
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                if "path" in r and str(r.get("unit", "")):
                    out.add(r["path"])
    return out


def run(sel_groups):
    units = plan()
    if sel_groups:
        units = [u for u in units if u[1] in sel_groups]
    tot_src = tot_zip = 0
    kept_locked = []
    t_all = time.time()
    for uid, g, lvl, ratio, files in units:
        gone = [f for f in files if not os.path.exists(lp(f[0]))]      # a path two selectors name (bm25.pkl inside *_hpr_clean) is archived once, by the first unit
        if gone:
            done_paths = archived_paths()
            for f in gone:
                assert rel(f[0]) in done_paths, "planned file vanished and is not in the ledger: " + f[0]
            print("  %s: %d file(s) already archived by an earlier unit, dropped from this unit" % (uid, len(gone)), flush=True)
            files = [f for f in files if f not in gone]
            if not files:
                continue
        src = sum(f[1] for f in files)
        est = int(src * ratio)
        fr = free()
        if fr < est + RESERVE:
            print("SKIP %s: %.2f GB source, est zip %.2f GB, free %.2f GB (need est + %.2f)" % (uid, src / 1e9, est / 1e9, fr / 1e9, RESERVE / 1e9), flush=True)
            ledger({"unit": uid, "status": "SKIPPED_NO_SPACE", "source_bytes": src, "free": fr, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            continue
        t0 = time.time()
        try:
            zp, meta = build_unit(uid, lvl, files)
        except OSError as e:
            print("ABORT %s: %s (sources untouched)" % (uid, e), flush=True)
            ledger({"unit": uid, "status": "ABORTED", "reason": str(e), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            continue
        verify_unit(zp, meta)
        for p, size, mt, _ in files:                     # sources unchanged since the plan
            st = os.stat(lp(p))
            assert st.st_size == size and st.st_mtime_ns == mt, "source changed since the plan: " + p
        zb, zs = os.path.getsize(lp(zp)), sha_file(zp)
        for m in meta:
            ledger(dict(m, unit=uid, zip=rel(zp), group=g))
        deleted, locked = 0, []
        for p, size, mt, _ in files:
            try:
                remove_file(p)
                deleted += 1
            except OSError as e:
                locked.append(rel(p))
                print("  KEPT (cannot delete): %s: %s" % (rel(p), e), flush=True)
        for d, floor in sorted({(os.path.dirname(p), fl) for p, _, _, fl in files}, key=lambda t: len(t[0]), reverse=True):
            cur = d
            while len(cur) > len(floor) and cur.startswith(floor + "/") and os.path.isdir(lp(cur)) and not os.listdir(lp(cur)):
                os.rmdir(lp(cur))                        # only directories the archive emptied, never the selector root's parent
                cur = os.path.dirname(cur)
        ledger({"unit": uid, "group": g, "status": "ZIPPED_VERIFIED_SOURCES_DELETED" if not locked else "ZIPPED_VERIFIED_SOME_SOURCES_KEPT",
                "zip": rel(zp), "zip_bytes": zb, "zip_sha256": zs, "files": len(files), "source_bytes": src, "sources_deleted": deleted,
                "sources_kept_locked": locked, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        tot_src += src
        tot_zip += zb
        kept_locked += locked
        print("OK   %-16s %6d files  %7.3f GB -> %7.3f GB  (%.0fs)  free now %.2f GB" % (uid, len(files), src / 1e9, zb / 1e9, time.time() - t0, free() / 1e9), flush=True)
    print("DONE %d units in %.0fs: source %.3f GB -> zip %.3f GB (saved %.3f GB); %d files kept (locked); free %.2f GB" % (
        len(units), time.time() - t_all, tot_src / 1e9, tot_zip / 1e9, (tot_src - tot_zip) / 1e9, len(kept_locked), free() / 1e9), flush=True)


def restore(unit_ids):
    led = [json.loads(ln) for ln in open(LEDGER, encoding="utf-8")]
    for uid in unit_ids:
        fm = {r["path"]: r for r in led if r.get("unit") == uid and "path" in r}
        zr = next(r for r in led if r.get("unit") == uid and r.get("status", "").startswith("ZIPPED"))
        zp = REPO + "/" + zr["zip"]
        assert sha_file(zp) == zr["zip_sha256"], "zip changed: " + zp
        with zipfile.ZipFile(lp(zp)) as z:
            for m, r in fm.items():
                dst = REPO + "/" + m
                if os.path.exists(lp(dst)):
                    print("  exists, left alone: " + m)
                    continue
                os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                h = hashlib.sha256()
                with z.open(m) as f, open(lp(dst), "wb") as o:
                    for b in iter(lambda: f.read(CH), b""):
                        h.update(b)
                        o.write(b)
                assert h.hexdigest() == r["sha256"], "restored file differs: " + m
                os.utime(lp(dst), ns=(r["mtime_ns"], r["mtime_ns"]))
        print("restored", uid, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "PLAN":
        tot = 0
        for uid, g, lvl, ratio, files in plan():
            src = sum(f[1] for f in files)
            tot += src
            print("%-16s %6d files %8.3f GB  est zip <= %.3f GB  level %d" % (uid, len(files), src / 1e9, src * ratio / 1e9, lvl))
        print("TOTAL source %.3f GB; free now %.2f GB" % (tot / 1e9, free() / 1e9))
    elif mode == "RUN":
        run(set(sys.argv[2:]))
    elif mode == "RESTORE":
        restore(sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
