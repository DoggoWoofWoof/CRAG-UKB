"""Phase 2 -- Freebase source availability scan. READ ONLY. Downloads nothing.

Single filesystem traversal; every target pattern matched in-memory.
Writes scratchpad/final_canonical_build/_audit/phase2_source_scan.json
"""
import hashlib
import json
import os
import re
import sys
import time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit",
                   "phase2_source_scan.json")

# (label, compiled regex on lowercased basename)
TARGETS = [
    ("FastRDFStore-data.zip", r"^fastrdfstore.*"),
    ("fb_en.txt",             r"^fb_en(\.|$)"),
    ("cvtnodes.bin",          r"^cvtnodes?.*\.bin$"),
    ("Freebase-Setup",        r"^freebase[-_ ]?setup"),
    ("Virtuoso DB",           r"^virtuoso.*|.*\.virtuoso$|^.*virtuoso\.db$"),
    ("object_names",          r"^object[-_]names.*"),
    ("object_types",          r"^object[-_]types.*"),
    ("id2name_parts",         r"^id2name.*"),
    ("entities_id_label",     r"^entities[-_]id[-_]label.*"),
    ("properties_id_label",   r"^properties[-_]id[-_]label.*"),
    ("FACC1",                 r"^facc1.*|.*facc1.*"),
    ("surface_map",           r"^surface[-_]map.*"),
    ("freebase dump",         r"^freebase.*(\.gz|\.zip|\.txt|\.nt|\.tsv|\.json)$"),
    ("freebase generic",      r"^freebase.*"),
    ("mid2name / entity2name", r"^(mid2name|entity2name|name2mid|id2entity|mid_to_name).*"),
    ("manual_fb_filter",      r"^manual[-_]fb[-_]filter.*|^manual[-_]filter[-_]fb.*"),
    ("NSM subgraph out",      r"^subgraph_hop[12]\.txt$|^(cwq|webqsp)_step[01]\.json$"),
    ("rdf ntriples bulk",     r".*\.nt(\.gz|\.bz2)?$"),
]
TARGETS = [(lab, re.compile(rx)) for lab, rx in TARGETS]

DIR_TARGETS = [
    ("Freebase-Setup dir", re.compile(r"^freebase[-_ ]?setup$")),
    ("virtuoso dir",       re.compile(r"^virtuoso.*")),
    ("FastRDFStore dir",   re.compile(r"^fastrdfstore.*")),
    ("freebase dir",       re.compile(r"^freebase.*|^fb_.*")),
    ("facc1 dir",          re.compile(r".*facc1.*")),
]

SKIP_DIR_NAMES = {
    "$recycle.bin", "system volume information", "windows", "winsxs",
    "node_modules", ".git",
}


def sha256_head(path, nbytes=1 << 20):
    """sha256 of the whole file if small, else of the first nbytes (labelled)."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return None, None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            if sz <= 64 << 20:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                return h.hexdigest(), "full"
            h.update(f.read(nbytes))
            return h.hexdigest(), "first_1MiB"
    except OSError:
        return None, None


def main():
    roots = [r for r in sys.argv[1:]] or [
        r"C:\Users\Swastik", r"C:\ProgramData", r"C:\Program Files",
        r"C:\Program Files (x86)", r"D:\\", r"E:\\",
    ]
    roots = [r for r in roots if os.path.isdir(r)]
    hits, ndirs, nfiles = [], 0, 0
    t0 = time.time()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                    onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in SKIP_DIR_NAMES]
            ndirs += 1
            for d in list(dirnames):
                dl = d.lower()
                for lab, rx in DIR_TARGETS:
                    if rx.match(dl):
                        hits.append({"target": lab, "kind": "dir",
                                     "path": os.path.join(dirpath, d)})
                        break
            for fn in filenames:
                nfiles += 1
                fl = fn.lower()
                for lab, rx in TARGETS:
                    if rx.match(fl):
                        p = os.path.join(dirpath, fn)
                        try:
                            st = os.stat(p)
                        except OSError:
                            continue
                        dg, scope = sha256_head(p)
                        hits.append({
                            "target": lab, "kind": "file", "path": p,
                            "size_bytes": st.st_size,
                            "mtime_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                            "sha256": dg, "sha256_scope": scope,
                        })
                        break
    rec = {
        "_what": "Phase 2 Freebase source availability scan (read-only, no downloads)",
        "roots_scanned": roots,
        "dirs_visited": ndirs,
        "files_visited": nfiles,
        "elapsed_s": round(time.time() - t0, 1),
        "hit_count": len(hits),
        "hits": hits,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print("SCAN_DONE dirs=%d files=%d hits=%d elapsed=%.1fs -> %s"
          % (ndirs, nfiles, len(hits), time.time() - t0, OUT))


if __name__ == "__main__":
    main()
