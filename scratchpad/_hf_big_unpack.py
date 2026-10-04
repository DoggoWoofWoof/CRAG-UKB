"""Host-side companion of _hf_big.py: turn the stored objects pulled by _fbx_hf_pull.py back into the original files and verify them (no token, no network).

  python -u scratchpad/_hf_big_unpack.py --src <pull dest> --dest <root that gets final_canonical/...> [--only SUBSTR ...] [--keep-stored] [--wheel scratchpad/zstandard.whl]

--src is the --dest that _fbx_hf_pull.py used (it contains final_canonical/<...>[.zst] and MANIFEST/HFBIG_INDEX.json).  Each object is decompressed (zstd) or copied (raw), its sha256 is compared with
the index's orig_sha (sha256 of the ORIGINAL bytes taken on the laptop) and only then renamed into place; a mismatch leaves the .tmp file and exits 1.  The stored object is deleted after a verified unpack
unless --keep-stored.  zstandard is not installed on the host: it is unpacked from the pushed wheel into <dest>/../_pylib (a private directory, no environment is modified)."""
import argparse
import hashlib
import json
import os
import sys
import zipfile


def need_zstd(wheel, libdir):
    try:
        import zstandard  # noqa: F401
        return
    except ImportError:
        pass
    if not os.path.isdir(os.path.join(libdir, "zstandard")):
        os.makedirs(libdir, exist_ok=True)
        with zipfile.ZipFile(wheel) as z:
            z.extractall(libdir)
    sys.path.insert(0, libdir)
    import zstandard  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--keep-stored", action="store_true")
    ap.add_argument("--wheel", default="scratchpad/zstandard.whl")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "MANIFEST", "HFBIG_INDEX.json")))["objects"]
    need_zstd(a.wheel, os.path.join(os.path.dirname(os.path.abspath(a.dest)), "_pylib"))
    import zstandard
    n = bad = skipped = 0
    for rel, v in sorted(idx.items()):
        if a.only and not any(o in rel for o in a.only):
            continue
        stored = os.path.join(a.src, v["stored_rel"])
        out = os.path.join(a.dest, "final_canonical", rel)
        if not os.path.exists(stored):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        h = hashlib.sha256()
        with open(stored, "rb") as fi, open(tmp, "wb") as fo:
            if v["mode"] == "zst":
                for chunk in zstandard.ZstdDecompressor().read_to_iter(fi, read_size=16 << 20):
                    h.update(chunk)
                    fo.write(chunk)
            else:
                while True:
                    b = fi.read(16 << 20)
                    if not b:
                        break
                    h.update(b)
                    fo.write(b)
        if h.hexdigest() != v["orig_sha"] or os.path.getsize(tmp) != v["orig_size"]:
            bad += 1
            print("MISMATCH", rel, flush=True)
            continue
        os.replace(tmp, out)
        if not a.keep_stored:
            os.remove(stored)
        n += 1
        if n % 50 == 0:
            print("  unpacked", n, flush=True)
    print("UNPACK: %d verified, %d mismatches, %d not present in src" % (n, bad, skipped), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
