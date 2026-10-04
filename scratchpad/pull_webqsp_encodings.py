# -*- coding: utf-8 -*-
"""
Pull the WebQSP encoder output from the Modal volume.

WHY NOT `modal volume get`
    `modal volume get crag-webqsp-enc / <dest>` fails with "No such file or directory" and the
    `**` form produced nothing in ten minutes. Downloading the 302 part directories one CLI
    call at a time works but pays process startup 302 times. The SDK does it in one process.

WHAT IT REFUSES TO DO
    Silently produce a short file. Every part is written to a .part file first and renamed only
    after its byte count matches what the volume reports, so an interrupted download leaves an
    obviously-unfinished name rather than a truncated .npy that numpy will happily open and
    misinterpret. Already-complete files are skipped, so this is resumable.
"""
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import modal  # noqa: E402

VOL = "crag-webqsp-enc"
DEST = "data/final_canonical/webqsp/_enc"


def main():
    t0 = time.time()
    from modal.volume import FileEntryType
    vol = modal.Volume.from_name(VOL)
    # listdir(recursive=True) returns the part DIRECTORIES, not their contents, so the tree is
    # walked one level explicitly. And the type test compares against the SDK's own enum:
    # FILE is 1 and DIRECTORY is 2, an ordering that is easy to get backwards, and a directory
    # entry reports a small nonzero size (18 bytes) so "size > 0" does not catch the mistake.
    files = []
    for top in vol.listdir("/"):
        if top.type == FileEntryType.FILE:
            files.append(top)
        elif top.type == FileEntryType.DIRECTORY:
            files.extend(e for e in vol.listdir(top.path) if e.type == FileEntryType.FILE)
    if not files:
        sys.exit("volume %s listed no files" % VOL)
    total = sum(e.size for e in files)
    print("%d files, %.2f GB" % (len(files), total / 1e9), flush=True)

    done_b = skipped = 0
    for i, e in enumerate(files):
        dst = os.path.join(DEST, e.path.lstrip("/").replace("/", os.sep))
        if os.path.exists(dst) and os.path.getsize(dst) == e.size:
            skipped += 1
            done_b += e.size
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        got = os.path.getsize(tmp)
        if got != e.size:
            os.remove(tmp)
            sys.exit("%s: got %d bytes, volume says %d" % (e.path, got, e.size))
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(tmp, dst)
        done_b += e.size
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            el = time.time() - t0
            print("  %4d/%d  %.2f/%.2f GB  %.0fs  %.1f MB/s"
                  % (i + 1, len(files), done_b / 1e9, total / 1e9, el,
                     (done_b / 1e6) / max(el, 1e-9)), flush=True)

    rec = {"volume": VOL, "dest": DEST, "n_files": len(files),
           "bytes": total, "skipped_already_present": skipped,
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open("scratchpad/webqsp_pull.json", "w", encoding="utf-8"), indent=1)
    print("\n%d files, %.2f GB (%d already present)  %.0fs"
          % (len(files), total / 1e9, skipped, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
