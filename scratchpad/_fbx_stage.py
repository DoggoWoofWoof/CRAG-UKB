"""Hard-link stage of the served Freebase tree for the Hugging Face upload (FREEBASE_SCALE lane transfer, user 2026-09-30).

upload_large_folder writes a .cache/huggingface/ directory INTO the folder it uploads; the frozen tree data/final_canonical/freebase must not
get one.  This mirrors the tree with NTFS hard links under data/_cache/fbx_stage/ (git-ignored, no extra bytes, the frozen files are only
ever read).  Refuses to run if the stage is non-empty.

Usage: python scratchpad/_fbx_stage.py        -> builds the stage, prints file count / bytes
       python scratchpad/_fbx_stage.py --drop -> removes the stage (only the links; the frozen files are untouched)
"""
import os
import sys

SRC = "C:/Users/Swastik/Desktop/CRAG/data/final_canonical/freebase"
DST = "C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_stage"


def main():
    if "--drop" in sys.argv:
        for r, ds, fs in os.walk(DST, topdown=False):
            for f in fs:
                os.remove(os.path.join(r, f))          # a link: the other name (the frozen file) keeps the data
            os.rmdir(r)
        print("stage removed")
        return
    assert not (os.path.isdir(DST) and os.listdir(DST)), "stage exists and is not empty: " + DST
    n = b = 0
    for r, ds, fs in os.walk(SRC):
        ds.sort()
        rel = os.path.relpath(r, SRC).replace("\\", "/")
        d = DST if rel == "." else DST + "/" + rel
        os.makedirs(d, exist_ok=True)
        for f in sorted(fs):
            s = os.path.join(r, f)
            os.link(s, os.path.join(d, f))
            n += 1
            b += os.path.getsize(s)
    print("staged %d files, %.3f GB (%d bytes) at %s" % (n, b / 1e9, b, DST))


if __name__ == "__main__":
    main()
