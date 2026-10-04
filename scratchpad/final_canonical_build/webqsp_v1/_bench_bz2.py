import bz2, time, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
P = ("data/final_canonical/freebase_v3/_acquisition/raw/"
     "quads_freebase-datadump-quadruples.tsv.bz2")
N = 1200000
KIND = {"/type/object/name": "name", "/common/topic/alias": "alias",
        "/type/object/type": "type", "/type/object/key": "key"}
KB = {k.encode(): v for k, v in KIND.items()}

t = time.time(); n = k = 0
with bz2.open(P, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n += 1
        f = line.rstrip("\n").split("\t")
        if len(f) >= 4 and KIND.get(f[1]):
            k += 1
        if n >= N:
            break
a = time.time() - t
print(f"text-mode, split every line : {a:6.1f}s  {N/a:>9,.0f} lines/s  kept {k:,}")

t = time.time(); n = k = 0
with bz2.open(P, "rb") as fh:
    for line in fh:
        n += 1
        if b"/type/object/" in line or b"/common/topic/alias" in line:
            f = line.rstrip(b"\n").split(b"\t")
            if len(f) >= 4 and KB.get(f[1]):
                k += 1
        if n >= N:
            break
b = time.time() - t
print(f"binary prefilter, split kept: {b:6.1f}s  {N/b:>9,.0f} lines/s  kept {k:,}"
      f"   speedup {a/b:.2f}x")

# how much of the wall time is decompression alone?  (upper bound on any parsing win)
t = time.time(); nb = 0
d = bz2.BZ2Decompressor()
with open(P, "rb") as fh:
    while nb < 60_000_000:
        c = fh.read(1 << 22)
        if not c:
            break
        nb += len(d.decompress(c))
c = time.time() - t
print(f"raw decompression only      : {c:6.1f}s for {nb/1e6:.0f} MB uncompressed "
      f"= {nb/1e6/c:.1f} MB/s")
