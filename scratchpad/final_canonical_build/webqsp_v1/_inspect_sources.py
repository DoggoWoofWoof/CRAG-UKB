"""What is actually inside the 2010 per-type TSV tar and the 2008 quadruples dump?

Streamed, never expanded to disk (21 GB free).  The tar is read with tarfile in stream mode so only
the members' headers and the first few members' heads are touched.
"""
import sys, io, os, bz2, tarfile, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = "data/final_canonical/freebase_v3/_acquisition/raw"
t0 = time.time()

print("===== 2008-03-28 quadruples: first lines and predicate mix on 400,000 lines")
import collections
p = collections.Counter()
n = 0
with bz2.open(f"{RAW}/quads2008_freebase-datadump-quadruples.tsv.bz2", "rt",
              encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n += 1
        if n <= 4:
            print("   ", repr(line.rstrip("\n"))[:160])
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2:
            p[f[1]] += 1
        if n >= 400000:
            break
print(f"  lines sampled {n:,}")
for k, v in p.most_common(10):
    print(f"     {k[:56]:<56} {v:,}")

print(f"\n===== 2010-07-16 per-type TSV tar: members ({time.time()-t0:.0f}s)")
fh = bz2.open(f"{RAW}/tsv_freebase-datadump-tsv.tar.bz2", "rb")
tf = tarfile.open(fileobj=fh, mode="r|")
shown = 0
tot = 0
for m in tf:
    if not m.isfile():
        continue
    tot += 1
    if tot <= 60:
        print(f"   {m.size:>13,}  {m.name}")
    if shown < 3 and m.size > 2_000_000:
        data = tf.extractfile(m).read(1400)
        print(f"      head of {m.name}:")
        for ln in data.decode("utf-8", "replace").split("\n")[:4]:
            print(f"        {ln[:150]!r}")
        shown += 1
    if tot >= 400:
        print("   ... (stopping the listing at 400 members)")
        break
print(f"  members seen {tot}  ({time.time()-t0:.0f}s)")
