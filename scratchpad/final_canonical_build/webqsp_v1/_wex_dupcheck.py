"""Why did 71,540 WEX name hits collapse to 40,305 nodes?  Duplicate guids, or a MID collision?

A collision would be a correctness failure: two different Freebase objects deriving to one MID would
put the wrong name on a node.  Duplicate guid rows are harmless.  This tells them apart, and prints
any colliding pair in full so the derivation can be judged on the evidence.
"""
import bz2, io, sys, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
P = ("data/final_canonical/freebase_v3/_acquisition/raw/"
     "wex_names_freebase-wex-2010-07-05-freebase_names.tsv.bz2")
AB = "0123456789bcdfghjklmnpqrstvwxyz_"
OFF = 0x8000000000000000


def enc(n):
    s = ""
    while n:
        s = AB[n & 31] + s
        n >>= 5
    return s or "0"


guid_seen = collections.Counter()
mid_guids = collections.defaultdict(set)
mid_names = collections.defaultdict(set)
n = 0
with bz2.open(P, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n += 1
        f = line.rstrip("\n").split("\t")
        if len(f) < 2 or not f[1]:
            continue
        h = f[0].strip()
        guid_seen[h] += 1
        mid = "m.0" + enc(int(h[16:], 16) - OFF)
        mid_guids[mid].add(h)
        mid_names[mid].add(f[1])

dup_guid = sum(1 for v in guid_seen.values() if v > 1)
coll = {m: g for m, g in mid_guids.items() if len(g) > 1}
print(f"lines {n:,}   distinct guids {len(guid_seen):,}   guids appearing more than once {dup_guid:,}")
print(f"distinct derived MIDs {len(mid_guids):,}   MIDs reached by >1 guid: {len(coll):,}")
diff = {m for m in coll if len(mid_names[m]) > 1}
print(f"of those, MIDs where the colliding guids carry DIFFERENT names: {len(diff):,}")
for m in list(diff)[:10]:
    print(f"  {m}  guids={sorted(mid_guids[m])}  names={sorted(mid_names[m])[:4]}")
for m in list(coll)[:5]:
    if m not in diff:
        print(f"  same-name collision {m}  guids={sorted(mid_guids[m])}  name={sorted(mid_names[m])}")
