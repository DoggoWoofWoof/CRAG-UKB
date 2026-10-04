"""Measure uri_render against a spread of the real EXTERNAL_URI population.

Samples every Nth shard rather than the head of one, because node_uid ordering correlates with
source order and therefore with language: the head of shard 0 is overwhelmingly en/de wiki, which
is exactly the population the decoder already handled. The failure tail lives elsewhere.
"""
import sys, io, glob, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "scratchpad/final_canonical_build/webqsp_v1")
import pyarrow.parquet as pq, pyarrow.compute as pc
from uri_render import render, REPL

V3 = "data/final_canonical/freebase_v3"
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 8

def suspect(t):
    return (not t) or REPL in t or any(ord(c) < 0x20 for c in t)

kinds = collections.Counter()
langs_bad = collections.Counter()
bad_examples = []
n = 0
files = sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet"))[::STRIDE]
for fp in files:
    t = pq.read_table(fp, columns=["node_id", "kind"])
    t = t.filter(pc.equal(t["kind"], "EXTERNAL_URI"))
    if not t.num_rows:
        continue
    for uri in t["node_id"].to_pylist():
        name, kind = render(uri)
        kinds[kind] += 1
        n += 1
        # strip the trailing " (Language Wikipedia)" / " (host)" before judging the text itself
        core = name.rsplit(" (", 1)[0] if name.endswith(")") else name
        if suspect(core) or kind == "WIKIPEDIA_UNDECODABLE":
            lang = uri.split("://", 1)[-1].split(".", 1)[0]
            langs_bad[lang] += 1
            if len(bad_examples) < 25:
                bad_examples.append((uri[:110], name[:70]))

print(f"sampled {n:,} EXTERNAL_URI node_ids from {len(files)} of 262 shards (stride {STRIDE})\n")
for k, c in kinds.most_common():
    print(f"  {c:>10,}  {100*c/n:6.3f}%  {k}")
tot_bad = sum(langs_bad.values())
print(f"\nunreadable or undecodable: {tot_bad:,} ({100*tot_bad/n:.4f}%)")
if langs_bad:
    print("  by wiki language:", dict(langs_bad.most_common(15)))
for u, nm in bad_examples:
    print(f"    {u}\n      -> {nm}")
