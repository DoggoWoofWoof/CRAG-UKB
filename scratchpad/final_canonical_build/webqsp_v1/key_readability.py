"""Does the FREEBASE_KEY_EXACT tier actually recover NAMES, or only identifiers?

The tier measurement counted nodes that HAVE a /type/object/key. Inspecting the keys shows most
are machine identifiers -- /dataworld/freeq/job_<uuid>_var_... is no more readable than the MID it
is standing in for. Counting those as recovered names would inflate the authoritative-resolution
figure with strings no reader can use, so this measures how many key-bearing nodes yield text that
is actually readable, and reports the two numbers separately.

Readable means: after undoing Freebase $XXXX escaping, the final path segment contains at least
two letter-runs of length >= 2, or one of length >= 4, and is not dominated by a UUID or hex blob.
"""
import sys, io, os, json, re, collections, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow.parquet as pq, pyarrow as pa

V3 = "data/final_canonical/freebase_v3"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
HEXBLOB = re.compile(r"^[0-9a-f]{16,}$", re.I)
LETTERS = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
ESC = re.compile(r"\$([0-9A-Fa-f]{4})")

def unescape(k):
    return ESC.sub(lambda m: chr(int(m.group(1), 16)), k)

VOWEL = re.compile(r"[aeiouyAEIOUYÀ-ɏ]")
# namespaces whose keys are structurally NOT names, whatever they look like: load-job artifacts,
# WordNet sense keys (a lemma wrapped in sense syntax), FDA article keys, schema path ids.
NOT_A_NAME = ("dataworld/freeq", "base/medicaldrugs/fda_load", "user/jamie/wordnet",
              "base/fbontology/metaschema", "base/wsjtopics", "wikipedia/images",
              "freebase/relevance", "authority/musicbrainz")

def readable(key):
    """-> (ok, text). text is the human-usable rendering when ok."""
    s = unescape(key)
    low = s.lstrip("/").lower()
    if any(low.startswith(p) for p in NOT_A_NAME):
        return False, ""
    seg = [p for p in s.split("/") if p]
    if not seg:
        return False, ""
    tail = seg[-1]
    if UUID.search(tail) or HEXBLOB.match(tail):
        return False, ""
    if tail.startswith(("job_", "articlekey_", "SB")):
        return False, ""
    runs = LETTERS.findall(tail)
    # a real name has at least one pronounceable word, not just consonant clusters like '0zf9nrd'
    if not any(len(r) >= 3 and VOWEL.search(r) for r in runs):
        return False, ""
    letterish = sum(c.isalpha() or c in " _-'" for c in tail)
    if letterish < len(tail) * 0.6:      # mostly punctuation/escapes -> a sense key, not a name
        return False, ""
    digits = sum(c.isdigit() for c in tail)
    if digits > len(tail) * 0.5:
        return False, ""
    return True, tail.replace("_", " ").strip()

t0 = time.time()
t = pq.read_table(f"{V3}/_acquisition/unnamed_keys.parquet")
nodes_readable, nodes_all = {}, set()
ns_ok, ns_bad = collections.Counter(), collections.Counter()
for nid, key in zip(t["node_id"].to_pylist(), t["key"].to_pylist()):
    nodes_all.add(nid)
    parts = [p for p in key.split("/") if p]
    ns = "/".join(parts[:2]) if len(parts) > 1 else "?"
    ok, text = readable(key)
    if ok:
        ns_ok[ns] += 1
        prev = nodes_readable.get(nid)
        if prev is None or key < prev[0]:      # deterministic: lexicographically smallest key wins
            nodes_readable[nid] = (key, text)
    else:
        ns_bad[ns] += 1

print(f"key rows                     {t.num_rows:,}")
print(f"distinct key-bearing nodes   {len(nodes_all):,}")
print(f"  -> READABLE key            {len(nodes_readable):,}  ({100*len(nodes_readable)/len(nodes_all):.3f}%)")
print(f"  -> identifier only         {len(nodes_all)-len(nodes_readable):,}")
print(f"\ntop namespaces yielding a readable key:")
for ns, c in ns_ok.most_common(12):
    print(f"  {c:>9,}  {ns}")
print(f"\ntop namespaces yielding NO readable key:")
for ns, c in ns_bad.most_common(8):
    print(f"  {c:>9,}  {ns}")
ex = list(nodes_readable.items())[:10]
print("\nexamples kept:")
for nid, (k, txt) in ex:
    print(f"  {nid:16s} {txt[:60]}")

if nodes_readable:
    ids = sorted(nodes_readable)
    out = pa.table({"node_id": pa.array(ids, pa.string()),
                    "display_name": pa.array([nodes_readable[i][1] for i in ids], pa.string()),
                    "source_key": pa.array([nodes_readable[i][0] for i in ids], pa.string())})
    pq.write_table(out, f"{V3}/_acquisition/key_names.parquet", compression="zstd")

rec = {"schema": "KEY_READABILITY/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "key_rows": t.num_rows, "distinct_key_bearing_nodes": len(nodes_all),
       "nodes_with_a_READABLE_key": len(nodes_readable),
       "nodes_with_identifier_only": len(nodes_all) - len(nodes_readable),
       "readable_pct": round(100 * len(nodes_readable) / len(nodes_all), 4),
       "namespaces_readable": dict(ns_ok.most_common(25)),
       "namespaces_identifier_only": dict(ns_bad.most_common(15)),
       "CORRECTION": ("V3_RESOLUTION_TIER_MEASURE.json reported FREEBASE_KEY_EXACT_recovered = "
                      "3,064,237. That counted nodes that HAVE a key, not nodes for which the key "
                      "is a name. The great majority are /dataworld/freeq/job_<uuid> load "
                      "artifacts, which carry no more information than the MID. Only the count "
                      "above may be reported as an authoritative NAME recovery."),
       "tie_break": "a node with several readable keys takes the lexicographically smallest, so the result is deterministic."}
with io.open(f"{V3}/V3_KEY_READABILITY.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{time.time()-t0:.0f}s")
