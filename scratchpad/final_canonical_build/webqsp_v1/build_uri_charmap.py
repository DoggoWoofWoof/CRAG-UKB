"""Measure, per language, which character each mangled low byte must have been.

The UTF-16 low-byte mangling dropped the high byte. For Cyrillic/Thai/Greek that byte is constant
across the whole script so a fixed offset recovers it, but for Latin-script languages it is not:
Romanian needs 0x01 for a-breve (U+0103) and 0x02 for t-comma (U+021B) in the SAME title, and
English needs 0x20 for an en dash (U+2013). The high byte is therefore not derivable from the byte
itself -- but it IS determined by which characters that language actually uses, and we can measure
that from the frozen metadata instead of guessing.

For each language, count the non-ASCII characters appearing in its Freebase names and in its
already-decodable Wikipedia titles, then group those characters by codepoint & 0xFF. Where one
character accounts for at least DOMINANCE of a byte's mass, that byte is unambiguous for that
language and is recorded. Where no character dominates, the byte is LEFT OUT and the title stays
undecoded -- which is what must happen for zh/ja/ko, where the high byte ranges over thousands of
values and any single choice would be invented text dressed up as a recovered title.
"""
import sys, io, os, json, glob, collections, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq, pyarrow.compute as pc
from uri_render import render, WIKI, _split_escapes

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/_acquisition/uri_charmap.json"
REC = f"{V3}/V3_URI_CHARMAP.json"
DOMINANCE = 0.70          # a byte is only resolved when one character owns this much of its mass
MIN_COUNT = 20            # and is seen at least this often, so rare noise cannot define a byte
NAME_STRIDE = 4           # row groups of name.parquet to sample
NODE_STRIDE = 16          # node shards to sample for clean wiki titles

t0 = time.time()
freq = collections.defaultdict(collections.Counter)   # lang -> Counter(char)

pf = pq.ParquetFile(f"{V3}/canonical/metadata/name.parquet")
rgs = list(range(0, pf.num_row_groups, NAME_STRIDE))
names_seen = 0
for i in rgs:
    t = pf.read_row_group(i, columns=["lexical", "lang"])
    for lex, lang in zip(t["lexical"].to_pylist(), t["lang"].to_pylist()):
        names_seen += 1
        if not lex or lex.isascii():        # C-speed skip; ~85% of names are pure ASCII
            continue
        c = freq[(lang or "").lower()]
        for ch in lex:
            if ord(ch) > 0x7F:
                c[ch] += 1
print(f"names scanned: {names_seen:,} from {len(rgs)}/{pf.num_row_groups} row groups ({time.time()-t0:.0f}s)")

titles_seen = 0
for fp in sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet"))[::NODE_STRIDE]:
    t = pq.read_table(fp, columns=["node_id", "kind"])
    t = t.filter(pc.equal(t["kind"], "EXTERNAL_URI"))
    for uri in t["node_id"].to_pylist():
        m = WIKI.match(uri)
        if not m:
            continue
        name, kind = render(uri)
        if kind != "WIKIPEDIA_TITLE":
            continue
        titles_seen += 1
        title = name.rsplit(" (", 1)[0]
        if title.isascii():
            continue
        c = freq[m.group(1).lower()]
        for ch in title:
            if ord(ch) > 0x7F:
                c[ch] += 1
print(f"clean wiki titles scanned: {titles_seen:,} ({time.time()-t0:.0f}s)")

charmap, stats = {}, {}
for lang, counter in freq.items():
    by_byte = collections.defaultdict(collections.Counter)
    for ch, n in counter.items():
        if ord(ch) <= 0xFFFF:
            by_byte[ord(ch) & 0xFF][ch] = n
    resolved, ambiguous = {}, 0
    for b, cands in by_byte.items():
        top, n = cands.most_common(1)[0]
        tot = sum(cands.values())
        if n >= MIN_COUNT and n / tot >= DOMINANCE:
            resolved[str(b)] = top
        else:
            ambiguous += 1
    if resolved:
        charmap[lang] = resolved
    stats[lang] = {"chars": len(counter), "bytes_resolved": len(resolved),
                   "bytes_ambiguous": ambiguous}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump({"DOMINANCE": DOMINANCE, "MIN_COUNT": MIN_COUNT, "map": charmap}, f, ensure_ascii=False)

interesting = ["ro", "pl", "en", "de", "vi", "lt", "lv", "cs", "it", "fr", "tr",
               "zh", "ja", "ko", "ru", "th"]
print(f"\n{'lang':6s} {'chars':>7s} {'resolved':>9s} {'ambiguous':>10s}")
for l in interesting:
    s = stats.get(l)
    if s:
        print(f"{l:6s} {s['chars']:>7,} {s['bytes_resolved']:>9,} {s['bytes_ambiguous']:>10,}")

rec = {"schema": "URI_CHARMAP/v1", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "purpose": "resolve the UTF-16 low-byte mangling for languages whose high byte is not constant",
       "DOMINANCE": DOMINANCE, "MIN_COUNT": MIN_COUNT,
       "names_scanned": names_seen, "clean_titles_scanned": titles_seen,
       "languages_with_a_map": len(charmap),
       "per_language": {k: stats[k] for k in sorted(stats) if k in charmap},
       "cjk_left_unresolved_on_purpose": {k: stats.get(k) for k in ("zh", "ja", "ko")},
       "honesty_note": ("a byte with no dominant character is deliberately left unresolved. For "
                        "zh/ja/ko the high byte ranges over thousands of values, so every choice "
                        "would be invented text presented as a recovered title; those URIs keep "
                        "their language-only rendering instead.")}
with io.open(REC, "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nlanguages with a map: {len(charmap)}   -> {OUT}\n{time.time()-t0:.0f}s")
