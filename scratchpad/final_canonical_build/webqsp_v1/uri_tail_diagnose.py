"""What is the residual undecodable EXTERNAL_URI tail actually encoded in?

Hypotheses tested per language, on real failing samples:
  A  legacy multi-byte CJK codepage (gb18030 / big5 / shift_jis / euc_jp / euc_kr / cp949)
  B  UTF-16 low byte with a single implied high byte (works only for scripts inside one 256-block;
     for hanzi/kanji/hangul the high byte VARIES per character, so the information is destroyed)
  C  a Latin single-byte codepage
Prints, per language, how many samples each hypothesis makes legible so the choice is measured.
"""
import sys, io, glob, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "scratchpad/final_canonical_build/webqsp_v1")
import pyarrow.parquet as pq, pyarrow.compute as pc
from uri_render import render, _split_escapes, WIKI, REPL

CJK = {"zh": ["gb18030", "big5", "gb2312"], "ja": ["shift_jis", "euc_jp", "cp932"],
       "ko": ["euc_kr", "cp949", "johab"],
       "pl": ["cp1250", "iso8859-2"], "cs": ["cp1250", "iso8859-2"], "ro": ["cp1250", "iso8859-2"],
       "lt": ["cp1257", "iso8859-13"], "lv": ["cp1257", "iso8859-13"], "hr": ["cp1250"],
       "sk": ["cp1250"], "vi": ["cp1258"], "en": ["cp1252", "latin-1"], "de": ["cp1252"],
       "it": ["cp1252"], "fr": ["cp1252"]}
BLOCKS = {"ja": [0x3000, 0x3040], "ko": [0xAC00, 0x3130], "zh": [0x4E00]}

def clean(t):
    return bool(t) and REPL not in t and not any(ord(c) < 0x20 for c in t)

samples = collections.defaultdict(list)
for fp in sorted(glob.glob("data/final_canonical/freebase_v3/canonical/nodes/*.parquet"))[::16]:
    t = pq.read_table(fp, columns=["node_id", "kind"])
    t = t.filter(pc.equal(t["kind"], "EXTERNAL_URI"))
    for uri in t["node_id"].to_pylist():
        name, kind = render(uri)
        if kind != "WIKIPEDIA_UNDECODABLE":
            continue
        m = WIKI.match(uri)
        lang = m.group(1).lower() if m else "?"
        if len(samples[lang]) < 200:
            samples[lang].append((uri, m.group(2) if m else ""))
    if sum(len(v) for v in samples.values()) > 2400:
        break

for lang, rows in sorted(samples.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(f"\n=== {lang}  ({len(rows)} failing samples) ===")
    wins = collections.Counter()
    demo = {}
    for uri, rest in rows:
        parts = _split_escapes(rest)
        raw = bytes(c if e else (ord(c) & 0xFF) for c, e in parts)
        for enc in CJK.get(lang, ["cp1252"]):
            try:
                d = raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
            if clean(d):
                wins[enc] += 1
                demo.setdefault(enc, (rest[:60], d[:40]))
        for base in BLOCKS.get(lang, []):
            try:
                d = "".join((chr(base + c) if c not in (0x20, 0x5F) else chr(c)) if e else c
                            for c, e in parts)
            except ValueError:
                continue
            if clean(d):
                wins[f"+0x{base:X}"] += 1
                demo.setdefault(f"+0x{base:X}", (rest[:60], d[:40]))
    if not wins:
        print("   nothing legible under any hypothesis")
    for enc, c in wins.most_common(6):
        src, out = demo[enc]
        print(f"   {c:>4}/{len(rows)}  {enc:12s}  {src}  ->  {out}")
