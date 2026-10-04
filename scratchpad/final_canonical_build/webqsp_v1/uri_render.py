"""Deterministic human-readable rendering for EXTERNAL_URI nodes (74,454,117 of them).

These are not an identity-recovery problem: most are Wikipedia article URLs that carry the title
in the path, so the readable text was always present in the identifier. Four encodings occur in
the real data and a naive unquote turns all but the first into mojibake:
  * plain percent-encoded UTF-8            %D0%9C%D1%83%D1%85  ->  Mux (Cyrillic)
  * DOUBLE percent-encoding                %25D9%25BE  ->  %D9%BE  ->  Persian pe
  * a legacy single-byte codepage          %E9 on an en wiki  ->  latin-1 e-acute
  * the UTF-16 LOW BYTE of each code unit  %1C%43%45 on ru  ->  +0x400  ->  Cyrillic Mux

The last form is the awkward one: the high byte (the script block) was dropped and is only
recoverable from the wiki language code. It cannot be detected by a trigger rule -- %25 is a
legitimate low byte for Cyrillic Ha, so a "looks double-encoded" test hijacks it, and a title
whose letters all sit above 0x20 leaves no control character to notice. So rather than guessing
which encoding applies, decode under ALL of them and keep whichever puts the most letters inside
the script block the wiki language implies. Latin-script wikis carry no such expectation, so
there the candidates are simply tried in order of prior likelihood.
"""
from urllib.parse import unquote_to_bytes
import re

WIKI = re.compile(r"^https?://([a-z0-9\-]+)\.wikipedia\.org/(?:wiki/)?(.+)$", re.I)
CURID = re.compile(r"^(?:index\.html)?\?*curid=(\d+)$", re.I)
LANG = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
        "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "zh": "Chinese", "nl": "Dutch",
        "pl": "Polish", "sv": "Swedish", "ca": "Catalan", "no": "Norwegian", "fi": "Finnish",
        "cs": "Czech", "hu": "Hungarian", "ko": "Korean", "ar": "Arabic", "fa": "Persian",
        "he": "Hebrew", "tr": "Turkish", "uk": "Ukrainian", "el": "Greek", "da": "Danish",
        "ro": "Romanian", "id": "Indonesian", "vi": "Vietnamese", "sk": "Slovak", "hr": "Croatian",
        "bg": "Bulgarian", "sr": "Serbian", "th": "Thai", "hi": "Hindi", "bn": "Bengali",
        "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi",
        "gu": "Gujarati", "pa": "Punjabi", "si": "Sinhala", "lo": "Lao", "ka": "Georgian",
        "hy": "Armenian", "am": "Amharic", "mk": "Macedonian", "be": "Belarusian", "kk": "Kazakh",
        "mn": "Mongolian", "ur": "Urdu", "ne": "Nepali", "et": "Estonian", "lv": "Latvian",
        "lt": "Lithuanian", "sl": "Slovenian", "bs": "Bosnian", "yi": "Yiddish"}

# the wiki language code tells us which legacy codepage a non-utf8 escape came from. Guessing by
# trial order instead rendered Rub%E9n as Rubjn (cp1251) where latin-1 gives the correct Ruben.
CP = {}
for _l in ("ru", "uk", "bg", "sr", "mk", "be", "kk"): CP[_l] = "cp1251"
for _l in ("pl", "cs", "hu", "sk", "hr", "sl", "ro", "bs"): CP[_l] = "cp1250"
for _l in ("lt", "lv", "et"): CP[_l] = "cp1257"
CP["el"] = "cp1253"; CP["tr"] = "cp1254"
for _l in ("he", "yi"): CP[_l] = "cp1255"
for _l in ("ar", "fa", "ur"): CP[_l] = "cp1256"

# lang -> (high byte to restore, block_lo, block_hi). block_lo/hi bound where that script's
# LETTERS live, which is what the score below counts; it is not always offset..offset+0x100
# (Greek letters start at 0x370, Hebrew at 0x590, Ethiopic runs to 0x137F).
SCRIPT = {}
for _l in ("ru", "uk", "bg", "sr", "mk", "be", "kk", "mn", "tg"): SCRIPT[_l] = (0x400, 0x400, 0x530)
SCRIPT.update({
    "el": (0x300, 0x370, 0x400), "he": (0x500, 0x590, 0x600), "yi": (0x500, 0x590, 0x600),
    "ar": (0x600, 0x600, 0x700), "fa": (0x600, 0x600, 0x700), "ur": (0x600, 0x600, 0x700),
    "hi": (0x900, 0x900, 0x980), "mr": (0x900, 0x900, 0x980), "ne": (0x900, 0x900, 0x980),
    "sa": (0x900, 0x900, 0x980), "bn": (0x980, 0x980, 0xA00), "as": (0x980, 0x980, 0xA00),
    "pa": (0xA00, 0xA00, 0xA80), "gu": (0xA80, 0xA80, 0xB00), "or": (0xB00, 0xB00, 0xB80),
    "ta": (0xB80, 0xB80, 0xC00), "te": (0xC00, 0xC00, 0xC80), "kn": (0xC80, 0xC80, 0xD00),
    "ml": (0xD00, 0xD00, 0xD80), "si": (0xD80, 0xD80, 0xE00), "th": (0xE00, 0xE00, 0xE80),
    "lo": (0xE80, 0xE80, 0xF00), "hy": (0x530, 0x530, 0x590), "ka": (0x10A0, 0x10A0, 0x1100),
    "am": (0x1200, 0x1200, 0x1380),
})
REPL = "�"

# For Latin-script languages the dropped high byte is NOT constant: Romanian needs 0x01 for
# a-breve and 0x02 for t-comma in the same title, English needs 0x20 for an en dash. Which
# character each low byte stood for is instead measured per language from the frozen metadata by
# build_uri_charmap.py -- and left UNRESOLVED wherever no character dominates that byte, which is
# what keeps zh/ja/ko hanzi from being invented. Missing file just disables this candidate.
_CHARMAP = None
def _charmap(lang):
    global _CHARMAP
    if _CHARMAP is None:
        import json, os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "data", "final_canonical", "freebase_v3",
                         "_acquisition", "uri_charmap.json")
        try:
            with open(p, encoding="utf-8") as f:
                _CHARMAP = {k: {int(b): c for b, c in v.items()}
                            for k, v in json.load(f)["map"].items()}
        except (OSError, ValueError, KeyError):
            _CHARMAP = {}
    return _CHARMAP.get(lang)

# ASCII bytes that legitimately appear percent-escaped in a URL path and so must pass through as
# themselves. A LETTER or DIGIT escaped instead of written plainly is meaningless in a real URL and
# is therefore treated as evidence of the mangling, not as itself.
_PASSTHROUGH = set(range(0x20, 0x30)) | set(range(0x3A, 0x41)) | set(range(0x5B, 0x61)) \
             | set(range(0x7B, 0x7F))

def _split_escapes(s):
    """-> list of (int_byte | str_char, was_escaped). Keeps the escaped/literal distinction that
    unquote_to_bytes throws away. The UTF-16 low-byte form needs it: hyphen, underscore, paren and
    comma appear LITERALLY in those URLs and must not be offset, which is what made a first
    attempt at this emit control characters in the middle of otherwise correct Cyrillic."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] == "%" and i + 2 < n:
            try:
                out.append((int(s[i + 1:i + 3], 16), True)); i += 3; continue
            except ValueError:
                pass
        out.append((s[i], False)); i += 1
    return out

def _score(text, sc):
    """How much does this decoding look like a real title in this language? -1 = not a candidate."""
    if not text:
        return -1.0
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        return -1.0          # no real title contains a control character
    if REPL in text:
        return -1.0
    if sc is None:
        return 0.5           # Latin-script wiki: no script expectation, so order decides
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.45          # digits and punctuation only -- plausible but uninformative
    lo, hi = sc[1], sc[2]
    return sum(1 for ch in letters if lo <= ord(ch) < hi) / len(letters)

def _decode(s, lang=None):
    """-> (text, ok). ok is False when no candidate decoding produced legible text."""
    lang = (lang or "").lower()
    sc = SCRIPT.get(lang)
    parts = _split_escapes(s)
    raw = bytes(c if e else (ord(c) & 0xFF) for c, e in parts)
    cand = []                             # in order of prior likelihood; ties keep the first
    if "%25" in s:
        try:
            cand.append(unquote_to_bytes(s.replace("%25", "%")).decode("utf-8"))
        except UnicodeDecodeError:
            pass
    try:
        cand.append(raw.decode("utf-8"))
    except UnicodeDecodeError:
        pass
    if lang in CP:
        try:
            cand.append(raw.decode(CP[lang]))
        except (UnicodeDecodeError, LookupError):
            pass
    cand.append(raw.decode("latin-1"))
    if sc:
        off, lo, hi = sc
        cand.append("".join(
            (chr(off + c) if (c not in (0x20, 0x5F) and lo <= off + c < hi) else chr(c))
            if e else c
            for c, e in parts))
    cm = _charmap(lang)
    if cm:
        out, usable = [], True
        for c, e in parts:
            if not e:
                out.append(c)
            elif c in cm:
                out.append(cm[c])
            elif c in _PASSTHROUGH:
                out.append(chr(c))
            else:
                usable = False       # one unresolvable byte invalidates the whole title: filling
                break                # the gap from a non-dominant character would be invention
        if usable:
            cand.append("".join(out))
    best, best_score = "", -1.0
    for t in cand:
        v = _score(t, sc)
        if v > best_score:
            best, best_score = t, v
    return (best, True) if best_score > -1.0 else ("", False)

def render(uri):
    """(display_name, subkind). Never returns empty."""
    m = WIKI.match(uri)
    if m:
        lang, rest = m.group(1), m.group(2)
        L = LANG.get(lang.lower(), lang)
        c = CURID.match(rest)
        if c:
            return f"{L} Wikipedia article {c.group(1)}", "WIKIPEDIA_CURID"
        title, ok = _decode(rest, lang)
        title = title.replace("_", " ").strip()
        if not ok or not title:
            return f"{L} Wikipedia article", "WIKIPEDIA_UNDECODABLE"
        return f"{title} ({L} Wikipedia)", "WIKIPEDIA_TITLE"
    try:
        after = uri.split("://", 1)[1]
        host = after.split("/", 1)[0]
        path = after.split("/", 1)[1] if "/" in after else ""
    except IndexError:
        return uri, "RAW"
    host_clean = host[4:] if host.startswith("www.") else host
    seg = [p for p in path.split("/") if p]
    tail = (_decode(seg[-1])[0] if seg else "")
    tail = tail.replace("_", " ").strip()
    if not tail:
        return host_clean, "HOST_ONLY"
    if len(tail) > 90:
        tail = tail[:90].rstrip() + "..."
    return f"{tail} ({host_clean})", "HOST_PATH"
