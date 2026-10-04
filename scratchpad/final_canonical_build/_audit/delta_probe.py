import unicodedata as U

A = "C:/Users/Swastik/Desktop/CRAG/scratchpad/final_canonical_build/_audit"
ents = [l.rstrip("\n") for l in open(A + "/union_entities.txt", encoding="utf-8")]
cls = [l.strip() for l in open(A + "/union_entity_class.txt", encoding="utf-8")]
D = 26603
BS = "\\"
tests = {
    "non_ascii": lambda e: any(ord(c) > 127 for c in e),
    "has_digit": lambda e: any(c.isdigit() for c in e),
    "leading_or_trailing_ws": lambda e: e != e.strip(),
    "contains_backslash_escape": lambda e: BS in e,
    "len_gt_100": lambda e: len(e) > 100,
    "startswith_g_dot": lambda e: e.startswith("g."),
    "looks_url_or_file": lambda e: ("http" in e or "www." in e
                                    or e.endswith((".com", ".org", ".net", ".jpg", ".gif", ".png"))),
    "pure_punct": lambda e: e.strip() != "" and not any(c.isalnum() for c in e),
    "empty_after_strip": lambda e: e.strip() == "",
}
for k, f in tests.items():
    n = sum(1 for e in ents if f(e))
    print(f"{k:26s} {n:>9,}  {'<== MATCHES DELTA' if n == D else ''}")
print(f"{'OTHER class':26s} {sum(1 for c in cls if c == 'OTHER'):>9,}")
print(f"{'g. MIDs':26s} {sum(1 for e, c in zip(ents, cls) if c == 'FREEBASE_MID' and e.startswith('g.')):>9,}")
print(f"{'m. MIDs':26s} {sum(1 for e, c in zip(ents, cls) if c == 'FREEBASE_MID' and e.startswith('m.')):>9,}")
print(f"TARGET DELTA {D:,}   TOTAL {len(ents):,}")
