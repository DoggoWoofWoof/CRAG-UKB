"""Fixed cases for uri_render, one per encoding hazard seen in the real EXTERNAL_URI population."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "scratchpad/final_canonical_build/webqsp_v1")
from uri_render import render

CASES = [
    ("http://en.wikipedia.org/wiki/Rub%E9n_Piaggio", "Rubén", "latin-1 on a Latin wiki"),
    ("http://pt.wikipedia.org/wiki/Malm%F6_(comuna)", "Malmö", "latin-1, pt"),
    ("http://de.wikipedia.org/wiki/Wang_Can", "Wang Can", "plain ascii"),
    ("http://ru.wikipedia.org/wiki/%D0%9C%D1%83%D1%85%D0%BE%D0%BC%D0%BE%D1%80",
     "Мухомор", "genuine utf-8 on a Cyrillic wiki"),
    ("http://ru.wikipedia.org/wiki/%1C%43%45%3E%3C%3E%40_%3A%3E%40%3E%3B%35%32%41%3A%38%39",
     "Мухомор", "utf-16 low byte, has control chars"),
    ("http://ru.wikipedia.org/wiki/%24%4D%40-%25%35%39%32%35%3D_(%42%30%43%3D%48%38%3F,_%1C%38%3D%3D%35%41%3E%42%30)",
     "Фэр-Хейвен", "utf-16 low byte with literal punctuation"),
    ("http://uk.wikipedia.org/wiki/%25%3E%40%32%30%42%41%4C%3A%56",
     "Хорватські", "starts %25 but is NOT double-encoded"),
    ("http://bg.wikipedia.org/wiki/%10%3B%31%38%3D_%25%3E%34%37%30", "Албин", "bg"),
    ("http://sr.wikipedia.org/wiki/%1B%38%3D%3A%3E%3B%3D", "Линколн", "sr"),
    ("http://th.wikipedia.org/wiki/%14%2D%01%44%21%49%2A%33%2B%23%31%1A%2D%31%25%40%08%2D%19%2D%19",
     "ดอกไม้", "utf-16 low byte, Thai (+0xE00 not +0x400)"),
    ("http://fa.wikipedia.org/wiki/%25D9%25BE%25D8%25B1%25DA%2586%25D9%2585",
     "پرچم", "genuinely double-encoded"),
    ("http://en.wikipedia.org/wiki/index.html?curid=43414153", "article 43414153", "curid"),
    ("http://zh.wikipedia.org/wiki/index.html?curid=3318811", "article 3318811", "curid, zh"),
]

def bad(t):
    return (not t) or "�" in t or any(ord(c) < 0x20 for c in t)

if __name__ == "__main__":
    fails = 0
    for uri, want, why in CASES:
        name, kind = render(uri)
        ok = (want in name) and not bad(name)
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  [{kind:20s}] {name[:60]:60s}  {why}")
    print(f"\n{len(CASES) - fails}/{len(CASES)} correct")
    sys.exit(1 if fails else 0)
