# -*- coding: utf-8 -*-
"""Layer-wide census of the RELATION HEAD each generated name is built on. BOUNDED MEMORY.

Why this supersedes head_census.py: a probe showed the head of a generated name is NOT drawn from a
small relation vocabulary in general. On shard 0, MEDIATOR_ROLE_NAME produced 1,089,561 distinct
heads over 1,621,502 rows and MERGE_SUCCESSOR_NAME 40,935 over 47,408, because for those rules the
head IS the neighbour title, which is free text and correct behaviour, not a defect. Counting every
head exactly across 64,038,024 rows is what OOM-killed rule_head_audit.py at shard 7 of 32.

So the census separates what the decision needs from what is merely diagnostic:

  EXACT, bounded by construction
    - rows per source
    - rows whose head is in the fixed BOOK_HEAD bookkeeping vocabulary
    - rows whose name is an ungrammatical preposition collision
  These are the numbers the admissibility rule is allowed to be built on.

  SKETCHED, with a stated error bound
    - the top heads per source, via Misra-Gries with K counters. Every head whose true frequency
      exceeds rows/K is guaranteed retained, and every reported count understates the true count by
      at most the reported max_undercount. This is used only to check whether the fixed bookkeeping
      vocabulary missed a high-frequency bookkeeping head, a heavy-hitter question that does not
      need exact tail counts.

The head is the text before the first template boundary, so only the leading relation phrase is
counted and never the neighbour title. A naive preposition-collision regex on the probe shard
flagged 41,285 rows of which 74 were real films titled e.g. Of Love and Shadows; the grammar test
here therefore fires only when the phrase before " of " ITSELF ends in a bookkeeping relation.
"""
import os, sys, io, json, glob, time, re
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
from collections import Counter

INF = "data/final_canonical/freebase_v3/inference_overlay_v1"
FILES = sorted(glob.glob(INF + "/*.parquet"))
K = 300000
EXACT_CAP = 400000

BOOK_HEAD = {"last referenced by", "freeq", "permission", "permissions", "creator", "created by",
             "timestamp", "id", "guid", "key", "namespace", "attribution", "provenance",
             "last referenced", "referenced by", "write permission", "read permission"}
SPLIT = re.compile(r" of | — | in | on | for | as | about | at ")
GRAM = re.compile(r"^(.*\b(?:by|of|in|to|for|with|from|at|on|as|into|over|under))\s+of\s")


class MG(object):
    """Misra-Gries heavy hitters, weighted updates, explicit undercount bound."""

    def __init__(self, k):
        self.k = k
        self.c = {}
        self.undercount = 0
        self.exact = True

    def add(self, key, n):
        c = self.c
        if key in c:
            c[key] += n
            return
        c[key] = n
        if len(c) > self.k:
            self.exact = False
            vals = np.fromiter(c.values(), dtype=np.int64, count=len(c))
            keep = self.k // 2
            cut = int(np.partition(vals, len(vals) - keep)[len(vals) - keep])
            self.undercount += cut
            self.c = {kk: vv - cut for kk, vv in c.items() if vv > cut}


t0 = time.time()
hh, rows_by_src, book_rows, grammar = {}, Counter(), Counter(), Counter()
book_ex, head_seen, head_overflow = {}, {}, set()

for i, f in enumerate(FILES):
    t = pq.read_table(f, columns=["inferred_name", "inference_source"])
    nm = t["inferred_name"].combine_chunks()
    src = pc.cast(t["inference_source"], "string").combine_chunks().to_numpy(zero_copy_only=False)
    dd = nm.dictionary_encode()
    dic = dd.dictionary.to_pylist()
    idx = dd.indices.to_numpy(zero_copy_only=False)
    hd = np.empty(len(dic), object)
    bk = np.zeros(len(dic), bool)
    gr = np.zeros(len(dic), bool)
    for j, v in enumerate(dic):
        v = v or ""
        m = SPLIT.search(v)
        h = v[:m.start()] if m else v
        hd[j] = h
        bk[j] = h.strip().lower() in BOOK_HEAD
        g = GRAM.match(v)
        gr[j] = bool(g) and (g.group(1).strip().lower() in BOOK_HEAD)
    for s in np.unique(src):
        sm = src == s
        rows_by_src[s] += int(sm.sum())
        u, c = np.unique(idx[sm], return_counts=True)
        H = hh.setdefault(s, MG(K))
        S = head_seen.setdefault(s, set())
        for kk, n in zip(u.tolist(), c.tolist()):
            H.add(hd[kk], int(n))
            if s not in head_overflow:
                S.add(hd[kk])
                if len(S) > EXACT_CAP:
                    head_overflow.add(s)
                    head_seen[s] = set()
            if bk[kk]:
                book_rows[s] += int(n)
                book_ex.setdefault(s, [])
                if len(book_ex[s]) < 8:
                    book_ex[s].append(dic[kk])
            if gr[kk]:
                grammar[s] += int(n)
    del t, nm, dd, idx, hd
    print("  [%d/%d] rows=%s bookkeeping=%s  %.0fs"
          % (i + 1, len(FILES), format(sum(rows_by_src.values()), ","),
             format(sum(book_rows.values()), ","), time.time() - t0), flush=True)

TOT = sum(rows_by_src.values())
BK = sum(book_rows.values())
per = {}
for s, H in hh.items():
    n = rows_by_src[s]
    top = sorted(H.c.items(), key=lambda kv: -kv[1])[:25]
    per[s] = {"rows": n,
              "bookkeeping_head_rows": int(book_rows.get(s, 0)),
              "bookkeeping_head_pct_of_source": round(100.0 * book_rows.get(s, 0) / n, 4),
              "ungrammatical_preposition_collision_rows": int(grammar.get(s, 0)),
              "distinct_heads": (len(head_seen.get(s, ())) if s not in head_overflow
                                 else ">%d" % EXACT_CAP),
              "head_counts_are_exact": bool(H.exact),
              "max_undercount": int(H.undercount),
              "guaranteed_retained_above_rows": int(n / H.k) + 1,
              "top_heads": [{"rows_at_least": v, "pct_at_least": round(100.0 * v / n, 4),
                             "head": k} for k, v in top],
              "bookkeeping_examples": book_ex.get(s, [])}
out = {"RECORD": "INFERENCE_RELATION_HEAD_CENSUS_V2",
       "supersedes": "head_census.py (unbounded head counter, the OOM shape that killed "
                     "rule_head_audit.py at shard 7/32)",
       "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "why": ("A generated name headed by a provenance, access-control or pseudo-key relation "
               "names the RECORD, not the node. The frozen floor already makes such nodes "
               "readable, so keeping these costs the layer meaning without buying coverage."),
       "exactness": {"EXACT": ["rows", "bookkeeping_head_rows",
                               "ungrammatical_preposition_collision_rows"],
                     "SKETCHED_MISRA_GRIES": ["top_heads"], "K": K,
                     "reading": "top_heads counts are lower bounds; true <= value+max_undercount"},
       "TOTAL_ROWS": TOT, "BOOKKEEPING_HEAD_ROWS": BK,
       "BOOKKEEPING_HEAD_PCT_OF_LAYER": round(100.0 * BK / TOT, 4),
       "UNGRAMMATICAL_PREPOSITION_COLLISION_ROWS": int(sum(grammar.values())),
       "BOOKKEEPING_HEAD_VOCABULARY": sorted(BOOK_HEAD),
       "BY_SOURCE": per, "elapsed_s": round(time.time() - t0, 1)}
json.dump(out, io.open("scratchpad/fb/head_census.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("\nTOTAL rows %s   bookkeeping-headed %s (%.4f%%)   ungrammatical %s"
      % (format(TOT, ","), format(BK, ","), 100.0 * BK / TOT, format(sum(grammar.values()), ",")))
for s, d in sorted(per.items(), key=lambda kv: -kv[1]["rows"]):
    print("\n== %s  rows=%s heads=%s exact=%s undercount<=%s  bookkeeping=%s (%.3f%%)  ungram=%s"
          % (s, format(d["rows"], ","), d["distinct_heads"], d["head_counts_are_exact"],
             format(d["max_undercount"], ","), format(d["bookkeeping_head_rows"], ","),
             d["bookkeeping_head_pct_of_source"],
             format(d["ungrammatical_preposition_collision_rows"], ",")))
    for e in d["top_heads"][:12]:
        print("   %10s %7.3f%%  %s"
              % (format(e["rows_at_least"], ","), e["pct_at_least"], (e["head"] or "")[:72]))
print("\nwrote scratchpad/fb/head_census.json")
