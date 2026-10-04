"""EXTERNAL_AUTHORITY_EXACT, route 3: the issuing authorities' own identifier endpoints.

    PYTHONHASHSEED=0 python .../authority_native.py <stage>

    stages:  mediawiki   1,622 Wikipedia/Commons page ids   (batched 50/request, minutes)
             musicbrainz 3,186 MBIDs                        (1 request/second, ~55 minutes)
             small       the long tail: stanford, thetvdb, openlibrary, discogs, ...

WHY THE STAGES ARE SEPARATE
  They have incomparable cost. MediaWiki resolves 50 ids per request and finishes in minutes;
  MusicBrainz asks clients to stay under one request per second, so the same number of ids takes
  the best part of an hour. Splitting them keeps the cheap recoveries from waiting behind the
  expensive ones, and lets the slow one run unattended without blocking anything.

STILL EXACT LOOKUP, STILL NO NAME MATCHING
  Every request here is keyed by the identifier Freebase stored. MediaWiki is queried by pageids=,
  MusicBrainz by /ws/2/<entity>/<mbid>. Nothing is searched for by name, and a 404 is recorded as a
  404 rather than retried as a search.

MUSICBRAINZ ENTITY TYPES COME FROM OUR OWN GRAPH, NOT FROM GUESSING
  2,415 of the MBIDs are stored bare, with no entity type in the key, and /ws/2 needs the type in
  the path. Rather than trying six endpoints per id -- which would multiply an already hour-long
  job by six -- the node's Freebase types are read from the frozen graph and mapped to the
  MusicBrainz entity type. A node typed /music/artist is looked up as an artist. Where the graph
  gives no usable type the endpoints are tried in descending order of observed frequency, and the
  order tried is recorded.

Writes only under _acquisition/. Touches neither the frozen graph nor the frozen overlay.
"""
import sys, io, os, re, json, time, urllib.request, urllib.parse, urllib.error, collections, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "mediawiki"
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

t = pq.read_table(f"{ACQ}/authority_pending_native.parquet")
PEND = list(zip(t["node_uid"].to_pylist(), t["freebase_mid"].to_pylist(),
                t["authority_namespace"].combine_chunks().cast(pa.string()).to_pylist(),
                t["authority_id"].to_pylist(), t["key"].to_pylist()))
print(f"pending rows: {len(PEND):,}", flush=True)


def get(url, attempts=4, accept="application/json"):
    for a in range(attempts):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(r, timeout=60) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None                    # a miss is a miss; do not fall back to searching
            if a == attempts - 1:
                return None
            time.sleep(2 ** a)
        except Exception:
            if a == attempts - 1:
                return None
            time.sleep(2 ** a)
    return None


rows, miss = [], collections.Counter()
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---------------------------------------------------------------- mediawiki: pageids, batched 50
if STAGE == "mediawiki":
    # /wikipedia/images/commons_id/660211 -> commons.wikimedia.org page 660211
    # /wikipedia/images/en_id/6351310     -> en.wikipedia.org  page 6351310
    # /wikipedia/en_id/46384040           -> en.wikipedia.org  article page id
    HOST = {"commons": "commons.wikimedia.org"}
    byhost = collections.defaultdict(list)
    for u, mid, ns, aid, key in PEND:
        if ns != "mediawiki":
            continue
        p = key.split("/")            # key here is the DECODED key string
        if "images" in p:
            sub = p[3] if len(p) > 3 else ""
            host = HOST["commons"] if sub.startswith("commons") else "en.wikipedia.org"
            pid = p[-1]
        else:
            lang = ns.replace("_id", "") if ns.endswith("_id") else "en"
            host = f"{lang}.wikipedia.org"
            pid = p[-1]
        if pid.isdigit():
            byhost[host].append((u, mid, pid, key))
        else:
            miss["not_a_pageid"] += 1
    for host, items in sorted(byhost.items()):
        uniq = sorted({i[2] for i in items}, key=int)
        got = {}
        for c in range(0, len(uniq), 50):
            ch = uniq[c:c + 50]
            url = (f"https://{host}/w/api.php?action=query&format=json&prop=info"
                   f"&pageids={'|'.join(ch)}")
            d = get(url)
            pages = ((d or {}).get("query") or {}).get("pages") or {}
            for pid, pg in pages.items():
                if "missing" in pg or "invalid" in pg:
                    continue
                if pg.get("title"):
                    got[str(pid)] = pg["title"]
            print(f"  {host}: {min(c+50, len(uniq)):,}/{len(uniq):,} resolved={len(got):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            time.sleep(0.15)
        for u, mid, pid, key in items:
            if pid in got:
                title = got[pid]
                rows.append((u, mid, f"mediawiki:{host}", pid, title,
                             "wikimedia-file" if title.startswith(("File:", "Image:"))
                             else "wikipedia-article",
                             f"https://{host}/?curid={pid}", now()))
            else:
                miss[f"gone:{host}"] += 1

# ---------------------------------------------------------------- musicbrainz: 1 request/second
elif STAGE == "musicbrainz":
    MB2FB = [("/music/artist", "artist"), ("/music/musical_group", "artist"),
             ("/music/recording", "recording"), ("/music/track", "recording"),
             ("/music/release", "release"), ("/music/album", "release-group"),
             ("/music/composition", "work"), ("/music/record_label", "label"),
             ("/music/label", "label")]
    ORDER = ["artist", "recording", "work", "release-group", "release", "label"]
    items = [(u, mid, aid, key) for (u, mid, ns, aid, key) in PEND if ns == "musicbrainz"]
    # entity type from the key where it is present, else from the node's Freebase types
    typed = {}
    for u, mid, aid, key in items:
        p = key.split("/")
        if len(p) >= 5 and p[3] in ("artist", "recording", "work", "release", "release_group",
                                    "release-group", "label"):
            typed[(u, aid)] = p[3].replace("_", "-")
    need = {u for (u, mid, aid, key) in items if (u, aid) not in typed}
    fbtype = collections.defaultdict(set)
    if need:
        for fp in sorted(glob.glob(f"{ACQ}/residue_types/*.parquet")):
            tt = pq.read_table(fp, columns=["node_uid", "type"])
            for a, b in zip(tt["node_uid"].to_pylist(), tt["type"].to_pylist()):
                if a in need:
                    fbtype[a].add(b)
        print(f"  freebase types found for {len(fbtype):,} of {len(need):,} untyped MBIDs",
              flush=True)
    class _NoRedir(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            raise urllib.error.HTTPError(req.full_url, code, newurl, hdrs, fp)
    _op = urllib.request.build_opener(_NoRedir)

    def mbid_type(mbid):
        """The entity type from MusicBrainz's own redirect. Still an exact id lookup: the server is
        told the MBID and answers with where that MBID lives. Nothing is searched for."""
        try:
            r = urllib.request.Request(f"https://musicbrainz.org/mbid/{mbid}",
                                       headers={"User-Agent": UA})
            _op.open(r, timeout=30)
            return None                       # a 200 here means no redirect happened
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                seg = str(e.reason).rstrip("/").split("/")
                return seg[-2] if len(seg) >= 2 and seg[-1] == mbid else None
            return None
        except Exception:
            return None
        finally:
            time.sleep(1.05)

    order_used = collections.Counter()
    seen = {}
    for n, (u, mid, aid, key) in enumerate(items):
        if aid in seen:                       # already looked up under another node; spend nothing
            v = seen[aid]
            if v:
                rows.append((u, mid, "musicbrainz", aid, v[0], v[1],
                             f"https://musicbrainz.org/{v[1]}/{aid}", now()))
            continue
        et = typed.get((u, aid))
        if not et:
            ts = fbtype.get(u, set())
            for fb, mb in MB2FB:
                if any(x == fb or x.startswith(fb + "/") for x in ts):
                    et = mb
                    break
        src = "from_key_or_graph" if et else ""
        if not et:
            # https://musicbrainz.org/mbid/<uuid> 302s to /<entity>/<uuid>. One request buys the
            # entity type exactly, instead of walking six /ws/2 endpoints until one answers.
            et = mbid_type(aid)
            src = "from_mbid_redirect" if et else "fallback_order"
        tries = [et] if et else ORDER
        order_used[src] += 1
        hit = None
        for et2 in tries:
            d = get(f"https://musicbrainz.org/ws/2/{et2}/{aid}?fmt=json")
            time.sleep(1.05)                    # MusicBrainz asks for <= 1 request/second
            if d and (d.get("name") or d.get("title")):
                hit = (d.get("name") or d.get("title"), et2)
                break
        seen[aid] = hit
        if hit:
            rows.append((u, mid, "musicbrainz", aid, hit[0], hit[1],
                         f"https://musicbrainz.org/{hit[1]}/{aid}", now()))
        else:
            miss["musicbrainz_404"] += 1
        if n % 100 == 0:
            print(f"  {n:,}/{len(items):,} resolved={len(rows):,} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  entity type source: {dict(order_used)}", flush=True)

# ---------------------------------------------------------------- the long tail
# EVERY ROUTE BELOW WAS PROBED BEFORE IT WAS WRITTEN, and the namespaces that are missing are
# missing because the probe showed there is no open identifier endpoint. What the probes returned:
#   searchworks.stanford.edu  /view/<id> is a JavaScript shell with no <title>; /view/<id>.json and
#                             the Blacklight /catalog/<id>.json both 404. No API. 106 rows.
#   openlibrary               545 of the 549 ids are pre-2009 legacy "thing" keys of the form
#                             <slug>_9202a8c04000641f8............ , not OLIDs. /b/<key>.json 404s
#                             and archive.org has no item of that name, so the OLID endpoints
#                             cannot accept them. The 4 genuine OLIDs are looked up.
#   giantbomb                 403 without an API key. 22 rows.
#   thetvdb                   /dereferrer/season/<id> DOES resolve, but every one of five sampled
#                             ids landed on .../seasons/official/0 with the title "<Series> -
#                             Unknown - Specials". The series is recovered exactly; the season
#                             number is not. Naming a season after its series would be wrong, so
#                             the series is recorded as a resolved external identity and no name is
#                             emitted -- the same ruling as the truncated freeq strings.
#   netflix, tvrage           the APIs are gone and the Wayback CDX has no capture of these URLs.
# Rows for authorities that are dead but ARE archived (imdb, google_plus, myspace) are written to
# authority_pending_archive.parquet instead of being guessed at here; they belong to the archive
# tier, not to EXTERNAL_AUTHORITY_EXACT, and mixing them into this file would misdescribe them.
elif STAGE == "small":
    DISCOGS = {"release": "releases", "master": "masters", "artist": "artists", "label": "labels"}
    ISFDB = {"title_id": "title.cgi", "pub_id": "pl.cgi", "series_id": "pe.cgi",
             "author_id": "ea.cgi"}
    ARCHIVE_URL = {"imdb": lambda k: "https://www.imdb.com/" + "/".join(k.split("/")[3:]) + "/",
                   "google": lambda k: "https://plus.google.com/" + k.split("/")[-1],
                   "myspace": lambda k: "http://www.myspace.com/" + k.split("/")[-1]}
    OG = re.compile(r'<meta[^>]+property=.og:title.[^>]+content=.(.*?).>', re.I)
    TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
    pend_arch = []

    def html(url, timeout=25):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=timeout) as f:
                return f.read(300000).decode("utf-8", "replace"), f.geturl()
        except Exception:
            return "", ""

    for u, mid, ns, aid, key in PEND:
        seg = key.split("/")
        if ns == "discogs":
            kind = DISCOGS.get(seg[3] if len(seg) > 3 else "")
            d = get("https://api.discogs.com/%s/%s" % (kind, aid)) if kind else None
            time.sleep(2.6)                     # Discogs asks for well under 1 request/second
            lab = (d or {}).get("title") or (d or {}).get("name")
            if lab:
                rows.append((u, mid, "discogs", aid, lab, seg[3],
                             "https://api.discogs.com/%s/%s" % (kind, aid), now()))
            else:
                miss["discogs_404"] += 1
        elif ns == "us":
            # /authority/us/gov/loc/na/n83162830 -- a Library of Congress name authority record
            d = get("https://id.loc.gov/authorities/names/%s.json" % aid)
            time.sleep(0.4)
            lab = None
            for node in (d or []):
                v = node.get("http://www.loc.gov/mads/rdf/v1#authoritativeLabel")
                if v and node.get("@id", "").endswith(aid):
                    lab = v[0].get("@value")
                    break
            if lab:
                rows.append((u, mid, "loc-name-authority", aid, lab, "authority-record",
                             "https://id.loc.gov/authorities/names/%s" % aid, now()))
            else:
                miss["loc_no_authoritative_label"] += 1
        elif ns == "openlibrary":
            m = re.fullmatch(r"OL\d+([AMW])", aid)
            if not m:
                miss["openlibrary_legacy_thing_key_no_endpoint"] += 1
                continue
            path = {"A": "authors", "M": "books", "W": "works"}[m.group(1)]
            d = get("https://openlibrary.org/%s/%s.json" % (path, aid))
            time.sleep(0.4)
            lab = (d or {}).get("title") or (d or {}).get("name")
            if lab:
                rows.append((u, mid, "openlibrary", aid, lab, path,
                             "https://openlibrary.org/%s/%s" % (path, aid), now()))
            else:
                miss["openlibrary_olid_404"] += 1
        elif ns == "isfdb":
            cgi = ISFDB.get(seg[3] if len(seg) > 3 else "")
            body, _ = html("https://www.isfdb.org/cgi-bin/%s?%s" % (cgi, aid)) if cgi else ("", "")
            time.sleep(0.6)
            t = TITLE.search(body)
            ttl = " ".join(t.group(1).split()) if t else ""
            if ttl and ttl.lower() != "error":
                rows.append((u, mid, "isfdb", aid, ttl, seg[3],
                             "https://www.isfdb.org/cgi-bin/%s?%s" % (cgi, aid), now()))
            else:
                miss["isfdb_record_gone"] += 1
        elif ns == "rovi":
            # /authority/rovi/music/album/mw0001446900 -- Rovi ids are AllMusic ids
            body, _ = html("https://www.allmusic.com/album/%s" % aid)
            time.sleep(1.0)
            t = OG.search(body)
            if t and t.group(1).strip():
                rows.append((u, mid, "allmusic", aid, t.group(1).strip(),
                             seg[4] if len(seg) > 4 else "album",
                             "https://www.allmusic.com/album/%s" % aid, now()))
            else:
                miss["allmusic_blocked_or_gone"] += 1
        elif ns == "thetvdb":
            body, url = html("https://www.thetvdb.com/dereferrer/season/%s" % aid)
            time.sleep(0.8)
            if re.search(r"/series/([^/]+)/", url or ""):
                miss["thetvdb_series_resolved_season_number_not"] += 1
            else:
                miss["thetvdb_dereferrer_failed"] += 1
        elif ns in ARCHIVE_URL:
            pend_arch.append((u, mid, ns, aid, ARCHIVE_URL[ns](key)))
            miss["deferred_to_archive_tier:%s" % ns] += 1
        else:
            miss["no_open_id_endpoint:%s" % ns] += 1

    if pend_arch:
        pend_arch.sort()
        pq.write_table(pa.table({
            "node_uid": pa.array([r[0] for r in pend_arch], pa.int64()),
            "freebase_mid": pa.array([r[1] for r in pend_arch]),
            "authority_namespace": pa.array([r[2] for r in pend_arch]).dictionary_encode(),
            "authority_id": pa.array([r[3] for r in pend_arch]),
            "url": pa.array([r[4] for r in pend_arch])}),
            "%s/authority_pending_archive.parquet" % ACQ, compression="zstd")

else:
    sys.exit(f"unknown stage {STAGE!r}")

if rows:
    rows.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in rows]),
        "authority_namespace": pa.array([r[2] for r in rows]).dictionary_encode(),
        "authority_id": pa.array([r[3] for r in rows]),
        "resolved_label": pa.array([r[4] for r in rows]),
        "resolved_entity_type": pa.array([r[5] for r in rows]),
        "source_url": pa.array([r[6] for r in rows]),
        "retrieved_at": pa.array([r[7] for r in rows])}),
        f"{ACQ}/external_authority_{STAGE}.parquet", compression="zstd")

rec = {"schema": f"EXTERNAL_AUTHORITY_NATIVE_{STAGE.upper()}/v1",
       "generated_utc": now(),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "RULE": "exact identifier lookup only; a 404 is recorded, never retried as a name search",
       "stage": STAGE,
       "resolved_rows": len(rows),
       "resolved_nodes": len({r[0] for r in rows}),
       "misses": dict(miss),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_EXTERNAL_AUTHORITY_{STAGE.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{STAGE}: resolved {len(rows):,} rows over {len({r[0] for r in rows}):,} nodes")
print(f"  misses: {dict(miss)}")
for r in rows[:12]:
    print(f"   {r[1]:<14} {r[2]:<26} {r[3][:20]:<22} {r[4][:46]!r}")
print(f"{time.time()-t0:.0f}s")
