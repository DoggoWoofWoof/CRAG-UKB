# -*- coding: utf-8 -*-
"""GENERATED_FLOOR names: one deterministic, evidence-based description for each of the 7,221,354
floor nodes (no original name, no cascade name, no URI/key rendering, no admissible inference).

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/floor_names.py

Every name carries the node's MID (or schema id), so it is unique, and it never reads like a label:
    "<Head> m.xxx (<what remains in the graph>)"
where <Head> is the most specific declared type ("Content Import", "Document", ...), "Freebase
object" for an untyped entity, or "<Type> mediator" / "Freebase mediator" for a CVT.  Rules, first
that fires (the rule id is stored per node):

    R0 SCHEMA_ID          a SCHEMA_* node: its humanised last segment + the schema id
    R1 MERGE_SUCCESSOR    dataworld.gardening_hint.replaced_by chain (<= 5 hops) ends at a named node
    R2 KEY_IDENTITY       a key that carries a decodable external identity (MusicBrainz NGS ids, a
                          Wikipedia article, an authority id, a USDA/NCES/IRS row id, a gx id, ...)
    R3 DESCRIPTION        a /common/topic/description remains (34 nodes): its first words
    R4 CONTEXT_OUT        an informative out-edge to a node that has a name (rel label: neighbour name)
    R5 CONTEXT_IN         an informative in-edge from a node that has a name (is the 'rel label' of subject)
    R6 KEY_OPAQUE         a key that is only an opaque Freeq variable / other key
    R7 CONTEXT_BOOKKEEPING only a bookkeeping edge (type.object.permission, ...) reaches a named node
    R8 CONTEXT_UNNAMED    the only neighbours are floor nodes themselves (rel label: their MID)
    R9 NO_REMAINING_FACTS nothing else in the universe mentions the node

"Informative" edges exclude the key relations (http://rdf.freebase.com/key/*, which are the keys
R2/R6 render), the type declaration (type.type.instance -- the type is already the Head) and the
bookkeeping relations.  Among informative edges the neighbour with an ACTUAL name (kind 0-2) is
preferred over an identifier (URI/key) over an inferred name, then the rarest relation in the
floor, then out-edges before in-edges, then the smallest neighbour name -- measured on
FLOOR_ANALYSIS.json (e.g. a Document keeps its own text before its source URI; a Content Import
keeps its URI before its blob).

Evidence: scratchpad/fb4/evidence/{floor_ids,floor_types,floor_keys,floor_desc,floor_edges}.parquet,
scratchpad/fb4/neighbour_names.parquet, scratchpad/fb4/schema_nodes.parquet.  Output:
scratchpad/fb4/floor_names.parquet (node_uid, name, rule, family) + scratchpad/fb4/FLOOR_NAMES.json.
"""
import collections
import io
import json
import os
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCR = "scratchpad/fb4"
FB = "data/final_canonical/freebase_v3"
EV = SCR + "/evidence"
KINDS = ["LITERAL", "EXTERNAL_URI", "ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER"]
MAX_NB = 80   # neighbour-name / key payload characters kept in a floor name
GENERIC_TYPES = {"common.topic", "type.object", "common.resource"}   # never the head when a specific type exists
BOOKKEEPING_RELS_LAST = {"type.object.permission", "type.object.id", "type.object.mid", "type.object.guid", "type.object.timestamp",
                         "type.object.creator", "type.object.attribution", "freebase.valuenotation.is_reviewed"}
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


# ---------------------------------------------------------------------------- text helpers
_ESC = re.compile(r"\$([0-9A-Fa-f]{4})")


def unescape_key(s):
    """Freebase key escaping: $XXXX is the code point in hex."""
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def humanise(seg):
    seg = seg.replace("_", " ").strip()
    if not seg:
        return seg
    if re.fullmatch(r"[0-9a-f]{8,}", seg) or re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", seg):
        return seg
    words = seg.split()
    return " ".join(w if (w.isupper() and len(w) > 1) else w[:1].upper() + w[1:] for w in words)


def clip(s, n=MAX_NB):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------- key identity renderers
MB_KINDS = {"recording_recording": "recording", "release_track_release_track": "release track", "artist_contribution": "artist contribution",
            "webpage_artist_webpage": "artist webpage", "webpage_recording_webpage": "recording webpage", "release_release": "release",
            "musical_group_membership_group_member": "group membership", "release_group_release_group": "release group", "artist_artist": "artist",
            "label_label": "label", "work_work": "work", "webpage_release_webpage": "release webpage", "webpage_label_webpage": "label webpage"}
_FREEQ = re.compile(r"^/dataworld/freeq/job_([0-9a-f-]{36})_var_(.*)$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def render_key(key):
    """(text, is_identity): identity keys name an external record the node was loaded from."""
    m = _FREEQ.match(key)
    if m:
        job, var = m.group(1), unescape_key(m.group(2))
        if var.startswith("mb_ngs:"):
            body = var[len("mb_ngs:"):]
            for k in sorted(MB_KINDS, key=len, reverse=True):
                for tag in ("_gid_", "_id_", "_cvt_id_"):
                    if body.startswith(k + tag):
                        ident = body[len(k) + len(tag):]
                        return "MusicBrainz %s %s" % (MB_KINDS[k], clip(ident, 100)), True
            return "MusicBrainz record %s" % clip(body, 100), True
        if var.startswith("en_wikipedia_org_wiki_"):
            return "Freeq import from en.wikipedia.org/wiki/%s" % clip(var[len("en_wikipedia_org_wiki_"):], 100), True
        mm = re.match(r"^(authority_usda|google_nces|organization_irs|organization_non_profit|business_business_operation)_(.*)$", var)
        if mm:
            return "Freeq import %s %s" % (mm.group(1).replace("_", " "), clip(mm.group(2), 80)), True
        if _HEX16.match(var):
            return "Freeq job %s variable %s" % (job[:8], var), False
        return "Freeq job %s variable '%s'" % (job[:8], clip(var, 80)), False
    if key.startswith("/authority/musicbrainz/"):
        return "MusicBrainz id %s" % clip(unescape_key(key[len("/authority/musicbrainz/"):]), 100), True
    m = re.match(r"^/authority/([^/]+)/(.*)$", key)
    if m:
        return "%s id %s" % (m.group(1), clip(unescape_key(m.group(2)), 100)), True
    m = re.match(r"^/user/jamie/gx/id/(\d+)$", key)
    if m:
        return "gx id %s" % m.group(1), True
    if key.startswith("/uri/"):
        return "URI %s" % clip(unescape_key(key[5:]), 100), True
    if key.startswith("/wikipedia/"):
        return "Wikipedia key %s" % clip(unescape_key(key[len("/wikipedia/"):]), 100), True
    return "key %s" % clip(unescape_key(key), 100), False


# ---------------------------------------------------------------------------- load evidence
floor = np.load(SCR + "/floor_uids.npy")
fkind = np.load(SCR + "/floor_kind.npy")
NF = len(floor)
log("floor", NF)


def rows_of(uids):
    r = np.searchsorted(floor, uids)
    assert np.array_equal(floor[r], uids)
    return r


ids = pq.read_table(EV + "/floor_ids.parquet")
node_id = np.empty(NF, dtype=object)
node_id[rows_of(ids["node_uid"].to_numpy())] = np.array(ids["node_id"].to_pylist(), dtype=object)
assert all(x is not None for x in node_id)

sn = pq.read_table(SCR + "/schema_nodes.parquet")
schema_label = {}
for nid, kd, dt in zip(sn["node_id"].to_pylist(), sn["kind"].to_pylist(), sn["display_text"].to_pylist()):
    if dt:
        schema_label[nid] = dt


def type_label(t):
    lab = schema_label.get(t)
    if lab and not lab.startswith("/") and lab.strip():
        return humanise(lab) if "_" in lab else lab
    return humanise(t.split(".")[-1])


def rel_label(r):
    lab = schema_label.get(r)
    if lab and lab.strip():
        return lab.replace("_", " ")
    return r.split(".")[-1].replace("_", " ")


# types per row: most specific = rarest type in the floor (ties: alphabetical); generic types only if alone
ty = pq.read_table(EV + "/floor_types.parquet")
trow = rows_of(ty["node_uid"].to_numpy())
tname = np.array(ty["type"].to_pylist(), dtype=object)
tfreq = collections.Counter(tname.tolist())
head_type = np.empty(NF, dtype=object)
o = np.argsort(trow, kind="stable")
starts = np.flatnonzero(np.r_[True, np.diff(trow[o]) > 0])
ends = np.r_[starts[1:], len(o)]
for s, e in zip(starts, ends):
    ts = sorted(tname[o[s:e]].tolist(), key=lambda t: (t in GENERIC_TYPES, tfreq[t], t))
    head_type[trow[o[s]]] = ts[0]
log("types on", int(sum(1 for x in head_type if x is not None)), "nodes")

# keys per row: best key = identity key first, then shortest
ke = pq.read_table(EV + "/floor_keys.parquet")
krow = rows_of(ke["node_uid"].to_numpy())
kkey = np.array(ke["key"].to_pylist(), dtype=object)
key_text = np.empty(NF, dtype=object)
key_is_identity = np.zeros(NF, dtype=bool)
o = np.argsort(krow, kind="stable")
starts = np.flatnonzero(np.r_[True, np.diff(krow[o]) > 0])
ends = np.r_[starts[1:], len(o)]
for s, e in zip(starts, ends):
    cands = sorted((render_key(k) for k in kkey[o[s:e]].tolist()), key=lambda ti: (not ti[1], len(ti[0]), ti[0]))
    key_text[krow[o[s]]], key_is_identity[krow[o[s]]] = cands[0]
log("keys on", int(sum(1 for x in key_text if x is not None)), "nodes; identity keys", int(key_is_identity.sum()))

# descriptions
de = pq.read_table(EV + "/floor_desc.parquet")
desc = {}
for u, lex in zip(de["node_uid"].to_numpy().tolist(), de["lexical"].to_pylist()):
    r = int(np.searchsorted(floor, u))
    if lex and (r not in desc or len(lex) > len(desc[r])):
        desc[r] = lex
log("descriptions on", len(desc), "nodes")

# neighbour names (non-floor endpoints of floor edges)
nb = pq.read_table(SCR + "/neighbour_names.parquet")
nb_uid = nb["node_uid"].to_numpy()
nb_name = np.array(nb["name"].to_pylist(), dtype=object)
nb_nk = nb["name_kind"].to_numpy()
assert np.all(np.diff(nb_uid) > 0)
log("neighbour names", len(nb_uid))

# edges, grouped by floor row (relations kept as rel_uid codes; names via the relations table)
rels_tbl = pq.read_table(FB + "/canonical/relations.parquet", columns=["rel_uid", "relation"])
rel_name = dict(zip(rels_tbl["rel_uid"].to_numpy().tolist(), rels_tbl["relation"].to_pylist()))
pf = pq.ParquetFile(EV + "/floor_edges.parquet")
E_row, E_dir, E_rel, E_other = [], [], [], []
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["node_uid", "rel", "other_uid", "direction"])
    E_row.append(rows_of(t["node_uid"].to_numpy()))
    E_dir.append(pc.equal(t["direction"], "out").to_numpy(zero_copy_only=False))
    E_rel.append(t["rel"].to_numpy())
    E_other.append(t["other_uid"].to_numpy())
E_row, E_dir, E_rel, E_other = (np.concatenate(x) for x in (E_row, E_dir, E_rel, E_other))
rel_freq = collections.Counter(E_rel.tolist())
KEY_PREFIX = "http://rdf.freebase.com/key/"
TYPE_DECL = "type.type.instance"


def rel_class(code):
    """0 informative, 1 bookkeeping (used only by R7), 2 never a context (key relations, the type declaration)."""
    name = rel_name[code]
    if name.startswith(KEY_PREFIX) or name == TYPE_DECL:
        return 2
    return 1 if name in BOOKKEEPING_RELS_LAST else 0


E_class = np.fromiter((rel_class(int(c)) for c in E_rel), dtype=np.int8, count=len(E_rel))
o = np.argsort(E_row, kind="stable")
E_row, E_dir, E_rel, E_other = E_row[o], E_dir[o], E_rel[o], E_other[o]
# neighbour lookup for every edge row
p = np.searchsorted(nb_uid, E_other)
p[p >= len(nb_uid)] = 0
E_named = nb_uid[p] == E_other
E_nbidx = np.where(E_named, p, -1)
E_other_floor = ~E_named
fp = np.searchsorted(floor, E_other)
fp[fp >= NF] = 0
assert np.array_equal(floor[fp] == E_other, E_other_floor), "an edge endpoint is neither a floor node nor a named neighbour"
starts = np.flatnonzero(np.r_[True, np.diff(E_row) > 0]) if len(E_row) else np.zeros(0, dtype=np.int64)
ends = np.r_[starts[1:], len(E_row)] if len(E_row) else starts
edge_start = np.full(NF, -1, dtype=np.int64)
edge_end = np.full(NF, -1, dtype=np.int64)
edge_start[E_row[starts]] = starts
edge_end[E_row[starts]] = ends
log("edges", len(E_row), "on", len(starts), "nodes")

# replaced_by successor per floor row (out-edge)
REPL = "dataworld.gardening_hint.replaced_by"
REPL_CODE = next((c for c, n in rel_name.items() if n == REPL), -1)
succ = {}
m = E_dir & (E_rel == REPL_CODE)
for r, other, named, nbi in zip(E_row[m].tolist(), E_other[m].tolist(), E_named[m].tolist(), E_nbidx[m].tolist()):
    succ.setdefault(r, (other, named, nbi))


def merge_successor(r):
    """Follow replaced_by through floor nodes (<= 5 hops); the name of the first named node, or None."""
    seen = {r}
    for _ in range(5):
        s = succ.get(r)
        if s is None:
            return None
        other, named, nbi = s
        if named:
            return nb_name[nbi]
        rr = int(np.searchsorted(floor, other))
        if rr in seen:
            return None
        seen.add(rr)
        r = rr
    return None


NB_CLASS = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3}   # actual < identifier < inferred (< floor, never named here)


def best_context(r, want_named, cls):
    """The most telling (relation, neighbour) among the node's edges of relation class `cls`: a neighbour with an
    ACTUAL name before an identifier before an inferred name, then the rarest relation in the floor, then out-edges
    before in-edges, then the smallest neighbour name."""
    a, b = edge_start[r], edge_end[r]
    if a < 0:
        return None
    best = None
    for j in range(a, b):
        if E_named[j] != want_named or E_class[j] != cls:
            continue
        rel = E_rel[j]
        if want_named:
            nbi = E_nbidx[j]
            key = (NB_CLASS[int(nb_nk[nbi])], rel_freq[rel], rel_name[rel], not E_dir[j], nb_name[nbi])
            val = (bool(E_dir[j]), rel_name[rel], nb_name[nbi])
        else:
            fr = int(np.searchsorted(floor, E_other[j]))
            key = (0, rel_freq[rel], rel_name[rel], not E_dir[j], node_id[fr])
            val = (bool(E_dir[j]), rel_name[rel], node_id[fr])
        if best is None or key < best[0]:
            best = (key, val)
    return None if best is None else best[1]


def head(r):
    k = KINDS[fkind[r]]
    t = head_type[r]
    if k == "CVT_MEDIATOR":
        return ("%s mediator" % type_label(t)) if t else "Freebase mediator"
    return type_label(t) if t else "Freebase object"


def context_text(is_out, rel, other_name):
    lab = rel_label(rel)
    return "%s: %s" % (lab, clip(other_name)) if is_out else "is the '%s' of %s" % (lab, clip(other_name))


# ---------------------------------------------------------------------------- the rules
names = np.empty(NF, dtype=object)
rules = np.empty(NF, dtype=object)
families = np.empty(NF, dtype=object)
samples = collections.defaultdict(list)
for r in range(NF):
    k = KINDS[fkind[r]]
    nid = node_id[r]
    if k.startswith("SCHEMA_"):
        what = {"SCHEMA_TYPE": "type", "SCHEMA_PROPERTY": "property", "SCHEMA_OTHER": "object"}[k]
        seg = nid.split(".")[-1]
        nm = "%s (Freebase schema %s %s)" % (humanise(seg), what, nid) if re.fullmatch(r"[A-Za-z][A-Za-z_]*", seg) else "Freebase schema %s %s" % (what, nid)
        rule, fam = "R0_SCHEMA_ID", "SCHEMA"
    else:
        h = head(r)
        s = merge_successor(r)
        if s is not None:
            nm = "%s (merged Freebase duplicate %s)" % (clip(s), nid)
            rule, fam = "R1_MERGE_SUCCESSOR", "MERGE"
        elif key_text[r] is not None and key_is_identity[r]:
            nm = "%s %s (%s)" % (h, nid, key_text[r])
            rule, fam = "R2_KEY_IDENTITY", "KEY"
        elif r in desc:
            nm = "%s %s (described as: %s)" % (h, nid, clip(desc[r], 100))
            rule, fam = "R3_DESCRIPTION", "DESCRIPTION"
        else:
            c = best_context(r, True, 0)
            if c is not None:
                nm = "%s %s (%s)" % (h, nid, context_text(*c))
                rule, fam = ("R4_CONTEXT_OUT" if c[0] else "R5_CONTEXT_IN"), "CONTEXT"
            elif key_text[r] is not None:
                nm = "%s %s (%s)" % (h, nid, key_text[r])
                rule, fam = "R6_KEY_OPAQUE", "KEY"
            else:
                c = best_context(r, True, 1)
                if c is not None:
                    nm = "%s %s (%s)" % (h, nid, context_text(*c))
                    rule, fam = "R7_CONTEXT_BOOKKEEPING", "CONTEXT"
                else:
                    c = best_context(r, False, 0) or best_context(r, False, 1)
                    if c is not None:
                        nm = "%s %s (%s)" % (h, nid, context_text(c[0], c[1], "Freebase object " + c[2]))
                        rule, fam = "R8_CONTEXT_UNNAMED", "CONTEXT"
                    else:
                        nm = "%s %s (no remaining facts)" % (h, nid)
                        rule, fam = "R9_NO_REMAINING_FACTS", "BARE"
    names[r], rules[r], families[r] = nm, rule, fam
    if len(samples[rule]) < 12 and (r % 97 == 0 or len(samples[rule]) < 3):
        samples[rule].append(nm)
    if r % 1_000_000 == 0:
        log("named", r)

# ---------------------------------------------------------------------------- checks + output
assert all(x for x in names)
assert all("Unnamed" not in x[:8] and not re.match(r"^[mg]\.[0-9a-z_]+$", x) for x in names)
uniq = len(set(names.tolist()))
rule_census = collections.Counter(rules.tolist())
fam_census = collections.Counter(families.tolist())
tbl = pa.table({"node_uid": pa.array(floor), "name": pa.array(names.tolist(), pa.string()), "rule": pa.array(rules.tolist(), pa.string()),
                "family": pa.array(families.tolist(), pa.string())})
pq.write_table(tbl, SCR + "/floor_names.parquet", compression="zstd")
by_kind_rule = collections.Counter("%s|%s" % (KINDS[k], ru) for k, ru in zip(fkind.tolist(), rules.tolist()))
rec = {
    "RECORD": "FLOOR_NAMES", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "floor_nodes": NF, "names_written": int(tbl.num_rows),
    "rules": {
        "R0_SCHEMA_ID": "SCHEMA_* node: humanised last segment + (Freebase schema <type|property|object> <id>)",
        "R1_MERGE_SUCCESSOR": "dataworld.gardening_hint.replaced_by chain (<= 5 hops) reaches a named node: '<successor name> (merged Freebase duplicate m.x)'",
        "R2_KEY_IDENTITY": "a key carrying a decodable external identity (MusicBrainz NGS id in a Freeq variable, /authority/*, a Wikipedia article import, "
                           "USDA/NCES/IRS/business row ids, /user/jamie/gx ids, /uri/*, /wikipedia/*): '<Head> m.x (<identity>)'",
        "R3_DESCRIPTION": "a /common/topic/description remains: '<Head> m.x (described as: <first 100 chars>)'",
        "R4_CONTEXT_OUT": "an informative out-edge to a node with a name: '<Head> m.x (<relation label>: <neighbour name>)' -- neighbour with an "
                          "ACTUAL name before an identifier before an inferred name, then the rarest relation in the floor, then alphabetical",
        "R5_CONTEXT_IN": "an informative in-edge from a named node ranks above any informative out-edge only by the same key: "
                         "'<Head> m.x (is the '<relation label>' of <subject name>)'",
        "R6_KEY_OPAQUE": "only an opaque key (Freeq job + hex variable, other): '<Head> m.x (Freeq job <8 hex> variable <var>)'",
        "R7_CONTEXT_BOOKKEEPING": "the only named neighbour is reached by a bookkeeping relation (type.object.permission, id, mid, guid, timestamp, "
                                  "creator, attribution, is_reviewed): the context form over that edge",
        "R8_CONTEXT_UNNAMED": "every neighbour is itself a floor node: the same context form with the neighbour's MID",
        "R9_NO_REMAINING_FACTS": "no edge, key or description in the universe: '<Head> m.x (no remaining facts)'",
        "excluded_as_context": "key relations (http://rdf.freebase.com/key/*: rendered by R2/R6 instead) and type.type.instance (the Head already "
                               "states the type)",
        "head": "the most specific declared type (rarest in the floor; common.topic/type.object/common.resource only when alone), "
                "'Freebase object' when untyped, '<Type> mediator' / 'Freebase mediator' for a CVT",
        "labels": "type and relation labels from the SCHEMA_* nodes' own display_text (scratchpad/fb4/schema_nodes.parquet), fallback: the id's last segment humanised",
        "clipping": "neighbour names and payloads are clipped to %d characters" % MAX_NB,
    },
    "by_rule": dict(rule_census.most_common()), "by_family": dict(fam_census.most_common()), "by_kind_rule": dict(by_kind_rule.most_common()),
    "evidence": {"floor_edges_rows": int(len(E_row)), "nodes_with_edges": int(len(starts)), "nodes_with_types": int(sum(1 for x in head_type if x is not None)),
                 "nodes_with_keys": int(sum(1 for x in key_text if x is not None)), "identity_keys": int(key_is_identity.sum()),
                 "nodes_with_description": len(desc), "named_neighbours": int(len(nb_uid)), "relations_seen": len(rel_freq),
                 "top_relations": [(rel_name[c], n) for c, n in rel_freq.most_common(25)]},
    "uniqueness": {"distinct_names": uniq, "all_unique": uniq == NF},
    "samples": {k: v for k, v in sorted(samples.items())},
    "seconds": round(time.time() - t0, 1),
}
json.dump(rec, open(SCR + "/FLOOR_NAMES.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("done", json.dumps(rec["by_rule"]), "unique", uniq == NF)
