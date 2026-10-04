"""
POINTER INDEX BUILDER
=====================
Installs the REV2 encoder patch as a POINTER INDEX rather than by copying vectors.

For each dataset and each model (dense, splade) we emit two parallel arrays of
length N_canonical, in canonical node order (the order of nodes.jsonl):

    src[i] : int8   0 = PHASE_C primary tree, 1 = REV2 PATCH, 2 = PHASE_C_VIEW398
    row[i] : int32  row index within that source

Resolver contract:
    src == 0 -> data/canonical/<tree>/encodings/<model>/docs/
                shard  = row // shard_size
                offset = row %  shard_size
    src == 1 -> data/final_canonical/_rev2_encoder_patch/<ds>/<model>.{npy,npz}
                offset = row
No vectors are copied. 2wiki dense alone is 17.6 GB and there is ~24 GB free.

The index is positional: pointer position i corresponds to line i of nodes.jsonl.
That is only sound if the reuse map is in canonical node order, so this script
asserts FULL-LENGTH order equality before writing anything.

WHICH ROW FIELD IS AUTHORITATIVE
    A reuse-map row carries two row numbers and they are not interchangeable:

      phase_c_row_index      the node's OWN row in the Phase-C tree
      <model>_reuse_source   "<family>:<row>", the row the reuse decision was MADE against

    Reuse is decided by token-ID equality under the frozen tokenizer, so when the map says
    reusable it has proved that the encoder input at <model>_reuse_source is identical to
    this node's canonical text. It has proved nothing about phase_c_row_index. The two agree
    for all but ~20k of 11.4M rows, which is precisely why keying on the own-row field looks
    correct and is not: TEXTUALIZATION_REV_2 rewrote 87,765 2wiki nodes, and for three
    (node, model) pairs the node's own row still holds the OLD pre-REV2 text while the reuse
    source holds the new one. Keying on the reuse source makes the index right by
    construction instead of right by coincidence.

    Consequence: under the own-row field the pointers were a bijection; under the reuse
    source they are many-to-one, because the source row is an equivalence-class
    representative. Many-to-one is the correct shape.

    The suffix is not the same kind of thing in every dataset -- hotpotqa and 2wiki carry a
    Phase-C row, metaqa/squad/musique carry a Phase-C document id -- so the mode is decided
    from the first row and then asserted on every later row.
"""
import json, io, os, sys, time
from array import array
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
PATCH = os.path.join(ROOT, "_rev2_encoder_patch")

# canonical dataset -> phase-C tree holding the primary encodings
TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
        "hotpotqa": "hotpotqa", "squad": "squad"}

SRC_PHASE_C = 0
SRC_PATCH = 1
SRC_VIEW398 = 2
SRC_NAME = {0: "PHASE_C", 1: "REV2_PATCH", 2: "PHASE_C_VIEW398"}

# reuse_source prefix -> src code. Anything not listed is a hard error.
PREFIX_SRC = {"PHASE_C": SRC_PHASE_C, "PHASE_C_UNIVERSE": SRC_PHASE_C,
              "PHASE_C_VIEW398": SRC_VIEW398}


def reuse_rows(ds):
    """Yield reuse-map rows in file order, skipping LEGACY_ONLY (null canonical id)."""
    ix = json.load(io.open("%s/%s/reuse_map/index.json" % (ROOT, ds), encoding="utf-8"))
    for sh in ix["shards"]:
        with io.open("%s/%s/%s" % (ROOT, ds, sh["file"]), encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                if o.get("canonical_node_id") is None:
                    continue
                yield o


def node_ids(ds):
    """Yield canonical node ids in nodes.jsonl order."""
    with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
        for ln in f:
            yield json.loads(ln)["node_id"]


def patch_positions(ds, model):
    """canonical_node_id -> row offset inside the REV2 patch array."""
    p = "%s/%s/%s_ids.json" % (PATCH, ds, model)
    if not os.path.exists(p):
        return {}
    ids = json.load(io.open(p, encoding="utf-8"))
    return {nid: i for i, nid in enumerate(ids)}


def build(ds):
    t0 = time.time()
    tree = TREE[ds]
    eix = {}
    for m in ("dense", "splade"):
        j = json.load(io.open(
            "data/canonical/%s/encodings/%s/docs/index.json" % (tree, m), encoding="utf-8"))
        # the encoding index carries per-shard row counts, not a total
        j["n_items"] = sum(int(sh["rows"]) for sh in j["shards"])
        eix[m] = j
    ppos = {m: patch_positions(ds, m) for m in ("dense", "splade")}

    # A reuse_source is "<family>:<suffix>" and the suffix is NOT the same thing in every
    # dataset: hotpotqa/2wiki carry a Phase-C ROW, metaqa/squad/musique carry a Phase-C
    # DOC ID. Decide the mode from the first row, then assert every later row agrees --
    # a silent mode change would resolve ids as rows and corrupt the whole index.
    first = next(reuse_rows(ds))
    suf = (first.get("dense_reuse_source") or ":").split(":", 1)[1]
    mode = "ROW" if suf.isdigit() else "DOC_ID"
    id2row = {}
    if mode == "DOC_ID":
        with io.open("data/canonical/%s/documents.jsonl" % tree, encoding="utf-8") as f:
            for k, ln in enumerate(f):
                id2row[json.loads(ln)["canonical_doc_id"]] = k

    src = {m: array("b") for m in ("dense", "splade")}
    row = {m: array("i") for m in ("dense", "splade")}
    used_patch = {m: set() for m in ("dense", "splade")}
    stats = {"n": 0, "order_mismatch": 0, "families": {},
             "reuse_source_ne_own_row": {"dense": 0, "splade": 0}}

    for i, (nid, o) in enumerate(zip(node_ids(ds), reuse_rows(ds))):
        # FULL-LENGTH order assertion -- the whole index is positional
        if nid != o["canonical_node_id"]:
            stats["order_mismatch"] += 1
            raise SystemExit("ORDER MISMATCH %s at %d: nodes=%s reuse=%s"
                             % (ds, i, nid, o["canonical_node_id"]))
        stats["n"] += 1
        for m in ("dense", "splade"):
            if o.get("%s_reusable" % m):
                rs = o.get("%s_reuse_source" % m) or ""
                fam = rs.split(":", 1)[0]
                stats["families"][fam] = stats["families"].get(fam, 0) + 1
                if fam not in PREFIX_SRC:
                    raise SystemExit("UNKNOWN REUSE FAMILY %s: %r" % (ds, rs))
                rtxt = rs.split(":", 1)[1] if ":" in rs else ""
                if not rtxt:
                    raise SystemExit("REUSE SOURCE HAS NO SUFFIX %s %s %s: %r"
                                     % (ds, m, nid, rs))
                if rtxt.isdigit() != (mode == "ROW"):
                    raise SystemExit("REUSE SOURCE MODE CHANGES MID-FILE %s %s %s: %r "
                                     "(mode=%s)" % (ds, m, nid, rs, mode))
                if mode == "ROW":
                    r = int(rtxt)
                else:
                    r = id2row.get(rtxt)
                    if r is None:
                        raise SystemExit("REUSE SOURCE DOC ID NOT IN PHASE-C %s %s %s: %r"
                                         % (ds, m, nid, rs))
                if r != o.get("phase_c_row_index"):
                    stats["reuse_source_ne_own_row"][m] += 1
                src[m].append(PREFIX_SRC[fam])
                row[m].append(r)
            else:
                if nid not in ppos[m]:
                    raise SystemExit("NEEDS ENCODE BUT NOT IN PATCH %s %s %s" % (ds, m, nid))
                src[m].append(SRC_PATCH)
                row[m].append(ppos[m][nid])
                used_patch[m].add(nid)

    out = {}
    d = "%s/%s/pointer_index" % (ROOT, ds)
    os.makedirs(d, exist_ok=True)
    for m in ("dense", "splade"):
        s = np.frombuffer(src[m], dtype=np.int8)
        r = np.frombuffer(row[m], dtype=np.int32)
        # every pointer must land inside its source
        n_pc = int((s == SRC_PHASE_C).sum())
        if n_pc:
            mx = int(r[s == SRC_PHASE_C].max())
            if mx >= eix[m]["n_items"]:
                raise SystemExit("PHASE_C ROW OUT OF RANGE %s %s %d>=%d"
                                 % (ds, m, mx, eix[m]["n_items"]))
        n_pt = int((s == SRC_PATCH).sum())
        if n_pt and len(ppos[m]) and int(r[s == SRC_PATCH].max()) >= len(ppos[m]):
            raise SystemExit("PATCH ROW OUT OF RANGE %s %s" % (ds, m))
        # the patch must be consumed EXACTLY -- no unused rows, no missing rows
        unused = set(ppos[m]) - used_patch[m]
        if unused:
            raise SystemExit("PATCH ROWS NEVER POINTED TO %s %s n=%d" % (ds, m, len(unused)))
        np.savez(os.path.join(d, "%s.npz" % m), src=s, row=r)
        out[m] = {"n_pointers": int(s.size),
                  "from_PHASE_C": n_pc,
                  "from_REV2_PATCH": n_pt,
                  "from_PHASE_C_VIEW398": int((s == SRC_VIEW398).sum()),
                  "distinct_source_rows": int(np.unique(r[s == SRC_PHASE_C]).size) if n_pc else 0,
                  "phase_c_tree": tree,
                  "phase_c_shard_size": eix[m]["shard_size"],
                  "phase_c_n_items": eix[m]["n_items"],
                  "patch_n_rows": len(ppos[m])}
    out["seconds"] = round(time.time() - t0, 1)
    out["n_canonical_nodes"] = stats["n"]
    out["reuse_families_seen"] = stats["families"]
    out["reuse_source_ne_own_row"] = stats["reuse_source_ne_own_row"]
    out["reuse_source_mode"] = mode
    print("== %s  n=%d  %.1fs" % (ds, stats["n"], time.time() - t0))
    for m in ("dense", "splade"):
        print("   %-6s phase_c=%-9d patch=%-6d distinct_src_rows=%-9d"
              % (m, out[m]["from_PHASE_C"], out[m]["from_REV2_PATCH"],
                 out[m]["distinct_source_rows"]))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
    man = {}
    for ds in which:
        man[ds] = build(ds)
    p = os.path.join(ROOT, "POINTER_INDEX.json")
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.setdefault("datasets", {}).update(man)
    old["resolver"] = {
        "arrays": "pointer_index/<model>.npz with src(int8) and row(int32), "
                  "position i == line i of nodes.jsonl",
        "src_codes": SRC_NAME,
        "PHASE_C": "data/canonical/<phase_c_tree>/encodings/<model>/docs/ ; "
                   "shard = row // phase_c_shard_size, offset = row % phase_c_shard_size",
        "REV2_PATCH": "data/final_canonical/_rev2_encoder_patch/<ds>/"
                      "{dense.npy,splade.npz} ; offset = row",
        "note": "many-to-one is legal: two canonical nodes with byte-identical "
                "encoder input share one source row (musique)",
    }
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=2)
    print("\nwrote", p)
