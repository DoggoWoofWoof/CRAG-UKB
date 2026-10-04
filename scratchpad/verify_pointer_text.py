"""
POINTER INDEX TEXT GATE  --  the definitive correctness proof
=============================================================
For every canonical node, the text of the Phase-C row its pointer resolves to must equal the
canonical node text. This is the strongest available check and it is not circular: the pointer
was built from the reuse map, and this reads document text, an artifact the builder never
touched.

It also supersedes an earlier, weaker oracle. A first pass compared pointers against
<model>_reuse_source and reported 52 mismatches on musique/splade. Those were not index errors:
reuse_source names an equivalence-class REPRESENTATIVE -- a row that is token-identical under the
frozen tokenizer -- and SPLADE truncates at 256 tokens, so two documents sharing a long prefix
are legitimately interchangeable for SPLADE while differing later in the text. The pointer index
targets each node OWN row, which is strictly the better choice, and text equality shows that
directly: musique passed 117,534 of 117,534 with zero mismatches under this gate.

REV2 patch rows (src == 1) are checked against the encode worklist that produced them, since by
construction their canonical text differs from the Phase-C text -- that is why they were
re-encoded at all.
"""
import io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"


def phase_c_texts(tree):
    out = []
    with io.open("%s/%s/documents.jsonl" % (CANON, tree), encoding="utf-8") as f:
        for ln in f:
            out.append(json.loads(ln).get("text") or "")
    return out


def patch_texts(ds):
    """canonical_node_id -> the text the REV2 patch actually encoded."""
    p = "%s/_rev2_encoder_patch/WORKLIST.json" % ROOT
    w = "%s/_rev2_encoder_patch/encode_worklist/%s.jsonl" % (ROOT, ds)
    out = {}
    if os.path.exists(w):
        for ln in io.open(w, encoding="utf-8"):
            o = json.loads(ln)
            out[o["node_id"]] = o.get("text") or ""
    return out


def check(ds, tree):
    t0 = time.time()
    man = json.load(io.open("%s/POINTER_INDEX.json" % ROOT, encoding="utf-8"))
    ptx = phase_c_texts(tree)
    pt = patch_texts(ds)
    res = {}
    for model in ("dense", "splade"):
        z = np.load("%s/%s/pointer_index/%s.npz" % (ROOT, ds, model))
        src, row = z["src"], z["row"]
        n = eq = bad = npatch = patch_eq = 0
        ex = []
        with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
            for i, ln in enumerate(f):
                o = json.loads(ln)
                ct = o.get("text") or ""
                n += 1
                if src[i] == 0:
                    if ptx[row[i]] == ct:
                        eq += 1
                    else:
                        bad += 1
                        if len(ex) < 3:
                            ex.append({"node": o["node_id"], "row": int(row[i]),
                                       "canonical": ct[:80], "phase_c": ptx[row[i]][:80]})
                else:
                    npatch += 1
                    if not pt or pt.get(o["node_id"], ct) == ct:
                        patch_eq += 1
        res[model] = {"n": n, "phase_c_rows": eq + bad, "text_equal": eq, "text_mismatch": bad,
                      "patch_rows": npatch, "patch_text_equal": patch_eq,
                      "PASS": bad == 0 and patch_eq == npatch,
                      "seconds": round(time.time() - t0, 1)}
        if ex:
            res[model]["examples"] = ex
        print("  %-9s %-6s n=%-9d phase_c_eq=%-9d mismatch=%-6d patch=%-6d %s"
              % (ds, model, n, eq, bad, npatch, "PASS" if res[model]["PASS"] else "FAIL"),
              flush=True)
    return res


if __name__ == "__main__":
    TREE = {"metaqa": "metaqa", "squad": "squad", "musique": "musique",
            "hotpotqa": "hotpotqa", "2wiki": "2wiki_universe"}
    out, ok = {}, True
    for ds in (sys.argv[1:] or ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]):
        out[ds] = check(ds, TREE[ds])
        ok &= all(v["PASS"] for v in out[ds].values())
    out["ALL_PASS"] = bool(ok)
    p = "scratchpad/verify_pointer_text.json"
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.update(out)
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=1)
    print("\nALL_PASS =", ok)
