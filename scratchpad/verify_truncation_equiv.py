"""
TRUNCATION-EQUIVALENCE PROOF
============================
The text gate found a small set of pointers whose canonical text is not byte-equal to the
Phase-C text at the pointed row:

    hotpotqa splade 1,594     2wiki splade 2     2wiki dense 1

These are not errors, and the counts say so before any tokenizer runs: hotpotqa has
splade_reusable 5,233,235 and dense_reusable 5,231,641, a difference of exactly 1,594, and
2wiki has 5,902,084 against 5,902,083, a difference of exactly 1. Those are the rows where the
reuse decision differed BETWEEN the two models -- which is possible only because reuse is
decided by token-ID equality under each frozen tokenizer, not by raw-text equality, and the two
tokenizers truncate at very different lengths (SPLADE/BERT at 256, gte-Qwen2 at 32768).

Counting is not proving, so this tokenizes the actual rows. For each exception the claim is:

    tokenizer(canonical_text)[:max_len] == tokenizer(phase_c_text)[:max_len]

If that holds, the two texts are the SAME INPUT as far as the encoder is concerned, the stored
vector is the correct vector for this node, and the pointer is right. If it fails for even one
row, the pointer index is wrong for that row and must not be frozen.
"""
import io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"
TREE = {"hotpotqa": "hotpotqa", "2wiki": "2wiki_universe"}
MAXLEN = {"dense": 32768, "splade": 256}
TOK = {"dense": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
       "splade": "naver/splade-cocondenser-ensembledistil"}


def collect(ds, model):
    """The exception rows: canonical text != Phase-C text at the pointed row."""
    tree = TREE[ds]
    ptx = []
    with io.open("%s/%s/documents.jsonl" % (CANON, tree), encoding="utf-8") as f:
        for ln in f:
            ptx.append(json.loads(ln).get("text") or "")
    z = np.load("%s/%s/pointer_index/%s.npz" % (ROOT, ds, model))
    src, row = z["src"], z["row"]
    out = []
    with io.open("%s/%s/nodes.jsonl" % (ROOT, ds), encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if src[i] != 0:
                continue
            o = json.loads(ln)
            ct = o.get("text") or ""
            pt = ptx[row[i]]
            if ct != pt:
                out.append({"node_id": o["node_id"], "row": int(row[i]),
                            "canonical": ct, "phase_c": pt})
    return out


def main():
    t0 = time.time()
    from transformers import AutoTokenizer
    res, allok = {}, True
    for ds in ("hotpotqa", "2wiki"):
        for model in ("dense", "splade"):
            ex = collect(ds, model)
            key = "%s/%s" % (ds, model)
            if not ex:
                res[key] = {"exceptions": 0, "PASS": True}
                print("  %-18s exceptions=0  PASS" % key, flush=True)
                continue
            tk = AutoTokenizer.from_pretrained(TOK[model], trust_remote_code=True)
            ml = MAXLEN[model]
            same = 0
            bad = []
            for e in ex:
                a = tk(e["canonical"], truncation=True, max_length=ml)["input_ids"]
                b = tk(e["phase_c"], truncation=True, max_length=ml)["input_ids"]
                if a == b:
                    same += 1
                elif len(bad) < 3:
                    bad.append({"node_id": e["node_id"], "row": e["row"],
                                "len_canonical": len(e["canonical"]),
                                "len_phase_c": len(e["phase_c"]),
                                "tok_canonical": len(a), "tok_phase_c": len(b)})
            ok = same == len(ex)
            allok &= ok
            res[key] = {"exceptions": len(ex), "token_identical": same,
                        "max_length": ml, "tokenizer": TOK[model],
                        "PASS": ok, "failures": bad}
            print("  %-18s exceptions=%-6d token_identical=%-6d %s"
                  % (key, len(ex), same, "PASS" if ok else "FAIL"), flush=True)
            if bad:
                print("     ", json.dumps(bad)[:400])
    res["ALL_PASS"] = bool(allok)
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, io.open("scratchpad/verify_truncation_equiv.json", "w", encoding="utf-8"),
              indent=1)
    print("\nALL_PASS =", allok)


if __name__ == "__main__":
    main()
