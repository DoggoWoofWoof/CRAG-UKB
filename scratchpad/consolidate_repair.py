"""
CONSOLIDATE THE DENSE REPAIR
============================
Joins the 19 Modal output parts back into one array per channel and proves the result before
anything is allowed to point at it.

Gates, all of which must pass:
  - every part is accounted for and no row is delivered twice
  - the delivered (row, id) pairs equal the repair worklist EXACTLY as a set and in count
  - the id at each repaired Phase-C row matches that channel ids_*.json, so the repair lands on
    the row it was cut from
  - every vector is finite and L2-normalised to within the float16 tolerance the clean rows show
  - no repaired vector is all-zero, which is the defect being fixed

Output per channel: dense.npy (rows in ascending Phase-C row order), dense_rows.json, dense_ids.json.
"""
import glob, io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PATCH = "data/final_canonical/_dense_repair_patch"
CANON = "data/canonical"


def ids_for(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "ids_*.json"))):
        out += json.load(io.open(f, encoding="utf-8"))
    return out


def main():
    t0 = time.time()
    man = json.load(io.open("scratchpad/REPAIR_WORKLIST.json", encoding="utf-8"))
    want = {}
    for key, c in man["channels"].items():
        ch = "%s__%s" % (c["tree"], c["kind"])
        want[ch] = {"key": key, "tree": c["tree"], "kind": c["kind"], "n": c["n"],
                    "rows": [json.loads(l)["row"] for l in io.open(c["file"], encoding="utf-8")],
                    "ids": [json.loads(l)["id"] for l in io.open(c["file"], encoding="utf-8")]}
    out, ok = {}, True
    for ch, w in sorted(want.items()):
        parts = sorted(glob.glob(os.path.join(PATCH, ch + "__p*")))
        V, R, I = [], [], []
        for p in parts:
            V.append(np.load(os.path.join(p, "dense.npy")))
            R += json.load(io.open(os.path.join(p, "dense_rows.json"), encoding="utf-8"))
            I += json.load(io.open(os.path.join(p, "dense_ids.json"), encoding="utf-8"))
        V = np.concatenate(V)
        r = {"parts": len(parts), "n": int(V.shape[0]), "expected": w["n"]}
        # delivered set must equal the worklist set, with no duplicates
        r["duplicate_rows"] = int(len(R) - len(set(R)))
        r["set_equal_worklist"] = bool(set(R) == set(w["rows"]) and len(R) == w["n"])
        r["id_pairs_match"] = bool(dict(zip(R, I)) == dict(zip(w["rows"], w["ids"])))
        # sort into ascending Phase-C row order so installation is a plain searchsorted
        order = np.argsort(np.asarray(R), kind="stable")
        V, R, I = V[order], [R[i] for i in order], [I[i] for i in order]
        # the repair must land on the row it was cut from
        chan_ids = ids_for(os.path.join(CANON, w["tree"], "encodings", "dense", w["kind"]))
        r["lands_on_correct_row"] = bool(all(chan_ids[rw] == idd for rw, idd in zip(R, I)))
        nr = np.linalg.norm(V.astype(np.float32), axis=1)
        r["finite"] = bool(np.isfinite(V).all())
        r["zero_rows"] = int((nr == 0).sum())
        r["l2_min"], r["l2_max"] = round(float(nr.min()), 6), round(float(nr.max()), 6)
        r["l2_ok"] = bool(np.abs(nr - 1.0).max() < 1e-2)
        r["dtype"], r["dim"] = str(V.dtype), int(V.shape[1])
        r["PASS"] = bool(r["n"] == r["expected"] and r["duplicate_rows"] == 0
                         and r["set_equal_worklist"] and r["id_pairs_match"]
                         and r["lands_on_correct_row"] and r["finite"]
                         and r["zero_rows"] == 0 and r["l2_ok"] and r["dim"] == 1536
                         and r["dtype"] == "float16")
        d = os.path.join(PATCH, ch)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "dense.npy"), V)
        json.dump(R, io.open(os.path.join(d, "dense_rows.json"), "w", encoding="utf-8"))
        json.dump(I, io.open(os.path.join(d, "dense_ids.json"), "w", encoding="utf-8"))
        out[ch] = r
        ok &= r["PASS"]
        print("  %-30s parts=%-2d n=%-7d l2=[%.6f,%.6f] %s"
              % (ch, r["parts"], r["n"], r["l2_min"], r["l2_max"],
                 "PASS" if r["PASS"] else "FAIL"), flush=True)
        if not r["PASS"]:
            print("     ", json.dumps(r))
    out["ALL_PASS"] = bool(ok)
    out["TOTAL_ROWS"] = sum(v["n"] for k, v in out.items() if isinstance(v, dict))
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, io.open("scratchpad/consolidate_repair.json", "w", encoding="utf-8"), indent=1)
    print("\nALL_PASS = %s  total rows %s" % (ok, format(out["TOTAL_ROWS"], ",")))


if __name__ == "__main__":
    main()
