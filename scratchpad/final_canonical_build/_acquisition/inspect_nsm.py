"""Route A step 2 -- open the NSM tarballs and settle whether the released graph is MID-keyed.

    python scratchpad/final_canonical_build/_acquisition/inspect_nsm.py

REFUSES TO RUN until NSM_ACQUISITION_PROVENANCE.json exists, because the pre-registration
requires provenance (bytes + sha256) to be on disk BEFORE extraction.

This is the [I] -> [V] conversion, and it is cheap.  WEBQSP_SOURCE_MATRIX.json records NSM's
layout as "*_simple.json (per-question graph, entity/relation mapped to global ids) + entities.txt
+ relations.txt", so the per-question graphs carry integer indices and the identity question is
entirely about what entities.txt CONTAINS.  Prior evidence that it is MIDs is [I], inferred from a
code audit of preprocess_step1.py plus the existence of GNN-RAG's entities_names.json; the file
itself has never been opened by this project.

Members are listed before anything is written to disk, and extraction uses tarfile's data filter
so a malicious or malformed member cannot escape the destination directory.
"""
import json, os, re, sys, tarfile, time

DST = "data/final_canonical/webqsp/_acquisition/nsm"
PROV = "data/final_canonical/webqsp/NSM_ACQUISITION_PROVENANCE.json"
OUT = "data/final_canonical/webqsp/NSM_MID_PRESERVATION.json"
TARS = ["webqsp.tgz", "CWQ.tgz"]

MID = re.compile(r"^[mg]\.[0-9a-z_]+$")          # Freebase machine id, the shape RoG leaves unresolved
CVT_ISH = re.compile(r"^[mg]\.[0-9a-z_]{6,}$")   # not a classifier, only a shape note
NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


def classify(tok):
    t = tok.strip()
    if not t:
        return "EMPTY"
    if MID.match(t):
        return "MID"
    if NUMERIC.match(t):
        return "NUMERIC_LITERAL"
    if t.startswith("http://") or t.startswith("https://"):
        return "URL"
    return "SURFACE"


def main():
    if not os.path.exists(PROV):
        sys.exit(f"REFUSING TO EXTRACT: {PROV} does not exist. The pre-registration requires "
                 f"provenance (bytes + sha256) recorded BEFORE extraction. Run fetch_nsm.py first.")
    prov = json.load(open(PROV, encoding="utf-8"))
    if not prov.get("all_ok"):
        sys.exit(f"REFUSING TO EXTRACT: provenance records a failed download: "
                 f"{[f['name'] for f in prov['files'] if f.get('status') != 'OK']}")

    t0 = time.time()
    report = {"schema": "NSM_MID_PRESERVATION/v1",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "provenance": PROV,
              "provenance_sha256": {f["name"]: f.get("sha256") for f in prov["files"]},
              "archives": {}}

    for tname in TARS:
        p = f"{DST}/{tname}"
        if not os.path.exists(p):
            report["archives"][tname] = {"status": "MISSING"}
            continue
        ent = {"status": "OK", "members": [], "extracted_to": None}
        with tarfile.open(p, "r:gz") as tf:
            members = tf.getmembers()
            ent["member_count"] = len(members)
            ent["members"] = [{"name": m.name, "size": m.size, "isdir": m.isdir()}
                              for m in members if not m.isdir()][:200]
            ent["total_uncompressed_bytes"] = sum(m.size for m in members)
            out = f"{DST}/extracted/{tname.replace('.tgz', '')}"
            os.makedirs(out, exist_ok=True)
            print(f"[{tname}] {len(members)} members, "
                  f"{ent['total_uncompressed_bytes'] / 1e9:.2f} GB uncompressed -> {out}", flush=True)
            try:
                tf.extractall(out, filter="data")        # py>=3.12: blocks path traversal / device files
            except TypeError:
                tf.extractall(out)                        # older python: no filter kwarg
            ent["extracted_to"] = out
        report["archives"][tname] = ent

    # ---- the decisive check: what is in entities.txt? ----
    findings = {}
    for root, _, files in os.walk(f"{DST}/extracted"):
        for fn in files:
            if fn not in ("entities.txt", "relations.txt"):
                continue
            p = os.path.join(root, fn).replace("\\", "/")
            counts, head = {}, []
            n = 0
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    n += 1
                    tok = line.rstrip("\n")
                    if len(head) < 15:
                        head.append(tok)
                    k = classify(tok)
                    counts[k] = counts.get(k, 0) + 1
            findings[p] = {
                "lines": n,
                "class_counts": counts,
                "class_fractions": {k: round(v / n, 6) for k, v in counts.items()} if n else {},
                "head_15": head,
                "mid_fraction": round(counts.get("MID", 0) / n, 6) if n else None,
            }
            print(f"[check] {p}  lines={n:,}  classes={counts}", flush=True)

    report["identity_files"] = findings
    ent_files = {k: v for k, v in findings.items() if k.endswith("entities.txt")}
    if ent_files:
        frac = max(v["mid_fraction"] or 0 for v in ent_files.values())
        report["MID_KEYED"] = frac > 0.5
        report["max_mid_fraction_over_entities_files"] = frac
        report["VERDICT"] = (
            "MID-KEYED CONFIRMED [V]. The prior [I] inference is now a measurement."
            if frac > 0.5 else
            "NOT MID-KEYED [V]. Route A's premise fails: NSM's released artifact does not carry MIDs, "
            "so it cannot be the MID preimage regardless of what the union count says.")
    else:
        report["MID_KEYED"] = None
        report["VERDICT"] = "entities.txt not found in either archive -- inspect the member listing."

    report["elapsed_s"] = round(time.time() - t0, 1)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({"MID_KEYED": report.get("MID_KEYED"),
                      "VERDICT": report.get("VERDICT"),
                      "entities_files": {k: {"lines": v["lines"], "mid_fraction": v["mid_fraction"]}
                                         for k, v in ent_files.items()},
                      "elapsed_s": report["elapsed_s"]}, indent=1))


if __name__ == "__main__":
    main()
