# -*- coding: utf-8 -*-
"""
Bring sections 6 and 7 to 6-of-6.

Section 7 was a five-dataset gate table. It also carried one wrong number: "13 materialised
families". The five have 12 (2+3+3+2+2); 13 is the count across all six. It reads as correct
now only by coincidence, so it is recomputed from the manifests rather than left alone.

Everything numeric is read from the records, never typed.
"""
import glob
import io
import json
import os
import sys

ROOT = "data/final_canonical"
H = ROOT + "/HANDOFF.md"


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def c(n):
    return format(int(n), ",")


five = load(ROOT + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json")
six = load(ROOT + "/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json")
we = load(ROOT + "/WEBQSP_ENCODING_SHARD_HASHES.json")
e2e = load(ROOT + "/SIX_DATASET_END_TO_END.json")
esh = load(ROOT + "/ENCODING_SHARD_HASHES.json")
w = six["webqsp"]

# families, counted from the manifests on disk
fam_n, fam_by_ds = 0, {}
for p in sorted(glob.glob("%s/*/graph/GRAPH_MANIFEST.json" % ROOT)):
    ds = p.replace(chr(92), "/").split("/")[-3]
    m = load(p)["families"]
    present = [k for k, v in m.items() if (v.get("present") if isinstance(v, dict) else bool(v))]
    fam_by_ds[ds] = len(present)
    fam_n += len(present)

gold_five = sum(five["datasets"][d]["gold_refs"] if "gold_refs" in five["datasets"][d]
                else 0 for d in five["datasets"])
ib = load(ROOT + "/ID_BRIDGE.json")["datasets"]
gold_five = sum(ib[d]["gold_refs"] for d in five["datasets"])

s = io.open(H, encoding="utf-8").read()

# ------------------------------------------------------------------ section 7
old7 = (
    "| gate | result | evidence |\n|---|---|---|\n"
    "| resolution text (all stores, both models, all 5) | **ALL_PASS** | `scratchpad/verify_resolution_text.json` |\n"
    "| post-repair staleness (20 channels) | **ALL_PASS**, 0 stale pointers | `scratchpad/verify_post_repair.json` |\n"
    "| id bridge + gold resolution | **PASS** all 5; 3,875,657 gold refs, **0 unresolved**, 0 rule violations | `ID_BRIDGE.json` |\n"
    "| graph endpoint alignment | **0 unresolved endpoints** in all 13 materialised families | `<ds>/graph/GRAPH_MANIFEST.json` |\n"
    "| encoding shard content hashes | 30 channels, 52.5 GB fingerprinted | `ENCODING_SHARD_HASHES.json` |\n")

new7 = (
    "**The five (verified before the 5/5 freeze, re-checked by hash since):**\n\n"
    "| gate | result | evidence |\n|---|---|---|\n"
    "| resolution text (all stores, both models, all 5) | **ALL_PASS** | `scratchpad/verify_resolution_text.json` |\n"
    "| post-repair staleness (20 channels) | **ALL_PASS**, 0 stale pointers | `scratchpad/verify_post_repair.json` |\n"
    "| id bridge + gold resolution | **PASS** all 5; %s gold refs, **0 unresolved**, 0 rule violations | `ID_BRIDGE.json` |\n"
    "| encoding shard content hashes | %d channels, %.1f GB fingerprinted | `ENCODING_SHARD_HASHES.json` |\n"
    "\n**Dataset 6:**\n\n"
    "| gate | result | evidence |\n|---|---|---|\n"
    "| webqsp artifacts, 15 checks against disk | **%s** | `webqsp/CANONICAL_V1_VERIFICATION.json` |\n"
    "| webqsp id bridge | **PASS**; node rule on all %s rows, topic-entity control **%.2f%%**, gold %.2f%% | `ID_BRIDGE.json` |\n"
    "| webqsp encoding assembly | rows contiguous, both pointer indexes rebuilt | `webqsp/ENCODING_ASSEMBLY.json` |\n"
    "| webqsp vector shards: content hash **and exhaustive row scan** | **%s** — %d channels, %.2f GB, %s rows scanned, **%d bad** | `WEBQSP_ENCODING_SHARD_HASHES.json` |\n"
    "\n**All six:**\n\n"
    "| gate | result | evidence |\n|---|---|---|\n"
    "| graph endpoint alignment | **0 unresolved endpoints** in all %d materialised families (%s) | `<ds>/graph/GRAPH_MANIFEST.json` |\n"
    "| end-to-end through the public API | **%s**, %d failures, %.0fs | `SIX_DATASET_END_TO_END.json` |\n"
    "| every frozen artifact re-hashed from disk | see the record; `DIVERGED` there is a real failure | `FROZEN_ARTIFACT_RECHECK.json` |\n"
    "| the two artifacts that did move, explained | **EXPLAINED_ADDITIVE** — stripping webqsp reproduces the frozen bytes exactly | `DIVERGED_ARTIFACT_EXPLANATION.json` |\n"
    "| this document checked against the data it describes | paths, totals, digests, families and the API, all compared | `HANDOFF_CLAIMS_CHECK.json` |\n"
    % (c(gold_five), esh["n_channels"], esh["total_bytes"] / 1e9,
       load(ROOT + "/webqsp/CANONICAL_V1_VERIFICATION.json")["VERDICT"],
       c(w["n_nodes"]), w["id_bridge"]["positive_control_pct"], w["id_bridge"]["gold_resolution_pct"],
       we["VERDICT"], we["n_channels"], we["total_bytes"] / 1e9,
       c(sum(v["rows_scanned"] for v in we["channels"].values())),
       sum(v["bad_rows"] for v in we["channels"].values()),
       fam_n, " ".join("%s %d" % (k, v) for k, v in sorted(fam_by_ds.items())),
       e2e["VERDICT"], len(e2e["failures"]), e2e["elapsed_s"]))

if old7 not in s:
    sys.exit("section 7 table anchor not found")
s = s.replace(old7, new7, 1)

# ------------------------------------------------------------------ section 6
old6 = ("| `scan_encoding_integrity.py` | Enumerates every damaged row per channel (dense: "
        "L2 = 0 or \\|L2−1\\| > 1e-2; splade: empty row) and the contiguous runs they form. "
        "This is what found the corruption. |\n")
new6 = old6 + (
    "| `hash_and_scan_webqsp_encodings.py` | The same two checks, applied to dataset 6's own "
    "stores, which neither original script covered. `scan_encoding_integrity.py`'s tree list "
    "predates `webqsp_rog_v1`, and `ENCODING_SHARD_HASHES.json`'s `webqsp/*` keys point at the "
    "**superseded** `data/canonical/webqsp/` tree — a different corpus, marked DAMAGED there, "
    "that dataset 6 does not use. Verify dataset 6 against `WEBQSP_ENCODING_SHARD_HASHES.json` "
    "instead. |\n"
    "| `verify_six_end_to_end.py` | Opens all six through `CanonicalEmbeddings`, `CanonicalGraph` "
    "and `families` — the public interface, from the package root, the way a stranger would. "
    "Checks dims, L2 norms, empty SPLADE rows, id alignment, graph endpoints, and each "
    "manifest against the `.npz` files on disk **in both directions**. A package can have every "
    "artifact correct and still be unusable. |\n"
    "| `verify_frozen_artifacts.py` | Re-hashes every artifact both freeze records declare. The "
    "records' own `RECORD_SHA256` proves the *records* are intact and says nothing about the "
    "*files*. Buckets: `MATCH`, `DOCS_DRIFT` (this document), `ADDITIVE` (cleared only by "
    "reproducing the frozen bytes after removing webqsp), `DIVERGED` (fails), `MISSING`. |\n"
    "| `explain_diverged_artifacts.py` | The additive test itself, and the semantic comparison "
    "used when it cannot decide. |\n"
    "| `verify_handoff_claims.py` | Checks **this file** against the data — every path it names, "
    "the corpus totals, every digest prefix it quotes, the encoder table, the family claims, and "
    "the quickstart API. The prose is the one artifact nothing else gated. |\n")
if old6 not in s:
    sys.exit("section 6 gates anchor not found")
s = s.replace(old6, new6, 1)

io.open(H, "w", encoding="utf-8", newline=chr(10)).write(s)
print("sections 6 and 7 updated")
print("  families across six: %d  %s" % (fam_n, fam_by_ds))
print("  five gold refs: %s" % c(gold_five))
print("  webqsp rows scanned: %s, bad: %d"
      % (c(sum(v["rows_scanned"] for v in we["channels"].values())),
         sum(v["bad_rows"] for v in we["channels"].values())))
