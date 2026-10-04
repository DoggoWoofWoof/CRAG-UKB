"""STEP 7b: freeze CRAG_FREEBASE_CANONICAL. Hashes, statistics, provenance, manifest.

    python scratchpad/final_canonical_build/webqsp_v1/v3_freeze_canonical.py [--force]

A freeze is a claim that these exact bytes are the dataset, so it is gated on STEP 6 having passed
against these exact bytes. --force writes the manifest anyway and records WHY it was forced and
which checks were failing at the time, because an unexplained freeze over a failing check is worse
than no freeze at all.

WHAT A MANIFEST IS FOR. Not to prove the build was correct -- the validation report does that -- but
to make a later reader able to tell whether the files in front of them are the ones that were
validated. That needs a digest per file, the row count per table, and the chain of records the
numbers came from. All three are here, and none of them is retyped: every figure is read out of the
artifact that measured it.

SOURCE PROVENANCE goes back to the byte level: the archive.org item, its published md5/sha1, the
size, the 2015 mtime, and the gzip member index the whole parallel build rests on. The interpreter
version and PYTHONHASHSEED are part of it too, because the node UID function depends on both and a
graph whose identity function is unrecorded cannot be rebuilt.
"""
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys
import time

import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
CAN = f"{V3}/canonical"
MANIFEST = f"{CAN}/CANONICAL_MANIFEST.json"
STATS = f"{CAN}/GRAPH_STATISTICS.json"
PROV = f"{CAN}/SOURCE_PROVENANCE.json"


def load(name, required=True):
    p = f"{V3}/{name}"
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"missing {p}")
        return None
    return json.load(open(p, encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return path, h.hexdigest(), os.path.getsize(path)


def main():
    t0 = time.time()
    force = "--force" in sys.argv
    val = load("V3_CANONICAL_VALIDATION.json")
    asm = load("V3_CANONICAL_ASSEMBLY.json")
    passa = load("V3_PASS_A_SCHEMA_METADATA.json")
    frz = load("V3_PASS_A_SCHEMA_FREEZE.json")
    passb = load("V3_PASS_B_REPORT.json")
    passc = load("V3_PASS_C_NODE_UNIVERSE.json")
    passd = load("V3_PASS_D_CANONICAL_EDGES.json")
    uidsp = load("V3_NODE_UID_SPACE.json")
    contig = load("V3_SUBJECT_CONTIGUITY.json")
    idir = load("V3_IDIR_ORACLE_COMPARISON.json", required=False)
    mirror = load("V3_RAW_MIRROR_PROVENANCE_LOCK.json")
    members = load("V3_GZIP_MEMBER_INDEX.json", required=False)
    contracts = {n: load(n)["CONTRACT_HASH"] for n in
                 ("V3_CANONICAL_CONTRACT.json", "V3_CANONICAL_CONTRACT_V2.json",
                  "V3_CANONICAL_CONTRACT_V3.json", "V3_CANONICAL_CONTRACT_V4.json")}

    failing = [k for k, v in val["CHECKS"].items() if not v.get("ok")]
    if not val["ALL_CHECKS_PASS"] and not force:
        raise SystemExit("STEP 6 has failing checks, refusing to freeze: " + ", ".join(failing))

    # The IDIR oracle gate was pre-registered with its thresholds fixed BEFORE the number was
    # known (the report records frozen_before_the_number_was_known). Until now this script only
    # copied the verdict into provenance, which means a STOP could be written into a manifest that
    # still says FROZEN -- an artifact that looks finished while a gate it declares is failing.
    # Only PASS is an automatic green light; INVESTIGATE and STOP both require --force, which is
    # recorded in the manifest along with the number that caused it.
    idir_verdict = (idir or {}).get("GATE", {}).get("VERDICT")
    idir_cov = (idir or {}).get("AGREEMENT", {}).get("coverage_pct")
    if idir_verdict in ("STOP", "INVESTIGATE") and not force:
        raise SystemExit(
            f"IDIR oracle gate returned {idir_verdict} at {idir_cov}% coverage, refusing to "
            f"freeze. This gate was pre-registered before the number was known. See "
            f"V3_IDIR_SOURCE_DIVERGENCE.json for the characterisation; freeze with --force only "
            f"as a recorded decision that the divergence is understood and accepted.")

    # ---------- digests ----------
    files = sorted(
        glob.glob(f"{CAN}/nodes/*.parquet") + glob.glob(f"{CAN}/edges/*.parquet")
        + glob.glob(f"{CAN}/metadata/*.parquet") + [f"{CAN}/relations.parquet"])
    files = [f for f in files if os.path.exists(f)]
    print(f"hashing {len(files)} files", flush=True)
    digests, total = {}, 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for p, h, n in ex.map(digest, files):
            digests[os.path.relpath(p, CAN).replace(os.sep, "/")] = {"sha256": h, "bytes": n}
            total += n
    print(f"  {total/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

    def rows(pattern):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(glob.glob(pattern)))

    tables = {
        "nodes": {"files": len(glob.glob(f"{CAN}/nodes/*.parquet")), "rows": rows(f"{CAN}/nodes/*.parquet")},
        "edges": {"files": len(glob.glob(f"{CAN}/edges/*.parquet")), "rows": rows(f"{CAN}/edges/*.parquet")},
        "relations": {"files": 1, "rows": rows(f"{CAN}/relations.parquet")},
        "metadata": {"files": len(glob.glob(f"{CAN}/metadata/*.parquet")),
                     "rows": rows(f"{CAN}/metadata/*.parquet")},
    }

    # ---------- statistics ----------
    stats = {
        "schema": "GRAPH_STATISTICS/v1",
        "name": "CRAG_FREEBASE_CANONICAL",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "NODES": val["STATISTICS"]["NODE_COUNTS"],
        "NODES_TOTAL": tables["nodes"]["rows"],
        "EDGES_TOTAL": tables["edges"]["rows"],
        "RELATIONS_TOTAL": tables["relations"]["rows"],
        "DISPLAY_TEXT_COVERAGE": val["STATISTICS"].get("DISPLAY_TEXT_COVERAGE"),
        "OUT_DEGREE": val["STATISTICS"].get("OUT_DEGREE"),
        "CONNECTIVITY": val["STATISTICS"].get("CONNECTIVITY"),
        "RELATION_FAMILIES": val["STATISTICS"].get("RELATION_FAMILIES"),
        "TOP_RELATIONS": val["STATISTICS"].get("TOP_RELATIONS"),
        "EDGE_OBJECT_KIND": val["STATISTICS"].get("EDGE_OBJECT_KIND"),
        "CVT": passc.get("CVT"),
        "LITERALS": {
            "distinct": val["STATISTICS"]["NODE_COUNTS"].get("LITERAL"),
            "identity": "(lexical_form, datatype, language). The visible string alone is NOT the "
                        "identity, so \"5\", \"5\"@en and \"5\"^^xsd:int are three nodes.",
        },
        "REVERSE_CANONICALISATION": passd.get("CANONICALISATION"),
        "PREREGISTERED_REVERSE_CHECKS": passd.get("PREREGISTERED_CHECKS"),
        "WHAT_THESE_NUMBERS_ARE_NOT": "this is the FULL Freebase universe as released, not a "
                                      "benchmark subgraph. It is not comparable to the 2.57M-entity "
                                      "RoG WebQSP graph, and neither number is a defect of the "
                                      "other: query subsetting is allowed, corpus subsetting by "
                                      "query is not, and neither graph was built by looking at a "
                                      "question.",
    }

    # ---------- provenance ----------
    prov = {
        "schema": "SOURCE_PROVENANCE/v1",
        "name": "CRAG_FREEBASE_CANONICAL",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "SOURCE_OF_RECORD": mirror["TARGET"],
        "custody": mirror.get("CUSTODY"),
        "local_copy": f"{V3}/_acquisition/raw/freebase-rdf-latest.gz",
        "GZIP_MEMBER_STRUCTURE": {
            "members": (members or {}).get("members_found") or (members or {}).get("members"),
            "why_it_matters": "the mirror is a concatenation of complete gzip members, so a worker "
                              "can take a byte range and decode it with no dependency on any other "
                              "worker. Every parallel pass in this build rests on that fact.",
        } if members else None,
        "LINES": {
            "total_rdf_statements": passb["LINES"]["stage1_lines_recorded"],
            "metadata_statements": passa["ROW_RECONCILIATION"].get("total_rows")
            if isinstance(passa.get("ROW_RECONCILIATION"), dict) else None,
            "structural_statements": passb["LINES"]["structural_lines"],
        },
        "SUBJECT_CONTIGUITY": {
            "distinct_subject_blocks": contig["DISTINCT_SUBJECTS_N"],
            "reopened_subject_blocks": contig.get("REOPENED_SUBJECT_BLOCKS_N"),
            "why_it_matters": "zero reopened blocks is what removed the source-side external sort "
                              "from this build and licensed every member-parallel pass.",
        },
        "NODE_IDENTITY": uidsp["UID_SCHEME"],
        "SUBJECT_UID_COLLISIONS": uidsp["SUBJECTS"],
        "SCHEMA_FREEZE_HASH": frz["FREEZE_HASH"],
        "CONTRACT_HASHES": contracts,
        "PASS_REPORTS": {
            "PASS_A": "V3_PASS_A_SCHEMA_METADATA.json + V3_PASS_A_VALIDATION.json",
            "PASS_A_FREEZE": "V3_PASS_A_SCHEMA_FREEZE.json",
            "PASS_B": "V3_PASS_B_REPORT.json",
            "PASS_C": "V3_PASS_C_NODE_UNIVERSE.json",
            "PASS_D": "V3_PASS_D_CANONICAL_EDGES.json",
            "TYPE_MIRROR_FINDING": "V3_TYPE_MIRROR_FINDING.json",
            "ASSEMBLY": "V3_CANONICAL_ASSEMBLY.json",
            "VALIDATION": "V3_CANONICAL_VALIDATION.json",
            "IDIR_ORACLE": "V3_IDIR_ORACLE_COMPARISON.json" if idir else "not run",
        },
        "INDEPENDENT_CROSS_CHECK": {
            "oracle": "IDIR Freebase (Zenodo 7909511), built from the same dump by an unrelated "
                      "parser. Frozen role: audit oracle and structural validation reference, "
                      "never the canonical graph.",
            "coverage_pct": (idir or {}).get("AGREEMENT", {}).get("coverage_pct"),
            "verdict": (idir or {}).get("GATE", {}).get("VERDICT"),
            "retained_orientation": (idir or {}).get("ORIENTATION_ANSWER"),
            "band_verification": (idir or {}).get("BAND_VERIFICATION"),
        },
        # A freeze that omits a known open question is worse than one that has none: the record is
        # what a later reader trusts, and silence in it reads as "nothing was found".
        "OPEN_FINDINGS": [
            {"record": "V3_TYPE_MIRROR_FINDING.json",
             "summary": "type membership is asserted in both directions by two predicates that "
                        "landed in two different layers: type.object.type in metadata "
                        "(254,946,431 rows) and its declared inverse type.type.instance in the "
                        "edge table (254,882,143 edges, 12.36% of all canonical edges). Nothing "
                        "is lost and neither layer is internally duplicated, but a consumer "
                        "reading both sees type membership twice.",
             "disposition": "OPEN. Both layers retained as built; no removal, merge or reweighting "
                            "was applied."}
            if os.path.exists(f"{V3}/V3_TYPE_MIRROR_FINDING.json") else None,
            {"record": "V3_IDIR_SOURCE_DIVERGENCE.json",
             "summary": "the pre-registered IDIR oracle gate returned STOP at 95.662% coverage. "
                        "The shortfall is characterised, not outstanding: 3.184% of IDIR triples "
                        "have an endpoint absent from our node universe, 0.740% sit on predicates "
                        "where IDIR simply holds more than our source does, and 0.427% is "
                        "residue. Our build is provably lossless from its source (PASS B accounts "
                        "for all 3,008,314,716 lines with zero dropped or malformed, and 784,928 "
                        "relations survive to the canonical table unchanged). The divergence is "
                        "SYMMETRIC -- we hold ~1.10M facts IDIR lacks against ~5.86M it holds "
                        "that we lack -- so neither source is a subset of the other and IDIR is "
                        "not a preprocessing of the same bytes.",
             "disposition": "DECIDED: FREEZE_WITH_DIVERGENCE_RECORDED. The gate was NOT "
                            "retargeted or relaxed -- it stands as written and still returns "
                            "STOP; this artifact records that it was overridden, and why. The "
                            "lock's MITIGATION clause named this comparison as the corroboration "
                            "of the Internet Archive mirror, and that corroboration is NOT "
                            "discharged, because its premise is falsified rather than merely "
                            "unmet. Transfer fidelity from the Internet Archive remains proven by "
                            "digest; IA-versus-Google does not, and cannot by this route. The "
                            "caveat travels with this dataset."}
            if os.path.exists(f"{V3}/V3_IDIR_SOURCE_DIVERGENCE.json") else None,
        ],
        "REPRODUCIBILITY": {
            "python": uidsp["UID_SCHEME"]["python"],
            "PYTHONHASHSEED": uidsp["UID_SCHEME"]["PYTHONHASHSEED"],
            "note": "the node UID is CPython hash() over the namespace-stripped identifier, so the "
                    "interpreter version and hash seed are part of the dataset identity, not of "
                    "the build environment. A rebuild under a different interpreter produces a "
                    "different but equally valid UID space; the strings, not the UIDs, are the "
                    "portable identity.",
        },
    }

    # The paper will describe this dataset, and the strongest supportable claim is narrower than
    # the one it would be natural to write. "We reconstructed Google's final Freebase dump" cannot
    # be supported: the IDIR comparison that was meant to corroborate the mirror against an
    # independent party instead showed the two describe different universes, and Google's own
    # endpoint returns 403. What IS supported is exact transfer fidelity from a named,
    # digest-locked Internet Archive artifact. The wording lives here rather than being recalled at
    # writing time on purpose -- the artifact should carry the limits of its own claim.
    prov["HOW_TO_DESCRIBE_THIS_DATASET"] = {
        "supported_claim":
            "We construct the canonical CRAG Freebase universe from the freebase-rdf-latest.gz "
            "snapshot preserved by the Internet Archive, whose downloaded artifact was verified "
            "against the archive-published size and checksums.",
        "and_separately":
            "IDIR Freebase (Zenodo 7909511) was used as an independent structural oracle. It "
            "reproduced 95.662% of its own FB+CVT-REV triples against our graph; the residual "
            "divergence is symmetric, so it corroborates structure without establishing shared "
            "provenance.",
        "NOT_supported": [
            "that this reconstructs the official final Google Freebase dump",
            "that the Internet Archive copy is byte-identical to what Google published",
            "that IDIR independently verifies this mirror's provenance",
        ],
        "why":
            "transfer fidelity from the Internet Archive is proven by MD5, SHA1, CRC32 and byte "
            "count against values locked before transfer. Equality with Google's original is a "
            "historical claim about an artifact its publisher no longer serves, and the single "
            "route to corroborating it -- an independent preprocessing of the same bytes -- turned "
            "out not to be one. See V3_IDIR_SOURCE_DIVERGENCE.json.",
        "instructed_by": "user, 2026-09-07, at the freeze decision",
    }

    manifest = {
        "schema": "CANONICAL_MANIFEST/v1",
        "name": "CRAG_FREEBASE_CANONICAL",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": ("FROZEN_UNDER_FORCE" if not val["ALL_CHECKS_PASS"]
                   else "FROZEN_WITH_ORACLE_GATE_OVERRIDDEN"
                        if idir_verdict in ("STOP", "INVESTIGATE") else "FROZEN"),
        "root": CAN,
        "TABLES": tables,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "VALIDATION": {
            "report": "V3_CANONICAL_VALIDATION.json",
            "ALL_CHECKS_PASS": val["ALL_CHECKS_PASS"],
            "checks": {k: v.get("ok") for k, v in val["CHECKS"].items()},
            "failing": failing,
        },
        "ASSEMBLY": {k: asm[k] for k in ("NODES", "EDGES", "RELATIONS", "NAME_INDEX")
                     if k in asm},
        "FILES": digests,
        "hash_algorithm": "sha256 over each file, whole file, no normalisation",
    }
    if force and val["ALL_CHECKS_PASS"] and idir_verdict in ("STOP", "INVESTIGATE"):
        manifest["ORACLE_GATE_OVERRIDE"] = {
            "gate": "IDIR oracle comparison",
            "verdict": idir_verdict,
            "coverage_pct": idir_cov,
            "pre_registered_thresholds": ">=99.9 PASS / 99.0-99.9 INVESTIGATE / <99.0 STOP",
            "frozen_before_the_number_was_known": True,
            "overridden_deliberately": True,
            "decision_record": "V3_IDIR_SOURCE_DIVERGENCE.json",
            "why": "the gate's premise is falsified, not merely unmet: IDIR's FB+CVT-REV is not a "
                   "preprocessing of the same bytes as our source, so no correct build of our "
                   "source could satisfy it. Measured symmetrically -- we hold facts IDIR lacks "
                   "as well as the reverse -- and our build is provably lossless from a "
                   "byte-verified source. Every validation check passes on its own terms.",
            "what_this_does_NOT_establish": "that the Internet Archive copy equals what Google "
                   "published. That was to be corroborated by this very comparison and now cannot "
                   "be, by this route. The custody caveat travels with this artifact UNDISCHARGED.",
        }

    if force and not val["ALL_CHECKS_PASS"]:
        manifest["FORCED"] = {
            "reason": "written with --force while checks were failing",
            "idir_oracle_gate": idir_verdict,
            "idir_oracle_coverage_pct": idir_cov,
            "failing_checks": failing,
            "reading": "this dataset has NOT passed STEP 6. Do not treat this manifest as a "
                       "statement that it did.",
        }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["MANIFEST_HASH"] = hashlib.sha256(payload).hexdigest()
    manifest["MANIFEST_HASH_NOTE"] = ("sha256 over this manifest with this field absent. Any later "
                                      "change is a new record, never an edit.")

    for path, body in ((STATS, stats), (PROV, prov), (MANIFEST, manifest)):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(body, fh, indent=2)
        os.replace(tmp, path)

    print(json.dumps({"status": manifest["status"], "TABLES": tables,
                      "total_gb": manifest["total_gb"],
                      "MANIFEST_HASH": manifest["MANIFEST_HASH"],
                      "NODES": stats["NODES"], "EDGES_TOTAL": stats["EDGES_TOTAL"]}, indent=1))


if __name__ == "__main__":
    main()
