"""Mirror the REPORT LAYER of data/final_canonical into results/data_audit/final_canonical_v1.

    python scratchpad/final_canonical_build/mirror_audit.py

data/ is gitignored, results/ is not, so the small json/md reports are mirrored while the bulky tables
(nodes.jsonl, queries/, eval_*.jsonl, node_id_map_legacy.json, reuse_map/) stay only under data/final_canonical/.
Superseded/alias subdirectories are mirrored too (their records are small and are part of the audit trail).
Nothing is deleted from the mirror: files that no longer exist upstream are listed in README.txt as stale.
"""
import os, json, shutil, time

SRC = "data/final_canonical"
DST = "results/data_audit/final_canonical_v1"
SUPERSEDED = f"{DST}/_superseded_mirror"
ROOT_FILES = ("MANIFEST.json", "CANONICALIZATION_AUDIT.json", "CANONICALIZATION_AUDIT.md",
              "SOURCE_CONTRACTS_FOR_REVIEW.md", "ARTIFACT_REUSE_REPORT.md", "DOWNSTREAM_REBUILD_POLICY.md",
              "_APPROVALS.json", "_APPROVAL_CRITERIA.md",
              # §F stale-hash sweep: the record that the rev1 -> rev2 re-pin left no live artifact still
              # asserting a superseded hash, and that every surviving mention is deliberate.
              "STALE_HASH_SWEEP.json",
              # the webqsp acceptance gate is the authority file for the whole webqsp lane and is
              # hand-maintained, so it must reach the tracked mirror (data/ is gitignored).
              "_WEBQSP_ACCEPTANCE_GATE.json",
              "_superseded_sidecars/SUPERSESSION.json", "_superseded_sidecars/GOLD_SEMANTICS_preSourceFix.json",
              # The TEXTUALIZATION_REV 1 supersession record is the ONLY place the pre-rev2 corpus hashes are
              # explained rather than merely stored.  Its directory lives under data/ (gitignored) because the
              # rev1 node tables are 9 GB, but the record itself is a few KB and must survive in the tracked
              # tree -- otherwise the hash history it documents is unreadable from the repo (2026-09-06).
              "_superseded_textualization_rev1/SUPERSEDED.md",
              "_superseded_textualization_rev1/2wiki/INSTALL_RECORD.json",
              # NOTE: there is deliberately no hotpotqa/INSTALL_RECORD.json.  hotpotqa's rev2 install
              # (2026-09-05) predates that record format, which 2wiki's install introduced a day later.
              # It is listed here only as a comment: writing one now would be a reconstruction of an event
              # nobody recorded, and leaving it in ROOT_FILES made every mirror run report a phantom
              # missing_upstream entry (2026-09-06).
              "_superseded_textualization_rev1/2wiki/build_info.json",
              "_superseded_textualization_rev1/2wiki/integrity_report.json",
              "_superseded_textualization_rev1/hotpotqa/build_info.json",
              "_superseded_textualization_rev1/hotpotqa/integrity_report.json")
DS_FILES = ("SOURCE_CONTRACT.json", "dataset_manifest.json", "integrity_report.json", "legacy_comparison.json",
            "query_independence_test.json", "build_info.json", "status.json", "FREEBASE_SOURCE_PROPOSAL.md",
            # the webqsp audits are written deliverables, not build outputs; without them here they
            # would exist only under the gitignored data/ tree.  The two SOURCE_*MATRIX files are named by
            # webqsp/status.json -> lane_status_2026_09_06.source_of_evidence as the evidence base for the
            # WEBQSP_ROG_MID = UNRESOLVED_SOURCE verdict; they were already present in the mirror but were
            # never in this list, so they would have drifted silently instead of being refreshed (2026-09-06).
            "WEBQSP_REPRESENTATION_AUDIT.md", "WEBQSP_FREEBASE_PIPELINE_AUDIT.md",
            "WEBQSP_SOURCE_LINEAGE_MATRIX.md", "WEBQSP_SOURCE_MATRIX.json",
            # per-shard sha256 verification for the 29 RoG parquet files, cited by
            # _WEBQSP_ACCEPTANCE_GATE.json.  These used to exist ONLY in the mirror, so the
            # stale sweep kept exiling them to _superseded_mirror/; they now have an upstream home.
            "ROG_WEBQSP_SOURCE_PROVENANCE.json", "ROG_CWQ_SOURCE_PROVENANCE.json",
            # empty-text forensics + the TEXTUALIZATION_REV 2 diff/verify records.  Written deliverables, like the
            # two webqsp audits above, so they get an upstream home rather than being exiled by the stale sweep.
            # 2wiki's per-record table is NOT listed: its upstream copy is the 106.7 MB full-fidelity version and
            # results/ is tracked, so only its .md/.json summaries are mirrored (hotpot's 94-row table is 0.1 MB
            # and small enough to track).
            "EMPTY_TEXT_AUDIT.md", "EMPTY_TEXT_SUMMARY.json",
            "EMPTY_TEXT_94_AUDIT.md", "EMPTY_TEXT_94_SUMMARY.json", "EMPTY_TEXT_94_AUDIT.jsonl",
            "REV2_DIFF_AUDIT.json", "REV2_VERIFY.json", "REV2_REUSE_ATTRIBUTION.json", "TEXTUALIZATION_REV2_REPORT.md",
            # --- V1 build + NSM acquisition branch, 2026-09-06 ---
            # The Route-A acquisition record (pre-registration written before any payload byte, the
            # bytes/sha256 provenance, the MID-preservation test that converted [I]->[V]) and the V1
            # section 1/2 outputs.  All are written deliverables; without them here they live only
            # under the gitignored data/ tree and the branch would be unreproducible from the repo.
            "NSM_ACQUISITION_PREREGISTRATION.json", "NSM_ACQUISITION_PROVENANCE.json",
            "NSM_MID_PRESERVATION.json", "NSM_UNION_RESULT.json", "NSM_NAMING_RESOURCE_AUDIT.json",
            "NSM_PROJECTION_DIAGNOSTIC.json", "NAME_MAPPING_COVERAGE.json",
            "ROG_ATTRIBUTABLE_COLLAPSE.json", "COLLISION_DEGREE_TEST.json",
            "ROG_UNION_REBUILD.json", "V1_TABLES_REPORT.json", "V1_RELATION_LABEL_ANALYSIS.json",
            # section 7 adaptive labels + section 3 classification: the freeze document, the run,
            # the audit and the node_kind freeze marker.  The freeze doc in particular must be in
            # the tracked tree -- it is the record that the thresholds predate the results.
            "V1_RELATION_LABELS.json", "V1_SECTION3_CLASSIFIER_FREEZE.json",
            "V1_CLASSIFICATION_REPORT.json", "V1_CLASSIFICATION_AUDIT.json",
            "V1_NODE_KIND_FROZEN.json",
            # the CVT encoding gate.  The preregistration is listed FIRST and deliberately:
            # it is the record that the decision cell was named before any number existed.
            "V1_CVT_GATE_PREREGISTRATION.json", "V1_RENDER_REPORT.json",
            "V1_CVT_GATE_RESULT.json", "V1_CVT_GATE_DIAGNOSTICS.json",
            # step 7: the ALL_NODE decision and the cost table it was taken on.
            "V1_ALL_NODE_COST_TABLE.json", "V1_RETRIEVAL_ROLE_DECISION.json",
            # V3 lane scoping: source assessment and the literal-deficit measurement.
            "V3_SOURCE_ASSESSMENT.json", "V3_LITERAL_DEFICIT.json",
            "V3_DESIGN_LOCK.json", "V3_PROBE_SETS.json", "V3_ACQUISITION_RECORD.json",
            "V3_EXTRACTION_RECORD.json", "V3_CHECK3_CHECK5_RELATIONS.json",
            "V3_CHECK3B_RELATION_MASS.json", "V3_CHECK3C_GOLD_INCIDENCE.json",
            "V3_CHECK2_CHECK4_JOIN.json", "V3_CHECK7_CVT_COVERAGE.json",
            "V3_CHECK6_METADATA_COVERAGE.json", "V3_WHY_CVTS_DROPPED.json",
            "V3_IDIR_VERDICT.json", "V3_CANONICAL_CONTRACT.json",
            "V3_RAW_MIRROR_PROVENANCE_LOCK.json", "V3_ORACLE_DISTILLATION.json",
            "V3_DISK_RECLAMATION.json",
    "V3_SOURCE_STRUCTURE_PROBE.json", "V3_CANONICAL_CONTRACT_V2.json",
            "V3_SUBJECT_CONTIGUITY.json", "V3_RAW_MIRROR_ACQUISITION.json",
            "V3_PASS_A_SCHEMA_METADATA.json", "V3_GZIP_MEMBER_INDEX.json",
            "V3_CANONICAL_CONTRACT_V3.json", "V3_PASS_A_SCHEMA_FREEZE.json",
            "V3_PASS_A_VALIDATION.json", "V3_NODE_UID_SPACE.json",
            "V3_PASS_B0_PROFILE.json", "V3_PASS_B_REPORT.json",
            # --- V3 passes C/D and the canonical freeze, 2026-09-07 ---
            # PASS C answers the node-count question the whole lane was built to answer, so its
            # report is a deliverable, not a build log.  The RECLAMATION record is listed with it
            # because it is the gate that licensed deleting 16.5 GB of bucket spill: without the
            # reconciliation it carries, the deletion is unaccounted for.
            "V3_PASS_C_NODE_UNIVERSE.json", "V3_PASS_C_RECLAMATION.json",
            "V3_CANONICAL_CONTRACT_V4.json", "V3_PASS_D_CANONICAL_EDGES.json",
            # PASS D measured that Freebase asserts type membership in both directions and that
            # the two directions landed in two different layers of this build.  Recorded as a
            # finding with an OPEN disposition, so the record has to be in the tracked tree:
            # it is the evidence for a decision that has not been taken yet.
            "V3_TYPE_MIRROR_FINDING.json",
            # the three freeze deliverables named by the roadmap, plus the two checks that gate
            # them.  IDIR comparison is listed even though IDIR is an oracle and not the graph:
            # its verdict is the external evidence that the raw build is right.
            "V3_CANONICAL_ASSEMBLY.json", "V3_CANONICAL_VALIDATION.json",
            "V3_IDIR_ORACLE_COMPARISON.json",
    "V3_IDIR_SOURCE_DIVERGENCE.json", "V3_IDIR_MISS_DIAGNOSTIC.json",
    "V3_IDIR_MISS_PARTITION.json", "V3_IDIR_RELATION_DEFICIT.json",
    "V3_IDIR_SYMMETRY.json", "V3_IDIR_ENTITY_COVERAGE.json",
    "V3_IDIR_G_NAMESPACE_DIAGNOSTIC.json",
            # these three are written INSIDE canonical/ beside the tables they describe, so they
            # are named by relative path the way reuse_map/index.json below is.  The tables
            # themselves are not mirrored: CANONICAL_MANIFEST carries their sha256s, which is
            # what the tracked tree needs to detect drift.
            "canonical/GRAPH_STATISTICS.json", "canonical/SOURCE_PROVENANCE.json",
            "canonical/CANONICAL_MANIFEST.json",
            # v1/*.parquet are NOT listed: 78 MB of build output, not a report.  V1_TABLES_REPORT
            # carries their counts, byte sizes and NODE_ORDER/CONTENT hashes, which is what the
            # tracked mirror needs to detect drift.
            "reuse_map/index.json")   # the per-node reuse shards stay under data/ ; only their index is small
SUB_FILES = ("SUPERSESSION.json", "ALIAS.json", "status.json", "dataset_manifest.json", "integrity_report.json",
             "build_info.json", "legacy_comparison.json", "query_independence_test.json", "SOURCE_CONTRACT.json")
DATASETS = ("metaqa", "2wiki", "musique", "squad", "hotpotqa", "webqsp",
            # V3 lane: CRAG_FREEBASE_CANONICAL. Records only -- the acquired archive and the
            # probe parquets stay under the gitignored data/ tree.
            "freebase_v3")
# small record-only subdirectories that are part of the audit trail
SUBDIRS = {"2wiki": ("_superseded_context_union_398354",), "hotpotqa": ("_legacy_alias_hotpotqa_clean",)}


def cp(s, d):
    if not os.path.exists(s):
        return None
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    return d


def main():
    os.makedirs(DST, exist_ok=True)
    written, missing = [], []
    for f in ROOT_FILES:
        (written if cp(f"{SRC}/{f}", f"{DST}/{f}") else missing).append(f)
    for ds in DATASETS:
        for f in DS_FILES:
            r = cp(f"{SRC}/{ds}/{f}", f"{DST}/{ds}/{f}")
            if r: written.append(f"{ds}/{f}")
        for sd in SUBDIRS.get(ds, ()):
            for f in SUB_FILES:
                r = cp(f"{SRC}/{ds}/{sd}/{f}", f"{DST}/{ds}/{sd}/{f}")
                if r: written.append(f"{ds}/{sd}/{f}")
    # A mirrored file whose upstream original is gone is STALE.  It is never deleted: it is MOVED into
    # _superseded_mirror/<same relative path> so the live tree has exactly one entry per live upstream file
    # (the same supersede-don't-delete rule the data tree uses), and that subtree is excluded from the scan.
    stale, moved = [], []
    for root, _, files in os.walk(DST):
        if f"{SUPERSEDED}/" in (root.replace("\\", "/") + "/"):
            continue
        for f in files:
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(DST) + 1:]
            if rel == "README.txt":
                continue
            if not os.path.exists(f"{SRC}/{rel}"):
                stale.append(rel)
                dest = f"{SUPERSEDED}/{rel}"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                k = 0
                while os.path.exists(dest):        # never overwrite and never delete: park the duplicate beside it
                    k += 1
                    dest = f"{SUPERSEDED}/{rel}.dup{k}"
                shutil.move(p, dest)
                moved.append(f"{rel} -> {dest[len(DST) + 1:]}")
    for root, dirs, files in os.walk(DST, topdown=False):     # drop directories emptied by the moves
        if root.replace("\\", "/") in (DST, SUPERSEDED) or f"{SUPERSEDED}/" in (root.replace("\\", "/") + "/"):
            continue
        if not os.listdir(root):
            os.rmdir(root)
    man = json.load(open(f"{SRC}/MANIFEST.json", encoding="utf-8"))
    with open(f"{DST}/README.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "Mirror of the report layer of data/final_canonical (Track B, canonical_v1).\n"
            "Bulky tables live only under data/final_canonical/ (gitignored): nodes.jsonl, queries/,\n"
            "eval_*.jsonl, node_id_map_legacy.json and the per-node reuse_map/shard_*.jsonl files\n"
            "(only reuse_map/index.json, which carries the totals, is mirrored here).\n\n"
            f"FULL_CANONICAL_V1_STATUS: {man.get('FULL_CANONICAL_V1_STATUS')}\n"
            f"WEBQSP_STATUS:            {man.get('WEBQSP_STATUS')}\n"
            f"CANONICAL_V1_STATUS:      {man.get('CANONICAL_V1_STATUS')}\n"
            f"BUILT:                    {', '.join(man.get('READY') or [])}\n"
            "BLOCKED:                  webqsp (BLOCKED_PENDING_FREEBASE_SOURCE; the builder refuses it\n"
            "                          unconditionally -- see webqsp/FREEBASE_SOURCE_PROPOSAL.md)\n"
            "NOTE:                     LOCKED_6_OF_6 must NOT be written while webqsp is blocked.\n"
            "                          2wiki was REBUILT on the full article universe (5,989,847); the superseded\n"
            "                          398,354-node table is preserved under\n"
            "                          data/final_canonical/2wiki/_superseded_context_union_398354/.\n"
            "                          hotpotqa_clean is a legacy alias only; one live directory per dataset.\n"
            f"\n{len(written)} files mirrored"
            + (f"\nSTALE (upstream gone) -- MOVED, never deleted, into _superseded_mirror/: {moved}"
               if moved else "\nno stale files")
            + f"\nMirrored {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    print(json.dumps({"written": len(written), "missing_upstream": missing,
                      "stale_moved_to_superseded_mirror": moved}, indent=1))


if __name__ == "__main__":
    main()
