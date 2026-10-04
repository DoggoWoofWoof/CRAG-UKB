Mirror of the report layer of data/final_canonical (Track B, canonical_v1).
Bulky tables live only under data/final_canonical/ (gitignored): nodes.jsonl, queries/,
eval_*.jsonl, node_id_map_legacy.json and the per-node reuse_map/shard_*.jsonl files
(only reuse_map/index.json, which carries the totals, is mirrored here).

FULL_CANONICAL_V1_STATUS: READY_5_OF_6_SOURCE_CONTRACTS
WEBQSP_STATUS:            WEBQSP_BLOCKED_ON_FREEBASE_SOURCE
CANONICAL_V1_STATUS:      BUILT_5_OF_6 (webqsp BLOCKED_PENDING_FREEBASE_SOURCE)
BUILT:                    2wiki, hotpotqa, metaqa, musique, squad
BLOCKED:                  webqsp (BLOCKED_PENDING_FREEBASE_SOURCE; the builder refuses it
                          unconditionally -- see webqsp/FREEBASE_SOURCE_PROPOSAL.md)
NOTE:                     LOCKED_6_OF_6 must NOT be written while webqsp is blocked.
                          2wiki was REBUILT on the full article universe (5,989,847); the superseded
                          398,354-node table is preserved under
                          data/final_canonical/2wiki/_superseded_context_union_398354/.
                          hotpotqa_clean is a legacy alias only; one live directory per dataset.

152 files mirrored
no stale files
Mirrored 2026-09-07T06:39:56Z
