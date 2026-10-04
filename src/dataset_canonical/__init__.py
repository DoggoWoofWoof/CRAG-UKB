"""Tooling for the canonical datasets under data/final_canonical/ (frozen 2026-09-12 by
CANONICAL_FREEZE.json): verify_canonical (re-verifies the tree), check_handoff (gates the
handoff prose), freeze_canonical (wrote the freeze), materialize_canonical (the 2026-09-12
materialisation, historical), freebase_v3_closure_status (the separate Freebase layer).

This package does NOT build datasets. The builders of record are
scratchpad/final_canonical_build/*, whose sha256 is pinned inside every
data/final_canonical/<ds>/build_info.json -- moving or editing them would invalidate the
provenance of all six datasets. Earlier tools live under _pre_v3/ (see its README).
"""
