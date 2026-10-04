"""H4_SK low-memory partitioning validation (experimental; never replaces the canonical partitioner).

Question: can canonical L1's H4_SK hard-partition stage run without the whole hypergraph in RAM,
with the H4_SK structure held exactly fixed and the L1 scientific contract unchanged?

Modules
  contract_freeze   step 1  record the exact current H4_SK / Mt-KaHyPar / P50 contract from the sources (no inferred values)
  hgr               H4_SK.npz -> hMETIS .hgr (weighted nets, canonical hyperedge order) + the representation-independent structure digest
  stream            H4_SK_STREAM_V1: node-centric shards over contiguous canonical-position ranges, resumable conversion, reconstruction gate
  freight           Candidate A: KaHIP/FREIGHT (freight_con for KM1) -- build, run, checkpoint/kill/resume equivalence tests, partition import
  l1_downstream     replay cache + frozen L1 BASE/SAFE on any partition tag (outputs under results/L1_LOWMEM, canonical artefacts untouched)
  report            H4_SK_LOW_MEMORY_PARTITIONING_REPORT

Everything writes under data/l1_lowmem/<ds>/ and results/L1_LOWMEM/; experiment partitions are stored beside the canonical
one under data/l1_canonical/<ds>/parts/ with the tag prefix LOWMEM__ (the canonical H4_SK.* and the L1 manifests are never touched).
"""
