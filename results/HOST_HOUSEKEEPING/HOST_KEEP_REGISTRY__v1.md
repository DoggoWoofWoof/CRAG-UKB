# Host keep registry (2026-10-05)

User rules: (1) reduce the host footprint before touching the laptop; (2) **keep enough data to run the experiments**.
`scratchpad/_housekeep.py` (every 3 h) only ever deletes what is (a) listed in a `_host_rm.py` PHASE, (b) present in the verified HF restore index, and (c) not in `data/_cache/HOUSEKEEP_PINS.json`.
Everything below that an experiment reads is therefore either not in a PHASE list or pinned.  Sizes are GiB measured on the host 2026-10-05 (robocopy dry run, junction-aware).

| host path | GiB | HF copy | read by | release condition |
|---|---:|---|---|---|
| `data/freebase` (the served tree, junction target of `ws/data/final_canonical/freebase`) | 73.78 | no (laptop master + this host; raw lineage `freebase_v3` on the laptop) | FREEBASE_SCALE encode (running), level-0 V-cycle gate (queued) | never while FREEBASE_SCALE is open |
| `ws/data/final_canonical/hotpotqa` + `2wiki` | 22.43 + 27.43 | yes, once `_hf_big.py UPLOAD g2_text` is verified (zstd) | text-L1 held-out evaluation (planned, user-gated), `_l1x_txt_score.py` | after the held-out read, or on the user's word: restore is `_hf_big.py URLS` -> `_fbx_hf_pull.py` -> `_hf_big_unpack.py` (~15 min) |
| `ws/data/final_canonical/webqsp` + `metaqa` | 8.83 + 0.45 | yes (`g1_small`) | WebQSP step-17 job (running), L3_HOST MetaQA inputs | keep |
| `ws/data/freebase_scale/names` | 11.74 | yes (old account index) | encode job (running) | after the encode finishes and its products are on HF |
| `ws/data/freebase_scale/enc` | 6.62 (growing) | yes, incremental (3-hourly sync) | encode job writes it | same as names |
| `models/hf` (gte-Qwen2 etc.) | 7.44 | public models, re-downloadable | encode job | after the encode |
| `ws/data/l1_canonical` | 1.46 | records on HF | host L1 harnesses | keep |
| `ws/results/L1_HOST/parts` | 4.50 | yes | level-0 regression (7 frozen WebQSP maps) | after the level-0 gate |
| `ws/results/FREEBASE_SCALE/{ml2,h2l}` | in 1.44 | yes | level-0 regression inputs — **pinned** | unpin (`HOUSEKEEP_PINS.json`) when the gate is done; the next cycle then deletes them |
| WSL `ext4.vhdx` (CSR 17.4 + fb_l1 3.1 + fb_work 2.8) | 31.88 | fb_l1/fb_work yes; CSR rebuildable (`_ooc_e1.py FB_BUILD`) | level-0 V-cycle gate (queued) | after the gate: delete CSR, `wsl --shutdown` is only possible when WSL is idle (compaction rule in `_housekeep.py`) |

Already released: `ooc_tmp`, `freebase_scale/ner` (15.45 GB), `FBX_CALIB/bundle_v1` (8.23 GB), `freebase_scale/pq` + `ivf_transfer` (2.64 GB, nothing queued reads them), WSL slack 36 GB (compaction).  Host free 20 -> 93 GB.

Rule for new jobs: before submitting any job whose inputs sit under a PHASE directory, add the directory to `HOUSEKEEP_PINS.json`; remove the pin when the job is done.
