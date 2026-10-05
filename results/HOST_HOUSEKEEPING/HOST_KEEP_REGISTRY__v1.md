# Host keep registry (2026-10-05, updated 10:30 after the level-0 gate)

User rules: (1) reduce the host footprint before touching the laptop; (2) **keep enough data to run the experiments**.
`scratchpad/_housekeep.py` (every 3 h) only ever deletes what is (a) listed in a `_host_rm.py` PHASE, (b) present in the verified HF restore index, and (c) not in `data/_cache/HOUSEKEEP_PINS.json`.
Everything below that an experiment reads is therefore either not in a PHASE list or pinned.  Sizes are GiB measured on the host 2026-10-05 (robocopy dry run, junction-aware).

| host path | GiB | HF copy | read by | release condition |
|---|---:|---|---|---|
| `data/freebase` (the served tree, junction target of `ws/data/final_canonical/freebase`) | 73.78 | no (laptop master + this host; raw lineage `freebase_v3` on the laptop) | FREEBASE_SCALE encode (running), level-0 V-cycle gate (queued) | never while FREEBASE_SCALE is open |
| `ws/data/final_canonical/hotpotqa` + `2wiki` | 22.43 + 27.43 | yes, once `_hf_big.py UPLOAD g2_text` is verified (zstd) | text-L1 held-out evaluation (planned, user-gated), `_l1x_txt_score.py` | after the held-out read, or on the user's word: restore is `_hf_big.py URLS` -> `_hf_big_hostrestore.py` (CHECK passed 2026-10-05: 1,484 objects, 0 mismatches; g2 sample restore 11 files verified, 10 bit-identical to the live copy; fetch 63 MB/s, so the whole group is ~20 min) |
| `ws/data/final_canonical/webqsp` + `metaqa` | 8.83 + 0.45 | yes (`g1_small`, third copy verified) | WebQSP step-17 job (running), L3_HOST MetaQA inputs | keep |
| `ws/data/freebase_scale/names` | 11.74 | yes (old account index) | encode job (running) | after the encode finishes and its products are on HF |
| `ws/data/freebase_scale/enc` | 6.62 (growing) | yes, incremental (3-hourly sync) | encode job writes it | same as names |
| `models/hf` (gte-Qwen2 etc.) | 7.44 | public models, re-downloadable | encode job | after the encode |
| `ws/data/l1_canonical` | 1.46 | records on HF | host L1 harnesses | keep |
| `ws/results/L1_HOST/parts` | 4.50 | yes | level-0 regression (7 frozen WebQSP maps) | after the level-0 gate |
| `ws/results/FREEBASE_SCALE/{ml2,h2l}` | 0.04 + 0.10 | yes (h2l: one record not yet on HF, so the guard blocks it until the next sync) | level-0 regression inputs | **unpinned 2026-10-05 10:07** (gate done); the housekeeping cycle deletes them once HF-verified |
| WSL `ext4.vhdx` (fb_l1 3.1 + fb_work 2.8 incl. the level-0 labels `lab0.i32` 1.2) | 15.02 | fb_l1 / fb_work yes (lab0 uploaded 2026-10-05 10:20, in the live HF restore index) | any follow-up that projects the Freebase partition (WebQSP bridge proxy, K-way sub-partition, SK arm) | keep; the CSR (`fb`, 18.2 GB, `aa.i32` + `xa.i64`) was **deleted 2026-10-05 10:25** after the gate and the vhdx compacted 32.2 -> 15.02 GB; rebuild with `python -u scratchpad/_ooc_e1.py FB_BUILD /home/student2/crag_ooc/fb 12` (streaming from the tree, resumable) |

Already released: `ooc_tmp`, `freebase_scale/ner` (15.45 GB), `FBX_CALIB/bundle_v1` (8.23 GB), `freebase_scale/pq` + `ivf_transfer` (2.64 GB, nothing queued reads them), WSL slack 36 GB (compaction), and after the level-0 gate (E7 report COMPLETE: peak RSS 55.6 GB, KM1 -2.74 % at level 0) the Freebase CSR 18.2 GB with a second compaction (-17.2 GB).  Host free 20 -> 110 GB.

Housekeeping bugs fixed 2026-10-05: the WSL idle check in `_housekeep.py` always failed (`.Replace([char]0,'')` binds the (char,char) overload, so every cycle reported WSL as not idle and automatic compaction never ran); `fstrim` is now called through the shell (`--exec` does not search PATH).

Rule for new jobs: before submitting any job whose inputs sit under a PHASE directory, add the directory to `HOUSEKEEP_PINS.json`; remove the pin when the job is done.
