# CRAG — if the lab host dies: restore on another host and keep going (v1, written 2026-10-03)

Goal: every experiment that is running or planned can be continued on a fresh host from (a) the laptop working tree, (b) the private HF dataset repo `Swastik9895/crag-host-backup`, (c) public model hubs.
Nothing below needs the old host. The HF token never goes on a shared host (presigned/signed URLs instead); on a machine you own you may use `HF_TOKEN` directly.

## 1. Where every piece lives

| piece | size | master | backup / restore source |
|---|---|---|---|
| code (`src/`, `scratchpad/`, `rx.toml`, docs) + result records (text/json ≤ 3 MB) | ~0.3 GB raw | laptop `C:\Users\Swastik\Desktop\CRAG` (git remote `crag-ukb` = github.com/DoggoWoofWoof/CRAG-UKB, branch `unified-crag-architecture`; pushing code there is authorised by the user, 2026-10-05; data stays out of git) | HF `CODE/crag_code_<stamp>.tar.gz` (58 MB, newest 3 kept); `tar xzf` into an empty directory |
| Freebase tree `data/final_canonical/freebase` (CANONICAL_FREEZE `58958f33…`) | 79.2 GB | laptop only | **not on HF** (account quota) — push from the laptop (`rx push`) or copy to a second HF account first |
| Hotpot / 2Wiki substrates `data/final_canonical/{hotpotqa,2wiki}` | 53.5 GB | laptop only | **not on HF** (relay repo deleted 2026-10-03 to free quota) — same as above |
| other canonical substrates (squad, musique, webqsp, metaqa) | small | laptop | laptop |
| models: `Alibaba-NLP/gte-Qwen2-1.5B-instruct` rev `a9af15a6…` (7.1 GB), `naver/splade-cocondenser-ensembledistil` (0.9 GB) | 8 GB | public HF hub | `huggingface_hub.snapshot_download` into `<ws>/models/…` |
| Freebase encode chunks (`data/freebase_scale/enc/…`, 1872 chunks) + codebook + train/holdout | ~16 GB at the end | host | HF groups `fbs_enc`, `fbs_enc_meta` (incremental, sync loop) |
| Freebase names / NER / PQ / IVF-transfer | 12.6 / 15.5 / 2.2 / 0.5 GB | host | HF groups `fbs_names`, `fbs_ner`, `fbs_pq`, `fbs_ivft` |
| calibration bundle `work/FBX_CALIB/bundle_v1` (8.2 GB incl. `ford.npy`) | 8.2 GB | host (regenerable: `_fbx_calib.py bundle`) | HF group `calib` |
| V-cycle state in WSL `/home/student2/crag_ooc/{fb_l1,fb_work}` (levels 4..1, surrogate, labels) | 4.7 GB | host | HF groups `wsl_fb_l1`, `wsl_fb_work` (root `wsl`) |
| Freebase CSR `/home/student2/crag_ooc/fb` (`aa.i32`, `xa.i64`) | 17.4 GB | host | **not on HF** (quota) — rebuild from the tree: `python -u scratchpad/_ooc_e1.py FB_BUILD <wsl dir>/fb 12` (streaming, resumable) |
| L1 text partition maps `results/L1_HOST/parts` (Hotpot/2Wiki K-grid) | 4.1 GB | host | HF group `l1parts` |
| completed FREEBASE_SCALE experiment outputs (ml2 / h2l / ivf / parts_S) and any other result > 8 MB | ~6 GB | host | HF groups `r_ml2 r_h2l r_ivf r_partsS recs_big work_big` |
| small records, logs, checkpoints (incl. L1 text scorer `data/_cache/txtscore_*.partial.npz`) | MB | host | laptop (sync loop, every 3 h) |

`MANIFEST/RESTORE_INDEX.json` in the HF repo lists every object (remote path, size, sha256, restore root `ws` | `wsl`, member list). Every object is verified size + sha256 on upload and again on restore.

## 2. Restore recipe (new host)

1. Provision: Windows + WSL2 Ubuntu-24.04 (or plain Linux), 1 GPU for encode, ≥ 130 GB RAM host for the V-cycle (level 0 needs a 41 GB slot), g++ ≥ 11 with OpenMP, Python 3.12; the pinned envs `mpr-cpu` / `mpr-cu128` (from the mpr project's rx lockfiles) or an equivalent with numpy, scipy, torch cu128, transformers, faiss, spaCy (`[envs.crag-ner]` in `rx.toml`).
2. Code: `tar xzf crag_code_*.tar.gz` (from HF or the laptop) into the project root; `rx.toml` lists what is pushed.
3. Big data from HF — on the laptop (has the token):
   ```
   python scratchpad/_hfx.py GETURLS Swastik9895/crag-host-backup urls.json [group,group,…]     # signed urls, valid ~1 h: run right before the fetch
   ```
   on the new host (no token needed; stdlib only):
   ```
   python -u scratchpad/_hfx.py FETCH urls.json <ws_root> /home/<user>/crag_ooc [group,group,…]  # resumable; sha256 + member sizes verified; extracts into the right root
   ```
   On a host you own: `HF_TOKEN=… python -u scratchpad/_hfx.py FETCH Swastik9895/crag-host-backup <ws_root> <wsl_root>`.
   Restore the groups an experiment needs (table above); a restore state file (`work/HOST_HOUSEKEEPING/restore_dl/restore_state.json`) makes it resumable.
4. From the laptop push what is laptop-only: `data/final_canonical/freebase` (needed by the encode job for node names AND by the CSR rebuild), the Hotpot/2Wiki substrates if L1 text work continues.
5. Models: download the two public checkpoints (revision pinned in the contract: `results/FREEBASE_SCALE/enc/FBX_QWEN_ENCODING_CONTRACT__v1.json`).

## 3. Resuming each experiment

- **Freebase encode (resumable by design):** restore `fbs_enc_meta`, `fbs_enc`, the tree, models, `fbs_names`; then `rx run … python -u scratchpad/_fbx_enc_order.py RETRY 20` (the job that is running now: 1.5 cpu / 6 GB / 0.5 gpu). It skips every chunk whose record + checksum verifies and continues; 1872 chunks, ~5.7 min each on the RTX 4500 Ada (≈ 130 h for the 1416 that remained at 2026-10-03 22:00).
- **FREEBASE_SCALE V-cycle level 0 (addendum 17):** restore `wsl_fb_l1`, `wsl_fb_work` into `/home/student2/crag_ooc`, rebuild the CSR (`FB_BUILD`, §1), then `python -u scratchpad/_ooc_vc.py VCYCLE /home/student2/crag_ooc/fb /home/student2/crag_ooc/fb_l1 /home/student2/crag_ooc/fb_work 4 levels=0-0 anon_gb=40` under `_host_yield.py`, reservation 41 GB. Gate records and the WebQSP reference maps (`parts_S`, `calib`) are in the groups above. Levels 4..1 are already done (`vc_l{1..4}.done.json`).
- **L1 text (Hotpot/2Wiki dev scoring + the one held-out eval):** restore `l1parts`, push the two substrates from the laptop, run `scratchpad/_l1x_txt_score.py` (resumable from `data/_cache/txtscore_*_dev_v1.partial.npz`, synced to the laptop by the sync loop).

## 4. The sync loop (keeps all of this current)

`python -u scratchpad/_host_sync.py --interval-min 180 --max-repo-gb 85` on the laptop (background). Each cycle: (1) small host files modified since the last cycle → laptop (never overwrites a diverging laptop file; host copy goes to `work/HOST_HOUSEKEEPING/conflicts/`), (2) new / settled (> 15 min old) big files in the groups above → HF via presigned URLs, committed and verified, (3) `SYNC_STATUS.json` / `sync_log.jsonl`. A quota guard stops uploads before the free account's ~100 GB limit; a second HF account is then needed. `python scratchpad/_code_snapshot.py` refreshes the HF code snapshot.

## 5. Known gaps (be honest)

- The Freebase tree (79 GB), Hotpot/2Wiki substrates (53 GB) and the 17 GB CSR have **no HF copy** (the free account holds ~100 GB; mpr's 10.6 GB mirror shares it). Laptop master + rebuild covers them; a second HF account would remove the laptop dependency.
- The laptop is the single master of the code tree beyond the HF snapshot; uncommitted work should be committed/pushed by the user.
- The sync loop only runs while the laptop is on and the loop process is alive.
- The HF token (`Swastik9895`, classic write) must stay valid while the loop runs; revoke at huggingface.co/settings/tokens when the encode + sync are finished (or replace by a fine-grained token scoped to `crag-host-backup`).
