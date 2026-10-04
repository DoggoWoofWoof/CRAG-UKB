"""Cascade step 5, control: is FACC1's reach of 1,623 a real result or a failed join?

    PYTHONHASHSEED=0 python .../facc1_control.py

WHY
  1,623 nodes out of 17,809,849 is small enough that it has to be distrusted until controlled. The
  P2671 zero was tested the same way and the test earned its keep: it showed the zero was real and
  identified WHY (disjoint identifier spaces), which no amount of staring at the code would have.

  The control is the same one: take every MID the source mentions and look it up in the FULL 302M
  universe, not just the residue. If the source's MIDs are largely present in the graph, then a low
  reach into the RESIDUE is a fact about the residue -- these entities were nameable, so they are
  not in the residue -- rather than a broken lookup.

BOTH FILES, BECAUSE THEY ARE NOT THE SAME MID SET
  entity_list  mid \\t surface \\t count      -- aliases plus FACC1 observation counts
  surface_map  surface \\t P(mid|surface) \\t mid
  Only entity_list has been scanned so far. surface_map is the larger file (2.49 GB vs 1.63 GB) and
  may carry MIDs entity_list does not, so its reach is measured separately rather than assumed
  equal. Its residue hits are spilled with their probabilities, because P(mid|surface) is the
  surface-ambiguity signal the conservative naming rule needs.

Streams from the zip. Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ZIP = f"{V3}/_acquisition/facc1/mentions.zip"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
ALLU = f"{V3}/canonical/_node_uids_sorted.npy"
OUT = f"{V3}/_acquisition/facc1_surfacemap_hits"
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
BATCH = 2_000_000
TAB = chr(9)
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
os.makedirs(OUT, exist_ok=True)
z = np.load(CACHE)
U, KC = z["U"], z["KC"]
NU = len(U)
ALL = np.load(ALLU)
NA = len(ALL)
print(f"residue {NU:,}   full universe {NA:,}  ({time.time()-t0:.0f}s)", flush=True)


def scan(entry, mid_field, keep_prob):
    """One streaming pass: distinct MIDs, presence in the universe, presence in the residue."""
    p = subprocess.Popen([SEVENZ, "x", "-so", ZIP, entry],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
    rows = bad = 0
    part = 0
    seen_chunks, uni_chunks, res_chunks = [], [], []
    b_mid, b_surf, b_p = [], [], []

    def flush():
        nonlocal part
        if not b_mid:
            return
        u = np.fromiter(map(hash, b_mid), dtype=np.int64, count=len(b_mid))
        seen_chunks.append(np.unique(u))
        j = np.searchsorted(ALL, u); np.clip(j, 0, NA - 1, out=j)
        m = ALL[j] == u
        if m.any():
            uni_chunks.append(np.unique(u[m]))
        j = np.searchsorted(U, u); np.clip(j, 0, NU - 1, out=j)
        idx = np.flatnonzero(U[j] == u)
        if idx.size:
            res_chunks.append(u[idx])
            if keep_prob:
                il = idx.tolist()
                pq.write_table(pa.table({
                    "node_uid": pa.array(u[idx], pa.int64()),
                    "node_id": pa.array([b_mid[i] for i in il]),
                    "surface": pa.array([b_surf[i] for i in il]),
                    "p_mid_given_surface": pa.array([b_p[i] for i in il], pa.float64())}),
                    f"{OUT}/part_{part:04d}.parquet", compression="zstd")
                part += 1

    for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""):
        rows += 1
        f = raw.rstrip("\r\n").split(TAB)
        if len(f) < 3:
            bad += 1
            continue
        mid = f[mid_field]
        if not mid or mid[0] not in "mg":
            bad += 1
            continue
        b_mid.append(mid)
        if keep_prob:
            b_surf.append(f[0])
            try:
                b_p.append(float(f[1]))
            except ValueError:
                b_p.append(float("nan"))
        if len(b_mid) >= BATCH:
            flush()
            b_mid, b_surf, b_p = [], [], []
            print(f"  {entry[:22]} rows={rows:,} ({time.time()-t0:.0f}s)", flush=True)
    flush()
    p.stdout.close(); p.wait()
    seen = np.unique(np.concatenate(seen_chunks)) if seen_chunks else np.array([], np.int64)
    uni = np.unique(np.concatenate(uni_chunks)) if uni_chunks else np.array([], np.int64)
    res = np.unique(np.concatenate(res_chunks)) if res_chunks else np.array([], np.int64)
    return rows, bad, seen, uni, res


out = {}
for entry, fld, kp in (("entity_list_file_freebase_complete_all_mention", 0, False),
                       ("surface_map_file_freebase_complete_all_mention", 2, True)):
    rows, bad, seen, uni, res = scan(entry, fld, kp)
    by = {}
    if res.size:
        kc = KC[np.searchsorted(U, res)]
        by = {KIND[i]: int((kc == i).sum()) for i in range(3)}
    out[entry] = {"rows": rows, "malformed": bad,
                  "distinct_mids": int(seen.size),
                  "present_in_302M_universe": int(uni.size),
                  "UNIVERSE_COVERAGE_PCT": round(100 * uni.size / seen.size, 3) if seen.size else 0,
                  "present_in_residue": int(res.size),
                  "residue_by_kind": by}
    print(f"\n{entry}\n  rows {rows:,}  distinct MIDs {seen.size:,}\n"
          f"  in universe {uni.size:,} ({out[entry]['UNIVERSE_COVERAGE_PCT']}%)\n"
          f"  in residue  {res.size:,}  {by}", flush=True)
    del seen, uni, res

n_ent = int((KC == 0).sum())
rec = {"schema": "FACC1_CONTROL/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "reads canonical/ and overlay_v1/, writes neither.",
       "WHY": ("a reach of 1,623 out of 17,809,849 is small enough to distrust until controlled. "
               "The control asks whether the source's MIDs are present in the graph AT ALL; if they "
               "are, a low reach into the residue is a fact about the residue, not a broken join."),
       "files": out,
       "residue_entity_mids": n_ent,
       "INTERPRETATION_RULE": ("high universe coverage + low residue reach = FACC1 mentions the "
                               "entities that were nameable, which is exactly the complement of the "
                               "residue. Low universe coverage would instead mean the join is "
                               "wrong and no conclusion could be drawn."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FACC1_CONTROL.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{time.time()-t0:.0f}s")
