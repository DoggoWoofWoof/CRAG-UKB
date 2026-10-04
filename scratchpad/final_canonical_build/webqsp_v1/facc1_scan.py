"""Cascade step 5, pass 1: how far can FACC1 mentions actually reach into the residue?

    PYTHONHASHSEED=0 python .../facc1_scan.py

WHY THIS PASS EXISTS SEPARATELY FROM THE NAMING PASS
  Every earlier step in this campaign was an ID-mapping source and they collectively reach 13,788
  nodes -- 0.077% of the residue. FACC1 is the first source of a different kind: it is keyed by
  observation of the MID in web text, not by anyone having curated a mapping for it. Whether that
  changes the order of magnitude is the question, and it is answerable by one streaming pass, so it
  is measured before any thresholds are chosen. Choosing thresholds first would invite tuning them
  against a yield I already knew.

THE TWO FILES ARE NOT THE SAME KIND OF THING
  entity_list_file_freebase_complete_all_mention   mid \\t surface \\t count \\t
        Per (mid, surface) OBSERVATION COUNTS. This is the "repeated support" signal.
  surface_map_file_freebase_complete_all_mention   surface \\t P(mid|surface) \\t mid
        A conditional probability, NOT a count: "be my weapon" splits 0.4286 / 0.1429 x4 over five
        MIDs. It measures how ambiguous a surface is, which is a confidence signal about whether the
        string names this entity at all. It is used in the naming pass, not here.

  The MIDs in both files are already in m./g. dot form, identical to node_id, so no conversion is
  applied and none can silently go wrong.

SURFACES ARE CASE-FOLDED
  Every sampled surface is lowercase ("ootaki, yuuko", "your song"). The proportion is counted here
  rather than assumed, because it decides whether a recovered display_name can carry original
  casing or must be published lowercase. Title-casing it would invent information the source does
  not contain.

Streams from the zip; the 4.12 GB of text never lands on disk. Reads the frozen graph and the
frozen overlay. Writes neither.
"""
import sys, io, os, json, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ZIP = f"{V3}/_acquisition/facc1/mentions.zip"
ENTRY = "entity_list_file_freebase_complete_all_mention"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
OUT = f"{V3}/_acquisition/facc1_hits"
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
print(f"unresolved population: {NU:,} "
      f"(ENTITY_MID {int((KC==0).sum()):,}, CVT {int((KC==1).sum()):,})", flush=True)

p = subprocess.Popen([SEVENZ, "x", "-so", ZIP, ENTRY],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
rows = kept = bad = 0
has_upper = 0
part = 0
hit_uids = []
b_mid, b_surf, b_cnt = [], [], []


def flush():
    """Batched membership test, then spill only the surviving rows."""
    global part, kept
    if not b_mid:
        return
    u = np.fromiter(map(hash, b_mid), dtype=np.int64, count=len(b_mid))
    j = np.searchsorted(U, u)
    np.clip(j, 0, NU - 1, out=j)
    idx = np.flatnonzero(U[j] == u)
    if idx.size:
        kept += idx.size
        il = idx.tolist()
        hit_uids.append(u[idx])
        pq.write_table(pa.table({
            "node_uid": pa.array(u[idx], pa.int64()),
            "node_id": pa.array([b_mid[i] for i in il]),
            "surface": pa.array([b_surf[i] for i in il]),
            "count": pa.array([b_cnt[i] for i in il], pa.int32())}),
            f"{OUT}/part_{part:04d}.parquet", compression="zstd")
        part += 1


for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""):
    rows += 1
    f = raw.rstrip("\r\n").split(TAB)
    if len(f) < 3:
        bad += 1
        continue
    mid, surf, cnt = f[0], f[1], f[2]
    if not surf:
        bad += 1
        continue
    if surf != surf.lower():
        has_upper += 1
    try:
        c = int(cnt)
    except ValueError:
        bad += 1
        continue
    b_mid.append(mid); b_surf.append(surf); b_cnt.append(c)
    if len(b_mid) >= BATCH:
        flush()
        b_mid, b_surf, b_cnt = [], [], []
        print(f"  rows={rows:,} kept={kept:,} ({time.time()-t0:.0f}s)", flush=True)
flush()
p.stdout.close()
p.wait()

UH = np.unique(np.concatenate(hit_uids)) if hit_uids else np.array([], dtype=np.int64)
by = {}
if UH.size:
    kc = KC[np.searchsorted(U, UH)]
    by = {KIND[i]: int((kc == i).sum()) for i in range(3)}
n_ent = int((KC == 0).sum())
rec = {"schema": "FACC1_REACH/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "source": {"artifact": "mentions.zip", "entry": ENTRY,
                  "sha256": "8ee49cb896e8e3edd50692c1818e683e9ceb454b289cd93cd8c6e870549c128e",
                  "format": "mid \\t surface \\t count \\t (trailing tab)"},
       "rows_scanned": rows, "malformed_rows": bad,
       "mention_rows_matching_residue": kept,
       "FACC1_REACH_N": int(UH.size),
       "reach_by_kind": by,
       "entity_mid_reach_pct": round(100 * by.get("ENTITY_MID", 0) / n_ent, 4) if n_ent else 0,
       "CASE": {"rows_with_any_uppercase": has_upper,
                "pct": round(100 * has_upper / max(rows - bad, 1), 4),
                "CONSEQUENCE": ("if this is ~0 the source is case-folded and a recovered "
                                "display_name cannot carry original casing. Title-casing it would "
                                "invent information the source does not contain, so it will be "
                                "published as the source holds it and the record will say so.")},
       "WHAT_REACH_MEANS": ("the number of residual nodes FACC1 mentions ANY surface form for. It "
                            "is an upper bound on what this source can name; the conservative "
                            "aggregation rule (repeated support, dominant normalised form, "
                            "surface-ambiguity confidence) is applied in a later pass and will "
                            "recover fewer."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FACC1_REACH.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nrows scanned {rows:,}  malformed {bad:,}")
print(f"mention rows matching residue: {kept:,}")
print(f"FACC1_REACH_N = {UH.size:,}   by kind: {by}")
print(f"ENTITY_MID reach: {rec['entity_mid_reach_pct']}% of {n_ent:,}")
print(f"surfaces with any uppercase: {has_upper:,} ({rec['CASE']['pct']}%)")
print(f"{time.time()-t0:.0f}s")
