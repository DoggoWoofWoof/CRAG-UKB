"""End-to-end test of build_semantic_v2 on a hand-built graph, before it is pointed at 2.06B edges.

Checks the things that would be expensive to discover on the real run:
  1. a mediator renders as "Marriage - spouse: A; spouse: B; from: 1992" -- a repeated relation
     with different neighbours must survive, since that IS the content of a mediator
  2. an anonymous neighbour is dropped and does not consume a slot
  3. a DENY relation never appears
  4. incoming relations are listed separately, under "referenced by"
  5. the 8-outgoing cap holds exactly
  6. the bucket shuffle round-trips: every selected neighbour resolves to a name
"""
import sys, os, io, json, shutil, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONHASHSEED"] = "0"
import build_semantic_v2 as S

ROOT = tempfile.mkdtemp(prefix="semv2_")
os.makedirs(f"{ROOT}/canonical/edges")
os.makedirs(f"{ROOT}/overlay_v1")

# uids all == 0 mod 128 so every node lands in bucket 0, which is where the test looks for names
M, A, B, L, U, P, T = (128 * i for i in range(1, 8))
CAP = [128 * (20 + i) for i in range(12)]          # 12 neighbours for the cap test
HUB = 128 * 50                                      # the node that has 12 outgoing edges

NAMES = {M: ("Marriage", "STRUCTURAL_INFERRED"),
         A: ("Barack Obama", "FREEBASE_CURRENT_EXACT"),
         B: ("Michelle Obama", "FREEBASE_CURRENT_EXACT"),
         L: ("1992", "LITERAL_SELF"),
         U: ("Anonymous Record", "STRUCTURAL_INFERRED"),
         P: ("Person X", "FREEBASE_CURRENT_EXACT"),
         T: ("Marriage Type", "FREEBASE_CURRENT_EXACT"),
         HUB: ("Hub Mediator", "STRUCTURAL_INFERRED")}
for i, c in enumerate(CAP):
    NAMES[c] = (f"Neighbour {i:02d}", "FREEBASE_CURRENT_EXACT")

lit = [L]
obj = [k for k in NAMES if k not in lit]
for nm, keys in (("lit_000", lit), ("obj_000", obj)):
    pq.write_table(pa.table({
        "node_uid": pa.array(keys, pa.int64()),
        "display_name": pa.array([NAMES[k][0] for k in keys]),
        "display_name_source": pa.array([NAMES[k][1] for k in keys]).dictionary_encode(),
    }), f"{ROOT}/overlay_v1/{nm}.parquet")

REL = {"people.marriage.spouse": 1001, "people.marriage.from": 1002,
       "people.marriage.location_of_ceremony": 1003, "type.object.type": 1004,
       "people.person.spouse_s": 1005, "film.performance.actor": 1006}
pq.write_table(pa.table({
    "rel_uid": pa.array(list(REL.values()), pa.int64()),
    "relation": pa.array(list(REL.keys())),
    "reverse_relation": pa.array([None] * len(REL), pa.string()),
    "role": pa.array(["UNPAIRED"] * len(REL)),
    "raw_edge_count": pa.array([1] * len(REL), pa.int64()),
}), f"{ROOT}/canonical/relations.parquet")

E = [(M, REL["people.marriage.spouse"], A),
     (M, REL["people.marriage.spouse"], B),
     (M, REL["people.marriage.from"], L),
     (M, REL["people.marriage.location_of_ceremony"], U),     # anonymous -> dropped
     (M, REL["type.object.type"], T),                         # DENY -> dropped
     (P, REL["people.person.spouse_s"], M)]                   # incoming for M
E += [(HUB, REL["film.performance.actor"], c) for c in CAP]   # 12 outgoing -> cap at 8
E.sort(key=lambda r: r[0])                                    # edges are partitioned by src
pq.write_table(pa.table({"src": pa.array([e[0] for e in E], pa.int64()),
                         "rel": pa.array([e[1] for e in E], pa.int64()),
                         "dst": pa.array([e[2] for e in E], pa.int64()),
                         "dst_kind": pa.array([0] * len(E), pa.int8()),
                         "flipped": pa.array([0] * len(E), pa.int8())}),
               f"{ROOT}/canonical/edges/e000.parquet")

S.V3 = ROOT
S.OVL = f"{ROOT}/overlay_v1"
S.SCR = f"{ROOT}/_scratch"
S.OUT = f"{ROOT}/semantic_v2"
S.EDGES = [f"{ROOT}/canonical/edges/e000.parquet"]
S.CACHE = f"{ROOT}/_pop.npz"
S.EXPECT_EDGES = len(E)
os.makedirs(S.SCR, exist_ok=True)

S.stage_a(); S.stage_b(); S.stage_c(); S.stage_d()

t = pq.read_table(f"{S.OUT}")
got = dict(zip(t["node_uid"].to_pylist(), t["semantic_text"].to_pylist()))
print()
for k, v in sorted(got.items()):
    if v:
        print(f"  {k:>6} -> {v}")

fail = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   {detail}"))
    if not cond:
        fail.append(name)


mt = got.get(M) or ""
check("mediator renders both spouses and the date",
      "spouse: Barack Obama" in mt and "spouse: Michelle Obama" in mt and "from: 1992" in mt, mt)
check("anonymous neighbour dropped", "Anonymous Record" not in mt, mt)
check("DENY relation absent", "Marriage Type" not in mt and "type" not in mt.split(" - ")[1], mt)
check("incoming listed separately", "referenced by - spouse s: Person X" in mt, mt)
check("head is the display_name", mt.startswith("Marriage - "), mt)

ht = got.get(HUB) or ""
n_out = ht.count("actor: ")
check("outgoing cap is exactly 8", n_out == 8, f"got {n_out}: {ht}")

sb = json.load(io.open(f"{S.SCR}/stage_b.json"))
sc = json.load(io.open(f"{S.SCR}/stage_c.json"))
check("every selected neighbour resolved to a name", sc["unresolved_neighbour_uids"] == 0, str(sc))
check("all edges scanned", sb["EDGES_COMPLETE"], str(sb))
check("anonymous drops counted", sb["candidate_slots_dropped_anonymous_neighbour"] >= 1, str(sb))
check("no name is None in output", all(v is None or "None" not in v for v in got.values()))

shutil.rmtree(ROOT, ignore_errors=True)
print()
print("RESULT:", "ALL PASS" if not fail else f"{len(fail)} FAILED: {fail}")
sys.exit(1 if fail else 0)
