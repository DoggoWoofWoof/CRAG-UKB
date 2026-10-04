"""Write the DEV_B preregistration record for the L1_P90_EXPLOIT typed-path selector.
No DEV_B number is computed or read here.  Snapshots the scratch modules (sha256-pinned)."""
import datetime
import hashlib
import json
import os
import shutil

import _l1x90_core as X

MODULES = ["_l1x90_core.py", "_l1x90_typed.py", "_l1x90_relwalk.py", "_l1x90_seeds.py", "_l1x90_universal.py", "_l1x90_devB_confirm.py"]
snap = os.path.join(X.OUT, "code_snapshot")
os.makedirs(snap, exist_ok=True)
mods = {}
for m in MODULES:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), m)
    dst = os.path.join(snap, m)
    shutil.copyfile(src, dst)
    mods[m] = X.sha_file(dst)
devA = {}
for n in ("metaqa", "squad", "musique", "metaqa_phg"):
    p = os.path.join(X.OUT, "universal_A_%s.json" % n)
    with open(p, encoding="utf-8") as f:
        o = json.load(f)
    devA[n] = {"cache_path": o["path"], "cache_sha256": o["cache_sha256"], "n_A": o["n_A"], "BASE_A_ALL": o["BASE_A"]["ALL"],
               "G1_A_ALL": o["gates"]["G1"]["A"]["ALL"], "G2_A_ALL": o["gates"]["G2"]["A"]["ALL"],
               "G2_gained_lost": [o["gates"]["G2"]["gained"], o["gates"]["G2"]["lost"]],
               "identical_to_BASE": o["gates"]["G2"]["identical_to_BASE"], "typed_graph": o["gates"]["G2"]["info"]["typed_graph"]}
rec = {
    "record": "L1_P90_EXPLOIT_PREREGISTRATION_DEV_B",
    "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "lane": "L1_P90_EXPLOIT (scratch; no contract file edited; nothing promoted; no commit)",
    "purpose": "Confirm on the held-out DEV_B halves that the universal typed-path P50 selector reaches ALL-gold block "
               "coverage > 0.90 overall on the canonical NAME_ONLY MetaQA substrate, without changing text corpora.",
    "populations": {
        "definition": "canonical replay-cache query rows; DEV_A = sha1(query_id)[:8] parity 0 (development, all numbers above were "
                      "read on DEV_A only), DEV_B = parity 1 (confirmation; no DEV_B number has been printed or read before this record)",
        "test_split": "never read",
    },
    "configuration_frozen": {
        "selector": "_l1x90_universal.universal_select(C, gate='G2')",
        "gate_primary": "G2: typed graph (>=2 distinct relation labels) AND query anchored (a corpus node name occurs verbatim in the question) -> "
                        "structural P50 ranking from the scheduled hard frontier walk (lexicographic: seed+answer-relation mass, then other typed + untyped walk mass), "
                        "back-filled by the frozen text ranking; otherwise the frozen BASE text ranking.",
        "gate_secondary_reported_only": "G1: as G2 but additionally requires >= 1 relation label matched in the question",
        "gate_choice_basis": "G2 vs G1 chosen on DEV_A (metaqa G2 0.9721 vs G1 0.9621); DEV_B is the only confirmation",
        "seeds": "lexical exact-name spans (longest, case-exact preferred, relation-word and stop-word spans excluded), cap 5, uniform mass; fallback dense top-1 + SPLADE top-1",
        "schedule": "schedule3(key='wh'): relation labels lexically matched (Porter + irregular lemmas + compound-suffix conflation), "
                    "positional masking of mention spans, attachment = no other matched relation token in the gap, sort (opaque, main-verb-after-mention, distance)",
        "walk": "hard typed frontier walk, H=3 (= MAX_HOPS of the frozen expansion), cumulative allowed relation set order[:k], unit mass per layer, "
                "answer channel = pushes along the LAST scheduled relation; hedge = other typed pushes + plain untyped frontier walk at EPS=1e-6",
        "budget": "P50 = 50 blocks (unchanged); partitions unchanged (frozen H4_SK Mt-KaHyPar; PHG arm = frozen Zoltan-PHG repair-1 partition)",
        "parameters": {"EPS": 1e-6, "H": 3, "seed_cap": 5, "P_MAIN": 50},
        "modules_sha256": mods,
        "snapshot_dir": snap,
    },
    "caches": devA,
    "dev_A_results_already_seen": {n: {"BASE": devA[n]["BASE_A_ALL"], "G2": devA[n]["G2_A_ALL"], "G1": devA[n]["G1_A_ALL"]} for n in devA},
    "decision_rule": {
        "primary": "metaqa DEV_B: ALL(G2) point estimate >= 0.90 AND exact McNemar (G2 vs BASE, two-sided) p < 0.01 with gained > lost -> P90_TYPED_PATH_CONFIRMED; else P90_TYPED_PATH_NOT_CONFIRMED",
        "secondary": ["metaqa_phg DEV_B (partition robustness): same statistics, reported, not part of the label",
                      "squad / musique DEV_B: selector must be identical to BASE (untyped graphs) -> reported as IDENTICAL_TO_BASE; no claim of improvement on text corpora",
                      "G1 on metaqa DEV_B reported for information"],
        "per_hop": "hop1/hop2/hop3 ALL reported; no per-hop label",
        "no_further_changes": "after this record no code, parameter, gate or population changes; a failed confirmation is reported as such",
    },
    "claims_not_made": [
        "no claim on the TEST split", "no claim on WebQSP/HotpotQA/2Wiki (not evaluated)", "no claim of L2/L3 survival",
        "no equivalence to legacy NAME+FACTS numbers", "not an L1 contract change: adopting the selector = new L1 contract hash (ruling required)",
    ],
}
p = os.path.join(X.OUT, "PREREGISTRATION_DEV_B.json")
X.wj(p, rec)
print("wrote", p)
for m, h in mods.items():
    print("  %-26s %s" % (m, h[:16]))
