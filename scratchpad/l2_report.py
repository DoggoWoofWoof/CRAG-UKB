"""Assemble results/L2/L2_E0_E3_RESULTS.md from the results JSON. Emits the canonical table
(all required columns) per dataset for VAL and TEST, plus DELTA-vs-previous and per-signal value class."""
import json

R = json.load(open("results/L2/L2_E0_E3_RESULTS.json"))
ROWS = [
    ("E0_dense", "E0 Dense"),
    ("E0s_splade", "E0s SPLADE"),
    ("E1a_dense_splade_RRF", "E1a Dense+SPLADE RRF"),
    ("E1b_dense_splade_ZNORM", "E1b Dense+SPLADE z-norm"),
    ("E2_dense_splade_offset_RRF", "E2 +Offset (RRF)"),
    ("E3_dense_splade_offset_mixture_RRF", "E3 +Mixture (RRF)"),
    ("E3b_bestE0E3_plus_graph_RRF", "E3b +cheap graph (RRF)"),
]
COLS = ["MRR", "R@1", "R@5", "R@10", "R@20", "R@50", "R@100", "ANY@10", "ANY@20", "ANY@50",
        "ALL@10", "ALL@20", "ALL@50", "best_gold_rank_mean", "best_gold_rank_median"]
DEN = "COND_ANY_GOLD_IN_SCOPE"


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def table(ds, split):
    lines = [f"#### {ds} — {split} (denominator: {DEN})", ""]
    head = "| config | " + " | ".join(COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    lines += [head, sep]
    res = R["datasets"][ds]["results"]
    for key, label in ROWS:
        if key not in res or split not in res[key]:
            continue
        m = res[key][split][DEN]
        cells = [fmt(m.get(c)) for c in COLS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


out = []
for ds in ("2wiki_clean", "musique_clean"):
    out.append(f"### {ds}")
    for split in ("test", "val"):
        out.append(table(ds, split))

open("results/L2/_L2_table_body.md", "w", encoding="utf-8").write("\n".join(out))
print("wrote results/L2/_L2_table_body.md")
