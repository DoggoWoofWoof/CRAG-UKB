"""Read coding_sheet.csv -> reporting-rate stats and the paper's bar figure.

Codes: 1 = reported, 0 = not, ? = unclear. Corpus-wide rates retain a common denominator for the
descriptive figure. Family-relevant rates separately expose control/recovery among iterative and
agentic systems and tool errors among explicitly tool-using agents.
Run:  <python> review_paper/make_figure.py
"""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = [
    ("S1_answer", "Answer\nEM/F1", "S"), ("S2_retrieval_recall", "Retrieval\nrecall", "S"),
    ("A1_evidence_complete", "Evidence\ncomplete", "A"), ("A2_stop_decision", "Control\ndecision", "A"),
    ("A3_recovery", "Recovery", "A"), ("A4_trajectory", "Trajectory", "A"),
    ("A5_tool_error", "Tool\nerror", "A"), ("C1_calls", "Retrieval\nuse/calls", "C"),
    ("C2_cost", "Tokens\n/cost", "C"), ("C3_latency", "Latency", "C"),
]
PH = {"S1_answer": "S1", "S2_retrieval_recall": "S2", "A1_evidence_complete": "A1",
      "A2_stop_decision": "A2", "A3_recovery": "A3", "A4_trajectory": "A4", "A5_tool_error": "A5",
      "C1_calls": "C1", "C2_cost": "C2", "C3_latency": "C3"}


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "coding_sheet.csv"), encoding="utf-8")))
    rows = [r for r in rows if r.get("paper", "").strip()]
    return rows


def stats(rows):
    N = len(rows)
    out = {}
    for key, _, _ in DIMS:
        vals = [(r.get(key) or "").strip() for r in rows]
        rep = sum(1 for v in vals if v == "1")
        unc = sum(1 for v in vals if v == "?")
        pct = math.floor(100 * rep / N + 0.5) if N else 0
        out[key] = {"reported": rep, "unclear": unc, "N": N, "pct": pct}
    return N, out


def relevant_family_stats(rows):
    iterative_agentic = [r for r in rows if r.get("group") in {"iterative", "agentic"}]
    tool_agents = [r for r in rows if r.get("group") == "agentic"]

    def reported(subset, key):
        return sum((r.get(key) or "").strip() == "1" for r in subset)

    return {
        "iterative_agentic": {
            "N": len(iterative_agentic),
            "A2_stop_decision": reported(iterative_agentic, "A2_stop_decision"),
            "A3_recovery": reported(iterative_agentic, "A3_recovery"),
        },
        "tool_agents": {
            "N": len(tool_agents),
            "A5_tool_error": reported(tool_agents, "A5_tool_error"),
        },
    }


def main():
    rows = load(); N, out = stats(rows); family = relevant_family_stats(rows)
    filled = sum(1 for r in rows if (r.get("S1_answer") or "").strip())
    print(f"coded rows: {filled}/{N}")
    print(f"{'dim':22s} {'reported':>8s} {'unclear':>8s} {'pct':>5s}")
    for key, _, g in DIMS:
        s = out[key]
        print(f"{key:22s} {s['reported']:>8d} {s['unclear']:>8d} {s['pct']:>4.0f}%")
    print("\n--- realm_paper.tex placeholder fills ---")
    print(f"{{{{N}}}} = {N}")
    for key, _, _ in DIMS:
        print(f"{{{{{PH[key]}_pct}}}} = {out[key]['pct']:.0f}\\%")
    print("\n--- family-relevant denominators ---")
    ia = family["iterative_agentic"]
    ta = family["tool_agents"]
    print(f"control quality: {ia['A2_stop_decision']}/{ia['N']} iterative/agentic papers")
    print(f"failed-retrieval recovery: {ia['A3_recovery']}/{ia['N']} iterative/agentic papers")
    print(f"tool-error rate: {ta['A5_tool_error']}/{ta['N']} tool-using agent papers")

    summary = {"corpus_N": N, "corpus_wide": out, "family_relevant": family}
    with open(os.path.join(HERE, "coding_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    if filled < N:
        print(f"\n[skip figure: only {filled}/{N} rows coded]")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[matplotlib unavailable: {e}]"); return
    colors = {"S": "#4C78A8", "A": "#E45756", "C": "#B0B0B0"}
    labels = [d[1] for d in DIMS]; pcts = [out[d[0]]["pct"] for d in DIMS]; cs = [colors[d[2]] for d in DIMS]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bars = ax.bar(range(len(DIMS)), pcts, color=cs, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of all 32 sampled papers reporting", fontsize=9); ax.set_ylim(0, 105)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 2, f"{p:.0f}", ha="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["S"], label="Outcome"), Patch(color=colors["A"], label="Process-level"),
                       Patch(color=colors["C"], label="Cost")], fontsize=8, frameon=False, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(HERE, f"fig_reporting_gap.{ext}"), dpi=200, bbox_inches="tight")
    print("\nwrote fig_reporting_gap.pdf/png")


if __name__ == "__main__":
    main()
