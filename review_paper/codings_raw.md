# Raw agent codings (assemble into coding_sheet.csv when all 4 batches land)
Cols: S1_answer, S2_retrieval_recall, A1_evidence_complete, A2_stop_decision, A3_recovery, A4_trajectory, A5_tool_error, C1_calls, C2_cost, C3_latency

## Batch 2 (DONE)
| Paper | grp | S1 | S2 | A1 | A2 | A3 | A4 | A5 | C1 | C2 | C3 | metrics | url |
|---|---|--|--|--|--|--|--|--|--|--|--|---|---|
| RaDeR | iterative | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nDCG@10(BRIGHT/RAR-b), MRR@10+R@1k(MSMARCO), recall/prec appx, downstream QA Acc%(TheoremQA) | aclanthology.org/2025.emnlp-main.1011 |
| Search-o1 | iterative | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Pass@1 reasoning sets; EM/F1 (NQ/TriviaQA/HotpotQA/2Wiki/MuSiQue/Bamboogle) | arxiv.org/abs/2501.05366 |
| ReAct | agentic | 1 | 0 | 0 | 0 | 0 | 1* | 0 | 0 | 0 | 0 | EM(HotpotQA),Acc(FEVER),Success(ALFWorld/WebShop); *manual trajectory success/failure-mode % (Table 2, qualitative) | arxiv.org/abs/2210.03629 |
| Toolformer | agentic | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | Acc/EM(LAMA/QA/math),ppl; tool-call frequency % (C1=freq not rounds) | arxiv.org/abs/2302.04761 |
| FrugalRAG | agentic | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | MBE(LLM-judge acc), Gold Evidence Recall%; Avg #Searches; Tokens+FLOPs+Latency(appx) | arxiv.org/abs/2507.07634 |
| SCMRAG | agentic | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Acc(PopQA/PubHealth/ARC/MultiHop-RAG),FactScore; top-n gen-acc curve (S2=0, labeled "retrieval" but plots gen-acc) | ifaamas.org/Proceedings/aamas2025/pdfs/p50.pdf |
| A-RAG | agentic | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | LLM-Acc,Contain-Acc(HotpotQA/2Wiki/MuSiQue/GraphRAG-Bench); retrieved-token counts (C1=0, sweeps max-steps not actual calls) | arxiv.org/abs/2602.03442 |
| MDR | dense | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | Answer+Joint+Sup EM/F1(HotpotQA); R@2/10/20 JOINT 2-passage recall; SP-EM; ~10x speed | arxiv.org/abs/2009.12756 |

Notes: MDR = only A1=1 (joint 2-passage recall). FrugalRAG = only paper strong on cost (C1/C2/C3). ReAct A4=1 is manual/qualitative (Table 2) — flag; keep as reported-but-qualitative. RaDeR: an early fetch hallucinated HotpotQA EM/F1 — corrected from PDF (A1=0 per its own Limitations).

## Batch 1 (DONE)
| Paper | grp | S1 | S2 | A1 | A2 | A3 | A4 | A5 | C1 | C2 | C3 | metrics | url |
|---|---|--|--|--|--|--|--|--|--|--|--|---|---|
| Self-Ask | iterative | 1 | 0 | 0 | 0 | 0 | 1* | 0 | 0 | 1 | 0 | Acc%(Bamboogle/2Wiki/MuSiQue); *compositionality gap (sub-Q correct, composition wrong); C2=generated-token proxy | arxiv.org/abs/2210.03350 |
| IRCoT | iterative | 1 | 1 | 0 | 0 | 0 | 1* | 0 | 0 | 0 | 0 | EM/F1(HotpotQA/2Wiki/MuSiQue/IIRC); gold-para recall (Fig3); *manual CoT factual-error count (40Q) | arxiv.org/abs/2212.10509 |
| DSP | iterative | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Answer EM/F1 (Open-SQuAD/HotpotQA), F1/nF1(QReCC); endpoint-only | arxiv.org/abs/2212.14024 |
| Least-to-Most | iterative | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Acc(last-letter/SCAN/GSM8K/DROP); acc-by-#steps; NO retrieval (pure prompting) | arxiv.org/abs/2205.10625 |
| ITER-RETGEN | iterative | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | EM/F1/Acc(HotpotQA/2Wiki/MuSiQue/Bamboogle); answer-recall across iters (Table6); avg #API calls + #paras (C2=call count) | arxiv.org/abs/2305.15294 |
| FLARE | iterative | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | EM/F1(2Wiki)+Disambig-F1/ROUGE/UniEval; retrieval frequency %(Fig5); A2=0 (trigger accuracy not measured) | arxiv.org/abs/2305.06983 |
| Self-RAG | iterative | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | acc(PopQA/TriviaQA/ARC),FactScore; citation prec/rec; retrieval-freq vs threshold(Fig3c) | arxiv.org/abs/2310.11511 |
| Adaptive-RAG | iterative | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | EM/F1/Acc(6 sets); classifier ROUTING accuracy(Table4, A2=1); Step count; relative Time(C3) | arxiv.org/abs/2403.14403 |

Notes: Adaptive-RAG = only A2=1 (measures its complexity-classifier routing accuracy, ~54.5%). IRCoT+ITER-RETGEN = retrieval recall (S2=1) but aggregate not joint (A1=0). Self-Ask/IRCoT A4=1 are qualitative/manual. NONE report A3(recovery) or A5(tool-error). "Retrieval frequency"→C1. Token/call-as-cost→C2 (proxy noted).

## Batch 3 (DONE)
| Paper | grp | S1 | S2 | A1 | A2 | A3 | A4 | A5 | C1 | C2 | C3 | metrics | url |
|---|---|--|--|--|--|--|--|--|--|--|--|---|---|
| GraphRAG | graph | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | LLM-judge win-rates (comprehensiveness/diversity/...); claim-faithfulness; context tokens. NO EM/F1, NO retrieval recall (summarization eval) | arxiv.org/abs/2404.16130 |
| HippoRAG | graph | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | EM/F1; Recall@2/5; **All-Recall@2/5 = %queries ALL support retrieved (A1)**; cost/speed relative (10-30x cheaper) | arxiv.org/abs/2405.14831 |
| HippoRAG2 | graph | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | F1/EM; passage recall@2/5 (aggregate→A1=0); index tokens/$/24h (offline cost) | arxiv.org/abs/2502.14802 |
| GNN-RAG | graph | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | Hit/Hits@1/F1; Answer-Coverage %(≥1 answer→A1=0); #LLM calls, tokens, $ | arxiv.org/abs/2405.20139 |
| Think-on-Graph | graph | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | Hits@1(9 sets); **path-overlap vs ground-truth SPARQL paths (A4 quantitative)**; call bound theoretical | arxiv.org/abs/2307.07697 |
| ToG-2 | graph | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | EM/Acc(7 sets); runtime vs ToG (relative→C3); 50-sample answer-source study (not path-correctness) | arxiv.org/abs/2407.10805 |
| KG2RAG | graph | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | Response F1/P/R; **Retrieval F1/P/R vs referenced facts (S2)**; tokens+calls; ms retrieval+gen latency (C3) | aclanthology.org/2025.naacl-long.449 |
| Clue-RAG | graph | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | Acc/F1 answer-only; indexing+per-query tokens; search depth D=1-4; no retrieval metric, no latency | arxiv.org/abs/2507.08445 |

Notes: GraphRAG S1=0 (query-focused summarization, LLM-judge only — a different eval paradigm). HippoRAG = joint All-Recall (A1). ToG = path-overlap vs ground-truth (A4, quantitative). Graph papers report MORE cost (C1/C2 high) but A2/A3/A5 still 0/8. C3(latency)=online only; offline indexing time noted but coded 0.

## Batch 4 (incorporated in `coding_sheet.csv`)

The remaining rows were consolidated directly into the final sheet. This file preserves working notes
from the first three coding batches; `coding_sheet.csv`, not this incomplete notebook, is the source of
the paper's counts.

## Final source-resolution pass (2026-08-03)

- `HippoRAG2/C3` changed from 0 to 1: Appendix F reports online time per query in addition to indexing cost.
- `MDR/A4` changed from 0 to 1: Figure 2 reports a quantified manual audit of 50 erroneous passage sequences.
- `LinearRAG/A1` changed from `?` to 1: Section 4.1 defines Evidence Recall as checking whether retrieved
  content contains all information necessary for the correct answer.
- The first two changes arose in the retained seven-paper repeat-coding audit. The LinearRAG ambiguity was
  resolved separately from the primary source before regenerating the final corpus statistics.
