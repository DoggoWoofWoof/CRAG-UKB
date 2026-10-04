#!/bin/sh
# HotpotQA H3/H4 on the 192 GB variant.  The 64 GB container stopped heart-beating three times
# on this corpus (16.9M directed edges vs WebQSP's 8.3M, which already peaked at 43.8 GB).
cd "C:/Users/Swastik/Desktop/CRAG" || exit 1
export MODAL_PROFILE=swathihrao28 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
python -m modal run scratchpad/modal_partition_hi.py::matrix \
  --dss hotpotqa_clean \
  --cells H3_MTKAHYPAR_GRAPH:G0_TOPOLOGY_C,H3_MTKAHYPAR_GRAPH:G1_LOCAL_HYPER_CLIQUE,H4_MTKAHYPAR_TRUE_HYPERGRAPH:G2_TRUE_HYPERGRAPH \
  >> scratchpad/_l1ov/log_mp_hotpot_hi.txt 2>&1
echo "HOTPOT_HI_DONE rc=$?" >> scratchpad/_l1ov/log_mp_hotpot_hi.txt
