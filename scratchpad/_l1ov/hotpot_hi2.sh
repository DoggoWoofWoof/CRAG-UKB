#!/bin/sh
# HotpotQA H3:G0 and H4:G2 -- the two cells that repeatedly hit Modal's 900 s runner-heartbeat
# timeout.  H3:G1 already landed (833 s), so it is not re-run.  The runner now calls Mt-KaHyPar
# in a child process so the parent can keep heart-beating; see modal_partition_hi.py.
cd "C:/Users/Swastik/Desktop/CRAG" || exit 1
export MODAL_PROFILE=swathihrao28 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
python -m modal run scratchpad/modal_partition_hi.py::matrix \
  --dss hotpotqa_clean \
  --cells H3_MTKAHYPAR_GRAPH:G0_TOPOLOGY_C,H4_MTKAHYPAR_TRUE_HYPERGRAPH:G2_TRUE_HYPERGRAPH \
  >> scratchpad/_l1ov/log_mp_hotpot_hi2.txt 2>&1
echo "HOTPOT_HI2_DONE rc=$?" >> scratchpad/_l1ov/log_mp_hotpot_hi2.txt
