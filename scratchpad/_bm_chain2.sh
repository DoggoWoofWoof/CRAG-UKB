set -x
until grep -q "wrote .*funnel_metaqa.json" scratchpad/_bm_funnel.log; do sleep 20; done
echo "=== funnel done -> B1 hotpotqa_clean ==="
python scratchpad/_l1bm_run.py build hotpotqa_clean B1_PARENT_DIVERSE
python scratchpad/_l1bm_step8.py hotpotqa_clean B1_PARENT_DIVERSE
echo "=== B1 webqsp ==="
python scratchpad/_l1bm_run.py build webqsp B1_PARENT_DIVERSE
python scratchpad/_l1bm_step8.py webqsp B1_PARENT_DIVERSE
echo "=== clean latency (nothing else running) ==="
python scratchpad/_l1bm_latency.py metaqa 400 3
echo "=== CHAIN2 ALL DONE ==="
