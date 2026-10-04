#!/bin/bash
set -x
for d in metaqa webqsp 2wiki_clean musique_clean; do
  python scratchpad/_l1ca_run.py $d
done
