#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath("."))
import json, numpy as np, torch, faiss, hashlib
from src.core.engine import CoreEngine
from src.experiments.overlap_retrain import _splits, _hard_membership
from src.experiments.l1_universal_head import _load, _train_universal
from src.experiments.l1_rerank100 import _feats, _rr

# Load historical 2000 IDs
hist=json.load(open('data/canonical/metaqa/splits/historical_2000_ids.json'))
hist_ids=set(hist['ids'])
print(f"historical 2000 hash {hist['hash']} n {len(hist_ids)}")

# Need to evaluate current model on those 2000 IDs vs full
# Use same _load path as D: _load(metaqa, gte_qwen, 8000,3000,2000) for historical, and 8000,3000,1000000 for full?
# But for exact reproduction, we need to use same heads as D
# Load full test to get mapping
eng=CoreEngine(source='metaqa', index_subdir='gte_qwen')
sp=_splits(eng, _hard_membership(eng))
test=sp['test']
# Build map node_id -> index in test
test_ids=[nd.node_id for nd,_,_ in test]
print(f"full test len {len(test_ids)}")
# Verify first 2000 match
assert test_ids[:2000]==hist['ids'], "mismatch"
print("verified first 2000 match historical")

# For stratified hop, already have
from collections import Counter
hops=[nd.metadata.get('hop') for nd,_,_ in test]
print("full hops", Counter(hops))
print("first2000 hops", Counter([nd.metadata.get('hop') for nd,_,_ in test[:2000]]))

# Now cheap check: we will not recompute L2 now, just report that subset evaluation will be done post-D
# Record checkpoint hashes
import glob
heads=sorted(glob.glob('data/ukb_storage/_head_cache/head_*.pt'))[-4:]
for p in heads:
    print(p, os.path.getsize(p), hashlib.sha256(open(p,'rb').read(1024*1024)).hexdigest()[:16])

# Record corpus hashes
for pp in ['data/ukb_storage/metaqa/gte_qwen/nodes.npy','data/ukb_storage/metaqa/splade_doc_embs.pkl','data/ukb_storage/metaqa/partition_map.json']:
    if os.path.exists(pp):
        print(pp, os.path.getsize(pp))

# Record GTE manifest
for c in ['metaqa_1hop','metaqa_2hop','metaqa_3hop']:
    man=f'data/canonical/{c}/encodings/dense/queries/manifest.json'
    if os.path.exists(man):
        j=json.load(open(man))
        print(c, j['source_sha256'][:16], j['n_items'])

print("READY for CURRENT_MODEL_ON_HISTORICAL_2000 evaluation (same K8 MAXK500 scope0)")
