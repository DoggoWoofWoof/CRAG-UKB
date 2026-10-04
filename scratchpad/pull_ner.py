import modal, os, sys
vol = modal.Volume.from_name("crag-data-volume")
# List files
for path in vol.list("data/ukb_storage/hotpotqa_clean/ner_edges_w_df25.pkl"):
    print(path)
