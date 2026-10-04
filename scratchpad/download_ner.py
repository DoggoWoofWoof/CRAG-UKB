import modal, os
vol = modal.Volume.from_name("crag-data-volume")
# Try to list
for entry in vol.list("data/ukb_storage/hotpotqa_clean/"):
    print(entry)
