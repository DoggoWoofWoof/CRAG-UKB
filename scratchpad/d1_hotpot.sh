set -u; log(){ echo "[$(date +%H:%M:%S)] $*"; }
B="https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main"
dl(){ curl -sSL -m 1200 -o "$2" "$1" && log "OK $(basename "$2") ($(stat -c%s "$2")B)" || log "FAIL $2"; }
dl "$B/distractor/train-00000-of-00002.parquet" data/original/hotpotqa/hf_distractor/train-00000-of-00002.parquet
dl "$B/distractor/train-00001-of-00002.parquet" data/original/hotpotqa/hf_distractor/train-00001-of-00002.parquet
dl "$B/distractor/validation-00000-of-00001.parquet" data/original/hotpotqa/hf_distractor/validation-00000-of-00001.parquet
dl "$B/fullwiki/validation-00000-of-00001.parquet" data/original/hotpotqa/hf_fullwiki/validation-00000-of-00001.parquet
dl "$B/fullwiki/test-00000-of-00001.parquet" data/original/hotpotqa/hf_fullwiki/test-00000-of-00001.parquet
log "HOTPOT DONE"
