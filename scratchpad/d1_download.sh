set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
dl() { # url dest
  local url="$1" dest="$2"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -gt 1000 ]; then
    log "SKIP exists $(basename "$dest") ($(stat -c%s "$dest") B)"; return 0; fi
  log "GET $url -> $dest"
  curl -sSL -m 1800 -o "$dest" "$url" && log "OK  $(basename "$dest") ($(stat -c%s "$dest" 2>/dev/null) B)" || log "FAIL $(basename "$dest")"
}
# 2WIKI official (April 2021 ids version) + hyperlink paragraphs (structural_native source)
dl "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1" data/original/2wiki/v1.0_ids_april2021/data_ids_april7.zip
dl "https://www.dropbox.com/s/wlhw26kik59wbh8/para_with_hyperlink.zip?dl=1" data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip
# HOTPOT official jsons
dl "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"        data/original/hotpotqa/v1.1/hotpot_train_v1.1.json
dl "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" data/original/hotpotqa/v1.1/hotpot_dev_distractor_v1.json
dl "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"   data/original/hotpotqa/v1.1/hotpot_dev_fullwiki_v1.json
dl "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"  data/original/hotpotqa/v1.1/hotpot_test_fullwiki_v1.json
# WEBQSP original (Microsoft)
dl "https://download.microsoft.com/download/F/5/0/F5012144-A4FB-4084-897F-CCDA3C58740E/WebQSP.zip" data/original/webqsp/WebQSP.zip
log "ALL DONE"
# RoG-webqsp parquet (provides per-question Freebase subgraphs; we currently have TEST only)
B="https://huggingface.co/datasets/rmanluo/RoG-webqsp/resolve/main/data"
dl "$B/train-00000-of-00002-d810a36ed97bc2cc.parquet" data/original/webqsp/rog_webqsp/train-00000-of-00002.parquet
dl "$B/train-00001-of-00002-e53244e71082a392.parquet" data/original/webqsp/rog_webqsp/train-00001-of-00002.parquet
dl "$B/validation-00000-of-00001-6ee6adc5b154643a.parquet" data/original/webqsp/rog_webqsp/validation-00000-of-00001.parquet
log "ROG DONE"
