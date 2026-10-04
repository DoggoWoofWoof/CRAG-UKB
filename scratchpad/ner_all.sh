cd /c/Users/Swastik/Desktop/CRAG
for DS in 2wiki 2wiki_universe hotpotqa; do
  echo "=========== NER build: $DS ==========="
  bash scratchpad/launch_ner.sh "$DS" 6
done
echo "ALL_NER_DONE"
