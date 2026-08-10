# Leaked / contaminated results (archived 2026-07-18)

These 2wiki + musique result files were computed on a **label-leaked substrate**
and are NOT valid. Kept for the before/after record only.

## The leak (audit finding)
On 2wiki/musique the ONLY doc-doc graph edges in `node.neighbors` were "bridge
edges" (`src/pipeline/loaders.py`) linking the co-supporting GOLD docs of each
question — built over ALL questions before the train/test split. So:
- **overlap1/overlap2 membership** put each gold in its co-gold's partition →
  FullCov's AND-over-golds degenerated to OR using test-question labels.
- **L3 traversal / pool_narrow** hopped those same annotation edges.
- **METIS partitions & degree-weighted centroids** were also computed on the
  leaked graph → even the "hard" baseline is mildly contaminated.

metaqa (KB-triple edges) and squad were NOT affected → kept in results/.

## Replacement
Clean, label-free substrate: source `2wiki_clean` (deduped corpus + title-mention
hyperlink edges, gold edges kept only on question nodes). See
`src/pipeline/build_clean.py` + `build_clean_index.py`. Reruns land in results/
under the `2wiki_clean` namespace.
