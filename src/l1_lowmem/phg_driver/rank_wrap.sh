#!/bin/bash
# per-rank wrapper: mpirun launches this; it records /usr/bin/time -v (peak RSS etc.) of its own rank's driver process.
# PHG_TIME_DIR must be exported through mpirun (-x PHG_TIME_DIR).
r="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
exec /usr/bin/time -v -o "${PHG_TIME_DIR}/time_rank${r}.txt" "$@"
