#!/usr/bin/env bash
# Generate a lantern's datasets on a many-core host, within a memory budget.
#
#   CBEAM_LANTERN=pl19_1200 bash run_gen.sh
#
# Run from the PLREC checkout, with the cbeam checkout next to it
# (../cbeam) and the lantern's 41 characterizations already made there
# (cbeam/run_characterize.sh). Steps:
#   1. measure the peak memory of one spectral piece (gen_datasets.py peak-memory);
#   2. run the single-wavelength dataset and the 410 spectral pieces
#      (41 wavelengths x 10 chunks), JOBS at a time, each single-threaded;
#   3. assemble the spectral dataset.
# JOBS is the number of pieces whose peak (plus 30%) fits in MEM_BUDGET_GB
# (default: half of this host's RAM), capped at the cores minus 2.
set -euo pipefail
cd "$(dirname "$0")"
: "${CBEAM_LANTERN:?set CBEAM_LANTERN to a batch_pipeline.LANTERNS name}"
export CBEAM_LANTERN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CBEAM_FFT_WORKERS=1 CBEAM_BACKEND=numpy
export PYTHONPATH="$(cd ../cbeam/src && pwd)${PYTHONPATH:+:$PYTHONPATH}"
LOGS=logs/$CBEAM_LANTERN
mkdir -p "$LOGS"

total_gb=$(awk '/MemTotal/ {printf "%d", $2 / 2^20}' /proc/meminfo)
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(( total_gb / 2 ))}
peak=""
for try in 1 2 3; do      # retried: a Julia start-up can fail transiently
    peak=$(python gen_datasets.py peak-memory 2> "$LOGS/peak.err" | awk '/PEAK_GB/ {print $2}') || true
    [ -n "$peak" ] && break
done
[ -n "$peak" ] || { echo "peak-memory measurement failed, see $LOGS/peak.err" | tee "$LOGS/plan.txt"; exit 1; }
JOBS=${JOBS:-$(python -c "import math,os; print(max(1, min(os.cpu_count()-2, math.floor($MEM_BUDGET_GB / ($peak * 1.3)))))")}
echo "lantern $CBEAM_LANTERN: piece peak ${peak} GB, budget ${MEM_BUDGET_GB} GB -> $JOBS parallel pieces" | tee "$LOGS/plan.txt"

# the single-wavelength dataset counts as one of the JOBS slots (4 FFT threads)
CBEAM_FFT_WORKERS=4 python gen_datasets.py single > "$LOGS/single.log" 2>&1 &

for w in $(seq 0 40); do for c in $(seq 0 9); do echo "$w $c"; done; done |
    xargs -P "$(( JOBS > 1 ? JOBS - 1 : 1 ))" -n 2 sh -c \
        'python gen_datasets.py spectral-chunk "$0" "$1" > "'"$LOGS"'/wl$0_c$1.log" 2>&1 || echo "$0 $1"'     > "$LOGS/failed.txt"
wait

# rerun the pieces that failed (e.g. a transient Julia start-up error), once
while read -r w c; do
    python gen_datasets.py spectral-chunk "$w" "$c" >> "$LOGS/rerun.log" 2>&1 || echo "FAILED AGAIN $w $c" | tee -a "$LOGS/plan.txt"
done < "$LOGS/failed.txt"

python gen_datasets.py spectral-assemble > "$LOGS/assemble.log" 2>&1
echo "ALLDONE $CBEAM_LANTERN" | tee -a "$LOGS/plan.txt"
