#!/usr/bin/env bash
# The whole campaign on one host, after a change that leaves the lantern
# characterizations valid but not the datasets (e.g. the PSF size):
#
#   setsid nohup bash run_campaign.sh <backup_suffix> [gpu] > logs/campaign.log 2>&1 < /dev/null &
#
# 1. moves the existing datasets, model caches and executed-notebook figures aside
#    (data/ml -> data/ml_<suffix>, cbeam's copies likewise) -- gen_datasets.py skips
#    pieces that exist, so stale ones must not be found;
# 2. run_lantern.sh steps 2-4 (datasets, PLREC notebooks, cbeam notebooks) for
#    pl7_1200, pl19_1200 and pl19_800, one after the other (half the RAM at most);
# 3. run_extras.sh: the pl19_800 studies.
# Run from PLREC with the environment active and the cbeam checkout next to it.
set -uo pipefail
cd "$(dirname "$0")"
SUFFIX=${1:?usage: bash run_campaign.sh <backup_suffix> [gpu]}
GPU=${2:-0}
PLREC=$(pwd); CBEAM=$(cd ../cbeam && pwd)
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "backing up the old datasets as *_$SUFFIX"
if [ -d data/ml ] && [ ! -e "data/ml_$SUFFIX" ]; then mv data/ml "data/ml_$SUFFIX"; fi
mkdir -p data/ml
if [ -d "$CBEAM/examples/data/ml" ] && [ ! -e "$CBEAM/examples/data/ml_$SUFFIX" ]; then
    mv "$CBEAM/examples/data/ml" "$CBEAM/examples/data/ml_$SUFFIX"
fi
mkdir -p "$CBEAM/examples/data/old_$SUFFIX"
mv "$CBEAM"/examples/data/mw_results_*.npz "$CBEAM/examples/data/old_$SUFFIX/" 2>/dev/null || true
for L in pl7_1200 pl19_1200 pl19_800; do
    for d in "runs/$L/figs" "$CBEAM/examples/runs/$L/figs"; do
        [ -d "$d" ] && mv "$d" "${d}_$SUFFIX"
    done
done

for L in pl7_1200 pl19_1200 pl19_800; do
    stamp "=== $L"
    STEPS=234 bash run_lantern.sh "$L" "$GPU" > "logs/$L.log" 2>&1
    stamp "$L: $(tail -1 "logs/$L.log")"
    grep -q "FAIL" "logs/$L.log" && stamp "$L had failures: $(grep FAIL "logs/$L.log" | tr '\n' ' ')"
done

stamp "=== pl19_800 extras"
bash run_extras.sh 2 > logs/extras.log 2>&1
stamp "extras: $(tail -1 logs/extras.log)"
stamp "CAMPAIGNDONE"
