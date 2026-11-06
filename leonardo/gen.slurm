#!/bin/bash
# Generate a lantern's datasets: array task k of n (one whole Booster node each)
# runs the spectral pieces (w, c) -- 41 wavelengths x 10 chunks -- with
# (10 w + c) % n == k, as independent single-threaded PLREC/gen_datasets.py
# processes; task 0 also makes the single-wavelength dataset. Pieces already
# made are skipped, so a resubmission resumes. Submitted by submit.sh.
#SBATCH --job-name=pl-gen
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SPECIAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=08:00:00

source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
k=${SLURM_ARRAY_TASK_ID:-0}; n=${NTASKS:-1}
LOGS="$LANTERN_ROOT/logs/$LANTERN/gen"; mkdir -p "$LOGS"
P=$(fit_on_node "${PEAK_PIECE[$LANTERN]}")
stamp "$LANTERN datasets, task $k/$n on $(hostname): $P pieces in parallel"

cd "$PLREC"
if [ "$k" = 0 ]; then                       # the single-wavelength dataset takes one slot
    CBEAM_FFT_WORKERS=4 python gen_datasets.py single > "$LOGS/single.log" 2>&1 &
    P=$(( P > 1 ? P - 1 : 1 ))
fi
python -c "print('\n'.join(f'{w} {c}' for w in range(41) for c in range(10) if (10 * w + c) % $n == $k))" |
    xargs -P "$P" -n 2 sh -c \
        'python gen_datasets.py spectral-chunk "$0" "$1" > "'"$LOGS"'/wl$0_c$1.log" 2>&1 || echo "$0 $1"' \
    > "$LOGS/failed_$k.txt"
wait
while read -r w c; do                       # retry failures once
    python gen_datasets.py spectral-chunk "$w" "$c" >> "$LOGS/wl${w}_c${c}.log" 2>&1 \
        || { stamp "FAILED $LANTERN piece $w $c"; exit 1; }
done < "$LOGS/failed_$k.txt"
if [ "$k" = 0 ] && ! grep -q "saved\|already present" "$LOGS/single.log"; then
    stamp "FAILED $LANTERN single dataset"; exit 1
fi
stamp "done"
