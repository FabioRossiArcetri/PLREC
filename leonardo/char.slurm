#!/bin/bash
# Characterize a lantern at its 41 wavelengths: array task k of n (one whole
# Booster node each) takes the wavelength indices i with i % n == k and runs
# them as independent single-threaded cbeam/characterize.py processes.
# Submitted by submit.sh (LANTERN, NTASKS exported).
#SBATCH --job-name=pl-char
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SPECIAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=02:00:00

source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
k=${SLURM_ARRAY_TASK_ID:-0}; n=${NTASKS:-1}
LOGS="$LANTERN_ROOT/logs/$LANTERN/char"; mkdir -p "$LOGS"
P=$(fit_on_node "${PEAK_CHAR[$LANTERN]}")
IDX=$(python -c "print(' '.join(str(i) for i in range(41) if i % $n == $k))")
stamp "$LANTERN characterization, task $k/$n on $(hostname): indices [$IDX], $P in parallel"

cd "$CBEAM"
printf '%s\n' $IDX | xargs -P "$P" -I{} sh -c \
    'python characterize.py --lantern "$LANTERN" {} > "'"$LOGS"'/char_{}.log" 2>&1 || echo {}' \
    > "$LOGS/failed_$k.txt"
# retry failures once (e.g. a transient Julia start-up error); cached ones are skipped
for i in $(cat "$LOGS/failed_$k.txt"); do
    python characterize.py --lantern "$LANTERN" "$i" >> "$LOGS/char_$i.log" 2>&1 \
        || { stamp "FAILED $LANTERN wavelength $i"; exit 1; }
done
stamp "done"
