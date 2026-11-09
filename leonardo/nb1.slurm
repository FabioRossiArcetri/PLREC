#!/bin/bash
# Run ONE pipeline notebook of PLREC on one GPU, inside a dbg window (30 min).
#
# Submitted by the pipeline driver with LANTERN and NB (notebook base name)
# exported, and ASSEMBLE=1 for the first notebook of a lantern, which then also
# assembles the spectral dataset from the 410 pieces before running.
#
# Kept as one notebook per job because the whole set (~1 GPU-hour) does not fit
# the 30-minute dbg QoS; the notebooks are also chained (each writes the model
# caches the next one reads).
#SBATCH --job-name=pl-nb
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SPECIAL
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00

source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
LOGS="$LANTERN_ROOT/logs/$LANTERN"
mkdir -p "$LOGS" "$PLREC/runs/$LANTERN"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PLREC"

if [ "${ASSEMBLE:-0}" = 1 ]; then
    stamp "$LANTERN: assembling the spectral dataset"
    python gen_datasets.py spectral-assemble > "$LOGS/assemble.log" 2>&1 \
        || { stamp "FAILED assemble (see $LOGS/assemble.log)"; exit 1; }
fi

stamp "$LANTERN: notebook $NB"
IPYTHONDIR="$PLREC/tools/ipython" \
PLREC_FIG_EXPORT="$PLREC/runs/$LANTERN/figs" \
PLREC_FIG_PREFIX="$NB" \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "runs/$LANTERN" "$NB.ipynb" > "$LOGS/nb_$NB.log" 2>&1 \
    || { stamp "FAILED $NB (see $LOGS/nb_$NB.log)"; exit 1; }
stamp "done $NB"
