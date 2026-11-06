#!/bin/bash
# Assemble the spectral dataset, then execute the pipeline notebooks of both
# repos for the lantern (PLREC/run_lantern.sh steps 3 and 4) on one A100.
# Submitted by submit.sh after the gen array.
#SBATCH --job-name=pl-finish
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SPECIAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00

source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
LOGS="$LANTERN_ROOT/logs/$LANTERN"; mkdir -p "$LOGS"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$PLREC"
stamp "$LANTERN: assembling the spectral dataset"
python gen_datasets.py spectral-assemble > "$LOGS/assemble.log" 2>&1 || { stamp "FAILED assemble"; exit 1; }
stamp "$LANTERN: notebooks"
# Slurm already restricts the job to its GPU, which is device 0 inside the job
STEPS=34 bash run_lantern.sh "$LANTERN" 0
grep -q FAIL "$PLREC/runs/$LANTERN"/*.log 2>/dev/null && stamp "some notebooks FAILED, see the logs"
stamp "done"
