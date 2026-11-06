#!/bin/bash
# Run setup_env.sh as a batch job on the serial partition: its nodes have
# internet like the login nodes, but no interactive CPU-time limit (the login
# nodes kill a process after ~10 min of CPU, too little for precompiling Julia
# packages such as PySR's SymbolicRegression).
#   cd /leonardo_work/IscrC_SPECIAL/lantern && sbatch PLREC/leonardo/setup.slurm [--extras]
#SBATCH --job-name=pl-setup
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_SPECIAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --output=setup-%j.out

export LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
bash "$LANTERN_ROOT/PLREC/leonardo/setup_env.sh" "$@"
