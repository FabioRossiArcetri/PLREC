#!/bin/bash
# Smoke test on a Booster compute node (debug QoS, <= 30 min), before a campaign:
#   sbatch PLREC/leonardo/smoke.slurm         (from $LANTERN_ROOT; LANTERN=pl7_1200 by default)
# one characterization at the central wavelength (index 20; it lands in the cache,
# so the campaign reuses it), the peak memory of a dataset piece, and jax on the GPU.
#SBATCH --job-name=pl-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=IscrC_SPECIAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=smoke-%j.out

export LANTERN=${LANTERN:-pl7_1200} LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
stamp "smoke test $LANTERN on $(hostname)"

cd "$CBEAM"
/usr/bin/time -f "CHAR wall %e s, peak %M kB" python characterize.py --lantern "$LANTERN" 20 2>&1 | tail -5
cd "$PLREC"
python gen_datasets.py peak-memory 2>&1 | tail -3
python -c "import jax, jax.numpy as jnp; print('jax', jax.devices()); print(float(jnp.ones(1000).sum()))"
stamp "smoke done"
