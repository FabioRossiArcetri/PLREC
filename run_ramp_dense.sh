#!/bin/bash
# the dense petal-0 ramp (gen_ramp_dense.py): 41 wavelengths in parallel, then assembled
cd /raid1/frossi/dev/work/PLREC
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
export CBEAM_LANTERN=pl19_1200 CBEAM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p logs/ramp_dense
date
seq 0 40 | xargs -P 21 -I{} bash -c 'python gen_ramp_dense.py piece {} > logs/ramp_dense/w_{}.log 2>&1; true'
python gen_ramp_dense.py assemble > logs/ramp_dense/assemble.log 2>&1
cat logs/ramp_dense/assemble.log
date
echo RAMPDONE
