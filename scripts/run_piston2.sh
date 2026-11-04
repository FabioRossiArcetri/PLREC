#!/bin/bash
# run_piston2.sh: the simulations (CPU) and trainings (GPU 1) behind piston2.ipynb,
# in parallel; restartable. Detach it from the session, e.g.
#   setsid nohup bash scripts/run_piston2.sh > data/ml/piston2/run.log 2>&1 < /dev/null & disown
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
mkdir -p data/ml/piston2
# twins: numpy propagation on the CPU, specula hidden (its log lines), jax kept off the GPU
PYTHONPATH=data/gen/nospecula CBEAM_BACKEND=numpy JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 \
  OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 bash -c "python scripts/piston2_twins.py \
    && python scripts/piston2_twins.py --family even && python scripts/piston2_twins.py --family lines" \
  > data/ml/piston2/twins.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/piston2_reref.py > data/ml/piston2/reref.log 2>&1 &
wait
echo "piston2 done $(date)"
