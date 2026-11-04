#!/bin/bash
# run_mlp_hpo.sh [workers]: every stage of scripts/mlp_hpo.py, the training
# stages with `workers` processes sharing GPU 1 (default 3). Restartable:
# finished runs are skipped. Detach it from the session, e.g.
#   setsid nohup bash scripts/run_mlp_hpo.sh > data/ml/mlp_hpo/run.log 2>&1 < /dev/null & disown
N=${1:-3}
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
# GPU 1 for jax, without preallocation so the workers can share it.
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p data/ml/mlp_hpo/logs
for stage in grid perwl seeds; do
  for ((k = 0; k < N; k++)); do
    python scripts/mlp_hpo.py $stage --worker $k --workers $N > data/ml/mlp_hpo/logs/${stage}_w$k.log 2>&1 &
  done
  wait
  echo "$stage done $(date)"
done
python scripts/mlp_hpo.py unwrap > data/ml/mlp_hpo/logs/unwrap.log 2>&1
echo "unwrap done $(date)"
