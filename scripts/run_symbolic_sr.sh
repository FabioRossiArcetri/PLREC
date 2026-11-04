#!/bin/bash
# run_symbolic_sr.sh: both symbolic-regression engines of scripts/symbolic_sr.py,
# concurrently on 32 cores each (about an hour). Restartable. Detach it, e.g.
#   setsid nohup bash scripts/run_symbolic_sr.sh > data/ml/symbolic/run.log 2>&1 < /dev/null & disown
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
mkdir -p data/ml/symbolic
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/symbolic_sr.py --engine operon > data/ml/symbolic/operon.log 2>&1 &
PYTHON_JULIACALL_THREADS=32 python scripts/symbolic_sr.py --engine pysr > data/ml/symbolic/pysr.log 2>&1 &
wait
echo "symbolic done $(date)"
