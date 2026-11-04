#!/bin/bash
# run_symbolic_sr_range.sh [R] [wait_pid]: both engines of scripts/symbolic_sr.py on
# the +-R nm synthetic problem (default 100); PySR starts after process wait_pid
# (e.g. a running PySR search) has exited, so the two PySR searches do not compete.
R=${1:-100}
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/symbolic_sr.py --engine operon --range $R > data/ml/symbolic/operon_r$R.log 2>&1
if [ -n "$2" ]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
PYTHON_JULIACALL_THREADS=32 python scripts/symbolic_sr.py --engine pysr --range $R > data/ml/symbolic/pysr_r$R.log 2>&1
echo "range $R done $(date)"
