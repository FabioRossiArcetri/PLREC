#!/bin/bash
# run_symbolic_template.sh: template symbolic regression (scripts/symbolic_template.py)
# over +-lambda/2 (400 nm) for the opposite class and piston 0, concurrently (15 min).
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_JULIACALL_THREADS=32
for c in opposite adjacent-0; do
  python scripts/symbolic_template.py --cls $c > data/ml/symbolic/template_$c.log 2>&1 &
done
wait
echo "template done $(date)"
