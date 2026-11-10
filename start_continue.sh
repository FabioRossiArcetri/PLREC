#!/bin/bash
# Verify the resumable trainer is present, then start the continuation campaign detached.
set -u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
cd /raid1/frossi/dev/work/PLREC || exit 1
python - <<'PY'
import inspect
import ml_reconstruct as ml
src = inspect.signature(ml.train_mlp)
print("init_params supported:", "init_params" in str(src))
PY
rm -f logs/mlp_big2/*.log
setsid nohup bash run_mlp_big_continue.sh > logs/mlp_big2_outer.log 2>&1 < /dev/null &
sleep 3
echo "launched"
