#!/bin/bash
# Big MLP campaign on sauron: 6 single-wavelength levels, then 21 per-wavelength models.
# 4 jobs at a time, 12 threads each (64 cores). Logs in logs/mlp_big/.
set -u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
cd /raid1/frossi/dev/work/PLREC || exit 1
LOGD=logs/mlp_big
mkdir -p "$LOGD" data/ml/pl19_1200/cache_big
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=12

echo "=== single-wavelength levels: $(date -Is)" >> "$LOGD/run.log"
printf "0\n30\n60\n90\n120\n150\n" | sed "s/^/single ao/" > "$LOGD/jobs_single.txt"
python - <<'PY' > "$LOGD/jobs_perlambda.txt"
import numpy as np, sys
sys.path.insert(0, '.')
import ml_reconstruct as ml
ds = ml.LanternDataset.load('data/ml/pl19_1200/spectral_ao120_21wl.npz')
for w, lam in enumerate(np.asarray(ds.wavelengths_nm, float)):
    print(w, lam)
PY
echo "jobs: $(wc -l < "$LOGD/jobs_single.txt") single, $(wc -l < "$LOGD/jobs_perlambda.txt") per-wavelength" >> "$LOGD/run.log"

# stage 1: the six levels, up to 4 in parallel
# the six levels are independent: one job each, 4 at a time
printf "0\n30\n60\n90\n120\n150\n" | \
  xargs -P 4 -I R bash -c 'python scripts/train_mlp_big.py --stage single --level R >> '"$LOGD"'/single_aoR.log 2>&1'
echo "single stage done $(date -Is)" >> "$LOGD/run.log"

# stage 2: the 21 per-wavelength models, 4 in parallel
awk '{printf "python scripts/train_mlp_big.py --stage perlambda --wl %s >> '"$LOGD"'/wl_%s.log 2>&1\n", $1, $1}' \
  "$LOGD/jobs_perlambda.txt" > "$LOGD/cmds_perlambda.txt"
xargs -P 4 -d '\n' -I CMD bash -c 'CMD' < "$LOGD/cmds_perlambda.txt"
echo "ALLDONE $(date -Is)" >> "$LOGD/run.log"
