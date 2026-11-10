#!/bin/bash
cd /raid1/frossi/dev/work/PLREC || exit 1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
python - <<'PY'
import numpy as np
d = np.load('data/ml/pl19_1200/cache_big/ao0-mlp-256x4.npz', allow_pickle=True)
print('keys:', list(d.files))
for k in d.files:
    a = d[k]
    print(f'  {k}: shape {getattr(a, "shape", None)} dtype {getattr(a, "dtype", None)}')
PY
echo "--- per-level results as printed during training ---"
grep -h "best epoch" logs/mlp_big/single_ao*.log logs/mlp_big2/single_*.log 2>/dev/null | head -12
echo "--- per-wavelength results ---"
grep -h "best epoch" logs/mlp_big/wl_*.log logs/mlp_big2/wl_*.log 2>/dev/null | head -8
