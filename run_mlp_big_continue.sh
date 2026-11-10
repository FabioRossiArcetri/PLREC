#!/bin/bash
# Continue every 4x256 model towards 20 000 epochs with patience 500 (4 jobs at a time).
set -u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
cd /raid1/frossi/dev/work/PLREC || exit 1
LOG=logs/mlp_big2
mkdir -p "$LOG"
export SR_TARGET=20000 SR_PATIENCE=500 SR_LR=3e-4 OMP_NUM_THREADS=14
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "CONTINUE START $(date -Is)" >> "$LOG/run.log"

python - <<'PY' > "$LOG/jobs.txt"
for r in (0, 30, 60, 90, 120, 150):
    print("single", r)
for w in range(21):
    print("perlambda", w)
PY
echo "jobs: $(wc -l < "$LOG/jobs.txt")" >> "$LOG/run.log"

awk '{printf "python scripts/continue_mlp_big.py --stage %s --%s %s >> '"$LOG"'/job_%s_%s.log 2>&1\n",
      $1, ($1 == "single" ? "level" : "wl"), $2, $1, $2}' "$LOG/jobs.txt" > "$LOG/cmds.txt"
xargs -P 4 -d '\n' -I CMD bash -c 'CMD' < "$LOG/cmds.txt"
echo "CONTINUE ALLDONE $(date -Is)" >> "$LOG/run.log"
