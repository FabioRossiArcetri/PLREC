#!/usr/bin/env bash
# The pl19_800-only studies (derivation, symbolic, psf_zoo, piston2, mlp_hpo) on
# one many-core host with GPUs -- the sauron counterpart of leonardo/extras.slurm:
#
#   data50k  -> { cpu: psf_zoo, piston2 twins, symbolic regression, residuals
#                 gpu: MLP hyperparameter search (one worker per GPU), unwrap,
#                      piston2 re-referencing }  -> notebooks (runs/pl19_800/)
#
#   setsid nohup bash run_extras.sh [n_gpus] > logs/extras.log 2>&1 < /dev/null &
# Run from PLREC with the environment active (PySR / pyoperon / jax / cbeam) and
# the pl19_800 datasets in place (run_lantern.sh pl19_800). STAGES (default
# "data50k cpu gpu notebooks"; cpu and gpu run concurrently) selects stages.
set -uo pipefail
cd "$(dirname "$0")"
PLREC=$(pwd); CBEAM=$(cd ../cbeam && pwd)
NGPU=${1:-2}
STAGES=${STAGES:-"data50k cpu gpu notebooks"}
export CBEAM_LANTERN=pl19_800 CBEAM_BACKEND=numpy PYTHONUTF8=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$CBEAM/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/ml/symbolic data/ml/piston2 data/ml/psf_zoo data/ml/mlp_hpo/logs logs
stamp() { echo "[$(date '+%F %T')] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
rc=0

if has data50k; then
    stamp "data50k"
    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 CBEAM_FFT_WORKERS=16 python - \
        > logs/extras_data50k.log 2>&1 <<'EOF' || { stamp "FAIL data50k"; exit 1; }
import os, time
from ml_reconstruct.simulate import make_single_dataset
from ml_reconstruct.presets import settings
out = "data/ml/single_800nm_50k.npz"
if os.path.exists(out):
    print("already present:", out)
else:
    t = time.time()
    ds = make_single_dataset(n=50000, n_train=40000, span_nm=settings("pl19_800")["single_span_nm"],
                             lantern="pl19_800", prop_chunk_size=5000)
    ds.save(out + ".tmp.npz"); os.replace(out + ".tmp.npz", out)
    print(f"saved {out} in {time.time() - t:.0f} s", flush=True)
EOF
    stamp "data50k done"
fi

cpu_stage() {
    export JAX_PLATFORMS=cpu
    ( export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CBEAM_FFT_WORKERS=8
      python scripts/psf_zoo.py > data/ml/psf_zoo/run.log 2>&1
      for fam in random even lines examples; do
          python scripts/piston2_twins.py --family $fam > "data/ml/piston2/twins_$fam.log" 2>&1
      done
      python scripts/symbolic_residuals.py > data/ml/symbolic/residuals.log 2>&1 ) &
    ( python scripts/symbolic_sr.py --engine operon > data/ml/symbolic/operon.log 2>&1
      python scripts/symbolic_sr.py --engine operon --range 100 > data/ml/symbolic/operon_r100.log 2>&1 ) &
    ( export PYTHON_JULIACALL_THREADS=16
      python scripts/symbolic_sr.py --engine pysr > data/ml/symbolic/pysr.log 2>&1
      python scripts/symbolic_sr.py --engine pysr --range 100 > data/ml/symbolic/pysr_r100.log 2>&1
      for c in opposite adjacent-0; do
          python scripts/symbolic_template.py --cls $c > "data/ml/symbolic/template_$c.log" 2>&1
      done ) &
    wait
    stamp "cpu stage done"
}

gpu_stage() {
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    for stage in grid perwl seeds data800; do
        for ((g = 0; g < NGPU; g++)); do
            CUDA_VISIBLE_DEVICES=$g python scripts/mlp_hpo.py $stage --worker $g --workers $NGPU \
                > "data/ml/mlp_hpo/logs/${stage}_w$g.log" 2>&1 &
        done
        wait
        stamp "mlp_hpo $stage done"
    done
    python scripts/mlp_hpo.py unwrap > data/ml/mlp_hpo/logs/unwrap.log 2>&1 || stamp "FAIL mlp_hpo unwrap"
    CUDA_VISIBLE_DEVICES=0 python scripts/piston2_reref.py > data/ml/piston2/reref.log 2>&1 || stamp "FAIL piston2_reref"
    stamp "gpu stage done"
}

if has cpu; then stamp "cpu stage"; cpu_stage > logs/extras_cpu.log 2>&1 & fi
if has gpu; then stamp "gpu stage"; gpu_stage > logs/extras_gpu.log 2>&1 & fi
wait

if has notebooks; then
    stamp "notebooks"
    out="$PLREC/runs/pl19_800"; mkdir -p "$out"
    for nb in derivation symbolic psf_zoo piston2 mlp_hpo; do
        if IPYTHONDIR="$PLREC/tools/ipython" PLREC_FIG_EXPORT="$out/figs" PLREC_FIG_PREFIX="$nb" \
           CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
           jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
               --output-dir "$out" "$nb.ipynb" > "$out/$nb.log" 2>&1; then stamp "    OK   $nb"
        else stamp "    FAIL $nb (see $out/$nb.log)"; rc=1; fi
    done
fi
stamp "EXTRASDONE (rc $rc)"
exit $rc
