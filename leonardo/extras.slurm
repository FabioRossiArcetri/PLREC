#!/bin/bash
# The pl19_800-only studies of PLREC (what scripts/run_*.sh ran on sauron), by
# stage -- submitted by submit_extras.sh with STAGE exported:
#   data50k    the 50000-sample 800 nm dataset (rng=0: its first 10000 rows are
#              single_800nm.npz), as data/gen/gen_single_50k.py did
#   cpu        psf_zoo, piston2 twins (4 families), symbolic regression
#              (run_symbolic_sr.sh, run_symbolic_sr_range.sh 100,
#              run_symbolic_template.sh) and symbolic_residuals
#   gpu        MLP hyperparameter search (run_mlp_hpo.sh: grid, perwl, seeds,
#              then data800 and unwrap), one worker per A100, + piston2 re-referencing
#   notebooks  derivation, symbolic, psf_zoo, piston2, mlp_hpo -> runs/pl19_800/
#SBATCH --job-name=pl-extras
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_SPECIAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=12:00:00

export LANTERN=pl19_800
source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
: "${STAGE:?STAGE not set}"
LOGS="$LANTERN_ROOT/logs/pl19_800/extras"; mkdir -p "$LOGS"
cd "$PLREC"
mkdir -p data/ml/symbolic data/ml/piston2 data/ml/psf_zoo data/ml/mlp_hpo/logs
stamp "extras stage $STAGE on $(hostname)"
rc=0
case $STAGE in
data50k)
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 CBEAM_FFT_WORKERS=16
    python - > "$LOGS/data50k.log" 2>&1 <<'EOF' || rc=1
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
    ;;
cpu)
    export JAX_PLATFORMS=cpu
    ( export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CBEAM_FFT_WORKERS=8
      python scripts/psf_zoo.py > data/ml/psf_zoo/run.log 2>&1
      for fam in random even lines examples; do
          python scripts/piston2_twins.py --family $fam > "data/ml/piston2/twins_$fam.log" 2>&1
      done
      python scripts/symbolic_residuals.py > data/ml/symbolic/residuals.log 2>&1 ) &
    # PySR / Operon: in env-sr (see env.sh --sr); they use the closed-form model, not cbeam
    ( source "$LANTERN_ROOT/PLREC/leonardo/env.sh" --sr
      python scripts/symbolic_sr.py --engine operon > data/ml/symbolic/operon.log 2>&1
      python scripts/symbolic_sr.py --engine operon --range 100 > data/ml/symbolic/operon_r100.log 2>&1 ) &
    ( export PYTHON_JULIACALL_THREADS=16
      source "$LANTERN_ROOT/PLREC/leonardo/env.sh" --sr
      python scripts/symbolic_sr.py --engine pysr > data/ml/symbolic/pysr.log 2>&1
      python scripts/symbolic_sr.py --engine pysr --range 100 > data/ml/symbolic/pysr_r100.log 2>&1
      for c in opposite adjacent-0; do
          python scripts/symbolic_template.py --cls $c > "data/ml/symbolic/template_$c.log" 2>&1
      done ) &
    wait
    ;;
gpu)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    N=4                                   # one worker per A100
    for stage in grid perwl seeds data800; do
        for ((g = 0; g < N; g++)); do
            CUDA_VISIBLE_DEVICES=$g python scripts/mlp_hpo.py $stage --worker $g --workers $N \
                > "data/ml/mlp_hpo/logs/${stage}_w$g.log" 2>&1 &
        done
        wait
        stamp "mlp_hpo $stage done"
    done
    python scripts/mlp_hpo.py unwrap > data/ml/mlp_hpo/logs/unwrap.log 2>&1 || rc=1
    CUDA_VISIBLE_DEVICES=0 python scripts/piston2_reref.py > data/ml/piston2/reref.log 2>&1 || rc=1
    ;;
notebooks)
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
    out="$PLREC/runs/pl19_800"; mkdir -p "$out"
    for nb in derivation symbolic psf_zoo piston2 mlp_hpo; do
        # figures also pickled to $out/figs for the slides (tools/ipython startup)
        if IPYTHONDIR="$PLREC/tools/ipython" PLREC_FIG_EXPORT="$out/figs" PLREC_FIG_PREFIX="$nb" \
           jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
               --output-dir "$out" "$nb.ipynb" > "$out/$nb.log" 2>&1; then stamp "OK   $nb"
        else stamp "FAIL $nb (see $out/$nb.log)"; rc=1; fi
    done
    ;;
*) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
stamp "stage $STAGE finished (rc $rc)"
exit $rc
