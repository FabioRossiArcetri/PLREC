#!/usr/bin/env bash
# Everything for one lantern preset, end to end, on sauron:
#
#   bash run_lantern.sh pl19_1200 [gpu]
#   STEPS=34 bash run_lantern.sh pl19_1200     # only some steps (default 1234)
#
# 1. cbeam/run_characterize.sh   41 characterizations (half the RAM at most)
# 2. PLREC/run_gen.sh            single + spectral datasets (half the RAM at most)
# 3. the PLREC pipeline notebooks -> PLREC/runs/<lantern>/
# 4. cbeam's example notebooks   -> cbeam/examples/runs/<lantern>/
#    (the ML ones on copies of the PLREC datasets)
# Steps run one after the other, so the memory budget holds throughout.
# Run from PLREC with the cbeam checkout next to it; detach it with
#   setsid nohup bash run_lantern.sh <lantern> > logs/<lantern>.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
LANTERN=${1:?usage: bash run_lantern.sh <lantern> [gpu]}
GPU=${2:-0}
STEPS=${STEPS:-1234}
PLREC=$(pwd); CBEAM=$(cd ../cbeam && pwd)
export CBEAM_LANTERN=$LANTERN PYTHONUTF8=1
export PYTHONPATH="$CBEAM/src${PYTHONPATH:+:$PYTHONPATH}"
stamp() { echo "[$(date '+%F %T')] $*"; }

if [[ $STEPS == *1* ]]; then stamp "1/4 characterization"; bash "$CBEAM/run_characterize.sh"; fi
if [[ $STEPS == *2* ]]; then stamp "2/4 datasets";         bash "$PLREC/run_gen.sh"; fi

# every figure the notebooks draw is also pickled to <outdir>/figs for the slides
# (tools/ipython startup; Presentation/Slides/slide_figs.py redraws them)
run_nbs() {   # <dir> <outdir> <notebooks...>
    local dir=$1 out=$2; shift 2
    mkdir -p "$out"
    for nb in "$@"; do
        if (cd "$dir" && IPYTHONDIR="$PLREC/tools/ipython" PLREC_FIG_EXPORT="$out/figs" \
                PLREC_FIG_PREFIX="${nb%.ipynb}" CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
                jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
                --output-dir "$out" "$nb" > "$out/${nb%.ipynb}.log" 2>&1); then
            stamp "    OK   $nb"
        else
            stamp "    FAIL $nb (see $out/${nb%.ipynb}.log)"
        fi
    done
}

if [[ $STEPS == *3* ]]; then
stamp "3/4 PLREC notebooks"
run_nbs "$PLREC" "$PLREC/runs/$LANTERN" \
    trainMLP.ipynb trainMLP_multiwavelength.ipynb PL_EML.ipynb multi_wavelength.ipynb workbench.ipynb
fi

if [[ $STEPS == *4* ]]; then
stamp "4/4 cbeam example notebooks"
# the two datasets, at the same relative paths in cbeam/examples (its presets
# use the same names: data/ml/single_800nm.npz for pl19_800, data/ml/<lantern>/single.npz else)
for f in $(python -c "import sys; sys.path.insert(0, '$PLREC'); from ml_reconstruct.presets import settings; s = settings(); print(s['single'], s['spectral'])"); do
    mkdir -p "$CBEAM/examples/$(dirname "$f")"
    cp "$PLREC/$f" "$CBEAM/examples/$f" 2>/dev/null || true
done
run_nbs "$CBEAM/examples" "$CBEAM/examples/runs/$LANTERN" \
    BatchedPiston.ipynb multi_wavelength.ipynb trainMLP.ipynb trainMLP_multiwavelength.ipynb PL_EML.ipynb Batched.ipynb
fi

stamp "ALLDONE $LANTERN"
