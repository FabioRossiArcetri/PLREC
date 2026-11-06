#!/usr/bin/env bash
# One-time setup of the lantern environment on Leonardo -- run on a LOGIN node
# (compute nodes have no internet). Everything lives under $LANTERN_ROOT
# (default /leonardo_work/IscrC_SPECIAL/lantern), separate from the SPECULA
# environments of the project:
#
#   $LANTERN_ROOT/miniforge3     private conda installation
#   $LANTERN_ROOT/env            Python 3.12 environment (versions as the local plrec env)
#   $LANTERN_ROOT/julia_depot    Julia packages and precompiled code (juliacall)
#   $LANTERN_ROOT/julia_env      juliapkg project (which Julia, which PythonCall)
#   $LANTERN_ROOT/env-sr, julia_env_sr   with --extras: clone of env + PySR/pyoperon,
#                                with its own Julia project (PySR needs an older juliacall)
#   $LANTERN_ROOT/wavesolve      jw-lin/wavesolve at the commit used locally
#   $LANTERN_ROOT/cbeam, PLREC   the code, copied by sync_to_leonardo.sh (run that first)
#
#   bash PLREC/leonardo/setup_env.sh            # idempotent: re-running skips what exists
#   bash PLREC/leonardo/setup_env.sh --extras   # + PySR / pyoperon for the pl19_800 extras
set -euo pipefail
L=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
WAVESOLVE_COMMIT=9b1272d
MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
mkdir -p "$L" && cd "$L"
[ -d "$L/cbeam" ] && [ -d "$L/PLREC" ] || { echo "copy the code first (sync_to_leonardo.sh)"; exit 1; }

# 1. private Miniforge
if [ ! -x "$L/miniforge3/bin/conda" ]; then
    curl -fsSL -o "$L/miniforge.sh" "$MINIFORGE_URL"
    bash "$L/miniforge.sh" -b -p "$L/miniforge3"
    rm -f "$L/miniforge.sh"
fi
source "$L/miniforge3/etc/profile.d/conda.sh"

# 2. Python environment (pinned to the versions validated on the local machine)
[ -x "$L/env/bin/python" ] || conda create -y -q -p "$L/env" python=3.12
conda activate "$L/env"
# a recent libstdc++ (the system's, RHEL 8, is too old for Julia 1.13); env.sh puts
# $L/env/lib first on LD_LIBRARY_PATH
[ -e "$L/env/lib/libstdc++.so.6" ] || conda install -y -q -p "$L/env" -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
python -m pip install -q --upgrade pip
EXTRAS=""; [ "${1:-}" = "--extras" ] && EXTRAS=1
# the main environment never carries PySR (it needs an older juliacall, see --extras)
python -m pip uninstall -q -y pysr pyoperon 2>/dev/null || true
python -m pip install -q -r "$L/PLREC/leonardo/requirements.txt"
# a Julia project left over from a PySR install in this environment: start it afresh
# (also juliapkg's own record, or it believes the project resolved and skips it)
if grep -q SymbolicRegression "$L/julia_env/Project.toml" 2>/dev/null || \
   { [ -d "$L/julia_env" ] && [ ! -f "$L/julia_env/Project.toml" ]; }; then
    rm -f "$L/julia_env/Project.toml" "$L/julia_env/Manifest.toml" "$L/julia_env/pyjuliapkg/meta.json"
fi

# 3. wavesolve (not on PyPI) and cbeam, editable from their checkouts
if [ ! -d "$L/wavesolve/.git" ]; then
    git clone -q https://github.com/jw-lin/wavesolve.git "$L/wavesolve"
fi
git -C "$L/wavesolve" checkout -q "$WAVESOLVE_COMMIT"
python -m pip install -q -e "$L/wavesolve" -e "$L/cbeam"

# 4. Julia: juliacall downloads Julia + PythonCall into the private depot, then
#    cbeam's FEvalsetup resolves and precompiles its FEval project (needs internet).
source "$L/PLREC/leonardo/env.sh" --online
python - <<'EOF'
import time; t = time.time()
import cbeam                                    # triggers juliapkg + FEval resolve/precompile
from juliacall import Main as jl
print("cbeam", cbeam.__file__, "| Julia", jl.seval("string(VERSION)"), f"| {time.time() - t:.0f} s")
EOF

# 4b. with --extras: env-sr = a clone of env + PySR/pyoperon, with its own Julia
#     project; PySR installs and precompiles SymbolicRegression.jl there
if [ -n "$EXTRAS" ]; then
    [ -x "$L/env-sr/bin/python" ] || conda create -y -q -p "$L/env-sr" --clone "$L/env"
    conda activate "$L/env-sr"
    python -m pip install -q -r "$L/PLREC/leonardo/requirements-extras.txt"
    source "$L/PLREC/leonardo/env.sh" --online --sr
    python -c "import pysr, pyoperon; from pysr import PySRRegressor; print('pysr', pysr.__version__, '| pyoperon ok')"
    source "$L/PLREC/leonardo/env.sh" --sr       # offline check, as in a job
    PYTHONPATH="$L/PLREC" python -c "from pysr import PySRRegressor; import ml_reconstruct; print('env-sr offline ok')"
fi

# 5. smoke test of the stack the jobs use (offline, as on a compute node)
source "$L/PLREC/leonardo/env.sh"
python - <<'EOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ["LANTERN_ROOT"], "PLREC"))
import numpy, scipy, jax, optax, sklearn, gmsh, pygmsh, nbconvert
from ml_reconstruct.simulate import bp
for name in bp.LANTERNS:
    p = bp.get_simulation_parameters(lantern=name)
    print(f"{name}: {p['n_output_positions']} ports, tag {bp.characterization_tag(p)}")
print("numpy", numpy.__version__, "| jax", jax.__version__, jax.devices())
EOF
echo "setup done: $L"
