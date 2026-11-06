# Environment of the lantern pipeline on Leonardo:  source PLREC/leonardo/env.sh [--online] [--sr]
# (jobs source it; setup_env.sh sources it with --online on a node with internet).
# Uses only $LANTERN_ROOT: nothing from the project's SPECULA environments.
#   --sr      the symbolic-regression environment (env-sr: the main one + PySR and
#             pyoperon, with its own Julia project), used only by the PySR/Operon
#             runs of the pl19_800 extras -- PySR needs an older juliacall/PythonCall
#             than the main pipeline, so the two are kept apart.
#   --online  allow juliapkg / Pkg to use the network (setup only)
export LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
_online=""; _sr=""
for _a in "$@"; do
    case $_a in --online) _online=1 ;; --sr) _sr=1 ;; esac
done
_env="$LANTERN_ROOT/env"; _jlproj="$LANTERN_ROOT/julia_env"
if [ -n "$_sr" ]; then _env="$LANTERN_ROOT/env-sr"; _jlproj="$LANTERN_ROOT/julia_env_sr"; fi

source "$LANTERN_ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$_env"
# the environment's libstdc++ before the system's (RHEL 8: no GLIBCXX_3.4.30, which
# Julia needs): otherwise whichever module loads libstdc++ first (gmsh, scipy, ...)
# decides, and juliacall then fails to start
export LD_LIBRARY_PATH="$_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Julia (juliacall): private depot (shared by both environments) and project;
# offline unless --online (compute nodes have no internet, setup_env.sh installs everything)
export JULIA_DEPOT_PATH="$LANTERN_ROOT/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$_jlproj"
if [ -n "$_online" ]; then
    unset PYTHON_JULIAPKG_OFFLINE JULIA_PKG_OFFLINE
else
    export PYTHON_JULIAPKG_OFFLINE=yes JULIA_PKG_OFFLINE=true
fi
export PYTHON_JULIACALL_THREADS=${PYTHON_JULIACALL_THREADS:-1} JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}

# the pipeline: numpy backend, one thread per process unless a job says otherwise
export CBEAM_BACKEND=numpy PYTHONUTF8=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} \
       MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} CBEAM_FFT_WORKERS=${CBEAM_FFT_WORKERS:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# NB: no MPLBACKEND=Agg -- it would drop the figures of the executed notebooks
unset _a _online _sr _env _jlproj
