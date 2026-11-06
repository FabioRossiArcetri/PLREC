# Running the lantern pipeline on Leonardo (CINECA)

The cbeam + PLREC computations -- 41-wavelength characterizations, datasets,
notebooks of both repos, and the pl19_800-only studies -- as Slurm jobs on
Leonardo's Booster partition (project `IscrC_SPECIAL`). These scripts only
*call* the existing entry points (`cbeam/characterize.py`,
`PLREC/gen_datasets.py`, `PLREC/run_lantern.sh`, `PLREC/scripts/*.py`); no
computation is split internally. What is distributed is what is already
independent: the wavelengths of a characterization and the 410 pieces of a
spectral dataset, many single-threaded processes on whole nodes.

## Layout on Leonardo

Everything lives under `LANTERN_ROOT=/leonardo_work/IscrC_SPECIAL/lantern`,
apart from the project's SPECULA environments (`specula_main`, ...):

| path | what |
|---|---|
| `miniforge3/`, `env/` | private conda + Python 3.12 environment (versions of the local `plrec` env, `requirements.txt`) |
| `julia_depot/`, `julia_env/` | Julia and its packages for juliacall (cbeam's FEval, PySR) |
| `wavesolve/` | jw-lin/wavesolve at the commit used locally (9b1272d) |
| `cbeam/`, `PLREC/` | copies of the local working trees (`sync_to_leonardo.sh`); data, caches and results inside them as locally |
| `logs/<lantern>/` | job logs (`char/`, `gen/`, `finish.out`, `extras/`) |

## One-time setup

From this machine (Git Bash), then on a Leonardo login node:

```bash
bash PLREC/leonardo/sync_to_leonardo.sh                    # local -> Leonardo, code only
ssh leonardo
cd /leonardo_work/IscrC_SPECIAL/lantern
sbatch PLREC/leonardo/setup.slurm --extras       # ~30-60 min; log: setup-<jobid>.out
```

Compute nodes have no internet, so the setup installs and precompiles
everything (Python packages, Julia, cbeam's FEval project, PySR) beforehand,
and the jobs run offline (`env.sh` sets `JULIA_PKG_OFFLINE`). It runs on the
serial partition, whose nodes have internet but, unlike the login nodes, no
~10-minute CPU limit per process (precompiling PySR's Julia package takes longer).
`env.sh` puts the environment's `libstdc++` first on `LD_LIBRARY_PATH`: the
system one (RHEL 8) is too old for Julia 1.13.

## Running

```bash
cd /leonardo_work/IscrC_SPECIAL/lantern
bash PLREC/leonardo/submit.sh pl19_800 pl7_1200 pl19_1200     # main pipeline
bash PLREC/leonardo/submit_extras.sh --after <pl19_800 finish job id>   # pl19_800 studies
squeue -A iscrc_special                                       # everything of the project
```

Per lantern, `submit.sh` chains three jobs with `afterok` dependencies:

| job | resources | what |
|---|---|---|
| `char-<lantern>` | array of 1-2 whole nodes | `characterize.py --lantern L i` for the 41 wavelengths, as many in parallel per node as cores and memory allow |
| `gen-<lantern>` | array of `share` whole nodes | `gen_datasets.py`: the single dataset + the 410 spectral pieces, split across the nodes |
| `fin-<lantern>` | 1 A100, 8 cores | `spectral-assemble`, then `run_lantern.sh` steps 3-4 (all pipeline notebooks of both repos -> `runs/<lantern>/`) |

**Node budget.** The project's group may use at most 8 nodes. `submit.sh`
counts the nodes already running or queued for `iscrc_special` (any user, e.g.
other GANtastic or SPECULA jobs), splits what is left between the lanterns, and
throttles each array to its share; `--max-nodes N` lowers the budget further.
`submit_extras.sh` adds at most 2 nodes. A failed step stops its lantern's
chain; fix and resubmit from that step with `--from gen` or `--from finish`
(pieces and wavelengths already done are skipped).

**Cost.** Roughly, per lantern: characterization 0.5-1 node-hour, datasets
2-4 node-hours (19 ports) or ~1 (7 ports), notebooks ~1 GPU-hour; the extras
~15-25 node-hours (mostly the MLP search). `saldo -b` shows the project's hours.

## Results back

```bash
bash PLREC/leonardo/fetch_from_leonardo.sh pl19_800 pl7_1200 pl19_1200          # notebooks + logs
bash PLREC/leonardo/fetch_from_leonardo.sh pl7_1200 --data                      # + datasets and models
```

## Notes

- The job scripts take per-job memory peaks measured on sauron (`common.sh`) and
  use at most 80% of a node's memory.
- The single dataset of a lantern runs as one of the parallel slots of the
  first `gen` task, with 4 FFT threads.
- `env.sh` deliberately does not set `MPLBACKEND=Agg`: it would drop the figures
  of the executed notebooks.
