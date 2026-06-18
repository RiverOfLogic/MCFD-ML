# MCFD-ML

MCFD-ML is a PyTorch project for multi-source cross-condition equipment fault diagnosis. It evaluates domain adaptation and domain generalization methods on bearing vibration datasets under changing speed and load conditions.

The main method is **MCFD-ML**, previously named MEDG in earlier internal code. The implementation files `src/MEDG.py` and `src/MEDGNet.py` are kept for backward compatibility, while experiment configs and reported method names now use `MCFD-ML`.

## Features

- Multi-source fault diagnosis on DIRG and MAFAULDA.
- Baselines: ERM, DANN, M-DANN, CDAN, MCD, MLDG, and CDDG.
- Main method: MCFD-ML with meta-learning, adversarial domain alignment, CORAL, domain supervision, HSIC disentanglement, and reconstruction.
- YAML-driven experiment scheduler with automatic GPU assignment.
- Repeated experiments with different random seeds.
- CSV outputs for raw runs and aggregated mean/std results.
- MCFD-ML ablation experiments.

## Repository Layout

```text
src/
  MEDG.py              # MCFD-ML training entrypoint, legacy filename
  MEDGNet.py           # Shared encoder and MCFD-ML network modules
  ERM.py               # Empirical risk minimization baseline
  DANN0.py             # Standard two-domain DANN baseline
  DANN.py              # Multi-domain DANN baseline
  CDAN.py              # CDAN baseline
  MCD.py               # MCD baseline
  MLDG.py              # MLDG baseline
  CDDG.py              # CDDG baseline
  config.py            # Default runtime config plus scheduler overrides
scripts/
  preprocess_dirg.py
  preprocess_mafault.py
  run_experiments.py   # YAML experiment scheduler
experiments/
  auto_train.yaml      # Full benchmark config
  mcfd_ml_ablation.yaml # MCFD-ML ablation config
```

## Datasets

Expected raw data locations:

```text
raw_data/DIRG/
raw_data/MAFDATA/
```

Processed data is saved as NumPy arrays:

```text
data/DIRG/
data/MAFAULDA/
```

Each processed dataset contains:

```text
train_x.npy, train_y.npy, train_info.npy
val_x.npy, val_y.npy, val_info.npy
test_x.npy, test_y.npy, test_info.npy
```

`*_x.npy` has shape `(N, channels, 2048)`. `*_info.npy` stores `[speed, load]` domain metadata.

## Installation

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default package is not suitable for your machine.

## Preprocessing

```bash
python scripts/preprocess_dirg.py --raw raw_data/DIRG --save data/DIRG
python scripts/preprocess_mafault.py --raw raw_data/MAFDATA --save data/MAFAULDA
```

DIRG uses 6 channels. MAFAULDA uses 8 channels.

## Training

Run a dry run first to inspect the expanded jobs:

```bash
python scripts/run_experiments.py --config experiments/auto_train.yaml --dry-run
```

Run the full benchmark:

```bash
python scripts/run_experiments.py --config experiments/auto_train.yaml
```

Resume an interrupted benchmark:

```bash
python scripts/run_experiments.py --config experiments/auto_train.yaml --resume
```

By default, `experiments/auto_train.yaml` runs 8 methods, 8 tasks, and 10 seeds. Tasks 1-4 use DIRG with 6 channels. Tasks 5-8 use MAFAULDA with 8 channels.

## MCFD-ML Ablations

Run MCFD-ML ablations:

```bash
python scripts/run_experiments.py --config experiments/mcfd_ml_ablation.yaml --dry-run
python scripts/run_experiments.py --config experiments/mcfd_ml_ablation.yaml
```

The default ablation config runs:

- `MCFD-ML-no-meta`
- `MCFD-ML-no-adv`
- `MCFD-ML-no-coral`
- `MCFD-ML-no-domain`
- `MCFD-ML-no-HSIC`
- `MCFD-ML-no-rec`

Full MCFD-ML results are expected to come from the main benchmark, then can be merged with ablation results for reporting.

## Outputs

The scheduler writes all outputs under the configured `output_dir`, for example:

```text
experiments/results/20260617_120000/
  raw_runs.csv
  summary.csv
  runs/
  logs/
  models/
  figures/
  runtime_configs/
```

`raw_runs.csv` stores each seed-level run. `summary.csv` stores mean and standard deviation by method, dataset, and task.

## GPU Scheduling

Set `gpus: auto` and `max_jobs_per_gpu: auto` to use memory-aware scheduling. The scheduler queries `nvidia-smi`, estimates each job's memory need, and launches jobs when enough free memory is available.

If you see CUDA out-of-memory errors, lower `gpu_scheduler.max_jobs_per_gpu_cap` or increase `gpu_scheduler.method_memory_mb` for the affected method.

## Custom Data Paths

Dataset paths are configured in YAML:

```yaml
datasets:
  DIRG:
    tasks: [1, 2, 3, 4]
    channels: 6
    path: data/DIRG
  MAFAULDA:
    tasks: [5, 6, 7, 8]
    channels: 8
    path: data/MAFAULDA
```

Use absolute paths or paths relative to the repository root.

## Compatibility Notes

- `MCFD-ML` is the public method name.
- `MEDG.py`, `MEDGNet.py`, and runtime keys such as `medg_ablation` remain for compatibility with existing scripts and checkpoints.
- Old YAML configs using `MEDG` still work, but new experiment outputs should use `MCFD-ML`.
