# DNA Supplementary Code Bundle

This folder contains the DNA supplementary code for the project. It includes:

- DNA oracle training code
- DNA optimization / evaluation / figure-generation code

The folder is organized so the copied scripts run from within this directory
without depending on the rest of the repository.

Companion folders are also provided:

- `../supplementary_code_protein`
- `../supplementary_code_rna`

These follow the same overall pattern as the DNA folder, but do not include
datasets, checkpoints, or result files.

## Data Layout

All large assets are centralized under:

`data/`

Please download the data needed to run dna_optimization from : https://drive.google.com/file/d/1uiubyhXh10nnawDdMMvB4U_czYzkb_VW/view?usp=sharing

## Included

- `optimize_dna.py`: main DNA optimization script for GRACE and baselines
- `summarize_dna_multistart.py`: summarize multistart DNA results
- `model/`: GRACE helper code used by the optimizer
- `Big_Oracles/DNA/train_cnn_dna.py`: train the main DNA CNN optimization oracles
- `Big_Oracles/DNA/train_dna_validation.py`: train the independent validation DNA oracles
- `Big_Oracles/DNA/eval_dna_test.py`: evaluate trained DNA CNNs on held-out test data
- `Big_Oracles/DNA/prepare_dna_multistart.py`: prepare the 5 multistart seeds
- `Big_Oracles/DNA/train_dna_cluster.sh`: example cluster launcher for main DNA oracle training
- `Big_Oracles/DNA/train_dna_validation_cluster.sh`: example cluster launcher for validation-oracle training
- `data/Big_Oracles/DNA/data/`: DNA assay tables and the multistart seed CSV
- `data/Big_Oracles/DNA/cnn_dna_models/`: bundled training/eval CSV artifacts for the main DNA oracles
- `data/Big_Oracles/DNA/cnn_dna_models_v2/`: bundled training/eval CSV artifacts for the validation DNA oracles
- `bio_constrain_bench/tasks/`: DNA task definitions
- `bio_constrain_bench/data/dna/`: DNA benchmark start sequences
- `data/results/dna/`: result JSONs used in the paper for the DNA task
- `paper_figures/`: DNA-runnable plotting/table scripts

## Oracle Weights

The trained DNA oracle checkpoints are expected under:

`data/Big_Oracles/DNA/cnn_dna_models/`

Expected layout:

```text
data/Big_Oracles/DNA/cnn_dna_models/
  hepG2/
    best_model.pt
    normalization.npy
  k562/
    best_model.pt
    normalization.npy
  sknsh/
    best_model.pt
    normalization.npy
```

An external data host link can be listed here if needed.

## Install

Core packages:

```bash
pip install -r requirements.txt
```

Additional packages for the `botorch_baseline` mode:

```bash
pip install -r requirements-botorch.txt
```

## Run A Single DNA Optimization

Example:

```bash
python optimize_dna.py \
  --sequence ACGTACGTACGT \
  --target k562 \
  --constraints hepG2 sknsh \
  --mode grace_lagrangian \
  --steps 500 \
  --K 8 \
  --lr 1e-2 \
  --constraint_eps 0.1 \
  --constraint_type absolute \
  --out data/results/dna/k562_vs_others/start_01/grace_lagrangian.json
```

Replace `--sequence` with a real enhancer sequence from the bundled start CSVs.

## Train The DNA Oracles

Train the main optimization oracles:

```bash
python Big_Oracles/DNA/train_cnn_dna.py --task all
```

Train the independent validation oracles used for reward-hacking checks:

```bash
python Big_Oracles/DNA/train_dna_validation.py --task all
```

Evaluate a trained oracle on its held-out test split:

```bash
python Big_Oracles/DNA/eval_dna_test.py --task sknsh
```

## Reproduce The Multistart Setup

Prepare the chosen starts:

```bash
python Big_Oracles/DNA/prepare_dna_multistart.py
```

Summarize bundled multistart runs:

```bash
python summarize_dna_multistart.py
```

## Figure / Table Scripts

Main DNA feasible-improvement figure:

```bash
python paper_figures/plot_feasible_by_task.py
```

DNA ablation feasible-improvement figure:

```bash
python paper_figures/plot_feasible_by_task_ablation.py
```

DNA ablation table:

```bash
python paper_figures/table_feasible_by_task_ablation.py
```

Single-run real-trajectory plot:

```bash
python paper_figures/plot_dna_single_run_trajectory.py \
  --run-dir data/results/dna/sknsh_vs_others/start_03
```
