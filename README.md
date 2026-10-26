# MSTA_CNN_P300
# MSTA-CNN for P300 Speller Decoding

A compact convolutional neural network for offline EEG-based P300 detection and character decoding, with reproducible comparisons of channel attention, temporal attention, and a backbone without attention.

**Author:** S. Ardalan Fatemi  
**Software version:** 2.0.2  
**License:** MIT

## Overview

MSTA-CNN combines temporal convolutions, depthwise spatial filtering, and separable convolutions with a residual dual-branch attention module. The attention branches modulate learned feature channels and pooled temporal representations.

This repository provides the implementation and reference resources for the revised **BCI Competition III Dataset II** analysis, including:

- EEG preprocessing and character-level data partitioning.
- Cross-validation and final within-subject training.
- Flash-level prediction and character decoding over 1–15 stimulus repetitions.
- Matched comparisons of four variants: no attention, channel-only, temporal-only, and full attention.
- Saved reference models for within-subject and cross-subject evaluation.
- Supplementary metrics, feature analysis, and regeneration of selected tables and figures.

The full model contains **3,185 total parameters**, including **3,105 trainable parameters**. The backbone without attention contains **2,001 total parameters**.

## Installation

The recorded execution environment used Windows 11, Python 3.11.0, TensorFlow 2.20.0, and Keras 3.12.0 on an Intel Core i5-12450H CPU with 16 GB RAM. GPU acceleration was not used.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/SeyedArdalanFatemi/MSTA_CNN_P300.git
cd MSTA_CNN_P300
python -m pip install -r requirements.txt
```

A separate Python virtual environment is recommended. Run the commands below from the directory containing `run.py`.

## Quick Start: Inspect the Package and Archived Results

Display the available commands:

```bash
python run.py
```

Check package integrity and run the included tests:

```bash
python run.py verify
python run.py self-test
python run.py smoke-test
```

- `verify` checks the release manifest and reference model bundles.
- `self-test` checks data-processing, decoding, aggregation, and supplementary-analysis logic without TensorFlow training.
- `smoke-test` performs short TensorFlow model checks, including updates, saving, and reloading.

These checks do **not** constitute a complete scientific reproduction.

To regenerate supplementary metrics and selected publication assets from the archived inputs:

```bash
python run.py supplementary
python run.py assets
```

Outputs are written to:

```text
runs/supplementary/
runs/assets/
```

The assets command generates seven tables and four figures. Figures are exported as PNG, PDF, and SVG. These two commands do not require the original EEG recordings or retraining.

## Dataset

The implementation expects the original MATLAB files for **BCI Competition III Dataset II**, recorded using the Wadsworth P300 speller paradigm:

```text
Subject_A_Train.mat
Subject_A_Test.mat
Subject_B_Train.mat
Subject_B_Test.mat
```

Obtain the recordings and official test-character labels from the authorized dataset source. Raw competition recordings are not distributed in this repository.

The implementation is designed for this dataset's structure and stimulus coding; it is not a generic loader for arbitrary EEG datasets.

Historical BCI Competition II results discussed in the manuscript were not regenerated under the revised protocol and are outside this package's reproduction scope.

## Evaluate the Saved Reference Models

Replace `D:\BCI_data` with the path to your dataset folder.

### Within-subject evaluation

```bash
python run.py test --data-dir "D:\BCI_data" --run-dir "runs/reference_test" --model-dir "references/primary"
```

This evaluates the supplied within-subject models on the corresponding subjects' official test sets.

To calculate supplementary metrics from the new evaluation outputs:

```bash
python run.py supplementary --evaluation-dir "runs/reference_test/evaluation/within_subject" --output-dir "runs/new_metrics"
```

### Cross-subject evaluation

```bash
python run.py transfer --data-dir "D:\BCI_data" --run-dir "runs/transfer_test" --model-dir "references/transfer"
```

This evaluates models trained on Subject A using Subject B's test data, and vice versa. The transfer models use separate source-derived training recipes; comparisons with the primary within-subject models are therefore reference comparisons rather than experiments differing only in subject identity.

## Retraining and Feature Analysis

Retraining is optional when the purpose is to inspect the archived results or evaluate the supplied reference weights.

Run the following stages in order:

```bash
python run.py all --data-dir "D:\BCI_data" --run-dir "runs/primary"

python run.py ablation --reference-run "runs/primary" --data-dir "D:\BCI_data" --output-dir "runs/ablation_d25"

python run.py features --reference-run "runs/primary" --ablation-run "runs/ablation_d25" --data-dir "D:\BCI_data" --output-dir "runs/features_d25"
```

The primary workflow includes 60 cross-validation training runs and 12 final within-subject models. The ablation and feature-analysis stages require the completed outputs of the preceding stages; summary JSON files alone are insufficient.

Use `python run.py ablation` for the retained **dropout = 0.25** ablation. The legacy `msta.py --profile ablation` configuration uses a different dropout setting and should not be substituted for this workflow.

### Optional transfer retraining

```bash
python run.py transfer-train --data-dir "D:\BCI_data" --run-dir "runs/transfer_refit"

python run.py transfer --data-dir "D:\BCI_data" --run-dir "runs/transfer_refit_test" --model-dir "runs/transfer_refit/final"
```

This reproduces the archived source-only training recipes. It does not repeat the historical configuration search.

## Reference Results

Official within-subject test accuracy at **15 stimulus repetitions**, summarized across training seeds **42, 123, and 2026**:

| Model | Subject A | Subject B | Mean across subjects |
|---|---:|---:|---:|
| Backbone without attention | 95.33% | 98.33% | 96.83% |
| MSTA-CNN with full attention | 93.67% | 98.33% | 96.00% |

Each seed-specific model was evaluated on 100 test characters per subject. Predictions were not ensembled across seeds.

In the separate matched training-set cross-validation analysis, mean accuracy at 15 repetitions was **95.49% without attention** and **95.88% with full attention**. Cross-validation and official test results must be interpreted separately.

The results show subject-, repetition-, and metric-dependent effects of attention. They do not establish consistent superiority over the backbone.

## Evaluation Scope

- Character-level partitions keep all flashes belonging to a character within the same partition.
- Normalization statistics are derived from the applicable training partition.
- Final training durations and learning-rate schedules are derived from training-data validation histories.
- All three training seeds are retained. Their variation describes training variability on the same data, not uncertainty across an independent participant population.
- The official test sets were examined during earlier development. The reported results are a retrospective benchmark evaluation, not an untouched confirmatory test.
- Preprocessing includes noncausal operations. The implementation supports offline analysis; online communication performance and clinical utility have not been established.
- Validation on the original execution environment is documented in `VALIDATION.md`. Identical numerical behavior on every platform is not guaranteed.

## Repository Contents

| Path | Purpose |
|---|---|
| `run.py` | Unified command-line entry point |
| `msta.py` | Core preprocessing, architecture, training, and evaluation |
| `requirements.txt` | Recorded Python dependencies |
| `references/` | Primary and transfer reference models and protocols |
| `analysis/ablation/` | Matched dropout-0.25 attention ablation |
| `analysis/features/` | Held-out feature extraction and analysis |
| `analysis/supplementary/` | Binary metrics and paired character-error analysis |
| `paper_assets/` | Archived inputs and scripts for selected tables and figures |
| `results/` | Archived numerical summaries |
| `evidence/` | Supporting validation and complexity records |
| `METHODS.md` | Implementation and protocol documentation |
| `VALIDATION.md` | Completed checks and validation limits |
| `RUN_FA.md` | Detailed Persian execution guide |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT license |
| `RIGHTS.md` | Scope of licensing and third-party rights |

## Citation and License

If you use this implementation in research, please cite the software using `CITATION.cff` and cite the associated article once its publication details are available.

The source code and accompanying software documentation are released under the **MIT License**. Citation is a scholarly request, not an additional restriction on that license.

Original datasets, third-party dependencies, and the associated journal article retain their respective rights. See `LICENSE` and `RIGHTS.md`.

## Questions and Reproducibility Reports

Please use the repository's Issues page to report problems. Include the command used, software versions, error message, and relevant output summary so that the issue can be reproduced.
