# Reliability Envelope Reproducibility Repository

This repository supports the manuscript:

Operational-Profile-Aware Reliability Envelopes for Complex Systems under Multi-Source Uncertain Information

It contains code, configuration files, controlled synthetic inputs, processed numerical tables, and figure source data for reproducing the reported Case A/B/C analyses. It is intended for GitHub deposition and can later be linked from the manuscript Code Availability statement.

## Contents

- `simulation.py`: top-level entry point.
- `scripts/`: controlled Case A/B simulation, C-MAPSS Case C analysis, manuscript-number generation, and diagnostic checks.
- `config/`: numerical experiment configuration.
- `data/`: synthetic controlled-case parameter and evidence tables.
- `results/`: processed numerical tables used by the manuscript.
- `figure_source_data/`: copy of the CSV result tables used as source data for manuscript figures.

## Python environment

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

The package was prepared with Python 3 and the versions listed in `requirements.txt`.

## C-MAPSS data

The raw NASA C-MAPSS text files are not redistributed in this repository. Obtain the public dataset from the NASA Ames Prognostics Data Repository:

https://data.nasa.gov/dataset/Turbofan-engine-degradation-simulation-data-set/vrks-gjie

Place the following files in a local directory:

- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

By default, the scripts look for these files at `../public_data/CMAPSS` relative to the repository root. To use another location, set the environment variable `CMAPSS_DIR`.

PowerShell example:

```powershell
$env:CMAPSS_DIR = "D:\data\CMAPSS"
python simulation.py
```

## Reproduction steps

From the repository root:

```bash
python simulation.py
```

This regenerates the controlled Case A/B outputs, the C-MAPSS Case C summaries, the manuscript-number macros, and the manuscript figures.

## Expected checks

- Case A/B CSV files are regenerated under `results/`.
- C-MAPSS FD002 and FD004 each have six operating-condition clusters in `results/cmapss_dataset_summary.csv`.
- `results/cmapss_crossfit_leakage_audit.csv` reports zero train-evaluation overlap.
- Reliability envelope columns contain finite values and satisfy lower <= upper.
- `results/cmapss_epsilon_monotonicity_summary.csv` reports nondecreasing width as the admissible set expands.

The finite scenario-wise envelopes are exact only over the declared finite scenarios encoded in the scripts and configuration. They are not global robust bounds and do not represent deployed industrial certification.
