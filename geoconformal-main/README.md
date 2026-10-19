# Geo-Causal Conformal Prediction (GCCP)

Research code for Geo-Causal Conformal Prediction (GCCP), a spatial conformal prediction framework developed from GeoSIMCP/geoconformal. GCCP introduces causal-information-guided similarity modeling into the calibration process to investigate whether causal information can improve spatial uncertainty quantification.

This repository is intended to support academic research, reproducible experiments, and future release of the code associated with the GCCP manuscript.

## Overview

Conformal prediction constructs prediction sets or intervals from calibration data and can provide distribution-free uncertainty statements under appropriate assumptions. For regression problems, conformal prediction commonly uses calibration nonconformity scores to determine the size of a prediction interval for a new observation.

Spatial prediction introduces additional challenges. Prediction errors may vary across geographic regions, observations may be spatially dependent, and the local data density or data-generating process may change across space. A single global calibration distribution may therefore provide uncertainty estimates that do not reflect local conditions.

Methods based on geographic distance address part of this problem by assigning greater importance to nearby calibration observations. However, geographic proximity alone may not capture similarity in the underlying prediction process. Two nearby observations can belong to different spatial regimes, while observations that are farther apart may exhibit similar process-level characteristics.

GCCP investigates whether adding causal-information-guided similarity to geographic information can improve the localization of conformal calibration. The goal is not to assume that every similarity measure has a universal causal interpretation, but to evaluate causal-information-based weighting under explicitly documented data, modeling, and experimental assumptions.

## Method Summary

The GCCP workflow consists of the following conceptual steps:

1. **Predictive model training**  
   A base predictive model is fitted using the training data. The base model may be selected according to the requirements of a particular experiment.

2. **Calibration score calculation**  
   Predictions on a calibration set are compared with observed responses to obtain nonconformity scores, such as absolute residuals.

3. **Geographic similarity modeling**  
   Geographic coordinates are used to characterize the spatial relationship between calibration observations and prediction locations.

4. **Causal-information-based similarity modeling**  
   Additional similarity information is constructed from causal or causally motivated analysis. The exact construction is experiment-specific and is documented with the corresponding experiment code.

5. **Weighted calibration distribution**  
   Geographic and causal-information-based similarities are used to assign weights to calibration scores. The resulting weighted empirical distribution is used to determine a local conformal threshold.

6. **Prediction interval evaluation**  
   Prediction intervals are evaluated using metrics such as empirical coverage, interval width, interval score, and other diagnostics defined by the experiment protocol.

The causal-information-based similarity used in an experiment should be interpreted according to that experiment's assumptions and estimation procedure. GCCP does not claim that a statistical similarity measure is universally causal or that causal information necessarily improves uncertainty quantification in every spatial setting.

## Repository Structure

The planned research repository structure is:

```text
GCCP/
├── geoconformal/
├── experiments/
│   ├── pm25/
│   ├── seattle/
│   └── simulation/
├── data/
├── results/
├── docs/
├── tests/
├── README.md
└── LICENSE
```

The `geoconformal/` directory contains the reusable conformal prediction implementation. The `experiments/` directory contains dataset-specific and simulation-specific workflows. Dataset documentation and, where permitted, data files are organized under `data/`. Selected generated tables and figures are organized under `results/`. Reproducibility notes, experiment protocols, and supplementary documentation are maintained under `docs/`.

## Installation

Create and activate a dedicated Python virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package and its dependencies:

```bash
python -m pip install --upgrade pip
pip install -e .
```

If an experiment-specific requirements file is provided, install it after the base package:

```bash
pip install -r requirements_geosimcp_repro.txt
```

A Conda environment configuration will be provided later. Exact dependency versions should be recorded before a formal paper release.

## Reproduction Workflow

The intended reproduction workflow is:

1. Install the Python dependencies and any experiment-specific R dependencies.
2. Obtain the required datasets and review their provenance and licensing conditions.
3. Run the documented data-preprocessing steps.
4. Generate or load the fixed training, calibration, validation, and test splits.
5. Run the baseline prediction and conformal prediction methods.
6. Run the geographic and causal-similarity analysis required by the experiment.
7. Run the GCCP calibration and prediction experiments.
8. Generate the documented result tables, diagnostics, and figures.

The exact commands, notebook execution order, random seeds, and configuration values are experiment-specific. They should be recorded in the corresponding experiment directory and in the reproducibility documentation before the final release.

## Experiments

### PM2.5

The PM2.5 experiments study spatial prediction and uncertainty quantification for China PM2.5 observations. The workflow includes dataset auditing, preprocessing, baseline prediction, geographic and causal-information analysis, GCCP calibration, and evaluation of prediction intervals.

The final experiment protocol, data source, preprocessing decisions, split definitions, and evaluation metrics will be documented with the PM2.5 experiment materials.

### Seattle Housing

The Seattle housing experiments study spatial prediction and uncertainty quantification for housing-related observations in the Seattle region. The workflow evaluates baseline methods and GCCP-style calibration under spatially structured train, calibration, validation, and test splits.

The experiment documentation will specify the sampled data, coordinate system, predictive features, split protocol, similarity construction, and evaluation procedure.

### Simulation

The simulation experiments evaluate GCCP under controlled data-generating settings. The intended simulations vary spatial structure, covariate or process heterogeneity, noise, sample size, and the relationship between geographic and causal-information-based similarity.

Simulation scenarios, parameter settings, random seeds, estimands, and evaluation metrics will be documented before the simulation results are treated as part of the final paper release.

## Reproducibility and Data Policy

Datasets used by this project may have different sources, licenses, and redistribution conditions. A dataset is included in the repository only when its licensing terms permit redistribution and the required attribution has been documented.

Restricted or non-redistributable datasets should be obtained separately by the user. The repository should provide the original source, download instructions, expected file names, preprocessing information, and checksums where possible.

Results should be traceable to the code, data version, split definition, random seed, and configuration that produced them. Generated results should be clearly distinguished from source data and reusable implementation code. Intermediate outputs should not be treated as final scientific results unless they are explicitly identified and documented.

## Citation

Citation information will be added after the GCCP manuscript has been completed.

## License

This repository is released under the MIT License.

Third-party datasets, software, figures, and other external materials may have separate licenses and usage restrictions. Those terms remain applicable to the corresponding materials.
