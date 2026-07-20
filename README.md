# Gillespie Repressilator and Stochastic Gene Expression

This project compares deterministic and stochastic descriptions of gene-regulatory dynamics.
It contains two related models:

1. A synthetic **repressilator**, simulated with ordinary differential equations and the Gillespie stochastic simulation algorithm.
2. A simple **mRNA–protein birth–death model**, simulated over repeated stochastic trajectories and compared with theoretical steady-state behavior.

The original notebook was reorganized into reusable Python modules and a clean analysis notebook. The scientific content was retained while duplicated code, deprecated NumPy types, inconsistent variables, and several reaction-definition errors were corrected.

## Main objectives

- Simulate the repressilator using deterministic ODEs.
- Simulate the same regulatory circuit using Gillespie SSA.
- Compare smooth deterministic trajectories with noisy stochastic trajectories.
- Measure oscillation period and amplitude.
- Explore the effects of transcription, repression, degradation, and Hill-coefficient parameters.
- Visualize parameter sweeps with heatmaps and three-dimensional surfaces.
- Simulate a simple stochastic gene-expression model over many trajectories.
- Compare mRNA copy-number distributions with a Poisson reference.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_analysis.py
├── notebooks/
│   └── gillespie_repressilator_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── repressilator.py
│   └── gene_expression.py
├── figures/
├── results/
└── legacy/
    └── Gillespie_original.ipynb
```

## Installation

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the analysis

Run the complete script:

```bash
python run_analysis.py
```

Or open the notebook:

```bash
jupyter notebook notebooks/gillespie_repressilator_analysis.ipynb
```

Generated plots are written to `figures/`, while numerical summaries are written to `results/`.

## Model notes

### Deterministic repressilator

The deterministic model contains three mRNA species and three proteins. Each protein represses transcription of the next gene in a cyclic network. Translation and degradation are represented with continuous rate equations.

### Stochastic repressilator

The stochastic model uses discrete reaction events for transcription, mRNA degradation, translation, and protein degradation. Waiting times and reaction identities are sampled according to Gillespie SSA.

### Simple gene-expression model

The second model contains four reactions:

- transcription;
- mRNA degradation;
- translation;
- protein degradation.

Repeated trajectories demonstrate that intrinsic noise can produce broad protein distributions even when the average trend is stable.

## Important corrections from the original notebook

- Added missing translation reactions to the stochastic repressilator.
- Corrected degradation propensities so they depend on the corresponding molecular counts.
- Prevented negative molecular populations.
- Used consistent mRNA and protein columns in deterministic/stochastic comparisons.
- Fixed parameter sweeps so `beta` and `n` are actually varied.
- Replaced deprecated `np.int` with built-in integer types.
- Removed duplicated oscillation-analysis functions and repeated sweep blocks.
- Added random-number generators for reproducible simulations.
- Added output saving and reusable functions.

## Authors

Ania Rotondi and Giulia Marchesani
