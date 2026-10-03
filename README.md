# sgd-magnet-tuner

[![Coverage Status](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/github/jgray-19/sgd-magnet-tuner/graph/badge.svg?token=Y1KZACDFPL)](https://codecov.io/github/jgray-19/sgd-magnet-tuner)

Tools for optimising accelerator magnet knob strengths using gradient-based
methods with MAD-NG. This README is short and focused — see the docs for
full details.

## Package overview

High-level modules (concise):

- `accelerators` / `config` — LHC, PSB and SPS optimiser accelerators and configuration dataclasses
- `training` — tracking fitters (`ArcByArcFitter`, `ACDMarkerFitter`, `KickerFitter`), data manager and worker orchestration
- `training_closed_twiss` — Levenberg–Marquardt closed-orbit / optics fitting
- `workers` / `mad` / `simulation` — MAD-NG tracking workers, interfaces and simulation helpers
- `measurements` / `noise` — measurement preparation shared by the PSB and LHC workflows (reconstruction, ACD marker rows, variances)
- `optimisers` — Adam / AMSGrad / L-BFGS implementations
- `analysis` / `dispersion` / `dataframes` / `io` — helpers and utilities

dp/p ↔ pt conversion comes from `pymadng_utils.physics` (or `accelerator.dp2pt`).
The LHC measurement workflows live in `lhc_measurements`; the PSB campaign
entry points live in `psb_md/scripts/optimisation/`.

Use the tests in `tests/training/` as compact examples of real workflows.

## Dependencies (external projects)

This project uses helper packages maintained in related repositories; install
them before running the end-to-end workflows:

- xtrack_tools: https://github.com/jgray-19/xtrack_tools
- tmom-recon:  https://github.com/jgray-19/tmom-recon
- pymadng-utils: https://github.com/jgray-19/pymadng-utils

Companion documentation for this stack is published under the same GitHub Pages
account with repository-name paths:

- sgd-magnet-tuner: https://jgray-19.github.io/sgd-magnet-tuner/
- pymadng-utils: https://jgray-19.github.io/pymadng-utils/
- tmom-recon: https://jgray-19.github.io/tmom-recon/
- xtrack_tools: https://jgray-19.github.io/xtrack_tools/

Install via pip from GitHub, for example::

```bash
pip install git+https://github.com/jgray-19/xtrack_tools.git
pip install git+https://github.com/jgray-19/tmom-recon.git
```

## Installation

Clone and install in editable mode::

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

For development (tests + docs):

```bash
pip install -e .[test,docs,tracking]
```

## Kicker mode

Kicker mode supports single-start tracking where the initial conditions come
from a kicker marker rather than a BPM. It enforces a single worker/track,
disables validation payloads, and only tracks forward (no sdir = -1).

Requirements:

- The input data must include the kicker marker name with x, px, y, py columns.
- The model sequence should include the kicker element so the sequence can be
	cycled to it.

Use `KickerFitter` with a `KickerConfig`:

```python
from aba_optimiser.training import KickerConfig, KickerFitter

fitter = KickerFitter(
    accelerator=accelerator,
    optimiser_config=optimiser_config,
    simulation_config=simulation_config,
    sequence_config=sequence_config,
    measurement_config=measurement_config,
    kicker_config=KickerConfig(kicker_name="KICKER.NAME", turns_after_kicker=1024),
)
knobs, uncertainties = fitter.run()
```

## Tests

Run tests with pytest::

```bash
pytest tests/
pytest tests/ --cov=aba_optimiser
```

## Docs

Build docs::

```bash
pip install -e .[docs]
cd docs && make html
```

View at `docs/_build/html/index.html`.
