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
- `training_closed_twiss` — Levenberg–Marquardt closed-orbit / optics fitting (`ClosedOrbitFitter`, `ClosedTwissFitter`)
- `workers` / `mad` — MAD-NG tracking and closed-orbit workers and interfaces
- `measurements` / `noise` — measurement preparation shared by the PSB and LHC workflows (reconstruction, ACD marker rows, variances)
- `optimisers` — Adam / AMSGrad / L-BFGS implementations
- `analysis` / `calibration` — degeneracy checks and BPM-gain / corrector calibration

dp/p ↔ pt conversion comes from `pymadng_utils.physics` (or `accelerator.dp2pt`).
The LHC measurement workflows live in `lhc_measurements`; the PSB campaign
entry points live in `psb_md/scripts/optimisation/`.

Use the tests in `tests/training/` as compact examples of real workflows.

### Closed-orbit fits at different machine states

A `ClosedOrbitSeries` can carry a `machine_state`: MAD-X globals (quadrupole
strengths such as `kbrqf`, corrector kicks such as `kbr3dhz2l4`, tune knobs; a dict or a
knobs file) at which its
orbits were measured. They are known inputs, not fitted, and differ freely between
series, so quadrupole beam-based alignment (fitting `dx`/`dy`) works by changing the
quadrupole strengths between measurements. Series are fitted either as absolute
orbits (`absolute_planes=("x", "y")`) or as changes from the fitter's own `machine_state`
(the reference) to the series' state, e.g. a changed corrector kick, in any mix. `ClosedOrbitFitter(machine_state=...)` sets a
default every series inherits; it replaces the old `tune_knobs`/`corrector_knobs` arguments.
`machine_state` is the one name for this everywhere (`GenericMadInterface`, `MeasurementDetails.interface_options`,
the fitters): a dict, a knobs file, or a TFS corrector table; `aba_optimiser.mad.merge_machine_states` combines several.
The fitter records its accepted iterations in `fitter.history` and `fitter.close()` shuts down its MAD process. See
`tests/training/test_closed_orbit_machine_state.py`.

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
