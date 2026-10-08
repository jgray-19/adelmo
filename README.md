# adelmo

[![Coverage Status](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/github/jgray-19/sgd-magnet-tuner/graph/badge.svg?token=Y1KZACDFPL)](https://codecov.io/github/jgray-19/sgd-magnet-tuner)

Estimation of accelerator magnet errors (strengths, misalignments and tilts) from
beam measurements, using gradient-based optimisation of MAD-NG models.

Two families of fit are provided:

| Fit | Data | Entry points |
|---|---|---|
| Tracking | Turn-by-turn BPM data, tracked through the model | `ArcByArcFitter`, `ACDMarkerFitter`, `KickerFitter` |
| Closed twiss | Closed orbit, phase advance, beta and dispersion | `ClosedOrbitFitter`, `ClosedTwissFitter` |

Supported machines: LHC, PSB, SPS and FCC.

Full documentation: <https://jgray-19.github.io/sgd-magnet-tuner/>

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

Optional dependency groups:

| Extra | Contents |
|---|---|
| `tracking` | omc3, cpymad, pyarrow, psutil (required by the tracking workflows) |
| `measurements` | tmom-recon (momentum reconstruction) |
| `test` | pytest, pytest-cov, pytest-xdist, xtrack-tools |
| `docs` | Sphinx and theme |
| `dev` | ruff, pre-commit |

```bash
pip install -e ".[test,docs,tracking]"
```

### Companion packages

Several workflows depend on companion packages installed from GitHub:

| Package | Purpose |
|---|---|
| [pymadng-utils](https://github.com/jgray-19/pymadng-utils) | Shared accelerator abstractions, dp/p and pt conversion, knob-file I/O |
| [tmom-recon](https://github.com/jgray-19/tmom-recon) | Transverse momentum and optics reconstruction |
| [xtrack_tools](https://github.com/jgray-19/xtrack_tools) | Tracking helpers and dataframe conversion (tests) |

```bash
pip install git+https://github.com/jgray-19/pymadng-utils.git
pip install git+https://github.com/jgray-19/tmom-recon.git
pip install git+https://github.com/jgray-19/xtrack_tools.git
```

## Usage

A tracking fit is configured with an accelerator and four configuration objects,
then run with `fitter.run()`. It returns a `FitResult`: the fitted knob values,
their 1-sigma uncertainties under the same knob names, and `diagnostics` saying why
and after how many epochs the fit stopped. Optional settings (starting knobs, true
strengths for diagnostics, output and checkpoint settings, callbacks) go in a
`FitterOptions`.

```python
from pathlib import Path

from adelmo.config import OptimiserConfig, SimulationConfig
from adelmo.fitting.config import OutputConfig, SequenceConfig
from adelmo.machine.accelerators import LHC
from adelmo.tracking import (
    ArcByArcFitter,
    FitterOptions,
    MeasurementConfig,
    MeasurementDetails,
)

accelerator = LHC(beam=1, sequence_file="lhcb1.seq", errors={"quad": {"k1"}})

fitter = ArcByArcFitter(
    accelerator=accelerator,
    optimiser_config=OptimiserConfig(
        max_epochs=200,
        warmup_epochs=10,
        warmup_lr_start=1e-6,
        max_lr=1e-4,
        min_lr=1e-6,
        gradient_converged_value=1e-12,
    ),
    simulation_config=SimulationConfig(num_workers=8, num_batches=4),
    sequence_config=SequenceConfig(magnet_range="$start/$end"),
    measurement_config=MeasurementConfig({Path("measurement.parquet"): MeasurementDetails()}),
    bpm_start_points=["BPM.12R1.B1"],
    bpm_end_points=["BPM.20R1.B1"],
    options=FitterOptions(output_config=OutputConfig(write_tensorboard_logs=False)),
)
result = fitter.run()
print(result.knobs, result.uncertainties, result.diagnostics.reason)
```

Further examples are in `tests/training/`, which exercise each fitter end to end.

### Tracking modes

| Class | Initial conditions | Tracking |
|---|---|---|
| `ArcByArcFitter` | BPM at the start of each range | Forward and backward over the configured BPM ranges |
| `ACDMarkerFitter` | AC-dipole `before`/`after` markers | Bidirectional, whole ring observed |
| `KickerFitter` | Kicker marker | Single worker, forward only, `turns_after_kicker` turns |

`KickerFitter` additionally takes a `KickerConfig(kicker_name, turns_after_kicker)`.
The measurement data must contain `x`, `px`, `y`, `py` at the kicker marker, and the
sequence must include the kicker element.

### Closed-twiss fits

The closed-twiss and closed-orbit fitters live in `adelmo.poco` (Parametric
Optimisation of Closed Orbits) and return the same `FitResult` as the tracking fitters.

`ClosedTwissFitter` fits knobs so that the model's periodic optics match measured
closed orbit, beta, phase and dispersion simultaneously, using a single parametric
MAD-NG `twiss`. It takes `measurements`, a mapping from the measurement momentum `pt` to a
measurement file or dataframe, and uses a Levenberg-Marquardt solver configured by
`LevenbergMarquardtConfig`.

### Closed-orbit fits

`ClosedOrbitFitter` fits knobs to one or more `ClosedOrbitSeries`. Each series
carries a `machine_state`: the MAD-X globals (quadrupole strengths, corrector kicks,
tune knobs) at which its orbits were measured. These are fixed inputs and may differ
between series. Varying the quadrupole strengths between series makes quadrupole
misalignments observable.

A series is fitted either as an absolute orbit (`absolute_planes`) or as the change
from the fitter's own `machine_state` to the series' state. `machine_state` accepts a
dictionary, a knobs file or a TFS corrector table; `adelmo.machine.mad.merge_machine_states`
combines several. Accepted iterations are recorded in `fitter.history`, and
`fitter.close()` shuts down the MAD-NG process.

See `tests/training/test_closed_orbit_machine_state.py`.

## Tests

```bash
pytest -m "not slow"          # fast suite
pytest -m slow                # convergence and end-to-end tests
pytest --cov=adelmo
```

Markers are listed in `pyproject.toml` and `tests/README.md`.

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
```

The output is written to `docs/_build/html/index.html`.

## Related repositories

The measurement and campaign workflows built on this package are maintained
separately: `lhc_measurements`, `psb_md`, `psb_loco`, `lhc_loco` and `fcc_loco`.
