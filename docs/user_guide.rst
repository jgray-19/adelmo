User Guide
==========

Tracking fits
-------------

A tracking fit recovers knob strengths by tracking measured initial conditions
through the MAD-NG model and comparing the result with the measured turn-by-turn
data. A fit is defined by an accelerator, four configuration objects and an optional
:class:`~adelmo.tracking.fitter.FitterOptions`. ``fitter.run()`` returns a
:class:`~adelmo.fitting.results.FitResult`: the fitted knobs, their 1-sigma
uncertainties under the same knob names, and ``diagnostics`` (why and after how many
epochs the fit stopped).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Object
     - Role
   * - :class:`~adelmo.machine.accelerators.base.Accelerator` subclass
     - Machine definition: sequence file, kinetic energy, BPM pattern and the knob
       families to fit (``errors``, ``misalignments``).
   * - :class:`~adelmo.config.OptimiserConfig`
     - Optimiser type (``adam`` or ``lbfgs``), epoch count, learning-rate schedule and
       convergence criterion.
   * - :class:`~adelmo.config.SimulationConfig`
     - Worker and batch counts, data and validation fractions, outlier screening.
   * - :class:`~adelmo.fitting.config.SequenceConfig`
     - Magnet range exposed to MAD-NG and BPMs to exclude.
   * - :class:`~adelmo.tracking.config.models.MeasurementConfig`
     - Measurement files, each with a :class:`~adelmo.tracking.config.models.MeasurementDetails`
       (MAD interface options, momentum offset, first BPM).
   * - :class:`~adelmo.tracking.fitter.FitterOptions` (optional)
     - Starting knobs, true strengths, output and checkpoint settings, callbacks.

Fitters
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Fitter
     - Initial conditions
     - Tracking
   * - :class:`~adelmo.tracking.fitter.ArcByArcFitter`
     - BPM at the start of each range
     - Forward and backward over the ``bpm_start_points`` x ``bpm_end_points`` ranges.
       Set ``acd_excited=True`` for AC-dipole data.
   * - :class:`~adelmo.tracking.fitter.ACDMarkerFitter`
     - AC-dipole ``before``/``after`` markers
     - Bidirectional; the whole ring is observed.
   * - :class:`~adelmo.tracking.fitter.KickerFitter`
     - Kicker marker
     - One worker, forward only, ``turns_after_kicker`` turns.

``KickerFitter`` also takes a
:class:`KickerConfig(kicker_name, turns_after_kicker) <adelmo.tracking.config.models.KickerConfig>`.
The measurement data must contain ``x``, ``px``, ``y`` and ``py`` at the kicker
marker, and the sequence must contain the kicker element.

Example
~~~~~~~

.. code-block:: python

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
       measurement_config=MeasurementConfig(
           {Path("measurement.parquet"): MeasurementDetails()}
       ),
       bpm_start_points=["BPM.12R1.B1"],
       bpm_end_points=["BPM.20R1.B1"],
       options=FitterOptions(output_config=OutputConfig(write_run_log=False)),
   )
   result = fitter.run()
   print(result.knobs, result.uncertainties, result.diagnostics.reason)

FitterOptions
~~~~~~~~~~~~~

``initial_knob_strengths``
    Starting values, in optimisation space. Knobs left out keep the model value.
``true_strengths``
    Reference values (file or dictionary) used to report the error of the fit.
``debug``
    Run MAD-NG in debug mode.
``output_config``
    :class:`~adelmo.fitting.config.OutputConfig`: run log, TensorBoard logging,
    uncertainty estimation and log files.
``checkpoint_config``
    :class:`~adelmo.tracking.config.models.CheckpointConfig`: periodic checkpoints and restart.
``initial_conditions_callback``
    Epoch-end hook that returns refreshed start coordinates for every worker, or
    ``None`` to keep them.
``loss_callback``
    Called each epoch with the loss values.

Validation and screening
~~~~~~~~~~~~~~~~~~~~~~~~

``SimulationConfig.validation_fraction`` (default 0.1) holds out a disjoint set of turns
per file. These are never used for training, so the validation loss is out-of-sample.
``data_fraction`` is applied to the remaining training turns.

With ``enable_preloop_outlier_screening`` (default on), the workers are evaluated at
the initial knobs before optimisation. BPMs and workers whose loss z-score exceeds
``bpm_loss_outlier_sigma`` or ``worker_loss_outlier_sigma`` are masked. The same mask
is applied to the validation workers.

Closed-twiss fits
-----------------

The closed-twiss and closed-orbit fitters live in :mod:`adelmo.poco` and return the
same :class:`~adelmo.fitting.results.FitResult` as the tracking fitters.

:class:`~adelmo.poco.fitter.ClosedTwissFitter` fits knobs so that the
periodic model optics match a measured closed twiss. Closed orbit, beta, phase and
dispersion are all obtained from one parametric MAD-NG ``twiss``, so they are fitted
simultaneously and no starting point is taken from the measurement. The solver is
Levenberg-Marquardt, configured by
:class:`~adelmo.optimisers.levenberg_marquardt.LevenbergMarquardtConfig`.

``measurements`` maps each measurement's ``pt`` to a file or dataframe. Parameters are
weighted by the inverse measurement variance; ``use_errors=False`` normalises every
observable family identically instead. ``prior_strengths`` adds a Gaussian prior.

Closed-orbit fits
-----------------

:class:`~adelmo.poco.closed_orbit.ClosedOrbitFitter` fits knobs to one or
more :class:`~adelmo.poco.closed_orbit.ClosedOrbitSeries`.

``machine_state``
    MAD-X globals (quadrupole strengths, corrector kicks, tune knobs) at which a
    series was measured. They are fixed inputs and may differ between series.
    Varying the quadrupole strengths between series makes quadrupole misalignments
    observable. Accepted forms are a dictionary, a knobs file or a TFS corrector
    table; :func:`adelmo.machine.mad.machine_state.merge_machine_states` combines several. Setting
    ``machine_state`` on the fitter provides a default inherited by every series.
``absolute_planes``
    Planes fitted as absolute orbits. The remaining planes are fitted as the change
    from the fitter's own ``machine_state`` to the series' state: the fitter subtracts
    each measurement's ``reference`` orbit (errors added in quadrature), which is
    required for those planes. Measured orbits are always passed in absolute.
``observables``
    Fitted quantities: ``x``, ``y``, and the phase advances ``mu1``, ``mu2``
    (``mu1``/``mu2`` are set per series).

Accepted iterations are stored in ``fitter.history``. Call ``fitter.close()`` to shut
down the MAD-NG process once the fit is complete.

:class:`~adelmo.poco.calibrated.CalibratedClosedOrbitFitter` extends this with BPM gains and corrector kick
calibrations as additional parameters, with Gaussian priors ``sigma_bpm`` and
``sigma_corrector``. Every ``k_<corrector>`` kick a series' ``machine_state`` sets gets a gain; a
series may set several or none (an off-momentum orbit). ``fit_corrector_gains=False`` fits the BPM
gains alone, which is the mode for absolute orbits.

Degeneracy analysis
-------------------

:mod:`adelmo.analysis.degeneracy_checker` evaluates the eigenspectrum of the
Gauss-Newton normal matrix at the initial knobs. Near-zero eigenvalues identify knob
combinations that the data cannot constrain, so the problem can be regularised or the
knob set reduced before running the fit. Use
:meth:`~adelmo.tracking.fitter.TrackingFitter.check_degeneracy` on a tracking fitter.

Monitoring a fit
----------------

Every fit writes a run log under ``OutputConfig.run_log_root`` (switch off with
``OutputConfig(write_run_log=False)``). Serve live plots of running and finished fits
with:

.. code-block:: bash

   pip install -e ".[monitor]"
   python -m adelmo.monitor RUN_LOG_ROOT --spec my_package.monitor:SPEC

``adelmo-monitor`` is the same command as ``python -m adelmo.monitor``. Then open
``http://localhost:5006``. On a remote machine, forward the port with
``ssh -L 5006:localhost:5006 host``. Without ``--spec`` only the generic panels are
shown. See :mod:`adelmo.monitor` and :mod:`adelmo.runlog`.

Testing
-------

.. code-block:: bash

   pytest -m "not slow"     # fast suite
   pytest -m slow           # convergence and end-to-end tests
   pytest --cov=adelmo

Markers are listed in ``tests/README.md``. Tests marked ``serial`` must not run in
parallel (do not combine with ``-n``). The tests under ``tests/training/`` double as
worked examples.
