User Guide
==========

Tracking fits
-------------

A tracking fit recovers knob strengths by tracking measured initial conditions
through the MAD-NG model and comparing the result with the measured turn-by-turn
data. A fit is defined by an accelerator and five configuration objects, and is run
with ``fitter.run()``, which returns the fitted knobs and their uncertainties.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Object
     - Role
   * - :class:`~aba_optimiser.machine.accelerators.Accelerator` subclass
     - Machine definition: sequence file, kinetic energy, BPM pattern and the knob
       families to fit (``errors``, ``misalignments``).
   * - :class:`~aba_optimiser.config.OptimiserConfig`
     - Optimiser type (``adam`` or ``lbfgs``), epoch count, learning-rate schedule and
       convergence criterion.
   * - :class:`~aba_optimiser.config.SimulationConfig`
     - Worker and batch counts, data and validation fractions, outlier screening.
   * - :class:`~aba_optimiser.tracking.SequenceConfig`
     - Magnet range exposed to MAD-NG and BPMs to exclude.
   * - :class:`~aba_optimiser.tracking.MeasurementConfig`
     - Measurement files, each with a :class:`~aba_optimiser.tracking.MeasurementDetails`
       (MAD interface options, momentum offset, first BPM).

Fitters
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Fitter
     - Initial conditions
     - Tracking
   * - :class:`~aba_optimiser.tracking.ArcByArcFitter`
     - BPM at the start of each range
     - Forward and backward over the ``bpm_start_points`` x ``bpm_end_points`` ranges.
       Set ``acd_excited=True`` for AC-dipole data.
   * - :class:`~aba_optimiser.tracking.ACDMarkerFitter`
     - AC-dipole ``before``/``after`` markers
     - Bidirectional; the whole ring is observed.
   * - :class:`~aba_optimiser.tracking.KickerFitter`
     - Kicker marker
     - One worker, forward only, ``turns_after_kicker`` turns.

``KickerFitter`` requires a :class:`~aba_optimiser.tracking.KickerConfig`. The
measurement data must contain ``x``, ``px``, ``y`` and ``py`` at the kicker marker, and
the sequence must contain the kicker element.

Example
~~~~~~~

.. code-block:: python

   from pathlib import Path

   from aba_optimiser.machine.accelerators import LHC
   from aba_optimiser.config import OptimiserConfig, SimulationConfig
   from aba_optimiser.tracking import (
       ArcByArcFitter,
       MeasurementConfig,
       MeasurementDetails,
       SequenceConfig,
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
   )
   knobs, uncertainties = fitter.run()

Optional fitter arguments
~~~~~~~~~~~~~~~~~~~~~~~~~

``initial_knob_strengths``
    Starting values, in optimisation space.
``true_strengths``
    Reference values (file or dictionary) used to report the error of the fit.
``optimise_knobs``
    Restrict the fit to a subset of global knob names.
``output_config``
    :class:`~aba_optimiser.tracking.OutputConfig`: TensorBoard logging, uncertainty
    estimation and log files.
``checkpoint_config``
    :class:`~aba_optimiser.tracking.CheckpointConfig`: periodic checkpoints and restart.
``initial_conditions_callback``
    Epoch-end hook that refreshes the workers' initial conditions.
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

:class:`~aba_optimiser.poco.ClosedTwissFitter` fits knobs so that the
periodic model optics match a measured closed twiss. Closed orbit, beta, phase and
dispersion are all obtained from one parametric MAD-NG ``twiss``, so they are fitted
simultaneously and no starting point is taken from the measurement. The solver is
Levenberg-Marquardt, configured by
:class:`~aba_optimiser.poco.LevenbergMarquardtConfig`.

``measurements`` maps each measurement's ``pt`` to a file or dataframe. Parameters are
weighted by the inverse measurement variance; ``use_errors=False`` normalises every
observable family identically instead. ``prior_strengths`` adds a Gaussian prior.

Closed-orbit fits
-----------------

:class:`~aba_optimiser.poco.ClosedOrbitFitter` fits knobs to one or
more :class:`~aba_optimiser.poco.ClosedOrbitSeries`.

``machine_state``
    MAD-X globals (quadrupole strengths, corrector kicks, tune knobs) at which a
    series was measured. They are fixed inputs and may differ between series.
    Varying the quadrupole strengths between series makes quadrupole misalignments
    observable. Accepted forms are a dictionary, a knobs file or a TFS corrector
    table; :func:`aba_optimiser.machine.mad.merge_machine_states` combines several. Setting
    ``machine_state`` on the fitter provides a default inherited by every series.
``absolute_planes``
    Planes fitted as absolute orbits. The remaining planes are fitted as the change
    from the fitter's own ``machine_state`` to the series' state.
``observables``
    Fitted quantities: ``x``, ``y``, and the phase advances ``mu1``, ``mu2``
    (``mu1``/``mu2`` are set per series).

Accepted iterations are stored in ``fitter.history``. Call ``fitter.close()`` to shut
down the MAD-NG process once the fit is complete.

``CalibratedClosedOrbitFitter`` extends this with BPM gains and corrector kick
calibrations as additional parameters, with Gaussian priors ``sigma_bpm`` and
``sigma_corrector``. Each series' ``machine_state`` must set exactly one ``k_<corrector>``
kick.

Degeneracy analysis
-------------------

:mod:`aba_optimiser.analysis.degeneracy_checker` evaluates the eigenspectrum of the
Gauss-Newton normal matrix at the initial knobs. Near-zero eigenvalues identify knob
combinations that the data cannot constrain, so the problem can be regularised or the
knob set reduced before running the fit. Use
:meth:`~aba_optimiser.tracking.TrackingFitter.check_degeneracy` on a tracking fitter.

Testing
-------

.. code-block:: bash

   pytest -m "not slow"     # fast suite
   pytest -m slow           # convergence and end-to-end tests
   pytest --cov=aba_optimiser

The markers are ``slow``, ``regression``, ``integration``, ``convergence``, ``e2e``,
``lhc``, ``sps``, ``psb`` and ``serial``. Tests marked ``serial`` must not run in
parallel. The tests under ``tests/training/`` double as worked examples.
