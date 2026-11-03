Architecture
============

Process model
-------------

Each fitter runs in a main process and a set of worker processes. A worker is a
``multiprocessing.Process`` that owns one MAD-NG instance. Workers do the expensive
work (tracking or closed-orbit solves, and the derivatives with respect to the knobs);
the main process sums their replies and updates the knobs.

There are two fitting engines, built the same way:

- **Tracking** (``tracking``): ``TrackingFitter`` and its subclasses
  ``ArcByArcFitter``, ``KickerFitter`` and ``ACDMarkerFitter`` fit
  turn-by-turn data with mini-batch gradient descent (Adam or L-BFGS, ``tracking.sgd``).
- **POCO** (``poco``): ``ClosedOrbitFitter``, ``ClosedTwissFitter`` and
  ``CalibratedClosedOrbitFitter`` fit closed-orbit and closed-twiss data with
  Levenberg-Marquardt (``poco.lm_loop``).

Both compose a :class:`~adelmo.fitting.setup.MachineSetup`: the MAD-NG model and its BPMs,
the knobs and their starting values, the output settings and the one
``SimulationConfig`` every other part reads. A tracking fit adds its BPM ranges
(``tracking.ranges``).

Worker protocol
---------------

Messages are typed dataclasses in :mod:`adelmo.fitting.protocol`. Every worker runs
the same loop, :meth:`AbstractWorker.run <adelmo.fitting.worker.AbstractWorker.run>`:

1. Receive ``Start(knobs)`` and build the MAD-NG session.
2. Answer each ``Evaluate(knobs, batch)`` with one ``GradReply`` (loss, gradient and,
   for POCO workers, the Hessian and normal matrix), and each ``Command`` (``DIAGNOSE``,
   ``VALIDATE``, ``APPLY_MASK``, ``SET_KNOBS``, ``UPDATE_INIT_COORDS``,
   ``SET_UNCERTAINTY_MODE``) with one ``Ack`` or ``LossReply``.
3. On ``Stop``, a training tracking worker replies with its ``UncertaintyPart``, then exits.

A worker sends exactly one reply per request. A failure sends one ``ErrorReply``
instead, and the worker exits without sending anything else; the main process raises
it. A NaN loss marks a worker that lost its particles or closed orbit: it is left out
of the sums and of the averaged loss
(:func:`~adelmo.fitting.reduction.reduce_replies`).

Run logs and monitoring
-----------------------

Every fit writes a run log (:mod:`adelmo.runlog`) under
``OutputConfig.run_log_root``, one directory per fit:

``stage.json``
    Name, start time, knob names and ``OutputConfig.run_metadata`` (how to rebuild
    the model). Written once.
``history.jsonl``
    One JSON object per step: ``step``, ``time`` and the logged scalars.
``knobs.f64``
    Raw float64 rows ``[step, knob_0, knob_1, ...]``; step -1 holds the starting knobs.
``result.json``
    How the fit ended. Written when it finishes.

Optics are not stored. :mod:`adelmo.monitor` recomputes them from the knobs on demand,
caches them, and serves live plots of a running or finished fit
(``python -m adelmo.monitor RUN_LOG_ROOT --spec module:SPEC``). The machine side
supplies an :class:`~adelmo.monitor.source.OpticsSource` (how to build the model and
twiss it) and any machine-specific panels, bundled in a
:class:`~adelmo.monitor.source.MonitorSpec`.

Run lifecycle
-------------

:func:`~adelmo.fitting.lifecycle.run_with_workers` gives every fitter the same shape: run the
body; on Ctrl-C return the best result so far; always stop the workers and close the
run log. For a tracking fit the body is:

1. Load the data (:class:`~adelmo.tracking.data_manager.DataManager`), split the turns into training and held-out
   validation batches with a seeded shuffle, and clamp ``num_batches`` to the data.
2. Start a :class:`~adelmo.tracking.session.TrackingSession`: build the worker
   payloads, start the training and validation
   :class:`~adelmo.fitting.pool.WorkerPool` and screen outlier BPMs and workers.
3. Run :class:`~adelmo.tracking.sgd.loop.SGDLoop`: per epoch, evaluate every batch and take an optimiser step after
   each; score the held-out validation loss; keep the knobs with the lowest loss
   (validation when available); stop on a converged loss or gradient.
4. Set the workers to the best knobs and stop them; their uncertainty parts give
   ``Cov = A⁻¹ B A⁻¹`` (:func:`~adelmo.fitting.uncertainty.sandwich_uncertainties`).

Each fit returns a :class:`~adelmo.fitting.results.FitResult`: knobs, 1-sigma
uncertainties keyed by the same knob names, and
:class:`~adelmo.fitting.results.FitDiagnostics` (why and after how many iterations it stopped).

Package layout
--------------

Imports only point down this list; the two engines never import each other
(``tests/test_package_layers.py`` checks it).

``adelmo.monitor``
    Live plots served from run logs; optics recomputed from the logged knobs. Imports
    only ``runlog``; the machine side plugs in through ``OpticsSource``.

``adelmo.momentum_reference``
    Momentum reference from blank measurements; uses both engines.
``adelmo.tracking``
    The tracking engine: fitters, ``TrackingSession``, BPM ranges, ``DataManager``,
    the tracking worker and its uncertainty parts. ``tracking.dispatch`` builds the
    worker payloads, plans turns and screens outliers; ``tracking.sgd`` holds the SGD
    loop, learning-rate schedule and checkpointing; ``tracking.config`` the
    measurement, kicker and checkpoint settings and the tracking plans.
``adelmo.poco``
    The POCO engine: Levenberg-Marquardt fitters for closed-orbit and closed-twiss
    data, priors, BPM-gain and corrector calibration, and their workers
    (``poco.workers``).
``adelmo.fitting``
    What both engines share: ``MachineSetup``, ``SequenceConfig`` and
    ``OutputConfig``, the worker protocol and base worker, the shared-memory
    reference, the worker pool, run lifecycle, reply reduction, loss weights,
    uncertainties and ``FitResult``.
``adelmo.optimisers``
    Adam, L-BFGS and Levenberg-Marquardt.
``adelmo.machine``
    ``machine.accelerators``: machine definitions (``LHC``, ``PSB``, ``SPS``,
    ``FCC``), knob families, BPM pattern and tune configuration. ``machine.mad``: the
    MAD-NG interfaces; ``GenericMadInterface`` builds the model,
    ``GradientDescentMadInterface`` adds the optimisation knobs and derivatives, and
    ``machine_state`` handles fixed machine-state inputs.
``adelmo.measurements``
    Measurement preparation shared by the PSB and LHC workflows: reconstruction,
    ACD marker rows, BPM noise and variances.
``adelmo.analysis``
    Degeneracy checks.
``adelmo.runlog``
    Run-log writer (:class:`~adelmo.runlog.writer.RunLog`) and reader
    (:class:`~adelmo.runlog.reader.StageLog`, :func:`~adelmo.runlog.reader.find_stages`).
``adelmo.config``
    ``OptimiserConfig`` and ``SimulationConfig``. Imports no other ``adelmo`` module.

Knob conventions
----------------

Fitters work in a single *optimisation space*. Inputs (``initial_knob_strengths``,
``true_strengths``), internal algorithms and reported results all use the same knob
coordinates.

dp/p and pt are related by ``pymadng_utils.physics`` (or ``accelerator.dp2pt``).
Momentum offsets are carried through MAD-NG ``pt``.

Quadrupole tilts are fitted through ``<element>.tilt`` knobs, seeded at 1e-9 rad because
a zero-angle rotation does not carry a differentiable parameter in MAD-NG.
