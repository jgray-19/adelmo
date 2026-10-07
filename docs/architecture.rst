Architecture
============

Process model
-------------

Each fitter runs in a main process and a set of worker processes. A worker is a
``multiprocessing.Process`` that owns one MAD-NG instance. Workers do the expensive
work (tracking or closed-orbit solves, and the derivatives with respect to the knobs);
the main process sums their replies and updates the knobs.

There are two fitting engines, built the same way:

- **Tracking** (``training.tracking``): ``TrackingFitter`` and its subclasses
  ``ArcByArcFitter``, ``KickerFitter`` and ``ACDMarkerFitter`` fit
  turn-by-turn data with mini-batch gradient descent (Adam or L-BFGS, ``training.sgd``).
- **POCO** (``poco``): ``ClosedOrbitFitter``, ``ClosedTwissFitter`` and
  ``CalibratedClosedOrbitFitter`` fit closed-orbit and closed-twiss data with
  Levenberg-Marquardt (``poco.lm_loop``).

Both compose a ``MachineSetup`` (``training.machine_setup``): the MAD-NG model, the
knobs and their starting values, the BPM ranges, the output settings and the one
``SimulationConfig`` every other part reads.

Worker protocol
---------------

Messages are typed dataclasses in ``workers.protocol``. Every worker runs the same
loop, ``AbstractWorker.run``:

1. Receive ``Start(knobs)`` and build the MAD-NG session.
2. Answer each ``Evaluate(knobs, batch)`` with one ``GradReply`` (loss, gradient and,
   for POCO workers, the Hessian and normal matrix), and each ``Command`` (``DIAGNOSE``,
   ``VALIDATE``, ``APPLY_MASK``, ``SET_KNOBS``, ``UPDATE_INIT_COORDS``,
   ``SET_UNCERTAINTY_MODE``) with one ``Ack`` or ``LossReply``.
3. On ``Stop``, a training tracking worker replies with its ``UncertaintyPart``, then exits.

A worker sends exactly one reply per request. A failure sends one ``ErrorReply``
instead, and the worker exits without sending anything else; the main process raises
it. A NaN loss marks a worker that lost its particles or closed orbit: it is left out
of the sums and of the averaged loss (``training.reduction.reduce_replies``).

Run lifecycle
-------------

``training.lifecycle.run_with_workers`` gives every fitter the same shape: run the
body; on Ctrl-C return the best result so far; always stop the workers and close the
TensorBoard writer. For a tracking fit the body is:

1. Load the data (``DataManager``), split the turns into training and held-out
   validation batches with a seeded shuffle, and clamp ``num_batches`` to the data.
2. Start a ``TrackingSession``: build the worker payloads, start the training and
   validation ``WorkerPool`` and screen outlier BPMs and workers.
3. Run ``SGDLoop``: per epoch, evaluate every batch and take an optimiser step after
   each; score the held-out validation loss; keep the knobs with the lowest loss
   (validation when available); stop on a converged loss or gradient.
4. Set the workers to the best knobs and stop them; their uncertainty parts give
   ``Cov = A⁻¹ B A⁻¹`` (``workers.common.sandwich_uncertainties``).

Each fit returns a ``FitResult``: knobs, 1-sigma uncertainties keyed by the same knob
names, and ``FitDiagnostics`` (why and after how many iterations it stopped).

Package layout
--------------

``aba_optimiser.accelerators``
    Machine definitions (``LHC``, ``PSB``, ``SPS``, ``FCC``): knob families, BPM
    pattern and tune configuration.
``aba_optimiser.mad``
    MAD-NG interfaces. ``GenericMadInterface`` builds the model;
    ``GradientDescentMadInterface`` adds the optimisation knobs and derivatives;
    ``machine_state`` handles fixed machine-state inputs.
``aba_optimiser.training``
    ``machine_setup``, ``pool``, ``lifecycle``, ``reduction`` and ``results`` are
    shared by both engines. ``training.tracking`` holds the tracking fitters, the
    worker session and data manager, and ``training.tracking.workers`` the payload
    construction, turn planning, outlier screening and uncertainty drain.
    ``training.sgd`` holds the SGD loop, learning-rate schedule and checkpointing.
``aba_optimiser.poco``
    Levenberg-Marquardt fitters for closed-orbit and closed-twiss data.
``aba_optimiser.workers``
    The worker protocol and base class, and the tracking, closed-twiss, closed-orbit
    and calibrated closed-orbit workers.
``aba_optimiser.optimisers``
    Adam, L-BFGS and Levenberg-Marquardt.
``aba_optimiser.measurements``, ``aba_optimiser.noise``
    Measurement preparation shared by the PSB and LHC workflows: reconstruction,
    ACD marker rows, BPM variances.
``aba_optimiser.analysis``, ``aba_optimiser.calibration``
    Degeneracy checks and BPM-gain and corrector calibration.

Knob conventions
----------------

Fitters work in a single *optimisation space*. Inputs (``initial_knob_strengths``,
``true_strengths``), internal algorithms and reported results all use the same knob
coordinates.

dp/p and pt are related by ``pymadng_utils.physics`` (or ``accelerator.dp2pt``).
Momentum offsets are carried through MAD-NG ``pt``.

Quadrupole tilts are fitted through ``<element>.tilt`` knobs, seeded at 1e-9 rad because
a zero-angle rotation does not carry a differentiable parameter in MAD-NG.
