Architecture
============

Process model
-------------

Each fitter runs in a main process and a set of worker processes. A worker is a
``multiprocessing.Process`` that owns one MAD-NG instance. Workers do the expensive
work (tracking or closed-orbit solves, and the derivatives with respect to the knobs);
the main process aggregates their results and updates the knobs.

Per epoch:

1. The main process sends the current knob values to every worker.
2. Each worker evaluates its loss and gradient (and, for the Gauss-Newton fitters, the
   Hessian contribution) against its share of the data and replies.
3. The main process sums the results and the optimiser takes a step.
4. Held-out validation workers report a loss which is used for early stopping.

After the final epoch the workers return the Hessian, from which the parameter
uncertainties are obtained.

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
    Tracking fitters, configuration, data management and the worker-management layer
    (``training.workers``: worker pool, payload construction, turn planning and
    outlier screening) and the optimisation loop (``training.optimisation``).
``aba_optimiser.poco``
    Gauss-Newton fitters for closed-orbit and closed-twiss data.
``aba_optimiser.workers``
    Worker process implementations: tracking, validation tracking, closed twiss,
    closed orbit and calibrated closed orbit.
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
