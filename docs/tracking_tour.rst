Tracking Fit Tour
=================

A tour of :class:`~adelmo.tracking.fitter.ArcByArcFitter` and
:class:`~adelmo.tracking.fitter.ACDMarkerFitter`: what each part of the code does, in the
order it runs. Underlined words are defined in the :ref:`tour-glossary` at the bottom
of the page.

The idea
--------

1. **Measure.** :term:`BPMs <BPM>` record the beam position (x, y) every
   :term:`turn`.
2. **Simulate.** :term:`MAD-NG` tracks each measured turn through the model. It
   returns the position at each BPM and its derivative with respect to each
   :term:`knob`.
3. **Fit.** Gradient descent changes the knobs until the model positions match the
   measured positions.

Package map
-----------

``adelmo.config``
    ``OptimiserConfig`` and ``SimulationConfig``.

``adelmo.tracking``
    The tracking fitters.

    * ``fitter.py``: the fitter classes
    * ``config/tracking.py``: the :term:`plan` for each mode
    * ``data_manager.py``: reads the measurement files
    * ``session.py``, ``dispatch/``: start the :term:`workers <worker>`
    * ``worker.py``: the tracking worker
    * ``sgd/loop.py``: the optimisation loop

``adelmo.fitting``
    Parts shared with the closed-orbit fitters.

    * ``setup.py``: model, BPM list, starting knobs
    * ``worker.py``: base worker
    * ``protocol.py``: messages between processes
    * ``reduction.py``: sums the worker results

``adelmo.machine``
    Accelerator models.

    * ``accelerators/``: LHC, SPS, PSB, FCC
    * ``mad/``: MAD-NG setup and tracking scripts

``adelmo.optimisers``
    Update rules: ``adam.py``, ``lbfgs.py``.

``adelmo.measurements``
    Preprocessing.

    * ``reconstruction.py``: raw data → fitter input
    * ``acd_pipeline.py``: optics analysis for :term:`ACD <AC dipole>` data

Not used by the tracking fitters: ``poco``, ``analysis``, ``runlog``, ``monitor``.

.. code-block:: text

   tracking ──▶ fitting ──▶ machine
      │
      └─▶ optimisers

.. _tour-preprocessing:

Preprocessing
-------------

Turns raw turn-by-turn BPM data into the :term:`parquet` file the fitter reads.
Entry point: :func:`~adelmo.measurements.reconstruction.process_single_dataframe`,
called once per :term:`bunch`.

.. code-block:: text

   raw BPM data (name, turn, x, y)
      ──▶ trim ──▶ variances ──▶ SVD clean ──▶ reconstruct px, py ──▶ parquet

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Step
     - Code
     - What it does
   * - ACD only: optics analysis
     - ``measurements/acd_pipeline.py``
     - omc3 (Harpy) measures the optics from the data and writes them to an analysis
       folder.
   * - Trim to the :term:`kick`
     - ``measurements/preprocessing.py``
     - Optional. Drops the turns before the kick.
   * - Assign variances
     - ``measurements/variances.py``
     - Adds ``var_x``, ``var_y`` columns: one value for every BPM, or a per-BPM noise
       table. Bad BPMs get zero weight.
   * - SVD clean
     - ``tmom_recon.svd``
     - Removes noise from x and y with an SVD.
   * - Reconstruct px, py
     - ``tmom_recon.calculate_pz``
     - Computes the momenta at each BPM from the positions and the measured optics.
   * - ACD only: add marker rows
     - ``measurements/reconstruction.py``
     - Adds rows for ``<acd>_before`` and ``<acd>_after``.

Output columns: ``name, turn, bunch_number, x, px, y, py, var_x, var_y, var_px,
var_py``.

Arc-by-arc vs ACD marker
------------------------

Both fitters run the same code. They differ in which parts of the ring are tracked.

.. raw:: html

   <div class="tour-rings"><div>

.. raw:: html
   :file: _static/tour/ring_arc_by_arc.svg

.. raw:: html

   </div><div>

.. raw:: html
   :file: _static/tour/ring_acd_marker.svg

.. raw:: html

   </div></div>

**Left, ArcByArcFitter:** tracks every start BPM to every end BPM, forward (teal) and
backward (orange), for one turn. With ``acd_excited=True``, a range containing the
:term:`AC dipole` is tracked round the other side of the ring (dashed).

**Right, ACDMarkerFitter:** two tracks round the whole ring: forward from
``<acd>_after``, backward from ``<acd>_before``.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * -
     - ArcByArcFitter
     - ACDMarkerFitter
   * - Setup function
     - ``arc_by_arc_setup``
     - ``acd_marker_setup``
   * - :term:`Plan <plan>` class
     - ``TrackingPlan`` or ``ACDArcByArcTrackingPlan``
     - ``ACDTrackingPlan``
   * - Tracks per :term:`batch`
     - starts × ends × 2
     - 2
   * - Start point
     - first BPM of the range
     - ACD marker
   * - BPMs compared
     - BPMs inside the range
     - all BPMs

Both setup functions live in ``tracking/config/tracking.py``.

Trace one fit
-------------

From ``fitter = ArcByArcFitter(...)`` to ``fitter.run()``. Paths are relative to
``src/adelmo/``.

1. Construction (main process)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Step
     - Code
     - What it does
   * - Pick the plan
     - ``tracking/fitter.py``
     - The fitter calls its setup function, which returns the :term:`plan`.
   * - Load the model
     - ``fitting/setup.py``
     - ``MachineSetup`` loads the machine in :term:`MAD-NG` and reads the knob names,
       BPM list and starting :term:`knob` values.
   * - Filter BPMs
     - ``tracking/ranges.py``
     - Removes bad BPMs from the start and end lists.
   * - Load the data
     - ``tracking/data_manager.py``
     - ``DataManager`` reads the parquet files and splits the turns into training
       :term:`batches <batch>` and validation turns.
   * - Create the loop
     - ``tracking/fitter.py``
     - ``SGDLoop`` creates the optimiser and learning-rate schedule.

2. Start the workers (main process)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Step
     - Code
     - What it does
   * - ``run()``
     - ``tracking/fitter.py``
     - Calls ``start_session()``, which creates a ``TrackingSession``.
   * - Build payloads
     - ``tracking/session.py``
     - For each tracking range and each batch, collects the BPMs, the measured data
       and the weights into one :term:`payload`.
   * - Spawn
     - ``tracking/session.py``
     - Starts one ``TrackingWorker`` process per payload, plus
       :term:`validation workers <validation worker>`. Each gets a ``Start(knobs)``
       message.

3. Worker startup (worker process, MAD-NG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Step
     - Code
     - What it does
   * - Wait for messages
     - ``fitting/worker.py``
     - Receives ``Start``, then answers messages until ``Stop``.
   * - Load MAD-NG
     - ``fitting/worker.py``
     - Loads its own copy of the machine and rotates the sequence to start at its
       first BPM or marker.
   * - Make knobs differentiable
     - ``tracking/worker.py``
     - Adds each knob as a :term:`TPSA` parameter.
   * - Load particles
     - ``tracking/worker.py``
     - Creates one particle per measured :term:`turn`, at the measured start
       coordinates.

4. Optimisation loop (all processes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Step
     - Code
     - What it does
   * - Send knobs
     - ``tracking/sgd/loop.py``
     - Sends ``Evaluate(knobs, batch)`` to every worker.
   * - Track
     - ``machine/mad/scripts.py``
     - MAD-NG tracks the batch and stores x, y and their knob derivatives at each BPM.
   * - Loss and gradient
     - ``tracking/worker.py``
     - loss = Σ w·(model − measured)². Sends back the loss and gradient as a
       ``GradReply``.
   * - Step
     - ``tracking/sgd/loop.py``
     - Sums the gradients from all workers and updates the knobs with Adam or
       L-BFGS.
   * - End of :term:`epoch`
     - ``tracking/sgd/loop.py``
     - Computes the validation loss, saves the best knobs, checks whether to stop,
       and prints one log line.

5. Finish (main process, workers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Step
     - Code
     - What it does
   * - Uncertainties
     - ``tracking/uncertainty.py``
     - Each worker sends its part of the uncertainty matrices, then stops.
   * - Return the result
     - ``tracking/fitter.py``
     - Returns a :class:`~adelmo.fitting.results.FitResult` with the fitted knobs and
       their uncertainties.

Messages between processes
--------------------------

Defined in ``fitting/protocol.py``. The main process talks to every
:term:`worker` with three messages: ``Start``, ``Evaluate`` and ``Stop``.

.. raw:: html
   :file: _static/tour/messages_overview.svg

Batch step
^^^^^^^^^^

Runs once per :term:`batch`.

.. raw:: html
   :file: _static/tour/batch_step.svg

.. _tour-glossary:

Glossary
--------

.. glossary::
   :sorted:

   AC dipole
      A magnet that drives the beam to oscillate. Short form: ACD.

   batch
      A group of measured turns. Each turn is one particle in a worker.

   BPM
      Beam position monitor. Measures the beam's x and y position each turn.

   bunch
      A group of particles in the beam. Each bunch is processed separately.

   epoch
      One pass over all batches.

   kick
      The moment the beam is excited. Turns before it are dropped.

   knob
      A fitted parameter, usually a magnet strength.

   MAD-NG
      The accelerator simulation code. Each worker runs its own copy.

   parquet
      The file format of the measurement data.

   payload
      Everything one worker needs to start: its BPMs, measured data and weights.

   plan
      The class that sets the tracking ranges for a mode.

   TPSA
      A number that carries its derivatives with it. Tracking with TPSA gives
      derivatives without extra runs.

   turn
      One trip of the beam around the ring.

   validation worker
      A worker that tracks held-out turns and reports only a loss.

   worker
      A separate process that tracks part of the data.
