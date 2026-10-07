API Reference
=============

The reference covers the public entry points and the modules with automated test
coverage. Measurement and campaign workflows are maintained in separate repositories.


Primary Entry Points
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   aba_optimiser.machine.accelerators.Accelerator
   aba_optimiser.machine.accelerators.LHC
   aba_optimiser.machine.accelerators.PSB
   aba_optimiser.machine.accelerators.SPS
   aba_optimiser.config.OptimiserConfig
   aba_optimiser.config.SimulationConfig
   aba_optimiser.tracking.ArcByArcFitter
   aba_optimiser.tracking.ACDMarkerFitter
   aba_optimiser.tracking.KickerFitter
   aba_optimiser.tracking.FitterOptions
   aba_optimiser.tracking.MeasurementConfig
   aba_optimiser.fitting.config.SequenceConfig
   aba_optimiser.fitting.config.OutputConfig
   aba_optimiser.fitting.results.FitResult


Accelerators And Configuration
------------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.machine
   aba_optimiser.machine.accelerators
   aba_optimiser.machine.accelerators.base
   aba_optimiser.machine.accelerators.lhc
   aba_optimiser.machine.accelerators.psb
   aba_optimiser.machine.accelerators.sps
   aba_optimiser.config


MAD Interface Layer
-------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.machine.mad
   aba_optimiser.machine.mad.aba_mad_interface
   aba_optimiser.machine.mad.optimising_mad_interface
   aba_optimiser.machine.mad.machine_state


Closed-Orbit Fitting (POCO)
---------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.poco
   aba_optimiser.poco.closed_orbit
   aba_optimiser.poco.calibrated
   aba_optimiser.poco.fitter
   aba_optimiser.poco.lm_loop
   aba_optimiser.poco.prior
   aba_optimiser.poco.calibration
   aba_optimiser.poco.workers
   aba_optimiser.poco.workers.closed_orbit
   aba_optimiser.poco.workers.closed_twiss
   aba_optimiser.poco.workers.calibrated


Tracking Fits
-------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.tracking
   aba_optimiser.tracking.fitter
   aba_optimiser.tracking.session
   aba_optimiser.tracking.ranges
   aba_optimiser.tracking.data_manager
   aba_optimiser.tracking.worker
   aba_optimiser.tracking.dispatch
   aba_optimiser.tracking.dispatch.payloads
   aba_optimiser.tracking.dispatch.screening
   aba_optimiser.tracking.dispatch.setup
   aba_optimiser.tracking.dispatch.turn_planner
   aba_optimiser.tracking.uncertainty
   aba_optimiser.tracking.sgd.loop
   aba_optimiser.tracking.sgd.scheduler
   aba_optimiser.tracking.sgd.checkpointing
   aba_optimiser.tracking.config
   aba_optimiser.tracking.config.helpers
   aba_optimiser.tracking.config.models
   aba_optimiser.tracking.config.tracking


Shared Fitting Base
-------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.fitting
   aba_optimiser.fitting.setup
   aba_optimiser.fitting.config
   aba_optimiser.fitting.protocol
   aba_optimiser.fitting.worker
   aba_optimiser.fitting.shared_reference
   aba_optimiser.fitting.pool
   aba_optimiser.fitting.lifecycle
   aba_optimiser.fitting.reduction
   aba_optimiser.fitting.weights
   aba_optimiser.fitting.uncertainty
   aba_optimiser.fitting.results


Optimisers And Numerical Helpers
--------------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.optimisers.base
   aba_optimiser.optimisers.adam
   aba_optimiser.optimisers.lbfgs
   aba_optimiser.optimisers.levenberg_marquardt


Measurement Preparation
-----------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.measurements
   aba_optimiser.measurements.acd_pipeline
   aba_optimiser.measurements.preprocessing
   aba_optimiser.measurements.reconstruction
   aba_optimiser.measurements.reference
   aba_optimiser.measurements.variances
   aba_optimiser.measurements.noise
