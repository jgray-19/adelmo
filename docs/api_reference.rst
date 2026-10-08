API Reference
=============

The reference covers the public entry points and the modules with automated test
coverage. Measurement and campaign workflows are maintained in separate repositories.


Primary Entry Points
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   adelmo.machine.accelerators.Accelerator
   adelmo.machine.accelerators.LHC
   adelmo.machine.accelerators.PSB
   adelmo.machine.accelerators.SPS
   adelmo.config.OptimiserConfig
   adelmo.config.SimulationConfig
   adelmo.tracking.ArcByArcFitter
   adelmo.tracking.ACDMarkerFitter
   adelmo.tracking.KickerFitter
   adelmo.tracking.FitterOptions
   adelmo.tracking.MeasurementConfig
   adelmo.fitting.config.SequenceConfig
   adelmo.fitting.config.OutputConfig
   adelmo.fitting.results.FitResult


Accelerators And Configuration
------------------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.machine
   adelmo.machine.accelerators
   adelmo.machine.accelerators.base
   adelmo.machine.accelerators.lhc
   adelmo.machine.accelerators.psb
   adelmo.machine.accelerators.sps
   adelmo.config


MAD Interface Layer
-------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.machine.mad
   adelmo.machine.mad.aba_mad_interface
   adelmo.machine.mad.optimising_mad_interface
   adelmo.machine.mad.machine_state


Closed-Orbit Fitting (POCO)
---------------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.poco
   adelmo.poco.closed_orbit
   adelmo.poco.calibrated
   adelmo.poco.fitter
   adelmo.poco.lm_loop
   adelmo.poco.prior
   adelmo.poco.calibration
   adelmo.poco.workers
   adelmo.poco.workers.closed_orbit
   adelmo.poco.workers.closed_twiss
   adelmo.poco.workers.calibrated


Tracking Fits
-------------

.. autosummary::
   :toctree: _autosummary

   adelmo.tracking
   adelmo.tracking.fitter
   adelmo.tracking.session
   adelmo.tracking.ranges
   adelmo.tracking.data_manager
   adelmo.tracking.worker
   adelmo.tracking.dispatch
   adelmo.tracking.dispatch.payloads
   adelmo.tracking.dispatch.screening
   adelmo.tracking.dispatch.setup
   adelmo.tracking.dispatch.turn_planner
   adelmo.tracking.uncertainty
   adelmo.tracking.sgd.loop
   adelmo.tracking.sgd.scheduler
   adelmo.tracking.sgd.checkpointing
   adelmo.tracking.config
   adelmo.tracking.config.helpers
   adelmo.tracking.config.models
   adelmo.tracking.config.tracking


Shared Fitting Base
-------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.fitting
   adelmo.fitting.setup
   adelmo.fitting.config
   adelmo.fitting.protocol
   adelmo.fitting.worker
   adelmo.fitting.shared_reference
   adelmo.fitting.pool
   adelmo.fitting.lifecycle
   adelmo.fitting.reduction
   adelmo.fitting.weights
   adelmo.fitting.uncertainty
   adelmo.fitting.results


Optimisers And Numerical Helpers
--------------------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.optimisers.base
   adelmo.optimisers.adam
   adelmo.optimisers.lbfgs
   adelmo.optimisers.levenberg_marquardt


Measurement Preparation
-----------------------

.. autosummary::
   :toctree: _autosummary

   adelmo.measurements
   adelmo.measurements.acd_pipeline
   adelmo.measurements.preprocessing
   adelmo.measurements.reconstruction
   adelmo.measurements.reference
   adelmo.measurements.variances
   adelmo.measurements.noise
