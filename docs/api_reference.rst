API Reference
=============

The reference covers the public entry points and the modules with automated test
coverage. Measurement and campaign workflows are maintained in separate repositories.


Primary Entry Points
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   aba_optimiser.accelerators.Accelerator
   aba_optimiser.accelerators.LHC
   aba_optimiser.accelerators.PSB
   aba_optimiser.accelerators.SPS
   aba_optimiser.config.OptimiserConfig
   aba_optimiser.config.SimulationConfig
   aba_optimiser.training.ArcByArcFitter
   aba_optimiser.training.ACDMarkerFitter
   aba_optimiser.training.KickerFitter
   aba_optimiser.training.MeasurementConfig
   aba_optimiser.training.SequenceConfig
   aba_optimiser.training.OutputConfig


Accelerators And Configuration
------------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.accelerators
   aba_optimiser.accelerators.base
   aba_optimiser.accelerators.lhc
   aba_optimiser.accelerators.psb
   aba_optimiser.accelerators.sps
   aba_optimiser.config


MAD Interface Layer
-------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.mad
   aba_optimiser.mad.aba_mad_interface
   aba_optimiser.mad.optimising_mad_interface
   aba_optimiser.mad.machine_state


Closed-Orbit Fitting
--------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.poco
   aba_optimiser.poco.closed_orbit
   aba_optimiser.poco.calibrated
   aba_optimiser.poco.fitter
   aba_optimiser.workers.closed_orbit
   aba_optimiser.workers.closed_twiss
   aba_optimiser.workers.calibrated_closed_orbit


Optimisation Runtime
--------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.training
   aba_optimiser.training.base_fitter
   aba_optimiser.training.tracking.fitter
   aba_optimiser.training.config
   aba_optimiser.training.config.helpers
   aba_optimiser.training.config.manager
   aba_optimiser.training.config.models
   aba_optimiser.training.config.tracking
   aba_optimiser.training.tracking.data_manager
   aba_optimiser.training.optimisation
   aba_optimiser.training.sgd.loop
   aba_optimiser.training.sgd.scheduler
   aba_optimiser.training.workers
   aba_optimiser.training.workers.manager
   aba_optimiser.training.tracking.workers.payloads
   aba_optimiser.training.tracking.workers.setup
   aba_optimiser.training.pool


Workers
-------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.workers
   aba_optimiser.workers.abstract_worker
   aba_optimiser.workers.common
   aba_optimiser.workers.tracking
   aba_optimiser.workers.tracking_position_only
   aba_optimiser.workers.tracking_validation


Optimisers And Numerical Helpers
--------------------------------

.. autosummary::
   :toctree: _autosummary

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
