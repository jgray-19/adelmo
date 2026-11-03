Measurement Preparation
=======================

The modules in :mod:`adelmo.measurements` turn long-form turn-by-turn BPM data
(``name``, ``turn``, ``x``, ``y``) into the parquet file the tracking fitters read.
They follow the conventions of ``tmom-recon`` and ``pymadng-utils``.

The step-by-step table is in the tour: :ref:`tour-preprocessing`.

Entry point
-----------

:func:`~adelmo.measurements.reconstruction.process_single_dataframe` processes one
bunch:

1. Optionally trim to the kick
   (:func:`~adelmo.measurements.preprocessing.trim_measurement_to_kick`).
2. Assign ``var_x`` / ``var_y``: uniform
   (:func:`~adelmo.measurements.variances.assign_uniform_variances`) or from a per-BPM
   noise table (:func:`~adelmo.measurements.variances.assign_known_noise_variances`).
3. SVD-clean the positions (``tmom_recon.svd.weighted_svd_clean_measurements``).
4. Estimate the measurement ``pt`` from the closed orbit, unless ``machine_deltap`` is
   given.
5. Reconstruct ``px`` / ``py`` with ``tmom_recon.calculate_pz``, using the measured
   phase, beta and alpha from the omc3 analysis directory.

Closed-orbit frame
------------------

:func:`~adelmo.measurements.reference.reconstruction_frame` builds the
:class:`~adelmo.measurements.reference.ReconstructionFrame` passed to
``process_single_dataframe``: the measured closed orbit at dp=0 and an ``orbit_mode``.

``"absolute"``
    No plane is dynamic. The result keeps the measured positions plus model angles.
``"dynamic"``
    Both planes are dynamic. The result stays relative to the measured orbit.

Mixing the two modes between planes is not supported.

AC-dipole data
--------------

* :func:`~adelmo.measurements.acd_pipeline.run_driven_and_compensated_optics` runs
  Harpy over every acquisition, then the driven and compensated optics analysis.
  Pass at least two acquisitions when the phase errors are used; one acquisition
  gives zero phase errors.
* ``process_single_dataframe`` takes ``ac_dipole_inputs_factory``, which returns the
  AC-dipole model details and configuration for each bunch.
* :func:`~adelmo.measurements.reconstruction.append_acd_marker_rows` appends the
  ``<acd>_before`` / ``<acd>_after`` rows from ``tmom-recon``.
  :class:`~adelmo.tracking.fitter.ACDMarkerFitter` uses these rows as initial
  conditions.

Variances
---------

After SVD cleaning, the BPM position variances are multiplied by ``rank / n_bpms``,
where ``rank`` is the number of SVD modes kept.

b2 errors
---------

``b2_errors`` (LHC only) adds a b2 error table to the quadrupole slot ``dknl[2]`` of
the loaded sequence
(:func:`~adelmo.machine.mad.optimising_mad_interface.apply_b2_errors_to_sequence`). It
raises ``ValueError`` without a ``machine_state``; the machine state carries the tune
knobs that compensate the b2 errors.
