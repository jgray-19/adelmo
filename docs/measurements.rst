Measurement Preparation
=======================

The modules in ``aba_optimiser.measurements`` prepare AC-dipole (ACD) measurements for
fitting. They follow the conventions of ``tmom-recon`` and ``pymadng-utils``.

Reconstruction
--------------

* Each measurement is reconstructed once with ``tmom_recon.calculate_acd_pz``. Marker
  momenta refreshed during a fit (:mod:`aba_optimiser.measurements.acd_pipeline`)
  recompute the same cleaned measurement after the magnets change.
* The reconstruction Twiss must be on-momentum. Momentum offsets are carried through
  MAD-NG ``pt``; an off-momentum Twiss would subtract a dispersive closed orbit from
  the measured positions and bias the reconstructed phase space.
* The LHC MAD-NG model is updated with both the natural and the driven AC-dipole tunes
  using ``update_model_with_madng(..., tunes=..., drv_tunes=...)``.

Output format
-------------

The saved parquet contains the BPM rows and the ``<acd>_before`` / ``<acd>_after``
marker rows emitted by ``tmom-recon``
(:func:`aba_optimiser.measurements.reconstruction.append_acd_marker_rows`). The
:class:`~aba_optimiser.training.ACDMarkerFitter` uses the marker rows as initial
conditions for bidirectional tracking.

Machine state
-------------

Tune and corrector knob files extracted for each measurement frequency are passed to
the ACD MAD-NG driver so the model matches the optics state of the corresponding
measurement.

When b2 dipole error tables are enabled, a tune knob file is required. Applying the b2
errors shifts the machine tunes, so the interface applies the error table and then
restores the tunes before creating the optimisation knobs.

Variances
---------

BPM position variances are reduced by the gain achieved by the SVD cleaning,
``rank / n_bpms``, since projecting onto ``rank`` retained modes leaves that fraction of
the noise variance. The weight of the cleaned data is raised accordingly.
