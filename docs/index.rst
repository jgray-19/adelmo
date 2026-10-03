aba_optimiser documentation
===========================

``aba_optimiser`` is a worker-based optimisation toolkit for accelerator magnet
studies. The current documentation is intentionally narrow: it focuses on the
API surface that is exercised by the automated tests and used by the main
fitter-driven runtime.

Use this site as a reference for:

* accelerator definitions and runtime configuration dataclasses
* MAD interface classes used to construct optimisation problems
* tracking fitter, worker, and optimiser APIs
* the shared measurement-preparation modules used by the PSB and LHC workflows

Workflow guides and campaign-specific scripts are intentionally left out of the
published docs until they have stronger validation coverage.

GitHub Dependencies
-------------------

Some parts of the repository rely on companion packages installed directly from
GitHub rather than PyPI-only dependencies:

* ``pymadng-utils`` provides shared accelerator abstractions, dp/p ↔ pt
  conversion (``pymadng_utils.physics``), plus MAD/MAD-X helper utilities such
  as knob file IO and interface glue used throughout the core runtime.
* ``tmom-recon`` provides transverse and longitudinal momentum reconstruction,
  AC-dipole measurement helpers, and optics reconstruction routines used by the
  measurement and optics-oriented code paths.
* ``xtrack-tools`` provides tracking helpers, environment initialisation, and
  dataframe conversion utilities used by the higher-fidelity fitter and
  simulation tests.

These dependencies are important because the tested end-to-end workflows in
this repository are built around a larger accelerator-tooling stack rather than
standalone numerical routines.

AC-Dipole Measurement Reconstruction
------------------------------------

The PSB (``psb_md``) and LHC squeeze (``lhc_measurements.squeeze``)
reconstructions follow the current ``tmom-recon`` and ``pymadng-utils``
conventions:

* Batch processing reconstructs each measurement once with
  ``tmom_recon.calculate_acd_pz``. Live marker-momentum refreshes during a fit
  (``aba_optimiser.measurements.acd_pipeline``) recompute the same cleaned
  measurement after magnet changes.
* The LHC MAD-NG model is updated with both the natural tunes and the driven
  AC-dipole tunes via ``update_model_with_madng(..., tunes=..., drv_tunes=...)``.
* The reconstruction Twiss is expected to be on-momentum. Momentum offsets are
  carried through MAD-NG ``pt``; using an off-momentum Twiss would subtract a
  dispersive closed orbit from the measured positions and bias the reconstructed
  phase space.
* The saved parquet keeps the usual BPM rows and appends the
  ``<acd>_before`` / ``<acd>_after`` marker rows emitted by ``tmom-recon``
  (``aba_optimiser.measurements.reconstruction.append_acd_marker_rows``) so
  downstream ACD optimisation can initialise bidirectional tracking at the
  reconstructed marker states.
* Tune and corrector knob files extracted for each measurement frequency are
  passed into the ACD MAD-NG driver, matching the optics state used for the
  corresponding turn-by-turn measurement.
* When b2 dipole error tables are enabled, the MAD interface requires a tune
  knob file. Applying b2 errors shifts the machine tunes, so the interface
  applies the error table and then restores the tunes before creating
  optimisation knobs.

Companion documentation:

* ``sgd-magnet-tuner``: `jgray-19.github.io/sgd-magnet-tuner <https://jgray-19.github.io/sgd-magnet-tuner/>`_
* ``pymadng-utils``: `jgray-19.github.io/pymadng-utils <https://jgray-19.github.io/pymadng-utils/>`_
* ``tmom-recon``: `jgray-19.github.io/tmom-recon <https://jgray-19.github.io/tmom-recon/>`_
* ``xtrack_tools``: `jgray-19.github.io/xtrack_tools <https://jgray-19.github.io/xtrack_tools/>`_

.. toctree::
   :maxdepth: 2
   :caption: API

   api_reference
