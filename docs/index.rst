aba_optimiser
=============

``aba_optimiser`` estimates accelerator magnet errors (strengths, misalignments and
tilts) from beam measurements by gradient-based optimisation of MAD-NG models.
Supported machines are the LHC, PSB, SPS and FCC.

Two families of fit are provided:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Fit
     - Data
     - Entry points
   * - Tracking
     - Turn-by-turn BPM data tracked through the model
     - :class:`~aba_optimiser.tracking.ArcByArcFitter`,
       :class:`~aba_optimiser.tracking.ACDMarkerFitter`,
       :class:`~aba_optimiser.tracking.KickerFitter`
   * - Closed twiss
     - Closed orbit, phase advance, beta and dispersion
     - :class:`~aba_optimiser.poco.ClosedOrbitFitter`,
       :class:`~aba_optimiser.poco.ClosedTwissFitter`

Installation
------------

Python 3.11 or later is required.

.. code-block:: bash

   git clone https://github.com/jgray-19/sgd-magnet-tuner.git
   cd sgd-magnet-tuner
   pip install -e ".[test,docs,tracking]"

Companion packages
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - `pymadng-utils <https://jgray-19.github.io/pymadng-utils/>`_
     - Shared accelerator abstractions, dp/p and pt conversion
       (``pymadng_utils.physics``), knob-file I/O.
   * - `tmom-recon <https://jgray-19.github.io/tmom-recon/>`_
     - Transverse momentum, AC-dipole and optics reconstruction.
   * - `xtrack_tools <https://jgray-19.github.io/xtrack_tools/>`_
     - Tracking helpers and dataframe conversion, used by the tests.

The measurement and campaign workflows built on this package are maintained in
separate repositories (``lhc_measurements``, ``psb_md``, ``psb_loco``, ``lhc_loco``
and ``fcc_loco``) and are not documented here.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide
   architecture
   measurements
   api_reference
