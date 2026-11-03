adelmo
=============

``adelmo`` estimates accelerator magnet errors (strengths, misalignments and
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
     - :class:`~adelmo.tracking.fitter.ArcByArcFitter`,
       :class:`~adelmo.tracking.fitter.ACDMarkerFitter`,
       :class:`~adelmo.tracking.fitter.KickerFitter`
   * - Closed orbit
     - Closed orbit and phase advance, at one or more machine states
     - :class:`~adelmo.poco.closed_orbit.ClosedOrbitFitter`,
       :class:`~adelmo.poco.calibrated.CalibratedClosedOrbitFitter`
   * - Closed twiss
     - Closed orbit, phase advance, beta and dispersion
     - :class:`~adelmo.poco.fitter.ClosedTwissFitter`

All fitters return a :class:`~adelmo.fitting.results.FitResult`.

Installation
------------

Python 3.11 or later is required.

.. code-block:: bash

   git clone https://github.com/jgray-19/sgd-magnet-tuner.git
   cd sgd-magnet-tuner
   pip install -e ".[test,docs,tracking]"

Optional dependency groups:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Contents
   * - ``tracking``
     - omc3, cpymad, pyarrow, psutil (required by the tracking workflows)
   * - ``measurements``
     - tmom-recon (momentum reconstruction)
   * - ``monitor``
     - Panel, for the live fit monitor (``python -m adelmo.monitor``)
   * - ``cern``
     - nxcals (CERN logging database access)
   * - ``test``
     - pytest, pytest-cov, pytest-xdist, xtrack-tools
   * - ``docs``
     - Sphinx and theme
   * - ``dev``
     - ruff, pre-commit

Build this documentation with:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

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

Install them from GitHub:

.. code-block:: bash

   pip install git+https://github.com/jgray-19/pymadng-utils.git
   pip install git+https://github.com/jgray-19/tmom-recon.git
   pip install git+https://github.com/jgray-19/xtrack_tools.git

The measurement and campaign workflows built on this package are maintained in
separate repositories (``lhc_measurements``, ``psb_md``, ``psb_loco``, ``lhc_loco``
and ``fcc_loco``) and are not documented here.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide
   tracking_tour
   architecture
   measurements
   api_reference
