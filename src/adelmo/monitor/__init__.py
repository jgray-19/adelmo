"""Live monitor for fits: plots from run logs, with optics recomputed from the knobs.

The monitor reads the run logs (:mod:`adelmo.runlog`) below a directory, so it
runs as its own process and never slows or blocks a fit. Optics are not logged:
an :class:`OpticsSource` from the machine side recomputes them from the logged
knobs in background threads, and :class:`OpticsCache` keeps every result on
disk. Panels are plain functions of a :class:`MonitorView`; a
:class:`MonitorSpec` bundles a machine's source and panels.

Serve it with::

    python -m adelmo.monitor RUN_LOG_ROOT --spec my_package.monitor:SPEC

and open ``http://localhost:5006`` (through ``ssh -L 5006:localhost:5006 host``
when it runs remotely). Requires the ``monitor`` extra (Panel).
"""

from adelmo.monitor.source import MonitorSpec, OpticsSource
from adelmo.monitor.view import MonitorView

__all__ = ["MonitorSpec", "MonitorView", "OpticsSource"]
