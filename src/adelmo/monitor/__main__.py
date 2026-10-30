"""``python -m adelmo.monitor RUN_LOG_ROOT [--spec module:attribute] ...``"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from adelmo.monitor.app import serve
from adelmo.monitor.source import MonitorSpec


def load_spec(reference: str) -> MonitorSpec:
    """Import a ``module:attribute`` reference to a :class:`MonitorSpec`."""
    module_name, _, attribute = reference.partition(":")
    spec = getattr(importlib.import_module(module_name), attribute or "SPEC")
    if not isinstance(spec, MonitorSpec):
        raise TypeError(f"{reference} is a {type(spec).__name__}, not a MonitorSpec")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory searched for run logs (recursively).")
    parser.add_argument(
        "--spec",
        help="module:attribute of a MonitorSpec giving the optics source and panels; "
        "without it only the generic panels are shown.",
    )
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--address", default="localhost")
    parser.add_argument("--refresh", type=float, default=5.0, help="Seconds between rescans.")
    parser.add_argument("--workers", type=int, default=1, help="Threads computing optics.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default: ROOT/.optics_cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    serve(
        args.root,
        load_spec(args.spec) if args.spec else None,
        port=args.port,
        address=args.address,
        refresh_seconds=args.refresh,
        workers=args.workers,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
