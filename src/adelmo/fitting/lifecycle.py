"""The run/interrupt/cleanup shape every fitter shares."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorboardX import SummaryWriter

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def run_with_workers(
    body: Callable[[], T],
    *,
    fallback: Callable[[], T],
    stop_workers: Callable[[], None],
    writer: SummaryWriter | None = None,
) -> T:
    """Run ``body``; on Ctrl-C return ``fallback()`` (the best result so far) instead.

    The workers are stopped and the TensorBoard writer closed however the run ends;
    any other exception propagates once they are.
    """
    try:
        return body()
    except KeyboardInterrupt:
        LOGGER.warning("KeyboardInterrupt: stopping the optimisation early with the best result so far.")
        return fallback()
    finally:
        stop_workers()
        if writer is not None:
            writer.close()
