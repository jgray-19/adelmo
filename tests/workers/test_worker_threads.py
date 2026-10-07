"""Worker processes limit their BLAS threads so that many workers do not oversubscribe the machine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from aba_optimiser.config import SimulationConfig
from aba_optimiser.fitting.worker import AbstractWorker


def _blas_threads() -> set[int]:
    return {lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"}


def _configure(worker_blas_threads: int | None) -> None:
    simulation_config = SimulationConfig(
        num_workers=1, num_batches=1, worker_blas_threads=worker_blas_threads
    )
    AbstractWorker.configure_worker_threads(SimpleNamespace(simulation_config=simulation_config))


def test_default_is_one_blas_thread() -> None:
    assert SimulationConfig(num_workers=1, num_batches=1).worker_blas_threads == 1


def test_configure_worker_threads_limits_blas() -> None:
    if not _blas_threads():
        pytest.skip("no BLAS library loaded")
    with threadpool_limits(limits=None):  # restore the original limits afterwards
        _configure(1)
        assert _blas_threads() == {1}


def test_none_leaves_library_default() -> None:
    before = _blas_threads()
    _configure(None)
    assert _blas_threads() == before
