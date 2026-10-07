"""Worker-count bound and the equal split of corrector settings over workers."""

from __future__ import annotations

import resource

from aba_optimiser.fitting import protocol
from aba_optimiser.fitting.protocol import distribute, machine_worker_limit


def test_distribute_balances_total_cost():
    costs = [5, 5, 5, 1, 1, 1, 1, 1, 1]
    groups = distribute(costs, 3)

    totals = [sum(costs[i] for i in group) for group in groups]
    assert max(totals) - min(totals) <= 1
    assert sorted(i for group in groups for i in group) == list(range(len(costs)))


def test_distribute_keeps_every_setting_whole_and_drops_empty_workers():
    groups = distribute([3, 2], 5)

    assert sorted(len(group) for group in groups) == [1, 1]


def test_limit_is_the_cpu_count_on_a_normal_host(monkeypatch):
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(
        protocol.resource, "getrlimit", lambda _which: (1_000_000, resource.RLIM_INFINITY)
    )

    assert machine_worker_limit() == (16, "cpus")


def test_limit_is_the_open_file_bound_on_a_large_host(monkeypatch):
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: set(range(1000)))
    monkeypatch.setattr(
        protocol.resource, "getrlimit", lambda _which: (1_000_000, resource.RLIM_INFINITY)
    )

    limit, reason = machine_worker_limit()
    assert reason == "open files"
    assert limit < 250


def test_a_low_file_limit_lowers_the_bound(monkeypatch):
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(protocol.resource, "getrlimit", lambda _which: (256, 256))

    assert machine_worker_limit() == (48, "open files")
