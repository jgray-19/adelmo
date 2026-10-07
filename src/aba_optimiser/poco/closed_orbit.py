"""Public closed-orbit fitter for LOCO-style machine-state comparisons."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from aba_optimiser.fitting.protocol import Evaluate, GradReply, distribute, machine_worker_limit
from aba_optimiser.machine.mad.machine_state import resolve_machine_state
from aba_optimiser.poco.fitter import (
    LevenbergMarquardtConfig,
    LMFitter,
    build_worker_payload,
    load_measurement,
    stamp_global_normalisation,
)
from aba_optimiser.poco.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.fitting.config import OutputConfig, SequenceConfig
    from aba_optimiser.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)
CLOSED_ORBIT_OBSERVABLES = ("x", "y")
#: BPM-to-BPM phase advance from a plain, no-corrector twiss -- see
#: ClosedOrbitSeries.observables and ClosedOrbitWorker's phase-only branch.
PHASE_OBSERVABLES = ("mu1", "mu2")
ALL_OBSERVABLES = CLOSED_ORBIT_OBSERVABLES + PHASE_OBSERVABLES


@dataclass(frozen=True)
class ClosedOrbitMeasurement:
    """One measured orbit at a signal momentum and its reference momentum."""

    orbit: pd.DataFrame
    pt: float = 0.0
    reference_pt: float = 0.0


@dataclass(frozen=True)
class ClosedOrbitSeries:
    """Measurements evaluated in one process at one machine state.

    ``machine_state`` maps MAD-X globals -- tune knobs, quadrupole strengths, corrector
    kicks (e.g. ``kbrqf``, ``kbr3dhz2l4``) -- to the values the machine had when these orbits
    were measured, either directly or as a knobs file. It is layered over the fitter's own
    ``machine_state`` default. They are known inputs, never fitted, and
    differ freely between series -- varying quadrupole strengths is what makes quadrupole
    ``dx``/``dy`` observable. Leave it empty to measure at the model's own settings.

    A relative plane is the change from the fitter's own ``machine_state`` (the reference) to this
    series' state, so changing a corrector is just listing its new kick here; the change is worked out internally.

    Each measurement keeps its own target and momentum. Combining measurements
    here changes process layout only; their residuals and normal equations are
    still evaluated and summed independently.
    """

    measurements: tuple[ClosedOrbitMeasurement, ...]
    machine_state: Path | Mapping[str, float] = field(default_factory=dict)
    absolute_planes: tuple[str, ...] = ()
    label: str = ""
    #: This series' own observables, overriding the fitter-wide default. A
    #: phase-only series (``("mu1", "mu2")``, one measurement)
    #: sets this; every existing orbit series leaves it empty.
    observables: tuple[str, ...] = ()


class ClosedOrbitFitter(LMFitter):
    """Fit magnet knobs to absolute or reference-subtracted closed orbits.

    A :class:`ClosedOrbitSeries` is the unit assigned to one MAD-NG process.
    Putting several measurements in one series changes only process layout:
    each keeps its own target, ``pt`` and ``reference_pt``, and contributes an
    independent residual/Jacobian block to the same global objective.

    With ``shared_reference`` (default) the reference orbit that reference-subtracted
    series are compared to is solved once per iteration by one extra worker, in
    parallel with the signal solves, instead of once per series.
    """

    worker_class = ClosedOrbitWorker
    log_suffix = "closed_orbit_opt"
    fit_label = "Closed-orbit"

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        series: list[ClosedOrbitSeries] | tuple[ClosedOrbitSeries, ...],
        observables: tuple[str, ...] = CLOSED_ORBIT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        machine_state: Path | Mapping[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strengths: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
        max_workers: int | None = None,
        shared_reference: bool = True,
    ) -> None:
        if not series:
            raise ValueError("series must contain at least one closed-orbit series")
        observables = tuple(observables)
        if not observables:
            raise ValueError("At least one closed-orbit observable must be fitted")
        unsupported = [name for name in observables if name not in CLOSED_ORBIT_OBSERVABLES]
        if unsupported:
            raise ValueError(
                f"ClosedOrbitFitter supports {CLOSED_ORBIT_OBSERVABLES}, got {unsupported}"
            )
        for item in series:
            if not item.measurements:
                raise ValueError(
                    f"Closed-orbit series {item.label or '<unnamed>'!r} has no measurements"
                )
            item_observables = item.observables or observables
            item_unsupported = [name for name in item_observables if name not in ALL_OBSERVABLES]
            if item_unsupported:
                raise ValueError(
                    f"Closed-orbit series {item.label or '<unnamed>'!r} requested "
                    f"{item_unsupported}; supported are {ALL_OBSERVABLES}"
                )

        self.observable_names = observables
        self.series = tuple(series)
        self.shared_reference = shared_reference
        self._reference_worker: int | None = None
        self.n_workers = self._worker_count(len(self.series), max_workers)
        super().__init__(
            accelerator,
            sequence_config,
            num_workers=self.n_workers,
            lm_config=lm_config,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            use_errors=use_errors,
            prior_strengths=prior_strengths,
            output_config=output_config,
        )

        self.default_machine_state = resolve_machine_state(machine_state) if machine_state is not None else {}
        self.worker_payloads = self._create_series_payloads(sequence_config, accelerator)

    @staticmethod
    def _worker_count(n_series: int, max_workers: int | None) -> int:
        limit, limited_by = machine_worker_limit()
        if max_workers is not None and max_workers > limit:
            LOGGER.warning("max_workers=%d exceeds the machine limit %d (%s)", max_workers, limit, limited_by)
        cap = limit if max_workers is None else min(max_workers, limit)
        count = max(1, min(n_series, cap))
        LOGGER.info("%d series on %d worker(s); machine limit %d (%s)", n_series, count, limit, limited_by)
        return count

    def _group_payloads(self, payloads):
        """Spread the series over the workers and, if it pays, add the shared-reference worker."""
        grouped = self._spread_payloads(payloads)
        if self.shared_reference:
            grouped = self._add_reference_worker(payloads, grouped)
        return grouped

    def _add_reference_worker(self, payloads, grouped):
        """Append a worker that solves the reference orbits for every series, and point the series at it."""
        needing = [data for _, data in payloads if data.needs_reference]
        if len(needing) < 2:
            return grouped  # nothing is solved twice
        coords = {tuple(o.name for o in data.measurements[0].observables) for data in needing}
        if len(coords) != 1:
            LOGGER.warning("Series fit different observables (%s); each solves its own reference", sorted(coords))
            return grouped
        config, first = next((c, d) for c, d in payloads if d.needs_reference)
        measurements = {m.reference_pt: m for data in needing for m in data.measurements}.values()
        reference = replace(first, measurements=list(measurements), shared_reference=False, reference_only=True)
        for _, data in payloads:
            data.shared_reference = True
        grouped = [*grouped, (config, [reference])]
        self._reference_worker = len(grouped) - 1
        LOGGER.info("Reference orbit solved once by worker %d for %d series", self._reference_worker, len(needing))
        return grouped

    def _evaluate_workers(self, channels, knobs):
        """As the base class; the reference worker's published orbit is relayed to the signal workers first."""
        if self._reference_worker is None:
            return super()._evaluate_workers(channels, knobs)
        channels.send_all(Evaluate(knobs))
        (reply,) = channels.recv_some([self._reference_worker])
        if not isinstance(reply, GradReply):
            raise RuntimeError(f"Unexpected reference worker payload: {reply!r}")
        reference = reply.extra  # None if the reference worker lost the closed orbit
        signal = [i for i in range(len(channels.workers)) if i != self._reference_worker]
        for index in signal:
            channels.send_to(index, reference)
        return channels.recv_some(signal)

    def _spread_payloads(self, payloads):
        """Spread the series over ``n_workers`` processes, each series whole in one."""
        if self.n_workers >= len(payloads):
            return [(config, [data]) for config, data in payloads]
        costs = [len(item.measurements) for item in self.series]
        groups = distribute(costs, self.n_workers)
        LOGGER.info(
            "Series per worker %s, momenta per worker %s",
            [len(group) for group in groups],
            [sum(costs[i] for i in group) for group in groups],
        )
        return [(payloads[group[0]][0], [payloads[i][1] for i in group]) for group in groups]

    def _create_series_payloads(self, sequence_config, accelerator):
        payloads = []
        for item in self.series:
            item_observables = item.observables or self.observable_names
            frames = [
                load_measurement(measurement.orbit, item_observables)
                for measurement in item.measurements
            ]
            common_bpms = [
                bpm
                for bpm in self.machine.all_bpms
                if all(bpm in frame.index for frame in frames)
            ]
            if len(common_bpms) < 2:
                raise ValueError(
                    f"Fewer than two common model BPMs in closed-orbit series "
                    f"{item.label or '<unnamed>'!r}"
                )

            measurement_data = []
            config = None
            for measurement, frame in zip(item.measurements, frames, strict=True):
                worker_config, base_data = build_worker_payload(
                    float(measurement.pt),
                    frame.loc[common_bpms],
                    item_observables,
                    self.machine.all_bpms,
                    sequence_config.magnet_range,
                    sequence_config.bad_bpms,
                    accelerator,
                    {},
                    self.use_errors,
                    self.machine.output_config.mad_logfile,
                    self.machine.output_config.python_logfile,
                )
                config = config or worker_config
                measurement_data.append(
                    ClosedOrbitMeasurementData(
                        observables=base_data.observables,
                        pt=float(measurement.pt),
                        reference_pt=float(measurement.reference_pt),
                    )
                )

            payloads.append(
                (
                    config,
                    ClosedOrbitSeriesData(
                        bpm_names=common_bpms,
                        measurements=measurement_data,
                        machine_state={**self.default_machine_state, **resolve_machine_state(item.machine_state)},
                        reference_state=dict(self.default_machine_state),
                        absolute_planes=tuple(item.absolute_planes),
                    ),
                )
            )
        stamp_global_normalisation(payloads)
        LOGGER.info(
            "Closed-orbit fit: %d worker series, %d independent measurements",
            len(payloads),
            sum(len(item.measurements) for item in self.series),
        )
        return self._group_payloads(payloads)
