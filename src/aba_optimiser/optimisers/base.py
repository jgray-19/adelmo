"""Abstract optimiser interface, registry and best-point tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from aba_optimiser.config import OptimiserConfig

T = TypeVar("T")


@dataclass
class BestTracker(Generic[T]):
    """The lowest loss seen so far and the point it was seen at."""

    loss: float = float("inf")
    value: T | None = None

    def record(self, loss: float, value: T) -> None:
        """Make ``value`` (at ``loss``) the best point."""
        self.loss = float(loss)
        self.value = value


class BaseOptimiser(ABC):
    """Common interface for all optimisers used by the training loop.

    Subclasses register under ``OPTIMISER_NAME``, are built from an
    :class:`~aba_optimiser.config.OptimiserConfig` by :meth:`from_config`, and list
    in ``KNOB_VECTORS`` / ``KNOB_VECTOR_LISTS`` the state entries that hold one value
    per knob, which :meth:`load_state_dict` remaps when the knob layout changed.
    """

    OPTIMISER_NAME: ClassVar[str]
    #: State entries that are one vector over the knobs (``None`` allowed).
    KNOB_VECTORS: ClassVar[tuple[str, ...]] = ()
    #: State entries that are a list of such vectors.
    KNOB_VECTOR_LISTS: ClassVar[tuple[str, ...]] = ()
    _REGISTRY: ClassVar[dict[str, type[BaseOptimiser]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "OPTIMISER_NAME", "")
        if name:
            BaseOptimiser._REGISTRY[name] = cls

    @classmethod
    def from_config(cls, config: OptimiserConfig, n_params: int) -> BaseOptimiser:
        """Create the optimiser ``config.optimiser_type`` names, for ``n_params`` parameters."""
        try:
            optimiser_cls = cls._REGISTRY[config.optimiser_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown optimiser type: {config.optimiser_type}. Available: {sorted(cls._REGISTRY)}"
            ) from exc
        return optimiser_cls(**optimiser_cls.config_kwargs(config, n_params))

    @classmethod
    @abstractmethod
    def config_kwargs(cls, config: OptimiserConfig, n_params: int) -> dict[str, Any]:
        """Constructor arguments taken from ``config``."""

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
        """Apply one optimisation update step."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Serialise optimiser state to a JSON-compatible dictionary."""

    @abstractmethod
    def _load_state(self, state: dict[str, Any]) -> None:
        """Restore optimiser state from a dictionary in the current knob layout."""

    def load_state_dict(
        self,
        state: dict[str, Any],
        remap: Callable[[list[float]], list[float]] | None = None,
    ) -> None:
        """Restore optimiser state; ``remap`` moves each per-knob vector onto the current knob layout."""
        if remap is not None:
            state = dict(state)
            for key in self.KNOB_VECTORS:
                if state.get(key) is not None:
                    state[key] = remap(state[key])
            for key in self.KNOB_VECTOR_LISTS:
                state[key] = [remap(vector) for vector in state.get(key, [])]
        self._load_state(state)

    @staticmethod
    def with_weight_decay(grads: np.ndarray, params: np.ndarray, weight_decay: float) -> np.ndarray:
        """L2 weight decay: the gradient of ``0.5 * weight_decay * |params|²`` added to ``grads``."""
        if weight_decay != 0:
            return grads + weight_decay * params
        return grads
