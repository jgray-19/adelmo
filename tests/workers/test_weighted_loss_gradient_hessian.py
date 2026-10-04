"""The Gauss-Newton Hessian / normal matrix of the closed-twiss workers match the direct weighted formulas."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.workers.closed_twiss import _weighted_loss_gradient_hessian
from aba_optimiser.workers.common import ObservableKind

SCALE = 5e13


def _reference(model, jacobian, targets, weights, raw_weights):
    """Straightforward four-product implementation (the pre-optimisation code)."""
    n_knobs = jacobian.shape[-1]
    grad, hessian, normal, loss = np.zeros(n_knobs), np.zeros((n_knobs,) * 2), np.zeros((n_knobs,) * 2), 0.0
    for i in range(len(model)):
        residual = np.where(weights[i] > 0.0, model[i] - targets[i], 0.0)
        grad += 2.0 * (weights[i] * residual) @ jacobian[i]
        hessian += 2.0 * (jacobian[i].T * weights[i]) @ jacobian[i]
        normal += (jacobian[i].T * raw_weights[i]) @ jacobian[i]
        loss += float(np.sum(weights[i] * residual**2))
    return grad, loss, hessian, normal


def _data(rng, n_obs=2, n_bpms=60, n_knobs=25):
    observables = [SimpleNamespace(kind=ObservableKind.POINTWISE) for _ in range(n_obs)]
    model, jacobian = rng.normal(size=(n_obs, n_bpms)), rng.normal(size=(n_obs, n_bpms, n_knobs))
    targets = list(rng.normal(size=(n_obs, n_bpms)))
    raw = rng.uniform(1e12, 1e16, size=(n_obs, n_bpms))
    raw[:, ::7] = 0.0  # invalid BPMs carry zero weight
    return observables, model, jacobian, targets, list(raw / SCALE), list(raw)


def test_matches_direct_formulas() -> None:
    observables, model, jacobian, targets, weights, raw = _data(np.random.default_rng(1))
    got = _weighted_loss_gradient_hessian(model, jacobian, observables, targets, weights, raw, SCALE)
    want = _reference(model, jacobian, targets, weights, raw)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-10)


def test_inconsistent_weight_scale_is_rejected() -> None:
    observables, model, jacobian, targets, weights, raw = _data(np.random.default_rng(2))
    with pytest.raises(ValueError, match="raw_weights / weight_scale"):
        _weighted_loss_gradient_hessian(model, jacobian, observables, targets, weights, raw, 2 * SCALE)
