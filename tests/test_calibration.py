import numpy as np
import pytest

from adelmo.poco.calibration import (
    CalibrationBlocks,
    add_series,
    apply_prior,
    back_substitute,
    reduce_blocks,
)

rng = np.random.default_rng(0)
N_OBS, N_BPM, N_Q, N_G = 2, 7, 5, 3


def make_series(n):
    out = []
    for k in range(n):
        out.append(
            {
                "model": rng.normal(size=(N_OBS, N_BPM)),
                "quad": rng.normal(size=(N_OBS, N_BPM)),  # second order in the kick: the orbit is nonlinear in it
                "jac": rng.normal(size=(N_OBS, N_BPM, N_Q)),
                "delta": 0.7 + 0.1 * k,
                "targets": rng.normal(size=(N_OBS, N_BPM)),
                "weights": rng.uniform(0.5, 2.0, size=(N_OBS, N_BPM)),
                "corr": k % N_G,
            }
        )
    return out


SERIES = make_series(6)
COLS = np.arange(N_OBS * N_BPM).reshape(N_OBS, N_BPM)


def full_params(q, g, b):
    return np.concatenate([q, g, b])


def orbit(s, q, g):
    """Orbit change at the kick K = (1 + g) * delta: linear in q, nonlinear in K. Returns (orbit, d/dq, d/dg)."""
    kick = (1 + g[s["corr"]]) * s["delta"]
    base = s["model"] + s["jac"] @ q
    return base * kick + s["quad"] * kick**2, s["jac"] * kick, s["delta"] * (base + 2 * s["quad"] * kick)


def residual_vector(q, g, b):
    res = []
    for s in SERIES:
        m, _, _ = orbit(s, q, g)
        res.append(np.sqrt(s["weights"]) * ((1 + b[COLS]) * m - s["targets"]))
    return np.concatenate([r.ravel() for r in res])


def blocks_at(q, g, b):
    blocks = CalibrationBlocks(N_Q, N_G, N_OBS * N_BPM)
    for s in SERIES:
        m, jac, d_gain = orbit(s, q, g)
        add_series(blocks, m, jac, d_gain[..., None], s["targets"], s["weights"], b[COLS], COLS, [s["corr"]])
    blocks.symmetrise()
    return blocks


def test_blocks_match_finite_difference():
    q, g, b = rng.normal(size=N_Q) * 0.1, rng.normal(size=N_G) * 0.1, rng.normal(size=N_OBS * N_BPM) * 0.1
    blocks = blocks_at(q, g, b)
    p0 = full_params(q, g, b)

    def loss(p):
        r = residual_vector(p[:N_Q], p[N_Q : N_Q + N_G], p[N_Q + N_G :])
        return float(r @ r)

    assert np.isclose(blocks.loss, loss(p0))
    grad = np.concatenate([blocks.grad_u, blocks.grad_b])
    fd = np.array([(loss(p0 + e) - loss(p0 - e)) / 2e-6 for e in np.eye(len(p0)) * 1e-6])
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-7)


def test_gauss_newton_hessian_matches_jacobian():
    q, g, b = rng.normal(size=N_Q) * 0.1, rng.normal(size=N_G) * 0.1, rng.normal(size=N_OBS * N_BPM) * 0.1
    blocks = blocks_at(q, g, b)
    p0 = full_params(q, g, b)
    jac = np.array([(residual_vector(*np.split(p0 + e, [N_Q, N_Q + N_G])) - residual_vector(*np.split(p0 - e, [N_Q, N_Q + N_G]))) / 2e-6 for e in np.eye(len(p0)) * 1e-6]).T
    hess = 2.0 * jac.T @ jac
    n_u = N_Q + N_G
    assert np.allclose(blocks.hess_uu, hess[:n_u, :n_u], rtol=1e-5, atol=1e-7)
    assert np.allclose(blocks.hess_ub, hess[:n_u, n_u:], rtol=1e-5, atol=1e-7)
    assert np.allclose(blocks.hess_bb, np.diag(hess[n_u:, n_u:]), rtol=1e-5, atol=1e-7)
    assert np.allclose(np.diag(np.diag(hess[n_u:, n_u:])), hess[n_u:, n_u:], atol=1e-7)


def test_schur_step_equals_dense_step():
    q, g, b = rng.normal(size=N_Q) * 0.1, rng.normal(size=N_G) * 0.1, rng.normal(size=N_OBS * N_BPM) * 0.1
    blocks = blocks_at(q, g, b)
    apply_prior(blocks, np.concatenate([q, g]), b, 0.01, 0.01, 1.0)  # the g/b scale degeneracy needs the prior
    n_u = N_Q + N_G
    hess = np.block([[blocks.hess_uu, blocks.hess_ub], [blocks.hess_ub.T, np.diag(blocks.hess_bb)]])
    grad = np.concatenate([blocks.grad_u, blocks.grad_b])
    dense = np.linalg.solve(hess, -grad)
    g_r, h_r = reduce_blocks(blocks)
    du = np.linalg.solve(h_r, -g_r)
    db = back_substitute(blocks, du)
    assert np.allclose(du, dense[:n_u])
    assert np.allclose(db, dense[n_u:])


def test_quad_prior_matches_finite_difference_and_leaves_zero_sigma_free():
    q, g, b = rng.normal(size=N_Q) * 0.1, rng.normal(size=N_G) * 0.1, rng.normal(size=N_OBS * N_BPM) * 0.1
    sigma_q = np.array([0.5, 0.0, 2.0, 0.1, 0.0])
    c = 1.7
    base = blocks_at(q, g, b)
    with_prior = blocks_at(q, g, b)
    apply_prior(with_prior, np.concatenate([q, g]), b, 0.0, 0.0, c, sigma_q)
    free = sigma_q == 0
    expected = c * np.sum((q[~free] / sigma_q[~free]) ** 2)
    assert np.isclose(with_prior.loss - base.loss, expected)
    d_grad = (with_prior.grad_u - base.grad_u)[:N_Q]
    assert np.allclose(d_grad, 2 * c * np.where(free, 0.0, q / np.where(free, 1.0, sigma_q) ** 2))
    d_hess = (with_prior.hess_uu - base.hess_uu)[:N_Q, :N_Q]
    assert np.allclose(d_hess, np.diag(2 * c / np.where(free, np.inf, sigma_q) ** 2))


def test_apply_prior_loss_only_matches_the_full_prior_loss():
    """A loss-only trial evaluation adds exactly the loss the full evaluation adds (no gradient/Hessian blocks needed)."""
    from types import SimpleNamespace

    rng = np.random.default_rng(3)
    n_q, n_g, n_b = 5, 3, 7
    u, b = rng.normal(size=n_q + n_g), rng.normal(size=n_b)
    sigma_q = np.array([0.5, 0.0, 1.0, 2.0, 0.1])
    full = CalibrationBlocks(n_q, n_g, n_b)
    full.loss = 12.5
    apply_prior(full, u, b, 0.02, 0.03, 1.7, sigma_q)
    shell = SimpleNamespace(loss=12.5, n_q=n_q)
    apply_prior(shell, u, b, 0.02, 0.03, 1.7, sigma_q, loss_only=True)
    assert shell.loss == pytest.approx(full.loss, rel=1e-14)
