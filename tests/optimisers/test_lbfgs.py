import numpy as np

from adelmo.optimisers.lbfgs import LBFGSOptimiser


class TestLBFGSOptimiser:
    def test_initialization(self):
        optim = LBFGSOptimiser(
            history_size=10,
            eps=1e-12,
            weight_decay=0.0,
            use_adaptive_lr=True,
            bb_clip=(1e-2, 1e2),
            ema_beta=0.8,
            eta_init=1.0,
        )
        assert optim.history_size == 10
        assert optim.eps == 1e-12
        assert optim.weight_decay == 0.0
        assert optim.use_adaptive_lr is True
        assert optim.bb_min == 1e-2
        assert optim.bb_max == 1e2
        assert optim.ema_beta == 0.8
        assert optim.eta_ema == 1.0
        assert optim.t == 0
        assert optim.prev_params is None
        assert optim.prev_grads is None
        assert len(optim.S) == 0
        assert len(optim.Y) == 0
        assert len(optim.RHO) == 0

    def test_step_first_step(self):
        optim = LBFGSOptimiser(max_grad_norm=None, max_step_norm=None)
        params = np.array([1.0, 2.0])
        grads = np.array([2.0, 4.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # First step, no history, so direction is -g
        expected_update = lr * optim.eta_ema * (-grads)
        expected_new_params = params + expected_update
        assert np.allclose(new_params, expected_new_params)
        assert optim.t == 1
        assert np.array_equal(optim.prev_params, params)
        assert np.array_equal(optim.prev_grads, grads)

    def test_step_with_history(self):
        optim = LBFGSOptimiser(history_size=5, max_grad_norm=None, max_step_norm=None)
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.1

        # First step
        params = optim.step(params, grads, lr)

        # Second step
        grads2 = np.array([1.0])  # new grads
        new_params = optim.step(params, grads2, lr)

        assert new_params is not None

        assert optim.t == 2
        assert len(optim.S) == 1
        assert len(optim.Y) == 1
        assert len(optim.RHO) == 1

    def test_adaptive_lr_bb1_calculation(self):
        """Test that BB1 multiplier is calculated correctly."""
        optim = LBFGSOptimiser(
            use_adaptive_lr=True,
            bb_clip=(0.0, float("inf")),
            ema_beta=0.0,
            max_grad_norm=None,
            max_step_norm=None,
        )  # Immediate update
        params = np.array([2.0])  # Start further from optimum
        grads = np.array([4.0])  # grad = 2*x for f=0.5*x^2
        lr = 0.5  # Larger lr

        # First step
        params = optim.step(params, grads, lr)  # Should move towards 0

        # Second step
        grads2 = np.array([2 * params[0]])  # New gradient
        optim.step(params, grads2, lr)

        # Calculate expected BB1
        s = params - 2.0  # Change in params
        y = grads2 - grads  # Change in grads
        sy_dot = np.dot(s, y)
        if sy_dot > 0:
            expected_bb1 = np.dot(s, s) / sy_dot
            assert np.isclose(optim.eta_ema, expected_bb1, rtol=1e-10)
        else:
            # If curvature poor, should keep previous eta_ema
            assert optim.eta_ema == 1.0

    def test_no_adaptive_lr(self):
        optim = LBFGSOptimiser(
            use_adaptive_lr=False,
            eta_init=2.0,
            max_grad_norm=None,
            max_step_norm=None,
        )
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.01

        optim.step(params, grads, lr)
        optim.step(params, grads, lr)

        # eta_ema should remain 2.0
        assert optim.eta_ema == 2.0

    def test_weight_decay(self):
        optim = LBFGSOptimiser(weight_decay=0.1, max_grad_norm=None, max_step_norm=None)
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # Effective grads = grads + weight_decay * params
        effective_grads = grads + 0.1 * params
        expected_update = lr * optim.eta_ema * (-effective_grads)
        expected_new_params = params + expected_update
        assert np.allclose(new_params, expected_new_params)

    def test_convergence_quadratic(self):
        optim = LBFGSOptimiser(history_size=10, max_grad_norm=None, max_step_norm=None)
        params = np.array([10.0])
        lr = 1.0  # Use lr=1.0 to see pure quasi-Newton effect

        # f(x) = 0.5 * x^2, so grad = x, Hessian = 1, inverse Hessian = 1
        # Newton step: x_new = x - H^-1 * grad = x - 1 * x = 0

        grads = params.copy()  # grad = x for f=0.5*x^2
        params = optim.step(params, grads, lr)

        # First step still behaves like gradient descent before any curvature history exists.
        assert abs(params[0]) < 10.0

        # Second step can exploit the stored curvature pair and should jump close to the optimum.
        grads = params.copy()
        params = optim.step(params, grads, lr)
        assert abs(params[0]) < 0.1

    def test_history_limit(self):
        optim = LBFGSOptimiser(history_size=2, max_grad_norm=None, max_step_norm=None)
        params = np.array([-1.0])  # start negative

        for i in range(5):
            grads = 2 * params  # gradient
            params = optim.step(params, grads, 0.01)

        # History should be limited to 2
        assert len(optim.S) == 2
        assert len(optim.Y) == 2
        assert len(optim.RHO) == 2

    def test_zero_gradients(self):
        optim = LBFGSOptimiser()
        params = np.array([1.0, 2.0])
        grads = np.array([0.0, 0.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # With zero grads, no change
        assert np.allclose(new_params, params)

    def test_default_clipping_limits_first_step_norm(self):
        optim = LBFGSOptimiser()
        params = np.array([1.0, 2.0])
        grads = np.array([2.0, 4.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)
        step = new_params - params

        assert np.isclose(np.linalg.norm(step), lr)

    def test_state_roundtrip_restores_update_trajectory(self):
        params = np.array([1.5, -0.5, 2.0])
        lr = 0.05

        optim = LBFGSOptimiser(history_size=4, eps=1e-12, weight_decay=0.03)
        grads1 = np.array([0.2, -0.4, 0.1])
        params_after_first = optim.step(params, grads1, lr)

        grads2 = np.array([-0.1, 0.3, -0.2])
        params_after_second = optim.step(params_after_first, grads2, lr)

        state = optim.state_dict()
        restored = LBFGSOptimiser()
        restored.load_state_dict(state)

        grads3 = np.array([0.15, -0.2, 0.05])
        expected_next = optim.step(params_after_second, grads3, lr)
        restored_next = restored.step(params_after_second, grads3, lr)

        assert np.allclose(restored_next, expected_next)
