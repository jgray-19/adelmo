import numpy as np

from adelmo.optimisers.adam import AdamOptimiser


class TestAdamOptimiser:
    def test_initialization(self):
        shape = (10,)
        optim = AdamOptimiser(
            shape=shape, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
        )
        assert optim.beta1 == 0.9
        assert optim.beta2 == 0.999
        assert optim.eps == 1e-8
        assert optim.weight_decay == 0.01
        assert optim.t == 0
        assert np.array_equal(optim.m, np.zeros(shape))
        assert np.array_equal(optim.v, np.zeros(shape))

    def test_bias_correction(self):
        """Test that bias correction works correctly for constant gradients."""
        shape = (1,)
        optim = AdamOptimiser(shape=shape, beta1=0.5, beta2=0.5, eps=0.0)
        params = np.array([1.0])
        grads = np.array([1.0])  # constant gradient
        lr = 0.1

        # After first step
        params = optim.step(params, grads, lr)
        # m = (1-beta1)*grads = 0.5*1 = 0.5
        # v = (1-beta2)*grads^2 = 0.5*1 = 0.5
        # m_hat = m / (1 - beta1^t) = 0.5 / 0.5 = 1
        # v_hat = v / (1 - beta2^t) = 0.5 / 0.5 = 1
        expected_update = lr * 1 / np.sqrt(1)  # 0.1
        assert np.isclose(params[0], 1.0 - expected_update)

        # After second step
        params = optim.step(params, grads, lr)
        # m = beta1*m + (1-beta1)*grads = 0.5*0.5 + 0.5*1 = 0.25 + 0.5 = 0.75
        # v = beta2*v + (1-beta2)*grads^2 = 0.5*0.5 + 0.5*1 = 0.25 + 0.5 = 0.75
        # m_hat = 0.75 / (1 - 0.5^2) = 0.75 / 0.75 = 1
        # v_hat = 0.75 / 0.75 = 1
        expected_update = lr * 1 / np.sqrt(1)  # 0.1
        assert np.isclose(params[0], 0.9 - expected_update)

    def test_weight_decay_application(self):
        """Test that weight decay is applied correctly to gradients."""
        shape = (1,)
        optim = AdamOptimiser(shape=shape, weight_decay=0.1, eps=0.0)
        params = np.array([1.0])
        grads = np.array([0.0])  # zero gradient, only weight decay
        lr = 1.0  # large lr to see effect

        new_params = optim.step(params, grads, lr)

        # Effective gradient = 0 + 0.1 * 1 = 0.1
        # With eps=0, update = lr * 0.1 / sqrt(0.1^2) = 1.0 * 0.1 / 0.1 = 1.0
        # new_params = 1.0 - 1.0 = 0.0
        assert np.isclose(new_params[0], 0.0)

    def test_adam_vs_sgd_on_constant_gradients(self):
        """Test that Adam behaves reasonably compared to SGD on constant gradients."""
        shape = (1,)
        optim = AdamOptimiser(shape=shape, beta1=0.0, beta2=0.0, eps=0.0)  # Like SGD
        params = np.array([1.0])
        grads = np.array([1.0])
        lr = 0.1

        new_params = optim.step(params, grads, lr)

        # With beta1=0, beta2=0, no bias correction needed
        # m = 0*0 + 1*1 = 1, m_hat = 1 / 1 = 1
        # v = 0*0 + 1*1 = 1, v_hat = 1 / 1 = 1
        # update = 0.1 * 1 / sqrt(1) = 0.1
        assert np.isclose(new_params[0], 1.0 - 0.1)

    def test_zero_gradients_no_change(self):
        shape = (2,)
        optim = AdamOptimiser(shape=shape)
        params = np.array([1.0, 2.0])
        grads = np.array([0.0, 0.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # With zero gradients, should not change much initially
        assert np.allclose(new_params, params, atol=1e-6)

    def test_large_eps_stabilizes(self):
        shape = (1,)
        optim = AdamOptimiser(shape=shape, eps=1.0)
        params = np.array([1.0])
        grads = np.array([1.0])
        lr = 0.1

        new_params = optim.step(params, grads, lr)

        # With large eps, denominator is sqrt(v_hat) + 1 ≈ 1 + 1 = 2
        # update = 0.1 * 1 / 2 = 0.05
        assert np.isclose(new_params[0], 1.0 - 0.05, atol=1e-3)

    def test_state_roundtrip_restores_update_trajectory(self):
        shape = (3,)
        params = np.array([1.0, -2.0, 0.5])
        grads1 = np.array([0.2, -0.4, 0.1])
        grads2 = np.array([-0.3, 0.2, 0.05])
        lr = 0.01

        optim = AdamOptimiser(shape=shape, beta1=0.85, beta2=0.95, eps=1e-8, weight_decay=0.02)
        params_after_first = optim.step(params, grads1, lr)

        state = optim.state_dict()
        restored = AdamOptimiser(shape=shape)
        restored.load_state_dict(state)

        expected_next = optim.step(params_after_first, grads2, lr)
        restored_next = restored.step(params_after_first, grads2, lr)

        assert np.allclose(restored_next, expected_next)
