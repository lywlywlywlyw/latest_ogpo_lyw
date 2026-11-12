"""Numerical checks for the offline TOPS source and JIT sampling contract."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ogpo.agents.modules.mppca import fit_mppca
from ogpo.agents.modules.pg_helper import sample_flow_actions_sde


def test_ppca_recovers_correlated_covariance():
    rng = np.random.default_rng(7)
    loading = np.array([[2.0], [1.0], [-1.0], [0.5]])
    x = rng.normal(size=(12000, 1)) @ loading.T + .3 * rng.normal(size=(12000, 4))
    model, info = fit_mppca(x, components=1, rank=1, source_std=1.0)
    covariance = np.asarray(model.scale_tril[0] @ model.scale_tril[0].T)
    expected = loading @ loading.T + .09 * np.eye(4)
    expected /= np.trace(expected) / 4
    np.testing.assert_allclose(covariance, expected, atol=.025)
    assert info['mppca_rank'] == 1
    np.testing.assert_allclose(np.trace(covariance) / 4, 1, atol=1e-5)


def test_mixture_jit_sampling_density_and_sde():
    rng = np.random.default_rng(8)
    x = np.concatenate([rng.normal(-2, .2, (1000, 4)), rng.normal(2, .3, (1000, 4))])
    model, _ = fit_mppca(x, rank=1)
    dist = model.distribution()
    samples = jax.jit(lambda key: dist.sample(seed=key, sample_shape=(30000,)))(jax.random.key(0))
    weights, means, factors = map(np.asarray, (model.weights, model.means, model.scale_tril))
    center = weights @ means
    covariance = sum(w * (l @ l.T + np.outer(m-center, m-center))
                     for w, m, l in zip(weights, means, factors))
    np.testing.assert_allclose(samples.mean(0), center, atol=.035)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=.04)
    values = np.asarray(samples[:8])
    expected = logsumexp(np.stack([np.log(w) + multivariate_normal.logpdf(values, m, l @ l.T)
                                  for w, m, l in zip(weights, means, factors)]), axis=0)
    np.testing.assert_allclose(dist.log_prob(values), expected, atol=1e-4)
    assert model.distribution((3, 2)).sample(seed=jax.random.key(1), sample_shape=(10,)).shape == (10, 3, 2, 4)
    # With zero velocity and a deterministic final step, a one-step SDE retains
    # the exact source density. This catches diagonal-Gaussian substitutions.
    @jax.jit
    def sample(key):
        return sample_flow_actions_sde(
            actor_fn=lambda obs, a, t, **kw: jnp.zeros_like(a), noise_fn=None,
            observations=jnp.zeros((8, 3)), rng=key, flow_steps=1, act_dim=4,
            use_constant_noise=True, clip_intermediate=False,
            base_dist=model.distribution((8,)))
    _, chains, logprob, _ = sample(jax.random.key(2))
    np.testing.assert_allclose(logprob, dist.log_prob(chains[:, 0]), atol=1e-5)


def test_small_dimension_and_degenerate_data():
    for x in (np.ones((20, 1)), np.ones((20, 3))):
        model, info = fit_mppca(x)
        assert info['mppca_rank'] == x.shape[-1] - 1
        assert np.isfinite(model.distribution().log_prob(x)).all()
        assert np.all(np.asarray(model.residual_variances) > 0)
