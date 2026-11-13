"""V3 tanh-bounded residuals, transformed densities, squared costs and CLI."""
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent, _box_safe_step_global
from ogpo.utils.config_loader import load_config, parse_cli_args
from test_dsrlresidualv3_entropy_config import small_config


def make_agent(limit, **kwargs):
    return DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
        small_config(latent_hard_clip=limit, residual_init_std=.02,
                     use_normal_base_encoder=True, **kwargs))


def with_mean(agent, mean):
    params = copy.deepcopy(agent.actor_network.params)
    params['modules_actor']['mean']['bias'] = jnp.asarray(mean, dtype=jnp.float32)
    return agent.replace(actor_network=agent.actor_network.replace(params=params))


@pytest.mark.parametrize('limit', [None, .1, 1.5])
@pytest.mark.parametrize('deterministic', [False, True])
def test_boxed_base_and_shared_alpha_sampling(limit, deterministic):
    agent = with_mean(make_agent(limit), [-2., -.5, .5, 2.])
    obs, key = jnp.zeros((16, 3)), jax.random.key(3)
    base_key, residual_key = jax.random.split(key)
    raw = agent.replace(config=agent.config.copy({'latent_hard_clip': None}))
    base_dist = agent.base_distribution(obs)
    base = base_dist.mean() if deterministic else base_dist.sample(seed=base_key)
    if limit is not None:
        base = jnp.clip(base, -limit, limit)
    gaussian = raw.residual_distribution(obs, base)
    u = gaussian.mode() if deterministic else gaussian.sample(seed=residual_key)
    proposal = u if limit is None else limit * jnp.tanh(u)
    z, actual_base, residual, _ = agent._sample_latents(obs, key, deterministic=deterministic)
    np.testing.assert_allclose(actual_base, base, atol=1e-6)
    np.testing.assert_allclose(z, base + residual, atol=1e-6)
    np.testing.assert_allclose(agent.sample_noise(obs, rng=key, deterministic=deterministic), z, atol=1e-6)
    if limit is None:
        np.testing.assert_allclose(residual, proposal, atol=1e-6)
    else:
        assert np.all(np.abs(z) <= limit + 1e-6)
        ratios = np.asarray(residual / proposal)
        np.testing.assert_allclose(ratios, np.repeat(ratios[:, :1], 4, axis=1), atol=1e-5)
        assert np.all(ratios >= -1e-6) and np.all(ratios <= 1 + 1e-6)
        for latent, alpha in zip(np.asarray(z), ratios[:, 0]):
            if alpha < 1 - 1e-5:
                assert np.any(np.isclose(np.abs(latent), limit, atol=1e-6))


def test_sample_log_prob_includes_tanh_and_scale_jacobians():
    limit = 1.5
    agent = with_mean(make_agent(limit), [-1., -.3, .3, 1.])
    obs, key = jnp.zeros((4, 3)), jax.random.key(5)
    z, base, residual, dist, logp = agent._sample_latents(obs, key, with_log_prob=True)
    _, residual_key = jax.random.split(key)
    gaussian = dist.distribution
    u, raw_logp = gaussian.sample_and_log_prob(seed=residual_key)
    tanh_logdet = 2 * (jnp.log(2.) - u - jax.nn.softplus(-2 * u))
    expected = raw_logp - tanh_logdet.sum(-1) - 4 * jnp.log(limit)
    proposal = limit * jnp.tanh(u)
    np.testing.assert_allclose(logp, expected, atol=1e-5)
    np.testing.assert_allclose(dist.log_prob(proposal), logp, atol=1e-4)
    assert np.all(np.abs(z) <= limit + 1e-6)
    np.testing.assert_allclose(z, base + residual, atol=1e-6)


def test_mode_retains_tanh_gradient_beyond_old_clip_interval():
    agent = make_agent(1.5)
    params = copy.deepcopy(agent.actor_network.params)
    def objective(mean):
        p = copy.deepcopy(params)
        p['modules_actor']['mean']['bias'] = mean
        return agent._sample_latents(jnp.zeros((1, 3)), jax.random.key(0),
                                    params=p, deterministic=True)[0].sum()
    mean = jnp.array([-2., -.3, .3, 2.])
    grad = jax.grad(objective)(mean)
    np.testing.assert_allclose(grad, 1.5 * (1 - jnp.tanh(mean) ** 2), atol=1e-6)
    assert np.all(grad > 0)


def test_actor_and_td_use_same_transformed_residual_and_cost():
    agent = with_mean(make_agent(.5, use_actor_entropy=True, entropy_backup=True),
                      [-.8, -.2, .2, .8])
    obs, key = jnp.zeros((4, 3)), jax.random.key(7)
    z, _, r, _, logp = agent._sample_latents(obs, key, with_log_prob=True)
    class Critic:
        def select(self, name):
            def q(observations, actions):
                np.testing.assert_allclose(actions, z, atol=1e-6)
                return jnp.zeros((2, len(observations)))
            return q
    _, info = agent.replace(critic_network=Critic()).actor_loss(
        {'observations': obs}, agent.actor_network.params, key)
    np.testing.assert_allclose(info['restrain_loss'], (r ** 2).sum(-1).mean() / 4)
    np.testing.assert_allclose(info['entropy'], -logp.mean())
    assert info['residual_tanh_squash_enabled'] == 1
    assert info['entropy_is_proposal'] == 1
    def decode(observations, latent):
        np.testing.assert_allclose(latent, z, atol=1e-6)
        return latent
    td_agent = SimpleNamespace(config=agent.config, _sample_latents=agent._sample_latents,
        noise_to_actions_public=decode, action_critic_network=Critic(), actor_network=agent.actor_network)
    target, next_z, next_logp = DSRLResidualV3Agent._action_td_target(td_agent,
        dict(next_observations=obs, rewards=jnp.zeros(4), masks=jnp.ones(4)), key)
    np.testing.assert_allclose(next_z, z)
    np.testing.assert_allclose(next_logp, logp)
    assert np.isfinite(target).all()
    np.testing.assert_allclose(agent.sample_actions(obs, rng=key),
                               agent.noise_to_actions_public(obs, z), atol=1e-6)


def test_saturated_samples_keep_losses_and_gradients_finite():
    agent = with_mean(make_agent(1.5, use_actor_entropy=True, entropy_backup=True),
                      [-30., -20., 20., 30.])
    batch = {'observations': jnp.zeros((4, 3))}
    (_, info), gradient = jax.value_and_grad(agent.actor_loss, argnums=1, has_aux=True)(
        batch, agent.actor_network.params, jax.random.key(1))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((info, gradient)))


def test_bounded_residual_real_update():
    agent = make_agent(1.5, use_actor_entropy=True, entropy_backup=True)
    obs = jnp.zeros((4, 3))
    batch = dict(observations=obs, next_observations=obs + .1,
                 true_actions=jnp.zeros((4, 4)), rewards=jnp.ones(4), masks=jnp.ones(4))
    agent, info = agent.batch_update(jax.tree.map(lambda x: x[None], batch))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(info))
    assert agent.sample_actions(jnp.zeros(3), rng=jax.random.key(0)).shape == (4,)


@pytest.mark.parametrize('limit', [0, -1, True, float('nan'), float('inf'), 'invalid'])
def test_invalid_clip_limit(limit):
    with pytest.raises(ValueError, match='latent_hard_clip'):
        make_agent(limit)


def test_legacy_config_defaults_to_unsquashed_gaussian():
    config = small_config()
    config.pop('latent_hard_clip')
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    assert agent.config['latent_hard_clip'] is None
    assert isinstance(agent.residual_distribution(jnp.zeros((2, 3)), jnp.zeros((2, 4))),
                      distrax.MultivariateNormalDiag)


@pytest.mark.parametrize('task', ['square', 'transport', 'toolhang'])
@pytest.mark.parametrize('value,expected', [(None, None), ('null', None), ('None', None), ('1.4', 1.4)])
def test_launcher_clip_option(task, value, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/dsrlresidualv3' / (task + '.sh')
    script = path.read_text().replace('${BASH_SOURCE[0]}', str(path))
    script = script.replace('"$PYTHON_BIN" ogpo/main.py', "printf '%s\\n' ogpo/main.py")
    args = [] if value is None else ['--latent_hard_clip=' + value]
    out = subprocess.check_output(['bash', '-c', script, str(path), *args], text=True)
    algo, overrides = parse_cli_args([line for line in out.splitlines() if line.startswith('--')])
    assert overrides['agent.latent_hard_clip'] == expected
    assert load_config(algo, cli_overrides=overrides)['agent']['latent_hard_clip'] == expected


@pytest.mark.parametrize('base,proposal,expected,alpha', [
    ([1.4, 0.], [1., 1.], [1.5, .1], .1),
    ([-1.4, 0.], [-1., 1.], [-1.5, .1], .1),
    ([1.5, 0.], [.2, .4], [1.5, 0.], 0.),
    ([0., 0.], [0., 0.], [0., 0.], 1.),
    ([2., 0.], [-.3, .5], [1.2, .5], 1.),
])
def test_lpds_projection_examples(base, proposal, expected, alpha):
    latent, actual_alpha = _box_safe_step_global(jnp.array(base), jnp.array(proposal), 1.5)
    np.testing.assert_allclose(latent, expected, atol=1e-6)
    np.testing.assert_allclose(actual_alpha, [alpha], atol=1e-6)


def test_prior_distillation_uses_the_same_box():
    agent = make_agent(.5)
    captured = []
    def decode(observations, latent):
        assert np.all(np.abs(latent) <= .5)
        captured.append(np.asarray(latent))
        return latent
    class Critic:
        def select(self, name):
            def q(observations, actions, params=None):
                np.testing.assert_array_equal(actions, captured[0])
                return jnp.zeros((2, len(observations)))
            return q
    surrogate = SimpleNamespace(config=agent.config, noise_to_actions_public=decode,
                                action_critic_network=Critic(), critic_network=Critic())
    loss, _ = DSRLResidualV3Agent.critic_loss(surrogate,
        {'observations': jnp.zeros((32, 3))}, {}, jax.random.key(0))
    assert loss == 0
    assert np.any(np.abs(captured[0]) == .5)


def test_base_only_and_normal_sampling_use_the_box():
    agent = make_agent(.5)
    obs, key = jnp.zeros((16, 3)), jax.random.key(9)
    _, base_key = jax.random.split(key)
    base = jnp.clip(agent.base_distribution(obs).sample(seed=base_key), -.5, .5)
    np.testing.assert_allclose(agent.sample_base_actions(obs, rng=key),
                               agent.noise_to_actions_public(obs, base), atol=1e-6)
    normal = jnp.clip(jax.random.normal(key, (16, 4)), -.5, .5)
    np.testing.assert_allclose(agent.sample_actions_normal(obs, rng=key),
                               agent.compute_flow_actions(obs, normal), atol=1e-6)
