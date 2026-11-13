"""V3 final latent clipping: composition, gradients, policy consumers and CLI."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent
from ogpo.utils.config_loader import load_config, parse_cli_args
from test_dsrlresidualv3_entropy_config import small_config


def composed_agent(limit, residual):
    base = jnp.array([[-2., -.5, .5, 2.]])
    base_dist = SimpleNamespace(mean=lambda: base, sample=lambda seed: base)
    dist = SimpleNamespace(mean=lambda: residual, sample=lambda seed: residual)
    return SimpleNamespace(config={'latent_hard_clip': limit},
        base_distribution=lambda obs: base_dist,
        residual_distribution=lambda obs, base_latent, params=None: dist)


@pytest.mark.parametrize('deterministic', [False, True])
@pytest.mark.parametrize('limit,expected', [
    (None, [[-.5, -1.5, 1.5, .5]]),
    (1., [[-.5, -1., 1., .5]]),
    (.5, [[-.5, -.5, .5, .5]]),
    (0., [[0., 0., 0., 0.]]),
])
def test_clip_sum_elementwise_including_boundaries(limit, expected, deterministic):
    residual = jnp.array([[1.5, -1., 1., -1.5]])
    agent = composed_agent(limit, residual)
    z, base, r, _ = DSRLResidualV3Agent._sample_latents(
        agent, jnp.zeros((1, 3)), jax.random.key(1), deterministic=deterministic)
    np.testing.assert_array_equal(z, expected)
    np.testing.assert_array_equal(base, [[-2., -.5, .5, 2.]])
    np.testing.assert_array_equal(r, residual)


def test_clip_gradients_inside_and_outside_interval():
    def sample(residual):
        agent = composed_agent(1., residual)
        return DSRLResidualV3Agent._sample_latents(
            agent, jnp.zeros((1, 3)), jax.random.key(1))[0].sum()
    # Final sum is [-2, -.25, .25, 2]: inside dimensions retain gradients.
    gradient = jax.grad(sample)(jnp.array([[0., .25, -.25, 0.]]))
    np.testing.assert_array_equal(gradient, [[0., 1., 1., 0.]])


def test_jitted_sampling_and_training_use_same_clipped_latent():
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
        small_config(latent_hard_clip=.1, use_normal_base_encoder=True,
                     use_actor_entropy=False, entropy_backup=False))
    obs = jnp.zeros((16, 3))
    key = jax.random.key(3)
    z, _, r, _ = agent._sample_latents(obs, key)
    assert np.any(np.abs(agent.replace(config=agent.config.copy({'latent_hard_clip': None}))
                         .sample_noise(obs, rng=key)) > .1)
    np.testing.assert_allclose(agent.sample_noise(obs, rng=key), z, atol=1e-6)
    assert np.all(np.abs(z) <= .1)
    np.testing.assert_allclose(agent.sample_actions(obs, rng=key),
                               agent.noise_to_actions_public(obs, z), atol=1e-6)
    assert agent.sample_noise(jnp.zeros(3), rng=key).shape == (4,)
    np.testing.assert_array_equal(agent.sample_noise(obs, rng=key, deterministic=True),
                                 np.zeros((16, 4)))

    class Critic:
        def select(self, name):
            def q(observations, actions):
                np.testing.assert_array_equal(actions, z)
                return jnp.zeros((2, len(observations)))
            return q
    loss_agent = agent.replace(critic_network=Critic())
    _, info = loss_agent.actor_loss({'observations': obs}, agent.actor_network.params, key)
    np.testing.assert_allclose(info['restrain_loss'], jnp.linalg.norm(r, axis=-1).mean())

    def decode(observations, latent):
        np.testing.assert_array_equal(latent, z)
        return latent
    td_agent = SimpleNamespace(config=agent.config, _sample_latents=agent._sample_latents,
                               noise_to_actions_public=decode, action_critic_network=Critic())
    _, next_z, _ = DSRLResidualV3Agent._action_td_target(td_agent,
        dict(next_observations=obs, rewards=jnp.zeros(16), masks=jnp.ones(16)), key)
    np.testing.assert_array_equal(next_z, z)


@pytest.mark.parametrize('limit', [-1, True, float('nan'), float('inf'), 'invalid'])
def test_invalid_clip_limit(limit):
    with pytest.raises(ValueError, match='latent_hard_clip'):
        DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
                                  small_config(latent_hard_clip=limit))


def test_legacy_config_defaults_to_no_clip():
    config = small_config()
    config.pop('latent_hard_clip')
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    assert agent.config['latent_hard_clip'] is None


@pytest.mark.parametrize('task', ['square', 'transport', 'toolhang'])
@pytest.mark.parametrize('value,expected', [(None, None), ('null', None), ('None', None), ('1.4', 1.4), ('0', 0)])
def test_launcher_clip_option(task, value, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/dsrlresidualv3' / (task + '.sh')
    script = path.read_text().replace('${BASH_SOURCE[0]}', str(path))
    script = script.replace('"$PYTHON_BIN" ogpo/main.py', "printf '%s\\n' ogpo/main.py")
    args = [] if value is None else ['--latent_hard_clip=' + value]
    out = subprocess.check_output(['bash', '-c', script, str(path), *args], text=True)
    algo, overrides = parse_cli_args([line for line in out.splitlines() if line.startswith('--')])
    assert overrides['agent.latent_hard_clip'] == expected
    assert load_config(algo, cli_overrides=overrides)['agent']['latent_hard_clip'] == expected
