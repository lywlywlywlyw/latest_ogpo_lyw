"""Sequential latent sampling, residual-only objectives and offline MPPCA checks."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent
from ogpo.agents.modules.flax_utils import save_agent
from ogpo.runners import dsrlresidualv3_runner as runner
from ogpo.utils.config_loader import config_to_agent_dict, load_config
from ogpo.utils.datasets import Dataset
from ogpo.utils.main_helper import save_rolling_checkpoint, load_rolling_checkpoint


def config_for_test(top=False):
    config = config_to_agent_dict(load_config('dsrlresidualv3'))
    config.update(actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16),
                  bc_flow_hidden_dims=(16, 16), horizon_length=2, batch_size=4,
                  flow_steps=2, action_chunking=True, use_top_distribution=top,
                  top_distribution_rank=1, top_distribution_max_iter=5,
                  use_base_distribution_in_qdistill=True, use_actor_entropy=True,
                  entropy_backup=True, base_bc_steps=1, q_min=-10., q_max=10.)
    return config


def trees_equal(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_residual_squared_cost_entropy_and_dual_direction():
    class Network:
        def select(self, name):
            if name == 'critic':
                return lambda observations, actions: jnp.zeros((2, len(observations)))
            if name == 'restrain_multiplier':
                return lambda params=None: jnp.float32(1) if params is None else params['multiplier']
            return lambda **kwargs: jnp.float32(1)
    config = dict(q_agg='min', online_log_std_min=None, online_log_std_max=None,
                  use_restrain_loss=True, restrain_threshold=.35,
                  use_actor_entropy=True, target_entropy=-2, noise_dim=2)
    batch = dict(observations=jnp.zeros((1, 3)))
    agent = SimpleNamespace(config=config, actor_network=Network(), critic_network=Network())
    def sample(obs, rng, params, with_log_prob=False):
        residual = params['residual'][None]
        base = jnp.full_like(residual, 100.)
        dist = distrax.MultivariateNormalDiag(jnp.zeros_like(residual),
                                              jnp.full_like(residual, .2))
        result = (base + residual, base, residual, dist)
        return (*result, dist.log_prob(residual)) if with_log_prob else result
    agent._sample_latents = sample
    def loss(params):
        return DSRLResidualV3Agent.actor_loss(agent, batch, params, jax.random.key(0))
    for residual, norm in (([.12, .16], .2), ([.36, .48], .6), ([3., 4.], 5.)):
        params = dict(multiplier=jnp.float32(1), residual=jnp.array(residual))
        (_, info), gradient = jax.value_and_grad(loss, has_aux=True)(params)
        cost = np.sum(np.square(residual)) / 2
        np.testing.assert_allclose(info['restrain_loss'], cost, atol=1e-6)
        np.testing.assert_allclose(gradient['multiplier'], .35 - cost, atol=1e-6)
        np.testing.assert_allclose(gradient['residual'], np.array(residual), atol=1e-6)
        expected_entropy = 2 * (.5 * np.log(2 * np.pi * np.e) + np.log(.2))
        np.testing.assert_allclose(info['entropy'], expected_entropy, atol=1e-6)
        np.testing.assert_allclose(info['actor_loss'], -expected_entropy, atol=1e-6)
        assert not any('kl' in key for key in info)


@pytest.mark.parametrize('top', [False, True])
def test_training_sampling_and_checkpoint(top, tmp_path, monkeypatch):
    config = config_for_test(top)
    obs, action = jnp.zeros(3), jnp.zeros(2)
    fresh = lambda: DSRLResidualV3Agent.create(9, obs, action, config)
    agent = fresh()
    rng = np.random.default_rng(7)
    dataset = Dataset.create(observations=rng.normal(size=(16, 3)).astype('float32'),
        next_observations=rng.normal(size=(16, 3)).astype('float32'),
        actions=rng.normal(size=(16, 2)).astype('float32'),
        rewards=np.zeros(16, dtype='float32'), masks=np.ones(16, dtype='float32'),
        terminals=np.tile([0., 0., 0., 1.], 4))
    logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    neural_before = jax.tree.map(np.array, agent.base_policy_network.params)
    # Exercise the optional flow reset/retraining as well as base fitting.
    monkeypatch.setattr(runner, '_run_evaluation', lambda *args, **kwargs: {'success': 0.})
    monkeypatch.setattr(runner, 'log_memory', lambda *args, **kwargs: None)
    pretrain_config = dict(config, use_invertionencoder_retrainflow=True,
                          invertionencoder_retrainflow_steps=1)
    agent, step = runner.run_residualv3_base_pretraining(
        agent, dataset, pretrain_config, logger, env_name='square-test')
    assert step == 2
    if top:
        trees_equal(neural_before, agent.base_policy_network.params)
        assert not np.allclose(agent.top_distribution.means, 0)
    if top:
        # Verify the actual flow-matching interpolation uses exactly the same
        # frozen MPPCA samples as the standalone inversion training pipeline.
        key = jax.random.key(20)
        _, target_key = jax.random.split(key)
        _, noise_key, time_key = jax.random.split(target_key, 3)
        x0 = agent.top_distribution.distribution((4,)).sample(seed=noise_key)
        t = jax.random.uniform(time_key, (4, 1))
        target_actions = jnp.ones((4, 4))
        def velocity(observations, xt, times, params=None):
            np.testing.assert_allclose(xt, (1 - t) * x0 + t * target_actions, atol=1e-6)
            np.testing.assert_allclose(times, t)
            return target_actions - x0
        surrogate = SimpleNamespace(config=agent.config,
            base_distribution=agent.base_distribution,
            bc_flow_network=SimpleNamespace(select=lambda name: velocity))
        flow_batch = dict(observations=jnp.zeros((4, 3)), actions=jnp.ones((4, 2, 2)),
                          valid=jnp.ones((4, 2)))
        loss, _ = DSRLResidualV3Agent.bc_loss(surrogate, flow_batch, {}, key, use_base_distribution=True)
        np.testing.assert_allclose(loss, 0, atol=1e-6)
    frozen = jax.tree.map(np.array, (agent.base_policy_network.params, agent.top_distribution))
    observations = jnp.zeros((4, 3))
    key = jax.random.key(1)
    z, base_latent, residual, dist = agent._sample_latents(observations, key)
    base_key, residual_key = jax.random.split(key)
    np.testing.assert_allclose(base_latent, agent.base_distribution(observations).sample(seed=base_key))
    conditional = agent.residual_distribution(observations, base_latent)
    np.testing.assert_allclose(residual, conditional.sample(seed=residual_key))
    np.testing.assert_allclose(z, base_latent + residual)
    np.testing.assert_allclose(agent.sample_noise(observations, rng=key), z, atol=1e-6)
    batch = dict(observations=observations, next_observations=observations + .1,
                 true_actions=jnp.zeros((4, 4)), rewards=jnp.ones(4), masks=jnp.ones(4))
    agent, info = agent.batch_update(jax.tree.map(lambda x: x[None], batch))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(info))
    trees_equal(frozen, (agent.base_policy_network.params, agent.top_distribution))
    # A real update must make the conditional residual respond to sampled base inputs.
    a = agent.residual_distribution(observations, jnp.zeros((4, 4))).mean()
    b = agent.residual_distribution(observations, jnp.ones((4, 4))).mean()
    assert not np.allclose(a, b, atol=1e-8)
    for deterministic in (False, True):
        for method in (agent.sample_noise, agent.sample_actions):
            result = method(observations, rng=key, deterministic=deterministic)
            assert result.shape == (4, 4)
            assert np.isfinite(result).all()
    assert agent.sample_actions(obs, rng=key).shape == (4,)
    assert agent.sample_base_actions(observations, rng=key, num_samples=3).shape == (4, 4)
    save_agent(agent, str(tmp_path), 5)
    restored = fresh().restore_actor_params(str(tmp_path / 'params_5.pkl'))
    trees_equal(agent.top_distribution, restored.top_distribution)
    np.testing.assert_allclose(agent.sample_noise(observations, rng=key),
                               restored.sample_noise(observations, rng=key))
    save_rolling_checkpoint(agent, {}, 5, 2, 9, 'test', algo_name='dsrlresidualv3', checkpoint_dir=str(tmp_path))
    restored, _, _, _, _ = load_rolling_checkpoint(9, 'test', fresh(), 'dsrlresidualv3', resume_path=str(tmp_path))
    trees_equal(agent.top_distribution, restored.top_distribution)


def test_config_registration_and_scripts():
    from ogpo.main import AGENTS
    assert AGENTS['dsrlresidualv3'] is DSRLResidualV3Agent
    config = config_to_agent_dict(load_config('dsrlresidualv3'))
    assert config['restrain_threshold'] == .35
    assert not any('kl' in k or k == 'use_one_variance' for k in config)
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'scripts/dsrlresidualv3').glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
        text = path.read_text()
        assert '--algo=dsrlresidualv3' in text
        assert 'restrain_threshold=0.35' in text
        setup = text.split('export CUDA_VISIBLE_DEVICES=')[0]
        command = setup + '\nprintf "%s\\n" "${optional_agent_args[@]}" "$restrain_threshold"'
        out = subprocess.check_output(['bash', '-c', command, str(path),
            '--use_top_distribution=True', '--top_distribution_rank=2',
            '--restrain_threshold=0.5'], text=True)
        assert '--agent.use_top_distribution=True' in out
        assert '--agent.top_distribution_rank=2' in out
        assert out.rstrip().endswith('0.5')
