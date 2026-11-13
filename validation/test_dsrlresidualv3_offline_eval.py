"""V3 retrain evaluation matches the standalone inversion ODE metric."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent
from ogpo.agents.modules.mppca import MPPCA
from ogpo.runners import dsrlresidualv3_runner as runner
from ogpo.utils.config_loader import AttrDict
from ogpo.utils.datasets import Dataset
from test_dsrlresidualv3_entropy_config import small_config


@pytest.mark.parametrize('top', [False, True])
def test_matches_standalone_eval_success_ode(top):
    from ogpo.train_invertion_offline import InvertionOfflineAgent
    config = small_config(use_top_distribution=top, top_distribution_rank=1,
                          use_normal_base_encoder=False, latent_hard_clip=.1)
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    if top:
        # Nontrivial task-level mixture, deliberately extending beyond the
        # online box so accidentally using online sampling is detectable.
        agent = agent.replace(top_distribution=MPPCA(
            weights=jnp.array([.3, .7]), means=jnp.array([[-2.] * 4, [2.] * 4]),
            loadings=jnp.zeros((2, 4, 1)), residual_variances=jnp.array([.04, .04]),
            scale_tril=jnp.broadcast_to(.2 * jnp.eye(4), (2, 4, 4))))
    obs, key = jnp.ones((4, 3)), jax.random.key(9)
    source_noise = agent.base_distribution(obs).sample(seed=key)
    assert np.any(np.abs(source_noise) > .1)
    actual = agent.sample_offline_flow_actions(obs, rng=key)

    # Execute the actual standalone implementation with the same source and
    # current flow. Its evaluation template's actor selects that current flow.
    class Flow:
        def select(self, name):
            assert name == 'actor'
            return agent.bc_flow_network.select('actor_bc_flow')
    policy = SimpleNamespace(config=AttrDict(dict(agent.config)), actor_network=Flow(),
        _encode_observations_for_flow=lambda x, is_encoded, images=None: x)
    standalone = SimpleNamespace(base_distribution=agent.base_distribution,
                                 _evaluation_policy=lambda: policy)
    expected = InvertionOfflineAgent.compute_flow_actions.__wrapped__(standalone, obs, rng=key)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual, agent.compute_flow_actions(obs, noises=source_noise), atol=1e-6)
    boxed = agent.compute_flow_actions(obs, noises=jnp.clip(source_noise, -.1, .1))
    assert not np.allclose(actual, boxed, atol=1e-6)

    # A changed residual actor and critics cannot affect this offline policy.
    params = jax.tree.map(lambda x: x, agent.actor_network.params)
    params['modules_actor']['mean']['bias'] = jnp.full(4, 100.)
    changed = agent.replace(actor_network=agent.actor_network.replace(params=params),
                            critic_network=None, action_critic_network=None)
    np.testing.assert_array_equal(changed.sample_offline_flow_actions(obs, rng=key), actual)


def test_offline_policy_accepts_single_observation():
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
        small_config(use_normal_base_encoder=False, latent_hard_clip=.1))
    obs, key = jnp.ones(3), jax.random.key(7)
    expected = agent.compute_flow_actions(obs, noises=agent.base_distribution(obs).sample(seed=key))
    actual = agent.sample_offline_flow_actions(obs, rng=key)
    assert actual.shape == (4,)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('budget,expected_updates', [(None, 10000), (10001, 10001)])
def test_logged_offline_success_and_early_stop_share_single_ode_evaluation(monkeypatch, budget, expected_updates):
    data = Dataset.create(observations=np.zeros((4, 3), np.float32),
        next_observations=np.zeros((4, 3), np.float32), actions=np.zeros((4, 2), np.float32),
        terminals=np.array([0., 0., 0., 1.]), rewards=np.zeros(4), masks=np.ones(4))
    calls, events, evaluations, saved = [], [], [], []
    class Agent:
        config = {'noise_dim': 4}
        def actions_to_noise(self, observations, actions):
            return actions
        def base_bc_update(self, batch):
            return self, {}
        def reset_flow_for_retraining(self, observations, actions):
            return self
        def bc_update(self, batch, use_base_distribution=False):
            calls.append(True)
            return self, {}
        def sample_offline_flow_actions(self, observations, rng=None):
            return jnp.zeros((len(observations), 4))
        def sample_base_actions(self, *a, **k):
            raise AssertionError('Best-of-N/base sampler must not be used')
        def sample_actions(self, *a, **k):
            raise AssertionError('Online residual policy must not be used')
    def evaluate(agent, env, envs, flags, action_dim, actor_fn, step, *a, **k):
        assert actor_fn.__self__ is agent
        assert actor_fn.__name__ == 'sample_offline_flow_actions'
        assert actor_fn(jnp.zeros((2, 3)), rng=jax.random.key(1)).shape == (2, 4)
        evaluations.append(step)
        return {'success': .25 if step < 10000 else .5, 'return': -123.}
    monkeypatch.setattr(runner, '_run_evaluation', evaluate)
    monkeypatch.setattr(runner, 'save_agent', lambda *a: saved.append(a[2]))
    monkeypatch.setattr(runner, 'log_memory', lambda *a, **k: None)
    config = dict(batch_size=2, horizon_length=2, action_chunking=True, discount=.99,
                  base_bc_steps=1, use_invertionencoder_retrainflow=True,
                  invertionencoder_retrainflow_steps=budget)
    _, final_step = runner.run_residualv3_base_pretraining(
        Agent(), data, config,
        SimpleNamespace(log=lambda info, prefix, step: events.append((dict(info), prefix, step))),
        start_step=123, env_name='square-mh-low_dim', save_dir='unused', step_data_log_interval=-1)
    assert len(calls) == expected_updates
    assert final_step == 124 + expected_updates
    expected_evals = [5000, 10000] if budget is None else [5000, 10000, 10001]
    assert evaluations == expected_evals  # Standalone retrain-local evaluation seeds.
    rows = [(info, step) for info, prefix, step in events if prefix == 'offline_eval']
    assert [info['success'] for info, _ in rows] == [.25] + [.5] * (len(rows) - 1)
    assert [info['retrain_step'] for info, _ in rows] == expected_evals
    assert [step for _, step in rows] == [124 + i for i in expected_evals]
    assert all(info['return'] == -123. for info, _ in rows)
    assert saved == ([10124] if budget is None else [])
