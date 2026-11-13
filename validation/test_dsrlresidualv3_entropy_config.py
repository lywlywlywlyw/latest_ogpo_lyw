"""V3 entropy precedence, CLI wiring, and explicit flow retraining budgets."""
import subprocess
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent
from ogpo.runners import dsrlresidualv3_runner as runner
from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
from ogpo.utils.datasets import Dataset


def small_config(**overrides):
    config = config_to_agent_dict(load_config('dsrlresidualv3'))
    config.update(actor_hidden_dims=(8,), value_hidden_dims=(8,),
                  bc_flow_hidden_dims=(8,), horizon_length=2, flow_steps=2,
                  action_chunking=True)
    config.update(overrides)
    return config


@pytest.mark.parametrize('enabled,low,serl,value,expected', [
    (True, False, False, None, -4.), (True, False, True, None, -2.),
    (True, True, True, None, -8.), (False, True, True, None, -4.),
    (True, False, False, -45, -45.), (True, True, True, -45.5, -45.5),
    (False, True, True, -45, -45.), (True, True, True, 0, 0.),
    (True, False, False, 1.25, 1.25),
])
def test_entropy_precedence(enabled, low, serl, value, expected):
    config = small_config(use_actor_entropy=enabled, use_low_target_entropy=low,
                          use_serl_target_entropy=serl, target_entropy_value=value)
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    assert agent.config['target_entropy'] == expected
    assert config['target_entropy_value'] == value


def test_legacy_config_without_new_key():
    config = small_config(use_actor_entropy=True, use_serl_target_entropy=True)
    config.pop('target_entropy_value')
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    assert agent.config['target_entropy_value'] is None
    assert agent.config['target_entropy'] == -2.


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '-45x'])
def test_invalid_entropy_override(value):
    with pytest.raises(ValueError, match='target_entropy_value'):
        DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
                                  small_config(target_entropy_value=value))


@pytest.mark.parametrize('task', ['square', 'transport', 'toolhang'])
@pytest.mark.parametrize('value,expected', [(None, None), ('null', None), ('None', None), ('-45.5', -45.5)])
def test_launcher_entropy_override(task, value, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/dsrlresidualv3' / (task + '.sh')
    # Expand the actual launcher command but print argv instead of training.
    script = path.read_text().replace('${BASH_SOURCE[0]}', str(path))
    needle = '"$PYTHON_BIN" ogpo/main.py'
    assert script.count(needle) == 1
    script = script.replace(needle, "printf '%s\\n' ogpo/main.py")
    args = [] if value is None else ['--target_entropy_value=' + value]
    out = subprocess.check_output(['bash', '-c', script, str(path), *args], text=True)
    algo, overrides = parse_cli_args([line for line in out.splitlines() if line.startswith('--')])
    assert algo == 'dsrlresidualv3'
    assert overrides['agent.target_entropy_value'] == expected
    assert load_config(algo, cli_overrides=overrides)['agent']['target_entropy_value'] == expected


@pytest.mark.parametrize('budget,expected_updates', [(5001, 5001), (None, 5000)])
def test_retrain_budget_and_success_early_stop(monkeypatch, budget, expected_updates):
    dataset = Dataset.create(observations=np.zeros((4, 3), np.float32),
        next_observations=np.zeros((4, 3), np.float32), actions=np.zeros((4, 2), np.float32),
        terminals=np.array([0., 0., 0., 1.]), rewards=np.zeros(4), masks=np.ones(4))
    calls = []
    class Agent:
        config = {'noise_dim': 4}
        def actions_to_noise(self, observations, actions):
            return actions
        def base_bc_update(self, batch):
            return self, {}
        def reset_flow_for_retraining(self, observations, actions):
            return self
        def bc_update(self, batch, use_base_distribution=False):
            assert use_base_distribution
            calls.append(True)
            return self, {}
        def sample_offline_flow_actions(self, *args, **kwargs):
            raise AssertionError('Evaluation is mocked')
    monkeypatch.setattr(runner, '_run_evaluation', lambda *a, **k: {'success': 1.})
    monkeypatch.setattr(runner, 'save_agent', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'log_memory', lambda *a, **k: None)
    config = dict(batch_size=2, horizon_length=2, action_chunking=True,
                  discount=.99, base_bc_steps=1, use_invertionencoder_retrainflow=True,
                  invertionencoder_retrainflow_steps=budget)
    _, step = runner.run_residualv3_base_pretraining(
        Agent(), dataset, config, SimpleNamespace(log=lambda *a, **k: None),
        env_name='square-mh-low_dim', save_dir='unused', step_data_log_interval=-1)
    assert len(calls) == expected_updates
    assert step == 1 + expected_updates
