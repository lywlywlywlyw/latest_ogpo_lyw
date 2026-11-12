"""Initial residual noise scale and launcher overrides for V3."""
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlresidualv3 import DSRLResidualV3Agent
from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.config_loader import load_config, parse_cli_args
from test_dsrlresidualv3_entropy_config import small_config


@pytest.mark.parametrize('std', [None, .02, .1, 1.])
def test_initial_residual_mean_and_std(std):
    config = small_config()
    if std is None:
        config.pop('residual_init_std')  # Legacy configs retain std=1.
    else:
        config['residual_init_std'] = std
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    observations = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    base = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    distribution = agent.residual_distribution(observations, base)
    np.testing.assert_array_equal(distribution.mean(), np.zeros((4, 4)))
    np.testing.assert_allclose(distribution.stddev(), 1. if std is None else std, rtol=1e-6)


@pytest.mark.parametrize('std', [0, -.1, True, None, float('inf'), float('nan'), 'invalid'])
def test_invalid_initial_std(std):
    with pytest.raises(ValueError, match='residual_init_std'):
        DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
                                  small_config(residual_init_std=std))


def test_initial_std_respects_existing_log_std_bounds():
    agent = DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
        small_config(residual_init_std=.02, online_log_std_min=-3., online_log_std_max=0.))
    distribution = agent.residual_distribution(jnp.zeros((2, 3)), jnp.zeros((2, 4)))
    np.testing.assert_allclose(distribution.stddev(), np.exp(-3.), rtol=1e-6)


def test_restore_keeps_trained_std_instead_of_reinitializing(tmp_path):
    def create(std):
        return DSRLResidualV3Agent.create(1, jnp.zeros(3), jnp.zeros(2),
            small_config(residual_init_std=std, use_normal_base_encoder=True))
    source = create(.1)
    save_agent(source, str(tmp_path), 1)
    restored = create(.02).restore_actor_params(str(tmp_path / 'params_1.pkl'))
    distribution = restored.residual_distribution(jnp.zeros((2, 3)), jnp.zeros((2, 4)))
    np.testing.assert_allclose(distribution.stddev(), .1, rtol=1e-6)


@pytest.mark.parametrize('task', ['square', 'transport', 'toolhang'])
@pytest.mark.parametrize('std', [None, '.02'])
def test_launcher_initial_std(task, std):
    path = Path(__file__).resolve().parents[1] / 'scripts/dsrlresidualv3' / (task + '.sh')
    script = path.read_text().replace('${BASH_SOURCE[0]}', str(path))
    script = script.replace('"$PYTHON_BIN" ogpo/main.py', "printf '%s\\n' ogpo/main.py")
    args = [] if std is None else ['--residual_init_std=' + std]
    out = subprocess.check_output(['bash', '-c', script, str(path), *args], text=True)
    algo, overrides = parse_cli_args([line for line in out.splitlines() if line.startswith('--')])
    expected = 1. if std is None else float(std)
    assert overrides['agent.residual_init_std'] == expected
    assert load_config(algo, cli_overrides=overrides)['agent']['residual_init_std'] == expected
